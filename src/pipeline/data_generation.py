import logging
from collections import deque
from typing import Dict, List, Any

import re
import os
import json
import random
import shutil
import sqlite3

from llm.models import async_llm_call

def data_generation(db_name: str, config: Dict[str, Any]):
    """
    Generate database data

    Args:
        db_name (str): database name
    """
    input_path = config['input_path']
    template_path = config['template_path']
    output_path = config['output_path']
    result_path = config['result_path']
    epoch = config['epoch']
    row_num = config['row_num']
    example_value_num = config['example_value_num']

    logging.info('Starting database data generation')

    try:
        schema_path = os.path.join(input_path, f'schema_preprocess/schema_{db_name}.json')
        with open(schema_path, "r") as file:
            schema = json.load(file)
        logging.debug(f'{db_name} SCHEMA loaded')
    except Exception as e:
        logging.error(f'Error loading database schema {db_name}: {e}')
        raise

    try:
        referenced_cols_path = os.path.join(input_path, f'schema_preprocess/referenced_cols_{db_name}.json')
        with open(referenced_cols_path, "r") as file:
            referenced_cols = json.load(file)
        logging.debug(f'{db_name} REFERENCED COLS got')
    except Exception as e:
        logging.error(f'Error loading referenced columns {db_name}: {e}')
        raise

    try:
        schema_prompt_path = os.path.join(input_path, f'schema_preprocess/schema_prompt_{db_name}')
        schema_prompt = load_schema_prompt(schema, schema_prompt_path)
        logging.debug(f'{db_name} SCHEMA PROMPT loaded')
    except Exception as e:
        logging.error(f'Error loading database schema prompt {db_name}: {e}')
        raise

    try:
        with open(template_path, "r") as file:
            logging.debug('Template found')
            template = file.read()
    except Exception as e:
        logging.error(f"Error loading template {template_path}: {e}")
        raise

    db_path = os.path.join(input_path, f'schema_generation/{db_name}.sqlite')
    db_new_path = os.path.join(result_path, f'{db_name}.sqlite')
    shutil.copy(db_path, db_new_path)
    
    topo_order = get_topo_order(schema)
    logging.debug(topo_order)

    foreign_key_value = dict.fromkeys(schema, [])
    value_generated = dict.fromkeys(schema, [])
    for idx in range(epoch):
        for table_name in topo_order:
            foreign_key_value_prompt = get_foreign_key_value_prompt(schema[table_name]['foreign_keys'], foreign_key_value, row_num)
            example_row_prompt = get_generated_row_prompt(value_generated[table_name], example_value_num) if value_generated[table_name] else get_example_row_prompt(db_new_path, table_name, example_value_num)
            prompt = template.replace('{ROW_NUM}', str(row_num)) \
                            .replace('{SCHEMA}', schema_prompt[table_name]) \
                            .replace('{FOREIGN_KEY_VALUE}', foreign_key_value_prompt) \
                            .replace('{EXAMPLE_ROW}', example_row_prompt) \
                            .replace('{EXAMPLE_FORMAT}', "\n".join([f'{col_name}: {{row1.{col_name}}}' for col_name in schema[table_name]['attributes']]))
            request_kwargs = f'[Data Generation]{db_name}{table_name}'

            response = async_llm_call(
                prompt=prompt,
                config=config,
                request_list=[request_kwargs],
                step="data_generation",
                sampling_count=1
            )[0][0]

            with open(os.path.join(output_path, f'{db_name}_{idx}_{table_name}.txt'), "w") as file:
                file.write(prompt)
                file.write('\n====================\n')
                file.write(response)

            executed_sql, new_row_cnt, new_value_generated = write_response_into_db(table_name, response, db_new_path)

            with open(os.path.join(output_path, f'sql_{db_name}_{table_name}.txt'), "a") as file:
                file.write(executed_sql)
            logging.info(f'Table {table_name} new {new_row_cnt} rows data added')

            value_generated[table_name] += new_value_generated

            if table_name in referenced_cols: # If table is referenced, record value of referenced cols
                foreign_key_value[table_name] = write_referenced_col_value(response, referenced_cols[table_name])
            logging.debug(f'Foreign key value: \n{foreign_key_value}')
        

def load_schema_prompt(schema, schema_prompt_path):
    schema_prompt = dict.fromkeys(schema)
    for table_name in schema:
        with open(os.path.join(schema_prompt_path, f'{table_name}.txt')) as file:
            schema_prompt[table_name] = file.read()
    return schema_prompt
    
def get_topo_order(schema):

    graph = dict.fromkeys(schema, [])
    in_degree = dict.fromkeys(schema, 0)

    for table_name in schema:
        foreign_keys = schema[table_name]['foreign_keys']
        for fk in foreign_keys:
            referenced_table = fk['referenced_table']
            graph[referenced_table].append(table_name)
            in_degree[table_name] += 1
    
    queue = deque([table for table in in_degree if in_degree[table] == 0])
    topo_order = []
    
    while queue:
        table = queue.popleft()
        # logging.debug(table)
        # logging.debug(queue)
        topo_order.append(table)
        
        for dependent_table in graph[table]:
            in_degree[dependent_table] -= 1
            if in_degree[dependent_table] == 0:
                queue.append(dependent_table)
    
    if len(topo_order) != len(schema):
        logging.debug(f'Wrong topo order: {", ".join(topo_order)}')
        raise ValueError("Found foreign key dependence cycle in schema")
    
    return topo_order

def get_foreign_key_value_prompt(foreign_keys, foreign_key_values, row_num):
    if not foreign_keys: # No foreign keys in this table
        return ''
    foreign_key_value_prompt = "\n\nFor foreign key, you can ONLY use values given in referenced tables as follow:\n"
    for fk in foreign_keys:
        column = fk.get('column')
        referenced_table = fk.get('referenced_table')
        referenced_column = fk.get('referenced_column')
        key_list = foreign_key_values[referenced_table][referenced_column]
        key_list_subset = random_select(key_list, row_num)
        foreign_key_value_prompt += f'{referenced_table}.{referenced_column}: {', '.join(key_list_subset)}\n'
    return foreign_key_value_prompt

def random_select(list, n):
    return random.sample(list, n) if n < len(list) else list

def get_generated_row_prompt(value_generated, example_value_num):
    generated_row_prompt = '\n\nHere are some newest data in database, you can check their formats and values for reference.\n'
    generated_value_num = min(len(value_generated), example_value_num)
    for i in range(generated_value_num):
        generated_row_prompt += f"{i + 1}\n"
        for col_value in value_generated[(i - example_value_num)]:
            generated_row_prompt += f'{col_value[0]}: {col_value[1]}\n'
        generated_row_prompt += '\n'
    
    return generated_row_prompt
    

def get_example_row_prompt(db_path, table_name, example_value_num):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"SELECT * FROM {table_name} LIMIT {example_value_num}")
    rows = cursor.fetchall()

    if not rows:
        return ''

    example_row_prompt = '\n\nHere are some existed data in database, you can check their formats and values for reference.\n'

    col_names = [description[0] for description in cursor.description]

    for i, row in enumerate(rows, start=1):
        example_row_prompt += f"{i}\n"
        for col_name, value in zip(col_names, row):
            example_row_prompt +=  f"{col_name}: {value}\n"
        example_row_prompt += '\n'

    conn.close()

    return example_row_prompt

def write_response_into_db(table_name, response, db_new_path):
    conn = sqlite3.connect(db_new_path)
    cursor = conn.cursor()

    rows = response.strip().split("\n\n")

    executed_sql = ''
    new_row_cnt = 0
    new_value_generated = []
    for row in rows:
        row_data = row.split("\n")[1:]
        
        try:
            col_values = [line.split(': ', 1) for line in row_data]
            placeholders = ", ".join(["?" for _ in col_values])
            sql = f'INSERT INTO `{table_name}` ({', '.join([f'`{col_value[0]}`' for col_value in col_values])}) VALUES ({placeholders})'
            values = [col_value[1] for col_value in col_values]
        except: 
            continue
        executed_sql += f'{sql}\n'
        executed_sql += f'VALUE: {', '.join(values)}\n'
        try:
            logging.debug(f'Run data insertion SQL command:\n{sql}\nVALUE: {values}')
            cursor.execute(sql, values)
            new_row_cnt += 1
            executed_sql += f'Exec success\n'
            new_value_generated.append(col_values)
        except:
            executed_sql += f'Exec failed\n'
            pass
    
    conn.commit()
    conn.close()
    return executed_sql, new_row_cnt, new_value_generated

def write_referenced_col_value(response, referenced_cols):
    referenced_col_value = dict.fromkeys(referenced_cols, [])
    for referenced_col in referenced_cols:
        pattern = re.compile(rf'^{re.escape(referenced_col)}:\s*(\S+)', re.MULTILINE)
        referenced_col_value[referenced_col] = pattern.findall(response)
    return referenced_col_value

