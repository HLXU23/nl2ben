import logging
from collections import deque
from typing import Dict, List, Any

import re
import os
import json
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

    for table_name in topo_order:
        foreign_key_value_prompt = get_foreign_key_value_prompt(schema[table_name]['foreign_keys'], foreign_key_value)
        example_row_prompt = get_example_row_prompt(db_path, table_name, example_value_num)
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

        with open(os.path.join(output_path, f'{db_name}_{table_name}.txt'), "w") as file:
            file.write(prompt)
            file.write('\n====================\n')
            file.write(response)

        write_response_into_db(table_name, response, db_new_path)
        logging.info(f'Table {table_name} new data added')
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

def get_foreign_key_value_prompt(foreign_keys, foreign_key_values):
    if not foreign_keys: # No foreign keys in this table
        return ''
    foreign_key_value_prompt = "\n\nFor foreign key, you can ONLY use values given in referenced tables as follow:\n"
    for fk in foreign_keys:
        column = fk.get('column')
        referenced_table = fk.get('referenced_table')
        referenced_column = fk.get('referenced_column')
        key_list = foreign_key_values[referenced_table][referenced_column]
        foreign_key_value_prompt += f'{referenced_table}.{referenced_column}: {', '.join(key_list)}\n'
    return foreign_key_value_prompt

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
    
    first_row_data = rows[0].split("\n")[1:]
    columns = [re.split(r":\s*", line)[0] for line in first_row_data]

    for row in rows:
        row_data = row.split("\n")[1:]
        values = [re.split(r":\s*", line, 1)[1] for line in row_data]
        values = [v.strip(' "') for v in values]
        
        placeholders = ", ".join(["?" for _ in columns])
        sql = f'INSERT INTO "{table_name}" ({', '.join([f'"{column}"' for column in columns])}) VALUES ({placeholders})'
        
        cursor.execute(sql, values)
    
    conn.commit()
    conn.close()

def write_referenced_col_value(response, referenced_cols):
    referenced_col_value = dict.fromkeys(referenced_cols, [])
    for referenced_col in referenced_cols:
        pattern = re.compile(rf'^{re.escape(referenced_col)}:\s*(\S+)', re.MULTILINE)
        referenced_col_value[referenced_col] = pattern.findall(response)
    return referenced_col_value

