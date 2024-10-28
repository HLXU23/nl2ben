import logging
from collections import deque
from typing import Dict, List, Any

import re
import os
import json
import random
import shutil
import sqlite3
import asyncio

from llm.models import batch_llm_call_async

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
        schema_path = os.path.join(input_path, f'data_rule_generation/schema_{db_name}.json')
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

    db_source = os.path.join(input_path, f'schema_generation/{db_name}.sqlite')
    db_path = os.path.join(result_path, f'{db_name}.sqlite')
    shutil.copy(db_source, db_path)
    
    topo_order = get_topo_order(schema)
    logging.debug(topo_order)
    
    new_row_cnt = dict.fromkeys(schema, 0)
    for epoch_idx in range(epoch):
        for table_name in topo_order:

            output_ai_path = os.path.join(output_path, f'{db_name}_{epoch_idx}_{table_name}.txt')
            if os.path.exists(output_ai_path):
                os.remove(output_ai_path)
            output_sql_path = os.path.join(output_path, f'sql_{db_name}_{epoch_idx}_{table_name}.txt')
            if os.path.exists(output_sql_path):
                os.remove(output_sql_path)

            prompts = []
            try:
                new_rows = get_k_new_rows(row_num, db_path, table_name, schema[table_name])
            except Exception as e:
                logging.error(f'Generating {table_name} data fail: {e}')
                raise
            for i in range(len(new_rows)):
                example_row_prompt = get_example_row_prompt(db_path, table_name, example_value_num)
                prompts += [template.replace('{SCHEMA}', schema_prompt[table_name]) \
                                 .replace('{EXAMPLE_ROW}', example_row_prompt) \
                                 .replace('{GENERATED_VALUE}', ''.join([f'{col_name}: {str(new_rows[i][col_name])}\n' for col_name in new_rows[i]]))]
            step = f'[Data Generation]{db_name}{table_name}'

            responses = asyncio.run(batch_llm_call_async(
                prompts=prompts,
                config=config,
                step=step
            ))

            for i in range(len(prompts)):
                prompt = prompts[i]
                response = responses[i][0]
                with open(output_ai_path, "a", encoding="utf-8") as file:
                    file.write('\n====================\n')
                    file.write('Human: \n')
                    file.write(prompt)
                    file.write('\n====================\n')
                    file.write('AI: \n')
                    file.write(response)

                col_values, executed_sql = write_response_into_db(table_name, response, db_path)
                with open(output_sql_path, "a") as file:
                    file.write(executed_sql)
                if col_values:
                    logging.info(f'Generation success.')
                    new_row_cnt[table_name] += 1
                else:
                    logging.info(f'Generation fail.')
            
    for table_name in schema:
        logging.info(f'New {new_row_cnt[table_name]} rows data added into table {table_name}') 
        

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
        for column in foreign_keys:
            referenced_table = foreign_keys[column]['referenced_table']
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

def get_k_new_rows(k, db_path, table_name, schema_table):
    new_rows = []
    for i in range(k):
        new_row = dict.fromkeys(schema_table['attributes'], '')
        for col_name in new_row:
            col_type = schema_table['attributes'][col_name]['type']
            if 'identifier' in schema_table['attributes'][col_name]:
                if col_name in schema_table['foreign_keys']:
                    try:
                        referenced_table = schema_table['foreign_keys'][col_name]['referenced_table']
                        referenced_col = schema_table['foreign_keys'][col_name]['referenced_column']
                        get_foreign_identifier(db_path, table_name, col_name, referenced_table, referenced_col)
                    except Exception as e:
                        logging.error(f'get new foreign identifier error: {e}')
                        return new_rows
                try:
                    new_row[col_name] = get_new_identifier(db_path, table_name, col_name) + i
                except Exception as e:
                    logging.error(f'get new primary identifier error: {e}')
                    return new_rows
            elif col_name in schema_table['foreign_keys']:
                try:
                    random_value = get_random_value(db_path, table_name, col_name)
                except Exception as e:
                    logging.error(f'get random value error: {e}')
                if random_value:
                    new_row[col_name] = random_value
                    continue
            elif col_type == 'BOOLEAN':
                new_row[col_name] = random.choice([True, False])
            elif 'categorical' in schema_table['attributes'][col_name]:
                new_row[col_name] = random.choice(schema_table['attributes'][col_name]['categorical'])
            elif 'numerical' in schema_table['attributes'][col_name]:
                l, u = schema_table['attributes'][col_name]['numerical']
                temp = random.uniform(l, u)
                if col_type in ['INT', 'INTEGER']:
                    new_row[col_name] = int(temp)
                else:
                    new_row[col_name] = temp
        new_rows.append(new_row)
    return new_rows

def get_max_value(db_path, table_name, col_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = f"SELECT MAX({col_name}) FROM {table_name}"
    cursor.execute(query)
    max_value = cursor.fetchone()[0]
    conn.close()
    return max_value

def get_min_value(db_path, table_name, col_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = f"SELECT MIN({col_name}) FROM {table_name}"
    cursor.execute(query)
    min_value = cursor.fetchone()[0]
    conn.close()
    return min_value

def get_foreign_identifier(db_path, table_name, col_name, referenced_table, referenced_col):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = f"""
    SELECT {referenced_col} 
    FROM {referenced_table} 
    WHERE {referenced_col} NOT IN (
        SELECT {col_name} 
        FROM {table_name}
    )
    """
    cursor.execute(query)
    results = [row[0] for row in cursor.fetchall()]
    conn.close()
    if results:
        return random.choice(results)
    else:
        return None

def get_new_identifier(db_path, table_name, col_name):
    max_value = get_max_value(db_path, table_name, col_name)
    return ((max_value) or 10000000) + 1

def get_random_value(db_path, table_name, col_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = f"SELECT DISTINCT `{col_name}` FROM `{table_name}`"
    cursor.execute(query)
    values = [row[0] for row in cursor.fetchall()]
    conn.close()
    if values:
        return random.choice(values)
    else:
        return None

def random_select(list, n):
    return random.sample(list, n) if n < len(list) else list
    

def get_example_row_prompt(db_path, table_name, example_value_num):

    example_rows = get_example_row(db_path, table_name, example_value_num)
    if not example_rows:
        return ''

    example_row_prompt = '\n\nHere are some existed data in database, you can check their formats and values for reference.\n'
    for idx, example_row in enumerate(example_rows, start=1):
        example_row_prompt += f"{idx}\n"
        for col_name in example_row:
            example_row_prompt +=  f"{col_name}: {example_row[col_name]}\n"
        example_row_prompt += '\n'

    return example_row_prompt

def get_example_row(db_path, table_name, example_value_num):

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name} LIMIT {example_value_num}")
    rows = cursor.fetchall()
    if not rows:
        return []
    col_names = [description[0] for description in cursor.description]

    example_rows = []
    for row in rows:
        example_row = {}
        for col_name, value in zip(col_names, row):
            example_row[col_name] = value
        example_rows.append(example_row)

    conn.close()
    return example_rows

def write_response_into_db(table_name, response, db_new_path):
    conn = sqlite3.connect(db_new_path)
    cursor = conn.cursor()

    executed_sql = ''
    row_data = response.split("\n")
    
    try:
        col_values = [line.split(': ', 1) for line in row_data]
        col_values = [(
            col_name.strip(), 
            re.sub(r"^['\"]|['\"]$", '', col_value.strip())
        ) for col_name, col_value in col_values]

        placeholders = ", ".join(["?" for _ in col_values])
        sql = f'INSERT INTO `{table_name}` ({', '.join([f'`{col_value[0]}`' for col_value in col_values])}) VALUES ({placeholders})'
        values = [col_value[1] for col_value in col_values]
    except: 
        executed_sql = 'SQL Generation Failed.\n\n'
        return False, executed_sql
    executed_sql += f'{sql}\n'
    executed_sql += f'VALUE: {', '.join(values)}\n'
    try:
        logging.debug(f'Run data insertion SQL command:\n{sql}\nVALUE: {values}')
        cursor.execute(sql, values)
        executed_sql += f'Exec success.\n\n'
    except Exception as e:
        executed_sql += f'Exec fail.\n{e}\n\n'
        col_values = False
    
    conn.commit()
    conn.close()
    return col_values, executed_sql

