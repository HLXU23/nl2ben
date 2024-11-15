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

def data_rule_generation(db_name: str, config: Dict[str, Any]):
    """
    Generate database data rule

    Args:
        db_name (str): database name
    """
    input_path = config['input_path']
    template_path = config['template_path']
    output_path = config['output_path']
    result_path = config['result_path']
    example_value_num = config['example_value_num']

    logging.info('Starting database data rule generation')

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
        schema_prompt_total = '\n'.join(schema_prompt[table_name] for table_name in schema_prompt)
        logging.debug(f'{db_name} SCHEMA PROMPT loaded')
    except Exception as e:
        logging.error(f'Error loading database schema prompt {db_name}: {e}')
        raise

    db_path = os.path.join(input_path, f'schema_generation/{db_name}.sqlite')

    templates = {
        'categorical_range_generation': '',
        'identifier_selection': '',
        'numerical_range_generation': ''
    }

    for template in templates:
        file_path = os.path.join(template_path, f'{template}.txt')
        try:
            with open(file_path, "r") as file:
                logging.debug('Template found')
                templates[template] = file.read()
        except Exception as e:
            logging.error(f"Error loading template {file_path}: {e}")
            raise
    
    result_path_root = os.path.join(output_path, f'{db_name}')
    if not os.path.exists(result_path_root):
        os.makedirs(result_path_root)

    # Identifier selection
    prompts = [templates['identifier_selection'].replace('{SCHEMA}', schema_prompt_total)]
    
    step = f'[Identifier Selection]{db_name}'

    response = asyncio.run(batch_llm_call_async(
        prompts=prompts,
        config=config,
        step=step
    ))[0][0]

    with open(os.path.join(output_path, f'{db_name}/identifier_{db_name}.txt'), "w") as file:
        file.write('\n====================\n')
        file.write('Human: \n')
        file.write(prompts[0])
        file.write('\n====================\n')
        file.write('AI: \n')
        file.write(response)

    rows = response.split('\n')
    for row in rows:
        try: 
            table_name, col_name = row.split('.')
            table_name = table_name.strip()
            col_name = col_name.strip()
            if col_name in schema[table_name]['primary_keys']:
                logging.info(f'Found identifier column: {col_name}')
                schema[table_name]['attributes'][col_name]['identifier'] = 1
        except:
            logging.info(f'Identifier column not primary: {col_name}')
            pass

    for table_name in schema:
        output_categorical_path = os.path.join(output_path, f'{db_name}/categorical_{table_name}.txt')
        if os.path.exists(output_categorical_path):
            os.remove(output_categorical_path)
        output_numerical_path = os.path.join(output_path, f'{db_name}/numerical_{table_name}.txt')
        if os.path.exists(output_numerical_path):
            os.remove(output_numerical_path)

        # Categorical columns
        prompts = []
        cols = []
        for col_name in schema[table_name]['attributes']:
            col_type = schema[table_name]['attributes'][col_name]['type']
            if not 'identifier' in schema[table_name]['attributes'][col_name]:
                unique_cnt, unique_values = get_unique_values(db_path, table_name, col_name, example_value_num)
                unique_value_prompt = '' if unique_cnt == 0 \
                                         else f'There are {unique_cnt} unique values in this column right now.\n' + \
                                              f'Example Values: {', '.join(unique_values)}'
                if unique_cnt >= 20: continue
                cols += [col_name]
                prompts += [templates['categorical_range_generation'].replace('{DB}', db_name) \
                                                                     .replace('{TABLE}', table_name) \
                                                                     .replace('{COLUMN}', col_name) \
                                                                     .replace('{SCHEMA}', schema_prompt[table_name]) \
                                                                     .replace('{UNIQUE_VALUE}', unique_value_prompt)]
                
        step = f'[Categorical]{db_name}{table_name}'

        responses = asyncio.run(batch_llm_call_async(
            prompts=prompts,
            config=config,
            step=step
        ))
                
        for i in range(len(prompts)):
            col_name = cols[i]
            prompt = prompts[i]
            response = responses[i][0]

            with open(output_categorical_path, "a", encoding="utf-8") as file:
                file.write('\n====================\n')
                file.write('Human: \n')
                file.write(prompt)
                file.write('\n====================\n')
                file.write('AI: \n')
                file.write(response)

            if ',' in response:
                categories = [category.strip() for category in response.split(',')]
                schema[table_name]['attributes'][col_name]['categorical'] = categories
                continue

        # Numerical columns
        prompts = []
        cols = []
        for col_name in schema[table_name]['attributes']:
            col_type = schema[table_name]['attributes'][col_name]['type']

            if not 'identifier' in schema[table_name]['attributes'][col_name] and \
               not 'categorical' in schema[table_name]['attributes'][col_name] and \
               col_type in ['INT', 'INTEGER', 'FLOAT', 'REAL']:
                example_row_prompt = get_example_row_prompt(db_path, table_name, example_value_num)
                cols += [col_name]
                prompts += [templates['numerical_range_generation'].replace('{DB}', db_name) \
                                                                   .replace('{TABLE}', table_name) \
                                                                   .replace('{COLUMN}', col_name) \
                                                                   .replace('{SCHEMA}', schema_prompt[table_name]) \
                                                                   .replace('{EXAMPLE_ROW}', example_row_prompt)]
        step = f'[Numerical]{db_name}{table_name}'

        responses = asyncio.run(batch_llm_call_async(
            prompts=prompts,
            config=config,
            step=step
        ))

        for i in range(len(prompts)):
            col_name = cols[i]
            prompt = prompts[i]
            response = responses[i][0]

            with open(output_numerical_path, "a", encoding="utf-8") as file:
                file.write('\n====================\n')
                file.write('Human: \n')
                file.write(prompt)
                file.write('\n====================\n')
                file.write('AI: \n')
                file.write(response)

            lower = get_min_value(db_path, table_name, col_name)
            upper = get_max_value(db_path, table_name, col_name)
            if ',' in response:
                l, u = response.split(', ', 1)
                try:
                    lower = min(l, lower)
                    upper = max(u, upper)
                except:
                    pass
            if isinstance(lower, str) and isinstance(upper, str):
                schema[table_name]['attributes'][col_name]['numerical'] = [float(lower), float(upper)]
            continue
            
    result_schema_path = os.path.join(result_path, f'schema_{db_name}.json')
    with open(result_schema_path, "w") as file:
        json.dump(schema, file, indent=4)

def load_schema_prompt(schema, schema_prompt_path):
    schema_prompt = dict.fromkeys(schema)
    for table_name in schema:
        with open(os.path.join(schema_prompt_path, f'{table_name}.txt')) as file:
            schema_prompt[table_name] = file.read()
    return schema_prompt

def get_max_value(db_path, table_name, col_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = f"SELECT MAX(`{col_name}`) FROM `{table_name}`"
    cursor.execute(query)
    max_value = cursor.fetchone()[0]
    conn.close()
    return max_value

def get_min_value(db_path, table_name, col_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = f"SELECT MIN(`{col_name}`) FROM `{table_name}`"
    cursor.execute(query)
    min_value = cursor.fetchone()[0]
    conn.close()
    return min_value

def random_select(list, n):
    return random.sample(list, n) if n < len(list) else list

def get_unique_values(db_path, table_name, col_name, example_num):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    count_query = f"SELECT COUNT(DISTINCT `{col_name}`) FROM `{table_name}`"
    cursor.execute(count_query)
    unique_cnt = int(cursor.fetchone()[0])
    
    example_query = f"SELECT DISTINCT `{col_name}` FROM `{table_name}` LIMIT {example_num}"
    cursor.execute(example_query)
    example_values = [str(row[0]) for row in cursor.fetchall()]
    
    conn.close()
    return unique_cnt, example_values

def get_example_row_prompt(db_path, table_name, example_value_num):

    example_row_prompt = '\n\nHere are some existed data in database, you can check their formats and values for reference.\n'
    example_rows = get_example_row(db_path, table_name, example_value_num)
    
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


