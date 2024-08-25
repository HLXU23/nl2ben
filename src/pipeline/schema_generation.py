import logging
from typing import Dict, List, Any

import re
import os
import json
import csv
import sqlite3

from llm.models import async_llm_call

def schema_generation(db_name: str, config: Dict[str, Any]):
    """
    Generate database schema based on domain and given parameters

    Args:
        db_name (str): database name

        db_params (Dict[str, Any]): parameters that describe attributes of required database.
            domain (str): the domain that required database is used for
            usage (str): description about the usge of required database
            min_table_num (int): the minimum number of tables contained by required database
            min_column_sum (int): the minimum number of columns contained by required database in total
            defined_table (Dict[str, Dict[str, Any]]): pre-defined tables and their attributes
            In defined_table[table_name]:
                table_description: predefined table description
                min_column_num: minimum columns this table should have
                defined_column: pre-defined columns and their attributes
                In defined_column[column_name]:
                    column_description: predefined column description
            other_requirements (str): Any other requirements you want the schema to satisfy.
    """

    params_root = config['params_root']
    template_path = config['template_path']
    output_path = config['output_path']
    result_path = config['result_path']

    logging.info('Starting database schema generation')

    try:
        filepath = os.path.join(params_root, f'{db_name}.json')
        with open(filepath, 'r', encoding='utf-8') as file:
            logging.debug(f'Find db [{db_name}] parameter.\n')
            db_params = json.load(file)
    except Exception as e:
        logging.error(f"Error loading schema {db_name}: {e}")
        raise

    try:
        with open(template_path, "r") as file:
            logging.debug(f'Template found')
            template = file.read()
    except Exception as e:
        logging.error(f"Error loading template {template_path}: {e}")
        raise
    
    try:
        requirements = get_requirements(db_params)
        logging.debug(f'Requirements got')
    except Exception as e:
        logging.error(f"Error loading parameter: {e}")
        raise
    
    prompt = template.replace('{NAME}', db_name) \
                     .replace('{REQUIREMENTS}', requirements)

    request_kwargs = f'[Schema Generation]{db_name}'

    response = async_llm_call(
        prompt=prompt,
        config=config,
        request_list=[request_kwargs],
        step="schema_generation",
        sampling_count=1
    )[0][0]

    with open(os.path.join(output_path, f'{db_name}.txt'), "w") as file:
        file.write('\n====================\n')
        file.write('Human: \n')
        file.write(prompt)
        file.write('\n====================\n')
        file.write('AI: \n')
        file.write(response)

    schema_result = parse_response_to_schema(response)

    result_sqlite_path = os.path.join(result_path, f'{db_name}.sqlite')
    executed_sql = create_db_from_schema(schema_result, result_sqlite_path)

    with open(os.path.join(output_path, f'sql_{db_name}.txt'), "w") as file:
        file.write(executed_sql)
    
    logging.info('Schema generation success')
    return

def get_requirements(db_params: Dict[str, Any]) -> str:
    """
    Get requirement part of prompt based on given parameters
    """
    requirements_prompt = ''
    if 'domain' in db_params:
        requirements_prompt += f'The database is used for {db_params['domain']} domain.\n'
    if 'usage' in db_params:
        requirements_prompt += f'{db_params['usage']}\n'
    if 'min_table_num' in db_params:
        requirements_prompt += f'Contains at least {db_params['min_table_num']} tables.\n'
    if 'min_column_sum' in db_params:
        requirements_prompt += f'Contains at least {db_params['min_column_sum']} columns in total.\n'
    if 'defined_table' in db_params:
        for table_name in db_params['defined_table']:
            table_attribute = db_params['defined_table'][table_name]
            requirements_prompt += f'Contains table {table_name}'
            if 'table_description' in table_attribute:
                requirements_prompt += f'(description: {table_attribute['table_description']})'
            requirements_prompt += f'.\n'
            if 'min_column_num' in table_attribute:
                requirements_prompt += f'Table {table_name} contains at least {table_attribute['min_column_num']} columns.\n'
            if 'defined_column' in table_attribute:
                for column_name in table_attribute['defined_column']:
                    column_attribute = table_attribute['defined_column'][column_name]
                    requirements_prompt += f'Table {table_name} contains column {column_name}'
                    if 'column_description' in column_attribute:
                        requirements_prompt += f'(description: {column_attribute['column_description']})'
                    requirements_prompt += '.\n'
    return requirements_prompt

def parse_response_to_schema(response):
    # Define regex patterns
    table_pattern = re.compile(r'(\w+): (.+)')
    column_pattern = re.compile(r'-\((\w+)\)([^:]+): (.+)')
    foreign_key_pattern = re.compile(r'(\w+)\((\w+)\.(\w+)\)')
    result = {}
    
    tables = response.split('\n\n')
    
    for table in tables:
        table_name_match = table_pattern.search(table)
        if table_name_match:
            table_name, table_desc = table_name_match.groups()
            result[table_name] = {
                "description": table_desc,
                "attributes": {},
                "primary_keys": [],
                "foreign_keys": {}
            }
            
            columns = column_pattern.findall(table)
            for col_type, col_info, col_desc in columns:
                primary_mark = False
                if col_info[-1] == '*':
                    col_info = col_info[:-1]
                    primary_mark = True
                if '(' in col_info and ')' in col_info:
                    foreign_key_match = foreign_key_pattern.search(col_info)
                    if foreign_key_match:
                        col_name, refer_table, refer_col = foreign_key_match.groups()
                        foreign_key_dict = {
                            "referenced_table": refer_table,
                            "referenced_column": refer_col
                        }
                        result[table_name]['foreign_keys'][col_name] = foreign_key_dict
                else:
                    col_name = col_info

                result[table_name]["attributes"][col_name] = {
                    "description": col_desc,
                    "type": col_type
                }
                if primary_mark:
                    result[table_name]["primary_keys"].append(col_name)
                    
    return result

def create_db_from_schema(schema_result, result_sqlite_path):
    if os.path.exists(result_sqlite_path):
            os.remove(result_sqlite_path)

    conn = sqlite3.connect(result_sqlite_path)
    cursor = conn.cursor()

    executed_sql = ''
    
    for table_name, table_info in schema_result.items():
        attributes = table_info.get('attributes', {})
        primary_keys = table_info.get('primary_keys', [])
        foreign_keys = table_info.get('foreign_keys', {})
        
        columns = []
        for attr_name, attr_info in attributes.items():
            column_type = attr_info.get('type', 'TEXT')
            columns.append(f'"{attr_name}" {column_type}')
        
        if primary_keys:
            columns.append(f"PRIMARY KEY ({', '.join([f'"{primary_key}"' for primary_key in primary_keys])})")
        
        for column in foreign_keys:
            referenced_table = foreign_keys[column].get('referenced_table')
            referenced_column = foreign_keys[column].get('referenced_column')
            if column and referenced_table and referenced_column:
                columns.append(f'FOREIGN KEY (`{column}`) REFERENCES `{referenced_table}` (`{referenced_column}`)')
        
        sql = f'CREATE TABLE `{table_name}` ({', '.join(columns)})'
        logging.debug(f'Run database creation SQL command:\n{sql}')
        cursor.execute(sql)
        executed_sql += f'{sql}\n'
        
    conn.commit()
    conn.close()
    return executed_sql