import logging
from collections import deque
from typing import Dict, List, Any

import re
import os
import json
import shutil
import sqlite3

def schema_preprocess(db_name: str, config: Dict[str, Any]):

    input_path = config['input_path']    
    result_path = config['result_path']

    logging.info('Starting schema preprocess')

    try: 
        db_path = os.path.join(input_path, f'schema_generation/{db_name}.sqlite')
        schema, referenced_cols = get_schema_from_sqlite(db_path)
        logging.debug('Schema got')
        logging.debug(schema)
    except Exception as e:
        logging.error(f'Error loading database {db_name}: {e}')
        raise

    result_schema_path = os.path.join(result_path, f'schema_{db_name}.json')
    with open(result_schema_path, "w") as file:
        json.dump(schema, file, indent=4)

    result_referenced_col_path = os.path.join(result_path, f'referenced_cols_{db_name}.json')
    with open(result_referenced_col_path, "w") as file:
        json.dump(referenced_cols, file, indent=4)

    schema_prompt = get_schema_prompt(schema)
    result_schema_prompt_path = os.path.join(result_path, f'schema_prompt_{db_name}')
    if not os.path.exists(result_schema_prompt_path):
        os.mkdir(result_schema_prompt_path)
    for table_name in schema_prompt:
        with open(os.path.join(result_schema_prompt_path, f'{table_name}.txt'), "w") as file:
            file.write(schema_prompt[table_name])
    
    
def get_schema_from_sqlite(db_path):

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Step 1: Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema = {}
    referenced_cols = {}

    for table in tables:
        table_name = table[0]
        schema[table_name] = {
            "description": "",  # 可以置空
            "attributes": {},
            "primary_keys": [],
            "foreign_keys": []
        }
        
        # Step 2: Get attributes and primary keys
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        for col in columns:
            col_name = col[1]
            schema[table_name]["attributes"][col_name] = {
                "description": "",
                "type": col[2]
            }
            if col[5] == 1:
                schema[table_name]["primary_keys"].append(col_name)
        
        # Step 3: Get foreign keys
        cursor.execute(f"PRAGMA foreign_key_list({table_name});")
        foreign_keys = cursor.fetchall()
        for fk in foreign_keys:
            schema[table_name]["foreign_keys"].append({
                "column": fk[3],
                "referenced_table": fk[2],
                "referenced_column": fk[4]
            })
            referenced_table = fk[2]
            referenced_column = fk[4]
            if referenced_table not in referenced_cols:
                referenced_cols[referenced_table] = []
            if referenced_column not in referenced_cols[referenced_table]:
                referenced_cols[referenced_table].append(referenced_column)
    
    # Close the connection
    conn.close()
    
    return schema, referenced_cols

def get_schema_prompt(schema):
    schema_prompt = {}
    
    for table_name, table_info in schema.items():
        attributes = table_info.get('attributes', {})
        primary_keys = table_info.get('primary_keys', [])
        foreign_keys = table_info.get('foreign_keys', [])
        
        columns = []
        
        for attr_name, attr_info in attributes.items():
            column_type = attr_info.get('type', 'TEXT')
            columns.append(f"    {attr_name} {column_type}")
        
        if primary_keys:
            columns.append(f"    PRIMARY KEY ({', '.join(primary_keys)})")
        
        for fk in foreign_keys:
            column = fk.get('column')
            referenced_table = fk.get('referenced_table')
            referenced_column = fk.get('referenced_column')
            if column and referenced_table and referenced_column:
                columns.append(f"    FOREIGN KEY ({column}) REFERENCES {referenced_table} ({referenced_column})")
        
        schema_prompt[table_name] = (
            f"CREATE TABLE {table_name} (\n"
            f"{',\n'.join(columns)}\n"
            ");"
        )

    return schema_prompt