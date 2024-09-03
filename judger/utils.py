import glob
import json
import os
from pathlib import Path
from typing import List, TypedDict, Dict

import sqlite3
import json

class SQLPair(TypedDict):
    question_id: int
    db_id: str
    question: str
    evidence: str
    SQL: str
    difficulty: str

class DataBench(TypedDict):
    database: str # JSON format
    question_sql_pairs: List[SQLPair]


def database_to_json(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Get the list of tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        db_dict = {}
        for table_name in tables:
            table_name = table_name[0]

            # Get the table schema (columns) with types and primary keys
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns_info = cursor.fetchall()
            columns = [
                {
                    "name": col[1],
                    "type": col[2],
                    "primary_key": col[5] == 1
                }
                for col in columns_info
            ]

            # Get foreign key information
            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            foreign_keys_info = cursor.fetchall()
            foreign_keys = [
                {
                    "column": fk[3],
                    "references_table": fk[2],
                    "references_column": fk[4]
                }
                for fk in foreign_keys_info
            ]

            # Get all data from the table
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()

            # Add table data and schema information to the dictionary
            db_dict[table_name] = {
                "columns": columns,
                "foreign_keys": foreign_keys,
                "data": [dict(zip([col["name"] for col in columns], row)) for row in rows]
            }

        return json.dumps(db_dict, indent=4)
    finally:
        cursor.close()
        conn.close()


def load_database_qsql(database_path: str, question_sql_path: str) -> DataBench:
    # Load the database and convert it to JSON format
    database_json_str = database_to_json(database_path)
    database_json = json.loads(database_json_str)

    # Load the question-SQL pairs from the JSON file
    with open(question_sql_path, 'r') as f:
        question_sql_pairs = json.load(f)

    # Create and return the DataBench object
    data_bench = DataBench(
        database=database_json,
        question_sql_pairs=question_sql_pairs
    )
    # print(database_json)
    # print("=" * 100)
    # print(question_sql_pairs)
    return data_bench

def parse_data_for_judging(data: DataBench) -> Dict[str, str]:
    res = {
        "database": data['database'],
        "question_SQL_pairs": json.dumps(data['question_sql_pairs'])  # Convert list to JSON string
    }
    return json.dumps(res)

    