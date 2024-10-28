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

def evidence_generation(db_name: str, config: Dict[str, Any]):
    """
    Generate database data

    Args:
        db_name (str): database name
    """
    input_path = config['input_path']
    template_path = config['template_path']
    output_path = config['output_path']
    result_path = config['result_path']
    evidence_num = config['evidence_num']

    logging.info('Starting evidence generation')

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

    schema_prompt_total = 'Schema of database used:\n' + '\n'.join(schema_prompt[table_name] for table_name in schema_prompt)

    prompt = template.replace('{EVIDENCE_NUM}', str(evidence_num)) \
                     .replace('{SCHEMA}', schema_prompt_total)
    request_kwargs = f'[Evidence Generation]{db_name}{schema_prompt_total}'
    response = async_llm_call(
        prompt=prompt,
        config=config,
        request_list=[request_kwargs],
        step="evidence_generation",
        sampling_count=1
    )[0][0]

    with open(os.path.join(output_path, f'{db_name}.txt'), "w") as file:
        file.write('\n====================\n')
        file.write('Human: \n')
        file.write(prompt)
        file.write('\n====================\n')
        file.write('AI: \n')
        file.write(response)

    evidences = get_evidences(response)
    with open(os.path.join(result_path, f'{db_name}.json'), "w") as file:
        json.dump(evidences, file, indent=4)
        

def load_schema_prompt(schema, schema_prompt_path):
    schema_prompt = dict.fromkeys(schema)
    for table_name in schema:
        with open(os.path.join(schema_prompt_path, f'{table_name}.txt')) as file:
            schema_prompt[table_name] = file.read()
    return schema_prompt

def get_evidences(response):
    evidences = []
    response_rows = response.split('\n\n')
    for response_row in response_rows:
        try:
            evidence_details = response_row.split('\n')
            evidence = {}
            for evidence_detail in evidence_details:
                if evidence_detail.startswith('Name: '):
                    evidence['Name'] = evidence_detail[6:]
                elif evidence_detail.startswith('Desc: '):
                    evidence['Description'] = evidence_detail[6:]
                elif evidence_detail.startswith('SQL: '):
                    evidence['SQL'] = evidence_detail[5:]
            evidences.append(evidence)
        except:
            pass
    return evidences

    



