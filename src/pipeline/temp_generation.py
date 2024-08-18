import logging
from typing import Dict, List, Any

import re
import os
import json
import sqlite3

from llm.models import async_llm_call

def temp_generation(db_name: str, config: Dict[str, Any]):
    """
    Generate database schema based on domain and given parameters

    Args:
        db_name (str): database name
    """

    input_path = config['input_path']
    template_path = config['template_path']
    evidence_path = config['evidence_path']
    output_path = config['output_path']
    result_path = config['result_path']
    ques_template_num = config['ques_template_num']
    example_value_num = config['example_value_num']

    logging.info('Starting template generation')
    try:
        schema_path = os.path.join(input_path, f'schema_preprocess/schema_{db_name}.json')
        with open(schema_path, "r") as file:
            schema = json.load(file)
        logging.debug(f'{db_name} SCHEMA loaded')
    except Exception as e:
        logging.error(f'Error loading database schema {db_name}: {e}')
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

    try:
        evidence_path_db = os.path.join(evidence_path, f'evidence_{db_name}.txt')
        with open(evidence_path_db, "r") as file:
            logging.debug('Evidence found')
            evidence_prompts = file.read().split('\n')
            evidence_prompts = [f'\n\nEvidence you should use:\n{evidence_prompt}' for evidence_prompt in evidence_prompts]
    except Exception as e:
        logging.error(f"No evidence found. Try generation without evidence.")
        evidence_prompt = ['\n\nYou should generate evidence on your own.\n']

    schema_prompt_total = 'Schema of database used:\n' + '\n'.join(schema_prompt[table_name] for table_name in schema_prompt)

    ques_templates = []

    for idx, evidence_prompt in enumerate(evidence_prompts):
        prompt = template.replace('{TEMP_NUM}', str(ques_template_num)) \
                         .replace('{SCHEMA}', schema_prompt_total) \
                         .replace('{EVIDENCE}', evidence_prompt)
        request_kwargs = f'[Temp Generation]{db_name}_idx'

        response = async_llm_call(
            prompt=prompt,
            config=config,
            request_list=[request_kwargs],
            step="temp_generation",
            sampling_count=1
        )[0][0]

        with open(os.path.join(output_path, f'{db_name}_{idx}.txt'), "w") as file:
            file.write(prompt)
            file.write('\n====================\n')
            file.write(response)
        
        ques_templates += get_templates(response)

    logging.info(f'Generate {len(ques_templates)} templates.')
    
    result_templates_path = os.path.join(result_path, f'ques_templates_{db_name}')
    if not os.path.exists(result_templates_path):
        os.mkdir(result_templates_path)
    with open(os.path.join(result_templates_path, 'generated_templates.json'), "a") as file:
        json.dump(ques_templates, file, indent=4)

def load_schema_prompt(schema, schema_prompt_path):
    schema_prompt = dict.fromkeys(schema)
    for table_name in schema:
        with open(os.path.join(schema_prompt_path, f'{table_name}.txt')) as file:
            schema_prompt[table_name] = file.read()
    return schema_prompt

def get_templates(response):
    pattern = re.compile(
        r"Q:\s*(?P<question>.*?)\s*E:\s*(?P<evidence>.*?)\s*A:\s*```sql\s*(?P<ans>.*?)\s*```", 
        re.DOTALL
    )
    matches = pattern.findall(response)
    data = [{'question': m[0].strip(), 'evidence': m[1].strip(), 'ans': m[2].strip()} for m in matches]
    return data
                

            
