import logging
from typing import Dict, List, Any

import re
import os
import json
import random
import sqlite3

from llm.models import async_llm_call

def ques_generation(db_name: str, config: Dict[str, Any]):

    input_path = config['input_path']
    result_path = config['result_path']
    ques_per_template = config['ques_per_template']

    logging.info('Starting question generation')

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

    ques_templates = []
    ques_templates_path = os.path.join(input_path, f'temp_generation/ques_templates_{db_name}')
    for file_name in os.listdir(ques_templates_path):
        if file_name.endswith('.json'):
            try:
                file_path = os.path.join(ques_templates_path, file_name)
                new_ques_templates = load_templates(file_path)
                if new_ques_templates:
                    logging.debug(f'Find question templates in {file_name}')
                    ques_templates += new_ques_templates
            except Exception as e:
                pass
    logging.info(f'Find {len(ques_templates)} question templates')

    db_path = os.path.join(input_path, f'data_generation/{db_name}.sqlite')
    questions = []
    for ques_template in ques_templates:
        questions += generate_ques_on_template(ques_template, schema, db_path, ques_per_template)
    
    logging.info(f'Generate {len(questions)} new questions totally.')

    raw_ques_path = os.path.join(result_path, f'raw_ques_{db_name}.json')
    with open(raw_ques_path, "w") as file:
        json.dump(questions, file, indent=4)

def load_schema_prompt(schema, schema_prompt_path):
    schema_prompt = dict.fromkeys(schema)
    for table_name in schema:
        with open(os.path.join(schema_prompt_path, f'{table_name}.txt')) as file:
            schema_prompt[table_name] = file.read()
    return schema_prompt

def load_templates(file_path: str):
    with open(file_path, "r") as file:
        items = json.load(file)
    ques_templates = []
    for item in items:
        try: 
            ques = item['question']
            ans = item['ans']
        except:
            continue
        try: 
            evid = item['evidence']
        except:
            evid = ''
        if isinstance(ques, str):
            ques_templates.append({
                'question': ques,
                'evidence': evid,
                'ans': ans
            })
        elif isinstance(ques, list):
            for q in ques:
                ques_templates.append({
                    'question': q,
                    'evidence': evid,
                    'ans': ans
                })
    return ques_templates

def generate_ques_on_template(ques_template: str, schema: Dict[str, Any], db_path: str, ques_per_template: int):
    question_temp = ques_template['question']
    evidence = ques_template['evidence']
    ans_temp = ques_template['ans']
    generated_ques = []
    ques_with_res_cnt = 0
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for _ in range(ques_per_template * 10):

        ans, related_tables = replace_placeholders_randomly(ans_temp, 'TABLE', [table_name for table_name in schema])
        cols = []
        for table_name in related_tables:
            cols += [col_name for col_name in schema[table_name]['attributes']]
        ans, related_cols = replace_placeholders_randomly(ans, 'COLUMN', cols)
        other_placeholders = re.findall(r"\{([^}]*)\}", ans)
        for other_placeholder in other_placeholders:
            if '.' in ans:
                try:
                    table_name, col_name = other_placeholder.split('.')
                    optional_values = get_values_from_table(db_path, table_name, col_name)
                except:
                    optional_values = []
            else:
                for table_name in schema:
                    if other_placeholder in schema[table_name]['attributes']:
                        optional_values = get_values_from_table(db_path, table_name, other_placeholder)
                        break
                optional_values = []
            if optional_values:
                ans, _ = replace_placeholders_randomly(ans, other_placeholder, optional_values)
        if ans in [item["ans"] for item in generated_ques]: continue
        try: 
            cursor.execute(ans)
            exec_result = cursor.fetchall()
        except Exception:
            continue
        
        if (not exec_result) or \
           (exec_result[0] == (None, ) or \
           (len(exec_result) == 1 and exec_result[0] == (0,))):
            continue
        # logging.debug(exec_result)
        new_ques = {
            "question": question_temp,
            "evidence": evidence,
            "ans": ans
        }
        generated_ques.append(new_ques)

        if not len(generated_ques) < ques_per_template:
            break
    if generated_ques:
        logging.info(f'{len(generated_ques)} executable with result.')
    return generated_ques

            
def get_values_from_table(db_path, table_name, col_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(f"SELECT {col_name} FROM {table_name}")
        values = cursor.fetchall()
        if not values:
            return []
        unique_values = list(set(row[0] for row in values))
        return unique_values

    except sqlite3.Error as e:
        return []

    finally:
        conn.close()

def replace_placeholders_randomly(template: str, placeholder: str, replacements: list):
    if not replacements:
        return template, set()
    new_template = template
    pattern = re.compile(r'\{' + re.escape(placeholder) + r'\}')
    used_replacements = set()
    while pattern.search(new_template):
        used_replacement = random.choice(replacements)
        used_replacements.add(used_replacement)
        new_template = pattern.sub(str(used_replacement), new_template, count=1)
    return new_template, used_replacements