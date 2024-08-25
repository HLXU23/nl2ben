import logging
from typing import Dict, List, Any

import re
import os
import json
import random
import sqlite3

from llm.models import async_llm_call

def ques_revision(db_name: str, config: Dict[str, Any]):

    input_path = config['input_path']
    template_path = config['template_path']
    output_path = config['output_path']
    result_path = config['result_path']

    try: 
        raw_ques_path = os.path.join(input_path, f'ques_generation/raw_ques_{db_name}.json')
        with open(raw_ques_path, "r") as file:
            raw_questions = json.load(file)
    except Exception as e:
        logging.error(f'Error loading raw questions: {e}')
    
    try:
        with open(template_path, "r") as file:
            logging.debug(f'Template found')
            template = file.read()
    except Exception as e:
        logging.error(f"Error loading template {template_path}: {e}")
        raise

    revised_questions = []
    for raw_question in raw_questions:
        evid = raw_question['evidence']
        ans = raw_question['ans']
        prompt = template.replace('{QUESTION}', raw_question['question']) \
                         .replace('{EVIDENCE}', evid) \
                         .replace('{ANS}', ans)
        request_kwargs = f'[Ques Revision]{db_name}'

        response = async_llm_call(
            prompt=prompt,
            config=config,
            request_list=[request_kwargs],
            step="ques_revision",
            sampling_count=1
        )[0][0]

        with open(os.path.join(output_path, f'{db_name}.txt'), "w") as file:
            file.write('\n====================\n')
            file.write('Human: \n')
            file.write(prompt)
            file.write('\n====================\n')
            file.write('AI: \n')
            file.write(response)

        revised_questions.append({
            "question": response,
            "evidence": evid,
            "ans": ans
        })

    revised_ques_path = os.path.join(result_path, f'revised_ques_{db_name}.json')
    with open(revised_ques_path, "w") as file:
        json.dump(revised_questions, file, indent=4)

        
