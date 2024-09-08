"""
Takes in a json file of answers and judge it based on what is speicfied in config
"""

import json
import numbers
from logging import Logger
import os
from pathlib import Path
from typing import Dict, List, Literal, _TypedDictMeta, get_type_hints, Optional

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from utils import (
    DataBench,
    load_database_qsql,
    parse_data_for_judging,
    parse_data_for_individual_judging
)
from logger import setup_logger

from config import (
    JUDGE_SYSTEM_PROMPT,
    INSTRUCTIONS,
    INSTRUCTIONS_INDIVIDUAL
)

load_dotenv()
CLIENT = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Note I only considered JSON-supported models
OPENAI_MODELS = Literal[
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4-turbo',
    'gpt-3.5-turbo',
]

ALLOWED_MODELS = {
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4-turbo',
    'gpt-3.5-turbo',
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def start(
    database_file: str,
    qsql_pairs_file: str,
    output_path: Optional[str],
    evaluator: OPENAI_MODELS = 'gpt-4o',
    distinct: bool = False,
):
    database_name = os.path.splitext(os.path.basename(database_file))[0]
    qsql_name = os.path.splitext(os.path.basename(qsql_pairs_file))[0]
    if not output_path:
        output_name = database_name + '_' + qsql_name + '.json'
        output_path = os.path.join(RESULTS_DIR, output_name)
    else:
        assert output_path.endswith('.json'), "Output file should have a .json extension"
        output_name = os.path.splitext(os.path.basename(output_path))[0]

    # Use the global logger
    logger = setup_logger(f'{output_name}')

    # Load synthetic database and question-sql pairs
    data: DataBench = load_database_qsql(database_file, qsql_pairs_file)

    logger.debug(f"Size of {database_name}: {len(data['database'].keys())}")
    logger.debug(f"Number of question-sql pairs to evaluate in {qsql_name}: {len(data['question_sql_pairs'])}")

    # pairs_failed = set()
    # # consider doing individually

    # call openAI
    success = True
    try:
        logger.info(f"Attempting to judge {database_name} and {qsql_name}...")
        response = judge(
            data, evaluator, logger, distinct
        )

    except Exception as error:
        success = False
        logger.error(f"Judging errored.")
        logger.error(error, exc_info=True)
    
    if success:
        with open(output_path, 'w') as json_file:
            json.dump(response, json_file, indent=4)
        logger.debug(f"Completed generation of {output_name}")
    else:
        logger.debug(f"Failed generation of {output_name}")


def judge(
    data_benchmark: DataBench,
    evaluator: OPENAI_MODELS,
    logger: Logger,
    distinct: bool
) -> json:
    json_responses = None

    if distinct:
        json_responses = {}
        formatted_data_arr = parse_data_for_individual_judging(data_benchmark)
        num_to_judge = len(formatted_data_arr)
        for i, formatted_data in enumerate(formatted_data_arr):
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": INSTRUCTIONS_INDIVIDUAL},
                {"role": "user", "content": formatted_data}
            ]
            response = call_openai_api(evaluator, messages, logger)
            try:
                single_response = json.loads(response)
                json_responses.update(single_response)
                logger.info(f"Completed {i+1}/{num_to_judge} of question-SQL pairs.")
            except Exception as error:
                logger.error('=' * 80)
                logger.error('Something wrong went converting output to JSON format.. \n')
                logger.error(f"MODEL's OUTPUT: {response}")
                logger.error('=' * 80)
                raise error

    else:
        formatted_data = parse_data_for_judging(data_benchmark)
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": INSTRUCTIONS},
            {"role": "user", "content": formatted_data}
        ]

        response = call_openai_api(evaluator, messages, logger)

        try:
            json_responses = json.loads(response)
        except Exception as error:
            logger.error('=' * 80)
            logger.error('Something wrong went converting output to JSON format.. LIKELY due to insufficient token limit..\n')
            logger.error(f"MODEL's OUTPUT: {response}")
            logger.error('=' * 80)
            raise error
    
    return json_responses


def call_openai_api(
    model: OPENAI_MODELS,
    messages: List[Dict[str, str]],
    logger: Logger,
):
    assert model in ALLOWED_MODELS, "Pick model that supports JSON object"
    # Refer to https://platform.openai.com/docs/guides/json-mode for models that support json_object.
    response = CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        response_format={'type': 'json_object'},
        n=1,
        temperature=0.05
    )

    output = response.choices[0].message.content
    return output


# def parse_response(response: str, typed_dict_cls: _TypedDictMeta) -> Dict:
#     response = dict.fromkeys(get_type_hints(typed_dict_cls).keys())

#     json_start = response.rfind('{')  # Find { starting from the back
#     json_end = response.rfind('}') + 1  # Find } starting from the back
#     json_string = response[json_start:json_end]  # Get the JSON string

#     response = json.loads(json_string)

#     for key, hint in get_type_hints(typed_dict_cls).items():
#         if key not in response:
#             raise Exception(f"key '{key}' not found in the response.")

#         if not isinstance(response[key], hint):
#             raise Exception(f"scores['{key}'] is not a {hint.__name__}.")

#     return response


if __name__ == '__main__':
    from jsonargparse import CLI

    CLI(start, as_positional=False)