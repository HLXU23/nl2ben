import json
import numbers
from logging import Logger
from os import environ
from pathlib import Path
from typing import Dict, List, Literal, _TypedDictMeta, get_type_hints

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from ragbench.jsonl_utils import RagBenchDataRow, load_jsonl_file, load_jsonl_files
from ragbench.logger import setup_logger
from ragbench.prompts.judge import (
    OpenAIJudgeResponse,
    judge_accuracy_system_prompt_v1,
    judge_helpfulness_system_prompt_v1,
    judge_relevance_system_prompt_v1,
    judge_depth_system_prompt_v1,
    judge_user_prompt_v1,
)

cumulative_stats = {
    'scores': 0,
    'n': 0,
    'failed': 0,
    'trues': 0,
    'falses': 0
}

OPENAI_MODELS = Literal[
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4-turbo',
    'gpt-4-turbo-preview',
    'gpt-4',
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-16k',
]

JSON_SUPPORTED_OPENAI_MODELS = [
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4-turbo',
    'gpt-3.5-turbo'
]

load_dotenv()
CLIENT = OpenAI(api_key=environ.get('OPENAI_API_KEY'))
BENCHMARK_DATASET = Path('dataset/80_validation_samples.jsonl')
JUDGE_SYSTEM_PROMPTS = [
    judge_accuracy_system_prompt_v1,
    judge_helpfulness_system_prompt_v1,
    judge_relevance_system_prompt_v1,
    judge_depth_system_prompt_v1,
]
JUDGE_USER_PROMPT = judge_user_prompt_v1


def start(
    answer_file: Path,
    output_filename='ragbench_score.jsonl',
    evaluator: OPENAI_MODELS = 'gpt-4o',
):
    # Use the global logger
    logger = setup_logger(f'ragbench_judge{answer_file}')

    # load dataset
    dataset_list = load_jsonl_file(BENCHMARK_DATASET)

    # load answers
    answers_list = load_jsonl_file(Path('generated_answers') / answer_file)

    assert len(dataset_list) == len(answers_list), 'The answers list is incomplete.'

    logger.debug(f'Total of {len(dataset_list)} samples to rate.')

    output_dir = Path('output/')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename

    files_done = set()
    # resume from checkpoint
    if output_path.is_file():
        output_list = load_jsonl_file(output_path)

        for json_row in output_list:
            files_done.add(json_row['filename'])

            # Update statistics
            if json_row['accuracy']:
                cumulative_stats['trues'] += 1
            else:
                cumulative_stats['falses'] += 1

            cumulative_stats['n'] += 1
            if json_row['average'] is not None:
                cumulative_stats['scores'] += json_row['average']

    logger.info(f"{cumulative_stats['n']} sample(s) already done.")

    # call openAI
    for index, datarow in enumerate(dataset_list):
        if datarow['filename'] in files_done:
            continue

        try:
            logger.info(f"Attempting to judge index {index}: {datarow['filename']}...")
            parsed_response: OpenAIJudgeResponse = judge(datarow, answers_list[index]['answer'], evaluator, logger)

        except Exception as error:
            logger.error(f"Filename: [{datarow['filename']}] errored.")
            logger.error(error, exc_info=True)
            cumulative_stats['failed'] += 1
            continue

        # Compute the average score
        scores: List[numbers.Real] = [
            score for score in parsed_response.values()
                if isinstance(score, numbers.Real) and not isinstance(score, bool)
        ]
        if len(scores) > 0:
            parsed_response['average'] = sum(scores) / len(scores)
        else:
            parsed_response['average'] = None

        # Add the filename into the dataset
        parsed_response['filename'] = datarow['filename']

        parsed_response_str = json.dumps(parsed_response)
        with open(output_path, 'a', encoding='utf-8') as responses_file:
            responses_file.write(parsed_response_str + '\n')

        # Update statistics
        cumulative_stats['n'] += 1
        logger.info(f"{cumulative_stats['n']}/{len(dataset_list)} inferences done")

        if parsed_response['accuracy']:
            cumulative_stats['trues'] += 1
        else:
            cumulative_stats['falses'] += 1

        if parsed_response['average'] is not None:
            cumulative_stats['scores'] += parsed_response['average']

    logger.debug(cumulative_stats)
    logger.debug(f"Final average: {cumulative_stats['scores'] / cumulative_stats['n']}")


def judge(
    rag_bench_datarow: RagBenchDataRow,
    answer: str,
    evaluator: OPENAI_MODELS,
    logger: Logger
) -> OpenAIJudgeResponse:
    # return dict.fromkeys(get_type_hints(ParsedResponse).keys())

    question = rag_bench_datarow['question']
    filename = rag_bench_datarow['filename']
    content = rag_bench_datarow['content']
    answer = answer

    parsed_responses = {}
    for judge_system_prompt in JUDGE_SYSTEM_PROMPTS:
        messages: List[ChatCompletionMessageParam] = []

        messages.append(judge_system_prompt)

        user_prompt = judge_user_prompt_v1(question, filename, content, answer)
        messages.append(user_prompt)

        for message in messages:
            print(message['content'])

        parsed_response = call_openai_api(
            evaluator, messages, filename, OpenAIJudgeResponse, logger
        )
        parsed_responses.update(parsed_response)

    return parsed_responses


def call_openai_api(
    model: OPENAI_MODELS,
    messages: List[Dict[str, str]],
    filename: str,
    typed_dict_cls: _TypedDictMeta,
    logger: Logger,
    formatted: bool = True,
):
    if model in JSON_SUPPORTED_OPENAI_MODELS:
        # Refer to https://platform.openai.com/docs/guides/json-mode for models that support json_object.
        response_format = {'type': 'json_object'} if formatted else {'type': 'text'}

        chat_completion = CLIENT.chat.completions.create(
            messages=messages,
            model=model,
            n=1,
            response_format=response_format,
            temperature=0,
        )
        response = chat_completion.choices[0].message.content
        if formatted:
            return json.loads(response)
        else:
            return response

    else:
        chat_completion = CLIENT.chat.completions.create(
            messages=messages,
            model=model,
            n=1,
            response_format={'type': 'text'},
            temperature=0,
        )

        response = chat_completion.choices[0].message.content

        try:
            return parse_response(response, typed_dict_cls)

        except Exception as error:
            logger.error('=' * 80)
            logger.error('Parsing error occurred.')
            logger.error(f'FILENAME: {filename}')
            logger.error(f'RESPONSE: {response}')
            logger.error('=' * 80)
            raise error


def parse_response(response: str, typed_dict_cls: _TypedDictMeta) -> Dict:
    response = dict.fromkeys(get_type_hints(typed_dict_cls).keys())

    json_start = response.rfind('{')  # Find { starting from the back
    json_end = response.rfind('}') + 1  # Find } starting from the back
    json_string = response[json_start:json_end]  # Get the JSON string

    response = json.loads(json_string)

    for key, hint in get_type_hints(typed_dict_cls).items():
        if key not in response:
            raise Exception(f"key '{key}' not found in the response.")

        if not isinstance(response[key], hint):
            raise Exception(f"scores['{key}'] is not a {hint.__name__}.")

    return response


if __name__ == '__main__':
    from jsonargparse import CLI

    CLI(start, as_positional=False)
