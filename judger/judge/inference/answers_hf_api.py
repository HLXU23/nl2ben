import json
import math
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from ragbench.inference.answers import get_filenames, get_prompt
from ragbench.jsonl_utils import RagBenchDataRow, load_jsonl_files
from ragbench.logger import setup_logger

load_dotenv()


def start(model_id: str, output_file: str, template: str):
    logger = setup_logger(f'ragbench_generate_answers_hf_api({template})')
    client = InferenceClient(
        model_id,
        token=os.environ.get('HF_API_KEY')
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # load dataset
    dataset_list: List[RagBenchDataRow] = load_jsonl_files('dataset')

    logger.debug(f'Total of {len(dataset_list)} samples to generate.')

    generated_answer_dir = Path('generated_answers')
    generated_answer_dir.mkdir(parents=True, exist_ok=True)
    generated_answer_filepath = generated_answer_dir / output_file

    files_done = get_filenames(generated_answer_filepath)
    num_files_done = len(files_done)
    logger.info(f'{num_files_done} sample(s) already done.')

    # generate
    total_response_tokens = 0
    for datarow in dataset_list:
        if datarow['filename'] in files_done:
            # skip the file
            continue

        try:
            logger.info(f"Attempting to generate answer for {datarow['filename']}...")
            answer, num_tokens = generate_answer(datarow, template, client, tokenizer)
            total_response_tokens += num_tokens

        except Exception as error:
            logger.error(
                f'Filename: [{datarow["filename"]}] errored. Answers generation is only partially complete.'
            )
            logger.error(error)
            continue

        num_files_done += 1
        logger.info(f'{num_files_done}/{len(dataset_list)} inferences done')

        with open(generated_answer_filepath, 'a', encoding='utf-8') as generated_answer_file:
            # the answer would be appended in the same sequence as the dataset
            # which could be extracted and benchmarked in sequence
            # so we don't need to store the content, just filename if we want to read it later
            json_string = json.dumps({'answer': answer, 'filename': datarow['filename']})
            generated_answer_file.write(json_string + '\n')

    logger.info(f'Total response tokens: {total_response_tokens}')


def generate_answer(
    datarow: RagBenchDataRow,
    template: str,
    client: InferenceClient,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
):
    system, prompt = get_prompt(datarow, template)

    # print('=' * 80)
    # print('Processing prompt:')
    # print('=' * 80)
    # print(system)
    # print(prompt)

    chat_completion = client.chat_completion(
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=4096,
        temperature=0.0,
    )
    answer = chat_completion.choices[0].message.content

    # print('=' * 20 + 'ANSWER' + '=' * 20)
    # print(answer)
    # print('=' * 80)

    output_tokens = tokenizer(answer, return_tensors='pt', add_special_tokens=True)['input_ids']
    num_tokens = math.prod(output_tokens.shape)

    return answer, num_tokens


if __name__ == '__main__':
    from jsonargparse import CLI

    CLI(start, as_positional=False)
