import json
from pathlib import Path
from typing import List

from lit_gpt.generate_lit import get_lit_inferences
from ragbench.inference.answers import get_filenames, get_prompt
from ragbench.jsonl_utils import RagBenchDataRow, load_jsonl_files
from ragbench.logger import setup_logger


def start(output_file: str, template: str, checkpoint_path: Path):
    # TODO: Merge this with generate_answers.start
    # TODO: Support inputting a list of templates to run all at once

    logger = setup_logger(f'ragbench_generate_answers_lit({template})')

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
    for datarow in dataset_list:
        if datarow['filename'] in files_done:
            # skip the file
            continue

        logger.info(f"Attempting to generate answer for {datarow['filename']}...")
        answer = generate_answer(datarow, template, checkpoint_path)

        num_files_done += 1
        logger.info(f'{num_files_done}/{len(dataset_list)} inferences done')

        with open(generated_answer_filepath, 'a', encoding='utf-8') as generated_answer_file:
            # the answer would be appended in the same sequence as the dataset
            # which could be extracted and benchmarked in sequence
            # so we don't need to store the content, just filename if we want to read it later
            json_string = json.dumps({'answer': answer, 'filename': datarow['filename']})
            generated_answer_file.write(json_string + '\n')


def generate_answer(
    datarow: RagBenchDataRow,
    template: str,
    checkpoint_path: Path,
    tokenizer_path: Path = Path('~/Llama-2-7b-hf/').expanduser()
):
    prompt = get_prompt(datarow, template)

    # print('=' * 80)
    # print('Processing prompt:')
    # print(prompt)
    # print('=' * 80)

    answer = get_lit_inferences(
        model_config_name='Llama-2-7b-hf',
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        prompts=[prompt],
        max_new_tokens=4096,
        temperature=0,
    )[0]

    # print('=' * 80)
    # print('Got answer:')
    # print(answer)
    # print('=' * 80)

    return answer


if __name__ == '__main__':
    from jsonargparse import CLI

    CLI(start, as_positional=False)
