import json
from pathlib import Path
from typing import List

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from ragbench.inference.answers import get_filenames, get_prompt
from ragbench.jsonl_utils import RagBenchDataRow, load_jsonl_files
from ragbench.logger import setup_logger
from ragbench.utils import load_tokenizer_and_pretrained_model


def start(model_id: str, output_file: str, template: str):
    logger = setup_logger(f'ragbench_generate_answers_hf_api({template})')
    
    tokenizer, model = load_tokenizer_and_pretrained_model(model_id, logger)

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
        answer = generate_answer(datarow, template, model, tokenizer)

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
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
):
    system, prompt = get_prompt(datarow, template)

    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': prompt}
    ]

    # print('=' * 80)
    # print('Processing prompt:')
    # print('=' * 80)
    # print(system)
    # print(prompt)

    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors='pt',
        add_generation_prompt=True
    ).cuda()

    generation_output = model.generate(
        input_ids,
        max_new_tokens=4096,
        do_sample=False,
    )
    generated_ids = generation_output[0][input_ids.shape[1] : -1]
    answer = tokenizer.decode(generated_ids)

    print('=' * 80)
    print('Got answer:')
    print('=' * 80)
    print(answer)

    return answer


if __name__ == '__main__':
    from jsonargparse import CLI

    CLI(start, as_positional=False)
