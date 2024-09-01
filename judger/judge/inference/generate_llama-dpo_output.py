import json
from typing import List, TypedDict, Dict, List, Optional
from pathlib import Path
from lit_gpt.prompts import PromptStyle, Llama3RagDPO
from ragbench.logger import setup_logger
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from ragbench.jsonl_utils import RagBenchDataRow, load_jsonl_files
from ragbench.utils import load_tokenizer_and_pretrained_model

class RagBenchDPODataRow(RagBenchDataRow):
    fictitious_filename_chosen: str
    fictitious_content_chosen: str
    chosen: str
    rejected: str

def start(
        model_name: str,
        model_path: Path,
        dataset_directory: Path,
        output_directory: Path
):
    logger = setup_logger(Path(__file__).name)

    dataset = load_jsonl_files(dataset_directory)
    logger.debug(f'Total of {len(dataset)} samples to generate.')

    resolved_output_directory = output_directory.resolve()
    resolved_output_directory.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_tokenizer_and_pretrained_model(model_path, logger)

    with open(resolved_output_directory / (model_name + '.jsonl'), 'w') as out_file:
        num_results = 0
        for index, datarow in enumerate(dataset):
            try:
                response = get_response(datarow, model, tokenizer, Llama3RagDPO())

                output_jsonl = {}
                output_jsonl['filename'] = datarow['filename']
                output_jsonl['content'] = datarow['content']
                output_jsonl['question'] = datarow['question']
                output_jsonl['response'] = response
                num_results += 1
            except Exception as error:
                logger.error(f"Filename: [{datarow['filename']}] errored.")
                logger.error(error, exc_info=True)

            out_file.write(json.dumps(output_jsonl) + '\n')
            logger.info(f'Answer has been generated and saved for file {index + 1} with name: {datarow["filename"]}')

        logger.info(f'Done, {num_results}/{len(dataset)} answers generated.')

def get_response(
    datarow: RagBenchDataRow,
    model: Optional[PreTrainedModel],
    tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast],
    prompt_style: PromptStyle
):
    if model is None or tokenizer is None:
        return None, None
    
    prompt = prompt_style.apply(datarow)['prompt']
    input = tokenizer(prompt, return_tensors="pt").to('cuda')
    generation_output = model.generate(
        input.input_ids,
        max_new_tokens=6000,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_ids = generation_output[0][input.input_ids.shape[1] : -1]
    answer = tokenizer.decode(generated_ids)

    return answer

if __name__ == '__main__':
    from jsonargparse import CLI
    CLI(start)
