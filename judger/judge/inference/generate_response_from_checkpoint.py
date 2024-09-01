import os
import json
import glob
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ProcessPoolExecutor
from jsonargparse import CLI, ArgumentParser
from lit_gpt.prompts import Llama3RagDPO
from ragbench.jsonl_utils import load_jsonl_files

PROMPT_STYLE = Llama3RagDPO()

def generate_responses(checkpoint_folder: str, device: str, data: list):
    """
    Generates responses from a Huggingface model for given data using a specified device (GPU/CPU).

    Parameters:
    checkpoint_folder (str): Path to the directory containing the model checkpoint.
    device (str): The device to use for model inference ('cuda' for GPU or 'cpu').
    data (list): A list of dictionaries containing prompt data.

    Returns:
    str: A message indicating the completion of processing for the given checkpoint on the specified device.
    """
    checkpoint_path = Path(checkpoint_folder)
    
    if not checkpoint_path.is_dir():
        return f'The path \"{checkpoint_folder}\" is not a directory or it does not exist. Please make sure you have indicated the correct checkpoint folder.'

    model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    responses = []
    
    for datarow in data:
        prompt_text = PROMPT_STYLE.apply(datarow)['prompt']
        inputs = tokenizer(prompt_text, return_tensors='pt').to(device)
        generation_output = model.generate(
            inputs.input_ids,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            temperature=None
        )
        generated_ids = generation_output[0][input.input_ids.shape[1] : -1]
        response = tokenizer.decode(generated_ids)

        responses.append({
            'checkpoint': str(checkpoint_path),
            'prompt': prompt_text,
            'response': response
        })
    
    response_file = f'responses_{checkpoint_path.name}.jsonl'
    with open(response_file, 'w') as f:
        for response in responses:
            f.write(json.dumps(response) + '\n')

    return f"Finished processing {checkpoint_path} on device {device}"

def start(parent_directory: str, prompts_file: str, num_gpus: int):
    """
    Distributes the workload of generating responses across multiple GPUs or CPU using a round-robin method.

    Parameters:
    parent_directory (str): Path to the parent directory containing all the checkpoint directories.
    prompts_file (str): Path to the JSONL file containing the prompts.
    num_gpus (int): Number of GPUs available for processing.
    """
    parent_directory = Path(parent_directory)

    dataset = load_jsonl_files(prompts_file)

    checkpoint_folders = glob.glob(str(parent_directory / 'checkpoint-*'))

    if num_gpus > 0:
        tasks = [(checkpoint_folder, f'cuda:{i % num_gpus}', dataset) for i, checkpoint_folder in enumerate(checkpoint_folders)]
    else:
        tasks = [(checkpoint_folder, f'cpu', dataset) for checkpoint_folder in checkpoint_folders]

    with ProcessPoolExecutor(max_workers=num_gpus if num_gpus > 2 else 1) as executor:
        executor.map(lambda p: generate_responses(*p), tasks)

if __name__ == "__main__":
    CLI(start)
