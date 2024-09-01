import json
import os
from typing import Tuple, TypedDict

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from lit_gpt.prompts import PromptStyle
from ragbench.jsonl_utils import RagBenchDataRow
from ragbench.utils import RandomInt

SEED = 888

random_binary_choice = RandomInt(1, 2, SEED)


class RagSet(TypedDict):
    filename: str
    content: str


def get_filenames(filepath):
    files_done = set()
    if os.path.isfile(filepath):
        with open(filepath, 'r', encoding='utf-8') as generated_answer_file:
            for line in generated_answer_file:
                line = line.strip()
                if line:
                    files_done.add(json.loads(line)['filename'])
    return files_done


def generate_answer(
    datarow: RagBenchDataRow,
    template: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
):
    prompt = get_prompt(datarow, template)

    # print('=' * 80)
    # print('Processing prompt:')
    # print('=' * 80)
    # print(prompt)

    model_inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=True).cuda()

    generation_output = model.generate(
        **model_inputs,
        max_new_tokens=4096,
        # Greedy decode
        do_sample=False,
        top_p=None,
        temperature=None, # Setting do_sample=False means the output will be deterministic, and to remove the runtime warning this should be set to None.
    )
    generated_ids = generation_output[0]
    answer = tokenizer.decode(generated_ids)
    answer = remove_prompt_from_answer(prompt, answer)

    # print('=' * 80)
    # print('Got answer:')
    # print('=' * 80)
    # print(answer)

    return answer


def get_prompt(datarow: RagBenchDataRow, template: str) -> str | Tuple[str, str]:
    rag_set, choice = prepare_rag_set(datarow)
    question = datarow['question']
    prompt = format_prompt(rag_set, question, template)
    return prompt


def prepare_rag_set(datarow: RagBenchDataRow) -> Tuple[Tuple[RagSet, RagSet], int]:
    real_content = datarow['content']
    real_filename = datarow['filename']

    # randomly choose which fictitious content
    choice = random_binary_choice.next()
    fictitious_content = datarow[f'fictitious{choice}_content']
    fictitious_filename = datarow[f'fictitious{choice}_filename']

    # choose whether the real or fictitious go first
    dice = random_binary_choice.next()
    if dice == 1:
        # real goes first
        return (
            {'filename': real_filename, 'content': real_content},
            {'filename': fictitious_filename, 'content': fictitious_content}
        ), choice

    else:
        # fake one goes first
        return (
            {'filename': fictitious_filename, 'content': fictitious_content},
            {'filename': real_filename, 'content': real_content}
        ), choice


def format_prompt(rag_set: Tuple[RagSet, RagSet], question: str, template: str) -> str | Tuple[str, str]:
    return PromptStyle.from_name(template).apply(
        '',
        system='',
        filename1=rag_set[0]['filename'],
        information1=rag_set[0]['content'],
        filename2=rag_set[1]['filename'],
        information2=rag_set[1]['content'],
        question=question
    )


def remove_prompt_from_answer(prompt: str, answer: str) -> str:
    assert prompt in answer, "Prompt not found in the answer. The model's output is likely wrong, or the output contains only the answer."
    response = answer.split(prompt)[-1]
    response = response.lstrip().removesuffix('</s>')
    return response


if __name__ == '__main__':
    raise NotImplementedError
