from typing import List

import pytest

from ragbench.inference.answers import format_prompt, prepare_rag_set
from ragbench.jsonl_utils import load_jsonl_files

templates: List[str] = [
    'rag-baseline',
    'rag-baseline-xml',
    'rag-enhanced',
    'rag-enhanced-xml',
    'llama-2-rag',
    'llama-2-rag-xml'
]


def test_generate_prompts():
    """Test the generation of all prompts with all prompt styles and validation samples."""
    dataset_list = load_jsonl_files('dataset')

    cnt = 0
    for template in templates:
        for datarow in dataset_list:
            rag_set, choice = prepare_rag_set(datarow)
            question = datarow['question']
            try:
                prompt = format_prompt(rag_set, question, template)
            except AssertionError:
                pytest.fail(f'ERROR: {rag_set} is invalid.')

            cnt += 1

    assert cnt == len(templates) * len(dataset_list)


# Run the tests with pytest
if __name__ == '__main__':
    test_generate_prompts()
