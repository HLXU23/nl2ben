import glob
import json
import os
from pathlib import Path
from typing import List, TypedDict


class RagBenchDataRow(TypedDict):
    filename: str
    question: str
    answer: str
    content: str
    content_before: str
    content_after: str
    fictitious1_filename: str
    fictitious1_content: str
    fictitious2_filename: str
    fictitious2_content: str


def process_json_dict(dict_obj: RagBenchDataRow):
    pass

    # The code below is no longer needed after cleaning the dataset
    # if '{' in dict_obj['question']:
    #     dict_obj['question'] = dict_obj['question'].replace(
    #         '\\', ''
    #     )
    #     try:
    #         question_dict = json.loads(dict_obj['question'])
    #         if 'question' in question_dict:
    #             dict_obj['question'] = question_dict['question']
    #         else:
    #             dict_obj['question'] = list(question_dict.values())[-1]
    #     except Exception:
    #         pass

    # dict_obj['question'] = (
    #     dict_obj['question']
    #         .removeprefix('1) ')
    #         .removeprefix('1. ')
    # )


def load_jsonl_file(file_path: Path) -> List[RagBenchDataRow]:
    """Loads the specified JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            dict_obj = json.loads(line)
            process_json_dict(dict_obj)
            data.append(dict_obj)

    assert len(data) > 0, \
        f'{file_path.name} is empty.'

    return data


def load_jsonl_files(directory: Path, empty_ok: bool = False) -> List[RagBenchDataRow]:
    """Recursively load all JSONL files within the directory and return a list of dictionaries."""
    all_dicts = []
    jsonl_files = glob.glob(os.path.join(directory, '**', '*.jsonl'), recursive=True)
    if not empty_ok:
        assert len(jsonl_files) > 0, \
            f'"{directory}" is empty.'

    for file in jsonl_files:
        all_dicts.extend(load_jsonl_file(Path(file)))

    if not empty_ok:
        assert len(all_dicts) > 0

    return sorted(all_dicts, key=lambda d: d['filename']) # Sort to make deterministic


def get_jsonl_paths(directory: Path) -> List[Path]:
    """Recursively searches for all JSONL paths from a directory."""
    jsonl_files = glob.glob(os.path.join(directory, '**', '*.jsonl'), recursive=True)
    jsonl_files.sort()
    return [Path(file).expanduser().resolve() for file in jsonl_files]


def rename_file(filepath: Path, new_name: str) -> Path:
    """Rename the file at the given filepath to the new_name and return the new file path."""
    directory = filepath.parent
    new_path = directory / new_name

    os.rename(filepath, new_path)
    return new_path
