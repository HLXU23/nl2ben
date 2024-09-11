"""
Group databases and corresponding question-SQL pairs of Bird dev together.
"""

import json
import json
from pathlib import Path
from typing import List, TypedDict, Dict
from collections import defaultdict

from utils import database_to_json

QUESTIONS_FILE = Path("bird_dev/dev.json")

def split_all_questions() -> Dict[str, List[Dict]]:
    dict = defaultdict(lambda: [])
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for obj in data:
        db_id = obj['db_id']
        dict[db_id].append(obj)
    return dict

def dump_questions(data: Dict[str, List[Dict]]):
    parent_folder = Path("bird_dev/")
    for db_id, qsql_pairs in data.items():
        folder = parent_folder / db_id
        assert folder.exists(), f"Something went wrong! {db_id} folder not found!"
        output_file = folder / 'questions.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qsql_pairs, f, ensure_ascii=False, indent=4)
                   

if __name__ == "__main__":
    dump_questions(split_all_questions())

