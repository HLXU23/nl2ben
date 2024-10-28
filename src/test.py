import logging
from collections import deque
from typing import Dict, List, Any

import re
import os
import json
import random
import shutil
import sqlite3
import asyncio
import logging
from llm.models import batch_llm_call_async

if __name__ == "__main__":

    test_prompts = [
        "1 * 1 = ?",
        "10 * 10 = ?",
        "100 * 100 = ?"
    ]

    test_config = {
        'engine': 'deepseek-coder',
        'temperature': 1
    }

    if not os.getenv('DEEPSEEK_API_KEY'):
        logging.error("Environment variable 'DEEPSEEK_API_KEY' is not set.")
    else:
        all_results = asyncio.run(batch_llm_call_async(test_prompts, test_config, 'test'))
        print(all_results)