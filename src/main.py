import os
import json
import sqlite3
import logging
import argparse
from datetime import datetime

from pipeline.schema_generation import schema_generation
from pipeline.schema_preprocess import schema_preprocess
from pipeline.data_rule_generation import data_rule_generation
from pipeline.data_generation import data_generation
from pipeline.evidence_generation import evidence_generation
from pipeline.temp_generation import temp_generation
from pipeline.ques_generation import ques_generation
from pipeline.ques_revision import ques_revision

pipeline_step_mapping = {
    'schema_generation': schema_generation,
    'schema_preprocess': schema_preprocess,
    'data_rule_generation': data_rule_generation,
    'data_generation': data_generation,
    'evidence_generation': evidence_generation,
    'temp_generation': temp_generation,
    'ques_generation': ques_generation,
    'ques_revision': ques_revision,
}

pipeline_config_path = './pipeline_configs.json'
run_config_path = './run_configs.json'
output_root = '../output/'
result_root = '../result/'
history_root = '../history/'

def main():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    try: 
        # Load pipeline configurations
        with open(pipeline_config_path, 'r') as file:
            pipeline_configs = json.load(file)
        print('Load pipeline configurations successfully.')
        # Load run configurations
        with open(run_config_path, 'r') as file:
            run_configs = json.load(file)
        print('Load run configurations successfully.')
    except Exception as e:
        print(f'Load pipeline configurations failed: {e}')

    db_name = next(iter(run_configs))

    folder_name = f'{db_name}-{current_time}'
    os.makedirs(os.path.join(history_root, folder_name), exist_ok=True)
    history_log_path = os.path.join(history_root, folder_name, f'{folder_name}.log')
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(history_log_path),
            logging.StreamHandler()
        ]
    )
    
    os.makedirs(os.path.join(output_root, folder_name), exist_ok=True)
    os.makedirs(os.path.join(result_root, folder_name), exist_ok=True)

    for pipeline_step in run_configs[db_name]:
        
        if pipeline_step not in pipeline_step_mapping:
            logging.error(f'Unknown pipeline step {pipeline_step}.')
            continue

        if run_configs[db_name][pipeline_step] == 1:
            
            pipeline_config = pipeline_configs[pipeline_step]

            pipeline_config['output_path'] = os.path.join(output_root, folder_name, pipeline_step)
            pipeline_config['result_path'] = os.path.join(result_root, folder_name, pipeline_step)

            os.makedirs(pipeline_config['output_path'], exist_ok=True)
            os.makedirs(pipeline_config['result_path'], exist_ok=True)

            logging.info(f'########### [{db_name}][{pipeline_step}] ###########')
            try: 
                pipeline_step_mapping[pipeline_step](db_name, pipeline_config)
                logging.info(f'[{db_name}][{pipeline_step}] Step success')
            except Exception as e:
                logging.error(f'[{db_name}][{pipeline_step}] Step failed: {e}')
                raise

if __name__ == '__main__':
    main()