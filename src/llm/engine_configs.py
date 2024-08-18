import json
import logging
import requests
from typing import Dict, Any

ENGINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "deepseek-coder": {
        "type": "llm",
        "uri": "https://api.deepseek.com/chat/completions",
        "api_key": ""
    }
}

def get_response(prompt: Any, config: Dict[str, Any]) -> Any:
    engine = config['engine']
    engine_config = ENGINE_CONFIGS[engine]
        
    if engine_config['type'] == 'llm':
        
        try: 
            uri = engine_config['uri']
            api_key = engine_config['api_key']
            temperature = config['temperature']
            logging.debug('Got LLM config')
        except Exception as e:
            raise KeyError(f'Loading LLM configurations failed: {e}')
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        params = {
            "model": engine,
            "messages": [
                {"role":"user", "content": prompt}
            ],
            "temperature": temperature,
        }
        logging.debug(f'send prmopt to {engine}')
        response = requests.post(uri, headers=headers, data=json.dumps(params))
        if response.status_code == 200:
            result = response.json().get('choices')[0].get('message').get('content')
            return result
        else:
            error_message = (
                f"Request to {engine} failed with status code {response.status_code}. "
                f"Response content: {response.text}"
            )
            raise ConnectionError(error_message)
