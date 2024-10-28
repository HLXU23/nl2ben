import os
import json
import aiohttp
import logging
import requests
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()
ENGINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "deepseek-coder": {
        "type": "llm",
        "uri": "https://api.deepseek.com/chat/completions",
        "api_key": os.getenv('DEEPSEEK_API_KEY')
    }
}

def get_response(prompt: Any, config: Dict[str, Any]) -> Any:
    try:
        engine = config['engine']
        engine_config = ENGINE_CONFIGS[engine]
    except KeyError as e:
        raise KeyError(f"Engine configuration missing or invalid: {e}")

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


async def get_response_async(prompt: Any, config: Dict[str, Any]) -> Any:
    """
    Asynchronously gets a response from the LLM API.

    Args:
        prompt (Any): The prompt to send to the LLM.
        config (Dict[str, Any]): Configuration for the LLM, must include 'engine', 'temperature', etc.

    Returns:
        Any: The response content from the LLM.

    Raises:
        KeyError: If required configuration is missing.
        ConnectionError: If the API request fails.
        ValueError: If the response format is invalid.
    """
    try:
        engine = config['engine']
        engine_config = ENGINE_CONFIGS[engine]
    except KeyError as e:
        raise KeyError(f"Engine configuration missing or invalid: {e}")

    if engine_config.get('type') == 'llm':
        try:
            uri = engine_config['uri']
            api_key = engine_config['api_key']
            temperature = config['temperature']
            logging.debug('LLM configuration loaded successfully.')
        except KeyError as e:
            raise KeyError(f'Loading LLM configurations failed: {e}')

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        params = {
            "model": engine,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
        }
        logging.debug(f'Sending prompt to {engine}')

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(uri, headers=headers, json=params) as response:
                    if response.status == 200:
                        resp_json = await response.json()
                        logging.debug(f'Received successful response: {resp_json}')

                        choices = resp_json.get('choices')
                        if not choices or not isinstance(choices, list):
                            raise ValueError("Invalid response format: 'choices' field missing or not a list.")
                        
                        message = choices[0].get('message')
                        if not message or 'content' not in message:
                            raise ValueError("Invalid response format: 'message' field missing or 'content' missing.")
                        
                        result = message['content']
                        return result
                    else:
                        resp_text = await response.text()
                        error_message = (
                            f"Request to {engine} failed with status code {response.status}. "
                            f"Response content: {resp_text}"
                        )
                        logging.error(error_message)
                        raise ConnectionError(error_message)
            except aiohttp.ClientError as e:
                # Handle client-side errors (e.g., network issues)
                error_message = f"Failed to connect to {uri}: {e}"
                logging.error(error_message)
                raise ConnectionError(error_message) from e
    else:
        error_message = f"Engine type '{engine_config.get('type')}' is not supported."
        logging.error(error_message)
        raise ValueError(error_message)
