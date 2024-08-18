import json
import queue
import random
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from llm.engine_configs import ENGINE_CONFIGS, get_response


def call_llm_chain(prompt: Any, config: Dict[str, Any], log_file_lock: threading.Lock, max_attempts: int = 5, backoff_base: int = 2, jitter_max: int = 60) -> Any:
    """
    Calls the LLM chain with exponential backoff and jitter on failure.

    Args:
        prompt (Any): The prompt to be passed to the chain.
        engine (Any): The engine to be used in the chain.
        request_kwargs (Dict[str, Any]): The request arguments.
        step (int): The current step in the process.
        log_file_lock (threading.Lock): The lock for logging into the file.
        max_attempts (int, optional): The maximum number of attempts. Defaults to 12.
        backoff_base (int, optional): The base for exponential backoff. Defaults to 2.
        jitter_max (int, optional): The maximum jitter in seconds. Defaults to 60.

    Returns:
        Any: The output from the chain.

    Raises:
        Exception: If all attempts fail.
    """
    for attempt in range(max_attempts):
        try:
            logging.debug(f'Human: \n{prompt}\n')
            output = get_response(prompt, config)
            # with log_file_lock:
            logging.debug(f'AI: \n{output}')
            return output
        except KeyError as e:
            raise e
        except Exception as e:
            if attempt < max_attempts - 1:
                logging.warning(f"Failed to invoke the engine {attempt + 1} times.\n")
                sleep_time = (backoff_base ** attempt) + random.uniform(0, jitter_max)
                time.sleep(sleep_time)
            else:
                logging.error(f"Failed to invoke the engine {attempt + 1} times.\n{type(e)} <{e}>\n")
                raise e

def threaded_llm_call(request_id: int, prompt: Any, config: Dict[str, Any], result_queue: queue.Queue, log_file_lock: threading.Lock) -> None:
    """
    Makes a threaded call to the LLM chain and stores the result in a queue.

    Args:
        request_id (int): The ID of the request.
        prompt (Any): The prompt to be passed to the chain.
        engine (Any): The engine to be used in the chain.
        request_kwargs (Dict[str, Any]): The request arguments.
        step (int): The current step in the process.
        result_queue (queue.Queue): The queue to store results.
        log_file_lock (threading.Lock): The lock for logging into the file.
    """
    try:
        result = call_llm_chain(prompt, config, log_file_lock)
        result_queue.put((request_id, result))  # Store a tuple of request ID and its result
    except Exception as e:
        result_queue.put((request_id, None))  # Indicate failure for this request

def async_llm_call(prompt: Any, config: Dict[str, Any], request_list: List[Dict[str, Any]], step: str, sampling_count: int) -> List[List[Any]]:
    """
    Asynchronously calls the LLM chain using multiple threads.

    Args:
        prompt (Any): The prompt to be passed to the chain.
        engine (Any): The engine to be used in the chain.
        request_list (List[Dict[str, Any]]): The list of request arguments.
        step (int): The current step in the process.
        sampling_count (int): The number of samples to be taken.

    Returns:
        List[List[Any]]: A list of lists containing the results for each request.
    """
    result_queue = queue.Queue()  # Queue to store results
    log_file_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=len(request_list) * sampling_count) as executor:
        for request_id, request_kwargs in enumerate(request_list):
            for _ in range(sampling_count):
                executor.submit(threaded_llm_call, request_id, prompt, config, result_queue, log_file_lock)
                time.sleep(0.2)  # Sleep for a short time to avoid rate limiting

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
        
    # Sort results based on their request IDs
    results = sorted(results, key=lambda x: x[0])
    sorted_results = [result[1] for result in results]

    # Group results by sampling_count
    grouped_results = [sorted_results[i * sampling_count: (i + 1) * sampling_count] for i in range(len(request_list))]

    return grouped_results
