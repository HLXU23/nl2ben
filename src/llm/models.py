import json
import queue
import random
import asyncio
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from llm.engine_configs import ENGINE_CONFIGS, get_response, get_response_async

# def call_llm_chain(
#     prompt: Any, 
#     config: Dict[str, Any], 
#     log_file_lock: threading.Lock, 
#     max_attempts: int = 5, 
#     backoff_base: int = 2, 
#     jitter_max: int = 60
# ) -> Any:
#     """
#     Calls the LLM chain with exponential backoff and jitter on failure.

#     Args:
#         prompt (Any): The prompt to be passed to the chain.
#         engine (Any): The engine to be used in the chain.
#         request_kwargs (Dict[str, Any]): The request arguments.
#         step (int): The current step in the process.
#         log_file_lock (threading.Lock): The lock for logging into the file.
#         max_attempts (int, optional): The maximum number of attempts. Defaults to 12.
#         backoff_base (int, optional): The base for exponential backoff. Defaults to 2.
#         jitter_max (int, optional): The maximum jitter in seconds. Defaults to 60.

#     Returns:
#         Any: The output from the chain.

#     Raises:
#         Exception: If all attempts fail.
#     """
#     for attempt in range(max_attempts):
#         try:
#             logging.debug(f'Human: \n{prompt}\n')
#             output = get_response(prompt, config)
#             logging.debug(f'AI: \n{output}')
#             return output
#         except KeyError as e:
#             logging.error(f"KeyError encountered: {e}")
#             raise e
#         except Exception as e:
#             if attempt < max_attempts - 1:
#                 logging.warning(f"Attempt {attempt} failed with error: {e}. Retrying...")
#                 sleep_time = (backoff_base ** attempt) + random.uniform(0, jitter_max)
#                 await asyncio.sleep(sleep_time)
#             else:
#                 logging.error(f"All {max_attempts} attempts failed. Last error: {type(e)} <{e}>")
#                 raise e

# def threaded_llm_call(
#     request_id: int, 
#     prompt: Any, 
#     config: Dict[str, Any], 
#     result_queue: queue.Queue, 
#     log_file_lock: threading.Lock
# ) -> None:
#     """
#     Makes a threaded call to the LLM chain and stores the result in a queue.

#     Args:
#         request_id (int): The ID of the request.
#         prompt (Any): The prompt to be passed to the chain.
#         engine (Any): The engine to be used in the chain.
#         request_kwargs (Dict[str, Any]): The request arguments.
#         step (int): The current step in the process.
#         result_queue (queue.Queue): The queue to store results.
#         log_file_lock (threading.Lock): The lock for logging into the file.
#     """
#     try:
#         result = call_llm_chain(prompt, config, log_file_lock)
#         result_queue.put((request_id, result))  # Store a tuple of request ID and its result
#     except Exception as e:
#         result_queue.put((request_id, None))  # Indicate failure for this request

# def llm_call(
#     prompt: Any, 
#     config: Dict[str, Any], 
#     request_list: List[Dict[str, Any]], 
#     step: str, 
#     sampling_count: int
# ) -> List[List[Any]]:
#     """
#     Asynchronously calls the LLM chain using multiple threads.

#     Args:
#         prompt (Any): The prompt to be passed to the chain.
#         engine (Any): The engine to be used in the chain.
#         request_list (List[Dict[str, Any]]): The list of request arguments.
#         step (int): The current step in the process.
#         sampling_count (int): The number of samples to be taken.

#     Returns:
#         List[List[Any]]: A list of lists containing the results for each request.
#     """
#     result_queue = queue.Queue()  # Queue to store results
#     log_file_lock = threading.Lock()

#     with ThreadPoolExecutor(max_workers=len(request_list) * sampling_count) as executor:
#         for request_id, request_kwargs in enumerate(request_list):
#             for _ in range(sampling_count):
#                 executor.submit(threaded_llm_call, request_id, prompt, config, result_queue, log_file_lock)
#                 time.sleep(0.2)  # Sleep for a short time to avoid rate limiting

#     results = []
#     while not result_queue.empty():
#         results.append(result_queue.get())
        
#     # Sort results based on their request IDs
#     results = sorted(results, key=lambda x: x[0])
#     sorted_results = [result[1] for result in results]

#     # Group results by sampling_count
#     grouped_results = [sorted_results[i * sampling_count: (i + 1) * sampling_count] for i in range(len(request_list))]

#     return grouped_results

async def call_llm_chain_async(
    prompt: str,
    config: Dict[str, Any],
    max_attempts: int = 5,
    backoff_base: int = 2,
    jitter_max: int = 60
) -> Any:
    """
    Asynchronously calls the LLM chain with retries using exponential backoff and jitter.

    Args:
        prompt (str): The prompt to send to the LLM.
        config (Dict[str, Any]): Configuration parameters for the LLM.
        max_attempts (int, optional): Maximum number of retry attempts. Defaults to 5.
        backoff_base (int, optional): Base for exponential backoff. Defaults to 2.
        jitter_max (int, optional): Maximum jitter in seconds. Defaults to 60.

    Returns:
        Any: The response from the LLM.

    Raises:
        Exception: If all retry attempts fail.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            logging.debug(f'Human Prompt:\n{prompt}\n')
            output = await get_response_async(prompt, config)
            logging.debug(f'AI Response:\n{output}\n')
            return output
        except KeyError as e:
            logging.error(f"KeyError encountered: {e}")
            raise e
        except Exception as e:
            if attempt < max_attempts:
                sleep_time = (backoff_base ** attempt) + random.uniform(0, jitter_max)
                logging.warning(
                    f"Attempt {attempt} failed with error: {e}. "
                    f"Retrying in {sleep_time:.2f} seconds..."
                )
                await asyncio.sleep(sleep_time)
            else:
                logging.error(
                    f"All {max_attempts} attempts failed. Last error: {type(e).__name__} <{e}>"
                )
                raise e

# Asynchronous function to handle multiple prompts concurrently
async def llm_call_async(
    request_list: List[str],  # Each element is a different prompt
    config: Dict[str, Any],
    step: str,
    sampling_count: int
) -> List[List[Any]]:
    """
    Asynchronously calls the LLM chain for multiple prompts concurrently using asyncio.

    Args:
        request_list (List[str]): List of different prompts to send.
        config (Dict[str, Any]): Configuration parameters for the LLM.
        step (str): Description of the current step.
        sampling_count (int): Number of samples to generate for each prompt.

    Returns:
        List[List[Any]]: A list where each sublist contains the sampling results for a prompt.
    """
    semaphore = asyncio.Semaphore(100)  # Limit the number of concurrent requests

    async def bound_call(prompt: str, prompt_id: int) -> Any:
        async with semaphore:
            try:
                result = await call_llm_chain_async(prompt, config)
                return result
            except Exception as e:
                logging.error(f"Prompt ID {prompt_id} failed with error: {e}")
                return None

    tasks = []
    for prompt_id, prompt in enumerate(request_list):
        for _ in range(sampling_count):
            task = asyncio.create_task(bound_call(prompt, prompt_id))
            tasks.append((prompt_id, task))

    results = [[] for _ in request_list]

    for prompt_id, task in tasks:
        result = await task
        results[prompt_id].append(result)

    return results

def chunked(iterable: List[Any], size: int) -> List[List[Any]]:
    """
    Splits a list into chunks of specified size.

    Args:
        iterable (List[Any]): The list to split.
        size (int): The size of each chunk.

    Returns:
        List[List[Any]]: A list of chunks.
    """
    return [iterable[i:i + size] for i in range(0, len(iterable), size)]

async def batch_llm_call_async(
    prompts: List[Any],
    config: Dict[str, Any],
    step: str,
    batch_size: int = 5,
    sampling_count: int = 1,
) -> List[List[Any]]:
    """
    Processes prompts in batches asynchronously.

    Args:
        prompts (List[Any]): List of prompts to process.
        config (Dict[str, Any]): Configuration parameters for the LLM.
        batch_size (int, optional): Number of prompts per batch. Defaults to 10.
        sampling_count (int, optional): Number of samples per prompt. Defaults to 1.

    Returns:
        List[List[Any]]: A list where each sublist contains the sampling results for a prompt.
    """
    step = "batch_processing"
    all_results = [[] for _ in prompts]

    batches = chunked(prompts, batch_size)

    for batch_num, batch in enumerate(batches, start=1):
        logging.info(f"Processing batch {batch_num}/{len(batches)} with {len(batch)} prompts.")
        try:
            batch_results = await llm_call_async(batch, config, step, sampling_count)
            start_index = (batch_num - 1) * batch_size
            for i, prompt_results in enumerate(batch_results):
                all_results[start_index + i].extend(prompt_results)
        except Exception as e:
            logging.error(f"An error occurred while processing batch {batch_num}: {e}")
            continue

    return all_results