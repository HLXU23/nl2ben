import random
from logging import Logger

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


class RandomInt:
    def __init__(self, start=0, end=1, seed=None):
        """
        Initializes the random integer generator with a specified range and seed.

        Parameters:
        start (int): The lower limit for the range (inclusive).
        end (int): The upper limit for the range (inclusive).
        seed (int, optional): The seed for the random number generator.
        """
        self.start = start
        self.end = end
        if seed is not None:
            random.seed(seed)

    def next(self):
        """
        Returns the next random integer from the specified range.

        Returns:
        int: A random integer from start to end.
        """
        return random.randint(self.start, self.end)


def convert_to_number(value) -> bool | int | float:
    # Check if the value is already an integer or a float
    if isinstance(value, (int, float)):
        return value

    # If the value is a string, try to convert it
    if isinstance(value, str):
        try:
            # First attempt to convert to float
            float_val = float(value)
            # If the string is a valid integer, convert to int instead
            if float_val.is_integer():
                return int(float_val)
            return float_val
        except ValueError:
            return False

    # If none of the conditions are met, return False
    return False


def load_tokenizer_and_pretrained_model(model_id: str, logger: Logger):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    logger.debug('Loading model...')
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # flash attention can only be used when model is fp16 or bfloat16
        attn_implementation='flash_attention_2',
        device_map=torch.device('cuda'),
    )
    logger.debug('Model loaded. Compiling it...')
    model = torch.compile(model)
    return tokenizer, model
