import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    namespace: str, logging_level=logging.DEBUG, logfile_name: Optional[str] = None
):
    # Create a custom logger
    logger = logging.getLogger(namespace)

    # Set the logging level
    logger.setLevel(logging.DEBUG)

    # Create handlers
    logfile = f'{logfile_name if logfile_name else namespace}.log'
    logfile_path = Path('logs/' + logfile)
    file_handler = logging.FileHandler(logfile_path)
    console_handler = logging.StreamHandler()

    # Set the logging level for the handlers
    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
