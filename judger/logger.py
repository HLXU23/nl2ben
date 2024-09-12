import logging
from pathlib import Path
from typing import Optional
import threading
import os

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = (BASE_DIR / 'logs').resolve()
os.makedirs(LOGS_DIR, exist_ok=True)

# Create a lock object for thread safety
log_lock = threading.Lock()

def setup_logger(
    namespace: str, logging_level=logging.DEBUG, logfile_name: Optional[str] = None
):
    with log_lock:
        # Create a custom logger
        logger = logging.getLogger(namespace)

        # Check if the logger already has handlers to prevent adding multiple
        if not logger.hasHandlers():
            # Set the logging level
            logger.setLevel(logging.DEBUG)

            # Create handlers
            logfile = f'{logfile_name if logfile_name else namespace}.log'
            logfile_path = LOGS_DIR / logfile
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