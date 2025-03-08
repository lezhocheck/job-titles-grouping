import logging
import os
import sys


def setup_logger(logger_name: str, logs_folder: str = './logs') -> logging.Logger:
    logger = logging.getLogger(logger_name)
    os.makedirs(logs_folder, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(logs_folder, f'{logger_name}.log'))
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.INFO) 
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger