import logging
import sys
import os
from datetime import datetime
from pathlib import Path

def setup_logging(log_level=logging.INFO, log_to_file=True):
    """Configure application-wide logging"""
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"mockup_generator_{timestamp}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # Create file handler if requested
    if log_to_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)
    
    return root_logger

def get_logger(name):
    """Get a logger with the specified name"""
    return logging.getLogger(name) 