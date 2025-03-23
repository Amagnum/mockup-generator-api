import cv2
import numpy as np
import os
import traceback
import time
import inspect
from functools import wraps
from pathlib import Path

# Create debug directory if it doesn't exist
debug_dir = Path("debug")
debug_dir.mkdir(exist_ok=True)

def save_debug_image(image, name, session_id=None):
    """Save an image for debugging purposes"""
    if image is None:
        return
    
    # Create session directory if provided
    if session_id:
        session_dir = debug_dir / session_id
        session_dir.mkdir(exist_ok=True)
        path = session_dir / f"{name}.png"
    else:
        path = debug_dir / f"{name}.png"
    
    # Convert to BGR if it's a single channel image
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        # For depth maps, normalize to 0-255 range
        if image.dtype == np.uint16:
            # Normalize 16-bit depth map for visualization
            normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image_to_save = normalized.astype(np.uint8)
        else:
            image_to_save = image
        
        # Convert to BGR
        image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_GRAY2BGR)
    else:
        image_to_save = image.copy()
    
    # Save the image
    cv2.imwrite(str(path), image_to_save)
    return str(path)

def debug_timing(func):
    """Decorator to measure and log function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Get the logger for the module where the function is defined
        from app.utils.logging_config import get_logger
        logger = get_logger(func.__module__)
        
        logger.debug(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper

def debug_exception(func):
    """Decorator to provide detailed exception information"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get the logger for the module where the function is defined
            from app.utils.logging_config import get_logger
            logger = get_logger(func.__module__)
            
            # Get the full stack trace
            stack_trace = traceback.format_exc()
            
            # Log the exception with detailed information
            logger.error(f"Exception in {func.__name__}: {str(e)}")
            logger.error(f"Stack trace:\n{stack_trace}")
            
            # Re-raise the exception
            raise
    return wrapper

def debug_function_args(func):
    """Decorator to log function arguments"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the logger for the module where the function is defined
        from app.utils.logging_config import get_logger
        logger = get_logger(func.__module__)
        
        # Get argument names
        arg_names = inspect.getfullargspec(func).args
        
        # Format positional arguments (excluding 'self' for methods)
        pos_args = args[1:] if arg_names and arg_names[0] == 'self' else args
        pos_arg_names = arg_names[1:] if arg_names and arg_names[0] == 'self' else arg_names
        
        # Create a dictionary of argument names and values
        arg_values = {name: value for name, value in zip(pos_arg_names, pos_args)}
        arg_values.update(kwargs)
        
        # Format argument values for logging (handle special cases like numpy arrays)
        formatted_args = {}
        for name, value in arg_values.items():
            if isinstance(value, np.ndarray):
                formatted_args[name] = f"ndarray(shape={value.shape}, dtype={value.dtype})"
            else:
                formatted_args[name] = value
        
        logger.debug(f"Calling {func.__name__} with args: {formatted_args}")
        return func(*args, **kwargs)
    return wrapper 