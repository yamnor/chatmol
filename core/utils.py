#!/usr/bin/env python3
# Common utility functions for both Streamlit app and batch processing
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Callable, Any, Optional

from core.error_handler import ErrorHandler

logger = logging.getLogger(__name__)


def execute_with_timeout(func: Callable, timeout_seconds: int, error_type: str = "timeout") -> Optional[Any]:
    """Execute a function with timeout control using ThreadPoolExecutor.
    
    Args:
        func: Function to execute
        timeout_seconds: Timeout in seconds
        error_type: Error type for error handling
        
    Returns:
        Function result or None if timeout/error occurred
    """
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            return future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        logger.warning(f"Function execution timed out after {timeout_seconds} seconds")
        return None
    except Exception as e:
        logger.error(f"Error in execute_with_timeout: {e}")
        return None
