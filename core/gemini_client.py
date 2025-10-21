#!/usr/bin/env python3
# Common Gemini API client functions for both Streamlit app and batch processing
import logging
from typing import Optional

from google import genai
from google.genai import types

from core.utils import execute_with_timeout
from core.error_handler import ErrorHandler

logger = logging.getLogger(__name__)


def call_gemini_api(prompt: str, client, model_name: str, 
                   use_google_search: bool = True, timeout: int = 10) -> Optional[str]:
    """Common function to call Gemini API with configurable options.
    
    Args:
        prompt: Prompt text to send to Gemini API
        client: Gemini client instance
        model_name: Model name to use
        use_google_search: Whether to enable Google Search tool
        timeout: Timeout in seconds
        
    Returns:
        Response text or None if failed
    """
    logger.info(f"Calling Gemini API with prompt length: {len(prompt)}")
    
    def api_call():
        """Execute API call with optional Google Search tool."""
        config = types.GenerateContentConfig()
        
        # Add Google Search tool if requested
        if use_google_search:
            search_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
            config.tools = [search_tool]
        
        # Generate content using the model
        return client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config
        )
    
    response = execute_with_timeout(
        api_call, 
        timeout, 
        "api_error"
    )

    if response is None:
        logger.warning("No response received from Gemini API")
        return None
    
    try:
        logger.info("Successfully received response from Gemini API")
        return response.text
    except Exception as e:
        logger.error(f"Error processing Gemini API response: {e}")
        return None
