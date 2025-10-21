#!/usr/bin/env python3
# Common Gemini API client functions for both Streamlit app and batch processing
import logging
from typing import Optional, Dict, Union

from google import genai
from google.genai import types

from core.utils import execute_with_timeout
from core.error_handler import ErrorHandler

logger = logging.getLogger(__name__)


def call_gemini_api(prompt: str, client, model_name: str, 
                   query_type: str = "molecular_search",
                   config_override: Optional[Dict] = None) -> Optional[Union[str, Dict]]:
    """Common function to call Gemini API with query-type specific parameters and grounding metadata support.
    
    Args:
        prompt: Prompt text to send to Gemini API
        client: Gemini client instance
        model_name: Model name to use
        query_type: Type of query ('molecular_search', 'molecular_analysis', 'similar_molecule_search')
        config_override: Optional configuration override dictionary
        
    Returns:
        Response text (str) for backward compatibility
    """
    logger.info(f"Calling Gemini API with prompt length: {len(prompt)}, query_type: {query_type}")
    
    # Import Config here to avoid circular imports
    from config.settings import Config
    
    # Get configuration for query type
    base_config = Config.GEMINI_CONFIG.get(query_type, Config.GEMINI_CONFIG['molecular_search'])
    
    # Override with custom config if provided
    if config_override:
        base_config = {**base_config, **config_override}
    
    def api_call():
        """Execute API call with query-type specific configuration."""
        config = types.GenerateContentConfig(
            temperature=base_config['temperature'],
            top_p=base_config['top_p'],
            top_k=base_config['top_k'],
            max_output_tokens=base_config['max_output_tokens']
        )
        
        # Add Google Search tool if requested
        if base_config['use_google_search']:
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
        base_config['timeout'], 
        "api_error"
    )

    if response is None:
        logger.warning("No response received from Gemini API")
        return None
    
    try:
        logger.info("Successfully received response from Gemini API")
        
        # For backward compatibility, return text only
        return response.text
        
    except Exception as e:
        logger.error(f"Error processing Gemini API response: {e}")
        return None
