#!/usr/bin/env python3
# Common molecular analysis functions for both Streamlit app and batch processing
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

from core.models import DetailedMoleculeInfo
from core.prompts import AIPrompts
from core.error_handler import is_no_result_response, parse_json_response

logger = logging.getLogger(__name__)


def calculate_retry_delay(attempt: int, retry_config: Dict) -> float:
    """Calculate retry delay with exponential backoff."""
    base_delay = retry_config.get('retry_delay_seconds', 5)
    exponential_backoff = retry_config.get('exponential_backoff', True)
    
    if exponential_backoff:
        return base_delay * (2 ** attempt)
    else:
        return base_delay


def get_molecule_name_variations(molecule_name: str) -> List[str]:
    """Get variations of molecule name for retry attempts."""
    variations = [
        molecule_name,
        molecule_name.lower(),
        molecule_name.upper(),
        molecule_name.replace('-', ' '),
        molecule_name.replace(' ', '-'),
        molecule_name.replace('_', ' '),
        molecule_name.replace(' ', '_'),
    ]
    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for variation in variations:
        if variation not in seen:
            seen.add(variation)
            unique_variations.append(variation)
    return unique_variations


def get_similar_molecule_prompt_variation(molecule_name: str, attempt: int) -> str:
    """Get different prompt variations for retry attempts."""
    prompt_variations = [
        AIPrompts.SIMILAR_MOLECULE_SEARCH,
        AIPrompts.SIMILAR_MOLECULE_SEARCH_STRUCTURAL,
        AIPrompts.SIMILAR_MOLECULE_SEARCH_FUNCTIONAL,
        AIPrompts.SIMILAR_MOLECULE_SEARCH_CLASSIFICATION,
    ]
    
    # Cycle through variations
    selected_prompt = prompt_variations[attempt % len(prompt_variations)]
    formatted_prompt = selected_prompt.format(molecule_name=molecule_name)
    
    return formatted_prompt


def should_retry(response_text: str, attempt: int, max_retries: int) -> bool:
    """Determine if we should retry based on response and attempt count."""
    if attempt >= max_retries:
        return False
    
    # Check if response indicates failure conditions
    failure_conditions = [
        not response_text,  # No response
        is_no_result_response(response_text),  # "該当なし"
        not parse_json_response(response_text)  # Invalid JSON
    ]
    
    return any(failure_conditions)


def get_fallback_similar_molecule(molecule_name: str, cache_manager) -> Optional[str]:
    """Get fallback similar molecule from cache when all retries fail."""
    try:
        # Try to get a random similar compound from cache
        normalized_name = molecule_name.lower()
        random_result = cache_manager.similar.get_random_similar_compound(normalized_name)
        if random_result:
            random_compound_name, random_description = random_result
            logger.info(f"Using cached similar compound as final fallback: {random_compound_name}")
            
            # Get proper Japanese and English names for the similar compound
            name_jp, name_en = cache_manager.get_compound_names_for_display(random_compound_name)
            
            # Use English name if Japanese name is not available
            if not name_jp:
                name_jp = name_en
            
            # Create a fallback response with proper names
            fallback_response = f'{{"name_jp": "{name_jp}", "name_en": "{name_en}", "description": "{random_description}"}}'
            return fallback_response
    except Exception as e:
        logger.warning(f"Fallback similar molecule search failed: {e}")
    
    return None


def analyze_molecule_properties(detailed_info: DetailedMoleculeInfo, 
                              molecule_name: str,
                              gemini_client,  # call_gemini_api function or GeminiClient instance
                              cache_manager) -> Optional[str]:
    """Analyze molecular properties and generate human-readable explanation.
    
    Args:
        detailed_info: Detailed molecule information from PubChem
        molecule_name: Name of the molecule to analyze
        gemini_client: Gemini API client (call_gemini_api function or GeminiClient instance)
        cache_manager: Cache manager instance
        
    Returns:
        Analysis result text or None if failed
    """
    logger.info(f"Getting Gemini analysis for molecule: {molecule_name}")
    
    # PubChemの詳細情報を整理
    properties_text = []
    if detailed_info.molecular_formula:
        properties_text.append(f"分子式: {detailed_info.molecular_formula}")
    if detailed_info.molecular_weight:
        properties_text.append(f"分子量: {detailed_info.molecular_weight:.2f}")
    if detailed_info.heavy_atom_count is not None:
        properties_text.append(f"重原子数: {detailed_info.heavy_atom_count}")
    if detailed_info.xlogp is not None:
        properties_text.append(f"LogP: {detailed_info.xlogp:.2f}")
    if detailed_info.tpsa is not None:
        properties_text.append(f"tPSA: {detailed_info.tpsa:.1f} Å²")
    if detailed_info.complexity is not None:
        properties_text.append(f"分子複雑度: {detailed_info.complexity:.1f}")
    if detailed_info.hbond_donor_count is not None:
        properties_text.append(f"水素結合供与体数: {detailed_info.hbond_donor_count}")
    if detailed_info.hbond_acceptor_count is not None:
        properties_text.append(f"水素結合受容体数: {detailed_info.hbond_acceptor_count}")
    if detailed_info.rotatable_bond_count is not None:
        properties_text.append(f"回転可能結合数: {detailed_info.rotatable_bond_count}")
    
    properties_str = "\n".join(properties_text)
    
    prompt = AIPrompts.MOLECULAR_ANALYSIS.format(
        molecule_name=molecule_name,
        properties_str=properties_str
    )
    
    # Call Gemini API using the provided client
    if hasattr(gemini_client, 'call_api'):
        # GeminiClient instance (batch processing)
        response = gemini_client.call_api(prompt, query_type="molecular_analysis")
        if isinstance(response, dict):
            response_text = response['text']
        else:
            response_text = response
    else:
        # call_gemini_api function (Streamlit app)
        response_text = gemini_client(prompt, query_type="molecular_analysis")
    
    if response_text:
        analysis_result = response_text.strip()
        
        # Save analysis result to cache using English name
        # For batch processing, we need to get the English name from the molecule_name
        # For Streamlit app, we can get it from current data if available
        cache_manager.save_analysis_result(molecule_name, analysis_result)
        
        logger.info(f"Analysis completed for: {molecule_name}")
        return analysis_result
    else:
        logger.warning(f"No analysis response for: {molecule_name}")
        return None


def find_similar_molecules(molecule_name: str,
                          gemini_client,
                          cache_manager,
                          current_data: Optional[Dict] = None,
                          is_batch_mode: bool = False,
                          retry_config: Optional[Dict] = None) -> Optional[str]:
    """Find molecules similar to the specified molecule.
    
    Args:
        molecule_name: Name of the molecule to find similar molecules for
        gemini_client: Gemini API client (call_gemini_api function or GeminiClient instance)
        cache_manager: Cache manager instance
        current_data: Current molecule data (for Streamlit app, contains name_en for cache key)
        is_batch_mode: Whether this is called from batch processing (enables retry logic)
        retry_config: Retry configuration for batch processing
        
    Returns:
        Similar molecule response text or None if failed
    """
    logger.info(f"Searching for similar molecules to: {molecule_name}")
    
    # Get English name from current data for cache key (Streamlit app specific)
    # For batch processing, use molecule_name directly if current_data is not provided
    english_name = None
    if current_data:
        english_name = current_data.get("name_en")
    else:
        # For batch processing, use molecule_name as english_name
        english_name = molecule_name
    
    # Set up retry configuration for batch mode
    if is_batch_mode and retry_config:
        max_retries = retry_config.get('max_retries', 3)
        molecule_name_variations = get_molecule_name_variations(molecule_name)
        logger.info(f"Batch mode enabled: max_retries={max_retries}, name_variations={len(molecule_name_variations)}")
    else:
        max_retries = 0
        molecule_name_variations = [molecule_name]
        logger.info("Streamlit mode: single attempt only")
    
    # Main retry loop
    for attempt in range(max_retries + 1):
        # Select molecule name variation for this attempt
        current_molecule_name = molecule_name_variations[attempt % len(molecule_name_variations)]
        
        if attempt > 0:
            # Calculate retry delay
            delay = calculate_retry_delay(attempt - 1, retry_config or {})
            logger.info(f"Retry attempt {attempt}/{max_retries} for: {molecule_name} (using variation: {current_molecule_name}, delay: {delay:.1f}s)")
            time.sleep(delay)
        else:
            logger.info(f"Initial attempt for: {molecule_name}")
        
        # Call Gemini API using the provided client
        # Use prompt variation for retry attempts if enabled
        if is_batch_mode and retry_config and retry_config.get('prompt_variation', False):
            similar_prompt = get_similar_molecule_prompt_variation(current_molecule_name, attempt)
        else:
            similar_prompt = AIPrompts.SIMILAR_MOLECULE_SEARCH.format(molecule_name=current_molecule_name)
        
        if hasattr(gemini_client, 'call_api'):
            # GeminiClient instance (batch processing)
            response = gemini_client.call_api(similar_prompt, query_type="similar_molecule_search")
            if isinstance(response, dict):
                response_text = response['text']
            else:
                response_text = response
        else:
            # call_gemini_api function (Streamlit app)
            response_text = gemini_client(similar_prompt, query_type="similar_molecule_search")
        
        # Check if we should retry
        if not should_retry(response_text, attempt, max_retries):
            # Success! Save to cache and return
            if response_text and english_name:
                parsed_data = parse_json_response(response_text)
                if parsed_data:
                    # Save English name and descriptions for similar compounds with timestamp
                    current_timestamp = datetime.now().isoformat()
                    similar_compounds = [{
                        "compound_name": parsed_data.get("name_en", ""), 
                        "name_jp": parsed_data.get("name_jp", ""),  # Include Japanese name from Gemini
                        "timestamp": current_timestamp,
                        "descriptions": [{
                            "description": parsed_data.get("description", ""),
                            "timestamp": current_timestamp
                        }]
                    }]
                    cache_manager.save_similar_molecules(english_name, similar_compounds, gemini_client=gemini_client)
            
            if attempt > 0:
                logger.info(f"Retry successful after {attempt + 1} attempts for: {molecule_name}")
            return response_text
    
    # All retries failed - try fallback
    logger.warning(f"All retry attempts failed for: {molecule_name}")
    
    # Try fallback from cache
    fallback_result = get_fallback_similar_molecule(molecule_name, cache_manager)
    if fallback_result:
        logger.info(f"Using fallback result for: {molecule_name}")
        return fallback_result
    
    # Final fallback - return original response or None
    logger.error(f"No similar molecules found after all attempts for: {molecule_name}")
    return None
