#!/usr/bin/env python3
# Common molecular analysis functions for both Streamlit app and batch processing
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from core.models import DetailedMoleculeInfo
from core.prompts import AIPrompts
from core.error_handler import is_no_result_response, parse_json_response

logger = logging.getLogger(__name__)


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
        properties_text.append(f"TPSA: {detailed_info.tpsa:.1f} Å²")
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
        response_text = gemini_client.call_api(prompt)
    else:
        # call_gemini_api function (Streamlit app)
        response_text = gemini_client(prompt, use_google_search=False)
    
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
                          current_data: Optional[Dict] = None) -> Optional[str]:
    """Find molecules similar to the specified molecule.
    
    Args:
        molecule_name: Name of the molecule to find similar molecules for
        gemini_client: Gemini API client (call_gemini_api function or GeminiClient instance)
        cache_manager: Cache manager instance
        current_data: Current molecule data (for Streamlit app, contains name_en for cache key)
        
    Returns:
        Similar molecule response text or None if failed
    """
    logger.info(f"Searching for similar molecules to: {molecule_name}")
    
    # Get English name from current data for cache key (Streamlit app specific)
    english_name = None
    if current_data:
        english_name = current_data.get("name_en")
    
    # Always call Gemini API for similar molecules (don't use cache for direct response)
    similar_prompt = AIPrompts.SIMILAR_MOLECULE_SEARCH.format(molecule_name=molecule_name)
    
    # Call Gemini API using the provided client
    if hasattr(gemini_client, 'call_api'):
        # GeminiClient instance (batch processing)
        response_text = gemini_client.call_api(similar_prompt)
    else:
        # call_gemini_api function (Streamlit app)
        response_text = gemini_client(similar_prompt, use_google_search=True)
    
    # Check if response indicates no results
    if is_no_result_response(response_text):
        logger.info(f"No similar molecules found from Gemini for: {molecule_name}")
        
        # Try to get a random similar compound from cache
        if english_name:
            random_result = cache_manager.similar.get_random_similar_compound(english_name)
            if random_result:
                random_compound_name, random_description = random_result
                logger.info(f"Using cached similar compound as fallback: {random_compound_name}")
                # Create a fallback response with the cached compound
                fallback_response = f'{{"name_jp": "{random_compound_name}", "name_en": "{random_compound_name}", "description": "{random_description}"}}'
                return fallback_response
            else:
                logger.info(f"No cached similar compounds found for: {english_name}")
                return response_text  # Return original "該当なし" response
        else:
            logger.info(f"No English name available for fallback search")
            return response_text  # Return original "該当なし" response
    
    if response_text and english_name:
        # Parse and save to cache using English name
        parsed_data = parse_json_response(response_text)
        if parsed_data:
            # Save English name and descriptions for similar compounds with timestamp
            current_timestamp = datetime.now().isoformat()
            similar_compounds = [{
                "compound_name": parsed_data.get("name_en", ""), 
                "timestamp": current_timestamp,
                "descriptions": [{
                    "description": parsed_data.get("description", ""),
                    "timestamp": current_timestamp
                }]
            }]
            cache_manager.save_similar_molecules(english_name, similar_compounds)
    
    return response_text
