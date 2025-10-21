# Batch processor main logic
import os
import random
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Import unified cache utilities from core
from core.cache_utils import normalize_compound_name, NameMappingCacheManager
from core.cache_managers import (
    PubChemCacheManager, QueryCacheManager, DescriptionCacheManager,
    SimilarMoleculesCacheManager, AnalysisCacheManager, FailedMoleculesCacheManager
)
from core.cache import UnifiedCacheManager
from core.models import DetailedMoleculeInfo
from core.prompts import AIPrompts
from core.pubchem import get_comprehensive_molecule_data
from core.error_handler import ErrorHandler, is_no_result_response, parse_json_response
from core.analysis import analyze_molecule_properties, find_similar_molecules
from core.utils import execute_with_timeout
from core.gemini_client import call_gemini_api

logger = logging.getLogger(__name__)

class GeminiClient:
    """Independent Gemini API client for batch processing with rate limiting."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite", 
                 use_google_search: bool = True, timeout: int = 10,
                 rate_limit_per_minute: int = 15):
        """Initialize Gemini client with rate limiting."""
        try:
            from google import genai
            from google.genai import types
            
            self.client = genai.Client(api_key=api_key)
            self.model = model
            self.use_google_search = use_google_search
            self.timeout = timeout
            self.types = types
            
            # Rate limiting
            self.rate_limit_per_minute = rate_limit_per_minute
            self.api_calls = []  # Track API call timestamps
            
            logger.info(f"Gemini client initialized with model: {model}, rate limit: {rate_limit_per_minute}/min")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Remove calls older than 1 minute
        self.api_calls = [call_time for call_time in self.api_calls 
                         if current_time - call_time < 60]
        
        # Check if we're at the rate limit
        if len(self.api_calls) >= self.rate_limit_per_minute:
            # Calculate wait time
            oldest_call = min(self.api_calls)
            wait_time = 60 - (current_time - oldest_call) + 1  # Add 1 second buffer
            
            logger.warning(f"Rate limit reached ({len(self.api_calls)}/{self.rate_limit_per_minute}). Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            
            # Update the list after waiting
            current_time = time.time()
            self.api_calls = [call_time for call_time in self.api_calls 
                             if current_time - call_time < 60]
    
    def call_api(self, prompt: str, query_type: str = "molecular_search") -> Optional[Dict]:
        """Call Gemini API with rate limiting, query-type specific parameters, and grounding metadata support."""
        # Check rate limit before making the call
        self._check_rate_limit()
        
        logger.info(f"Calling Gemini API with prompt length: {len(prompt)}, query_type: {query_type}")
        
        # Import Config here to avoid circular imports
        from config.settings import Config
        
        # Get configuration for query type
        base_config = Config.GEMINI_CONFIG.get(query_type, Config.GEMINI_CONFIG['molecular_search'])
        
        def api_call():
            """Execute API call with query-type specific configuration."""
            config = self.types.GenerateContentConfig(
                temperature=base_config['temperature'],
                top_p=base_config['top_p'],
                top_k=base_config['top_k'],
                max_output_tokens=base_config['max_output_tokens']
            )
            
            # Add Google Search tool if requested
            if base_config['use_google_search']:
                search_tool = self.types.Tool(
                    google_search=self.types.GoogleSearch()
                )
                config.tools = [search_tool]
            
            # Generate content using the model
            return self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config
            )
        
        try:
            response = execute_with_timeout(api_call, base_config['timeout'], "api_error")
            
            # Record successful API call
            self.api_calls.append(time.time())
            
            if response is None:
                logger.warning("No response received from Gemini API")
                return None
            
            logger.info("Successfully received response from Gemini API")
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return None

class BatchProcessor:
    """Main batch processor class."""
    
    def __init__(self, config):
        """Initialize batch processor."""
        self.config = config
        self.gemini_client = GeminiClient(
            api_key=config.get('api.key'),
            model=config.get('api.model'),
            use_google_search=config.get('api.use_google_search'),
            timeout=config.get('api.timeout'),
            rate_limit_per_minute=config.get('api.rate_limit_per_minute', 15)
        )
        self.cache_manager = UnifiedCacheManager(config.get('cache.base_directory', 'cache'), config.config)
        self.execution_log = {
            'session_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': datetime.now().isoformat(),
            'total_iterations': 0,
            'successful_iterations': 0,
            'failed_iterations': 0,
            'iterations': [],
            'errors': []
        }
    
    
    def select_random_query(self, queries: List[Dict[str, str]]) -> Dict[str, str]:
        """Select a random query from the list."""
        return random.choice(queries)
    
    def process_single_query(self, query_obj: Dict[str, str], iteration: int) -> Dict[str, Any]:
        """Process a single query through the complete pipeline."""
        query_text = query_obj['text']
        query_icon = query_obj['icon']
        
        start_time = time.time()
        iteration_data = {
            'iteration': iteration,
            'selected_query': query_text,
            'selected_icon': query_icon,
            'timestamp': datetime.now().isoformat(),
            'status': 'processing'
        }
        
        try:
            # Step 1: Gemini API call
            logger.info(f"Iteration {iteration}: Processing query: {query_text}")
            prompt = AIPrompts.MOLECULAR_SEARCH.format(user_input=query_text)
            gemini_response = self.gemini_client.call_api(prompt, query_type="molecular_search")
            
            if not gemini_response:
                iteration_data.update({
                    'status': 'failed',
                    'error_type': 'gemini_api_error',
                    'error_message': 'No response from Gemini API'
                })
                # Save failed query to cache
                try:
                    self.cache_manager.failed_molecules.save_failed_query(
                        query_text, 
                        'No response from Gemini API', 
                        'gemini_api_error'
                    )
                except Exception as cache_error:
                    logger.warning(f"Failed to save failed query to cache: {cache_error}")
                return iteration_data
            
            # Handle response
            response_text = gemini_response
            
            # Parse Gemini response
            parsed_data = parse_json_response(response_text)
            if not parsed_data or is_no_result_response(response_text):
                iteration_data.update({
                    'status': 'failed',
                    'error_type': 'no_result',
                    'error_message': 'No valid compound found'
                })
                # Save failed query to cache
                try:
                    self.cache_manager.failed_molecules.save_failed_query(
                        query_text, 
                        'No valid compound found', 
                        'no_result'
                    )
                except Exception as cache_error:
                    logger.warning(f"Failed to save failed query to cache: {cache_error}")
                return iteration_data
            
            english_name = parsed_data.get('name_en', '').strip()
            japanese_name = parsed_data.get('name_jp', '').strip()
            description = parsed_data.get('description', '').strip()
            
            if not english_name:
                iteration_data.update({
                    'status': 'failed',
                    'error_type': 'invalid_data',
                    'error_message': 'No English name found'
                })
                # Save failed query to cache
                try:
                    self.cache_manager.failed_molecules.save_failed_query(
                        query_text, 
                        'No English name found', 
                        'invalid_data'
                    )
                except Exception as cache_error:
                    logger.warning(f"Failed to save failed query to cache: {cache_error}")
                return iteration_data
            
            iteration_data.update({
                'gemini_response': parsed_data,
                'pubchem_success': False,
                'xyz_available': False,
                'analysis_performed': False,
                'similar_search_performed': False
            })
            
            # Step 2: PubChem data retrieval
            logger.info(f"Iteration {iteration}: Getting PubChem data for: {english_name}")
            success, detailed_info, cid, error_msg = get_comprehensive_molecule_data(english_name)
            
            if not success or not detailed_info:
                iteration_data.update({
                    'status': 'failed',
                    'error_type': 'pubchem_error',
                    'error_message': error_msg or 'PubChem data retrieval failed'
                })
                # Save failed molecule to cache
                try:
                    self.cache_manager.failed_molecules.save_failed_molecule(
                        english_name, 
                        query_text, 
                        error_msg or 'PubChem data retrieval failed', 
                        'pubchem_error'
                    )
                except Exception as cache_error:
                    logger.warning(f"Failed to save failed molecule to cache: {cache_error}")
                return iteration_data
            
            iteration_data['pubchem_success'] = True
            
            # Step 3: Check XYZ data availability
            if not detailed_info.xyz_data:
                logger.warning(f"Iteration {iteration}: No XYZ data available for {english_name}")
                iteration_data.update({
                    'status': 'failed',
                    'error_type': 'xyz_unavailable',
                    'error_message': 'XYZ coordinates not available'
                })
                # Save failed molecule to cache
                try:
                    self.cache_manager.failed_molecules.save_failed_molecule(
                        english_name, 
                        query_text, 
                        'XYZ coordinates not available', 
                        'xyz_unavailable'
                    )
                except Exception as cache_error:
                    logger.warning(f"Failed to save failed molecule to cache: {cache_error}")
                return iteration_data
            
            iteration_data['xyz_available'] = True
            
            # Step 4: Save to cache
            logger.info(f"Iteration {iteration}: Saving to cache: {english_name}")
            compounds = [{"compound_name": english_name, "timestamp": datetime.now().isoformat()}]
            # Save PubChem data first so other cache managers can find it
            self.cache_manager.save_pubchem_data(english_name, detailed_info, cid)
            self.cache_manager.save_name_mapping(japanese_name, english_name)
            # Then save dependent caches
            self.cache_manager.save_query_compound_mapping(query_obj, compounds)
            self.cache_manager.save_compound_description(english_name, description)
            
            # Step 5: Molecular analysis
            logger.info(f"Iteration {iteration}: Performing molecular analysis for: {english_name}")
            analysis_result = analyze_molecule_properties(detailed_info, english_name, self.gemini_client, self.cache_manager)
            if analysis_result:
                iteration_data['analysis_performed'] = True
            
            # Step 6: Similar molecule search with retry
            logger.info(f"Iteration {iteration}: Finding similar molecules for: {english_name}")
            current_data = {"name_en": english_name}
            
            # Get retry configuration for similar molecule search
            retry_config = self.config.get('similar_molecule_retry', {})
            if retry_config.get('enabled', False):
                logger.info(f"Similar molecule retry enabled: {retry_config}")
            
            similar_result = find_similar_molecules(
                english_name, 
                self.gemini_client, 
                self.cache_manager, 
                current_data,
                is_batch_mode=True,
                retry_config=retry_config
            )
            if similar_result:
                iteration_data['similar_search_performed'] = True
                # Add retry statistics if available
                if retry_config.get('enabled', False):
                    iteration_data['similar_search_retry_enabled'] = True
                    iteration_data['similar_search_max_retries'] = retry_config.get('max_retries', 3)
            
            processing_time = time.time() - start_time
            iteration_data.update({
                'status': 'success',
                'compound': english_name,
                'processing_time_seconds': round(processing_time, 2)
            })
            
            logger.info(f"Iteration {iteration}: Successfully processed {english_name} in {processing_time:.2f}s")
            return iteration_data
            
        except Exception as e:
            processing_time = time.time() - start_time
            iteration_data.update({
                'status': 'failed',
                'error_type': 'general_error',
                'error_message': str(e),
                'processing_time_seconds': round(processing_time, 2)
            })
            
            # Save failed molecule to cache
            try:
                if 'english_name' in locals() and english_name:
                    self.cache_manager.failed_molecules.save_failed_molecule(
                        english_name, 
                        query_text, 
                        str(e), 
                        iteration_data.get('error_type', 'general_error')
                    )
            except Exception as cache_error:
                logger.warning(f"Failed to save failed molecule to cache: {cache_error}")
            
            logger.error(f"Iteration {iteration}: Error processing query '{query_text}': {e}")
            return iteration_data
    
    def run_batch_processing(self, queries: List[Dict[str, str]]) -> Dict[str, Any]:
        """Run batch processing on the given queries."""
        max_iterations = self.config.get('execution.max_iterations', 10)
        max_duration_minutes = self.config.get('execution.max_duration_minutes', 30)
        continue_on_xyz_failure = self.config.get('execution.continue_on_xyz_failure', True)
        
        start_time = time.time()
        max_duration_seconds = max_duration_minutes * 60
        
        logger.info(f"Starting batch processing: {max_iterations} iterations, {max_duration_minutes} minutes max")
        
        for iteration in range(1, max_iterations + 1):
            # Check duration limit
            elapsed_time = time.time() - start_time
            if elapsed_time > max_duration_seconds:
                logger.info(f"Reached maximum duration limit: {max_duration_minutes} minutes")
                break
            
            # Select random query
            selected_query = self.select_random_query(queries)
            
            # Process query
            iteration_result = self.process_single_query(selected_query, iteration)
            
            # Update execution log
            self.execution_log['iterations'].append(iteration_result)
            self.execution_log['total_iterations'] += 1
            
            if iteration_result['status'] == 'success':
                self.execution_log['successful_iterations'] += 1
            else:
                self.execution_log['failed_iterations'] += 1
                self.execution_log['errors'].append({
                    'iteration': iteration,
                    'query': selected_query['text'],
                    'icon': selected_query['icon'],
                    'error_type': iteration_result.get('error_type'),
                    'error_message': iteration_result.get('error_message'),
                    'timestamp': iteration_result['timestamp']
                })
                
                # Check if we should stop on error
                if not continue_on_xyz_failure and iteration_result.get('error_type') == 'xyz_unavailable':
                    logger.info("Stopping due to XYZ data unavailability")
                    break
            
            # Log progress
            logger.info(f"Progress: {iteration}/{max_iterations} iterations completed")
        
        # Finalize execution log
        self.execution_log['end_time'] = datetime.now().isoformat()
        self.execution_log['total_duration_seconds'] = round(time.time() - start_time, 2)
        
        logger.info(f"Batch processing completed: {self.execution_log['successful_iterations']}/{self.execution_log['total_iterations']} successful")
        
        return self.execution_log
