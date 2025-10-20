# Batch processor main logic
import os
import random
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Import unified cache utilities
from tools.utils.cache_utils import normalize_compound_name, NameMappingCacheManager
from tools.utils.updated_cache_managers import (
    PubChemCacheManager, QueryCacheManager, DescriptionCacheManager,
    SimilarMoleculesCacheManager, AnalysisCacheManager, FailedMoleculesCacheManager
)
from ..utils.shared_models import DetailedMoleculeInfo
from ..utils.shared_prompts import AIPrompts
from ..utils.pubchem_client import get_comprehensive_molecule_data
from ..utils.error_handler import ErrorHandler, is_no_result_response, parse_json_response

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
    
    def call_api(self, prompt: str) -> Optional[str]:
        """Call Gemini API with rate limiting and timeout protection."""
        # Check rate limit before making the call
        self._check_rate_limit()
        
        logger.info(f"Calling Gemini API with prompt length: {len(prompt)}")
        
        def api_call():
            """Execute API call with optional Google Search tool."""
            config = self.types.GenerateContentConfig()
            
            # Add Google Search tool if requested
            if self.use_google_search:
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
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(api_call)
                response = future.result(timeout=self.timeout)
            
            # Record successful API call
            self.api_calls.append(time.time())
            
            if response is None:
                logger.warning("No response received from Gemini API")
                return None
            
            logger.info("Successfully received response from Gemini API")
            return response.text
            
        except FutureTimeoutError:
            logger.warning(f"Gemini API timeout after {self.timeout} seconds")
            return None
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return None

class CacheManager:
    """Cache manager for batch processing using unified cache utilities."""
    
    def __init__(self, cache_base_dir: str = "cache", config: Optional[Dict] = None):
        """Initialize cache manager with configuration support."""
        self.cache_base_dir = cache_base_dir
        
        # Load cache configuration from config or use defaults
        cache_config = self._load_cache_config(config)
        
        self.name_mappings = NameMappingCacheManager(cache_base_dir)
        self.pubchem = PubChemCacheManager(cache_config['data_sources']['pubchem'], cache_base_dir)
        self.queries = QueryCacheManager(cache_config['data_sources']['queries'], cache_base_dir)
        self.descriptions = DescriptionCacheManager(cache_config['data_sources']['descriptions'], cache_base_dir)
        self.analysis = AnalysisCacheManager(cache_config['data_sources']['analysis'], cache_base_dir)
        self.similar = SimilarMoleculesCacheManager(cache_config['data_sources']['similar'], cache_base_dir)
        self.failed_molecules = FailedMoleculesCacheManager(cache_config['data_sources']['failed_molecules'], cache_base_dir)
    
    def _load_cache_config(self, config: Optional[Dict]) -> Dict:
        """Load cache configuration from config or use defaults."""
        # Default configuration matching main.py structure
        default_config = {
            'data_sources': {
                'pubchem': {
                    'enabled': True,
                    'directory': 'pubchem',
                    'max_age_days': 36500,
                    'max_items_per_file': 25
                },
                'queries': {
                    'enabled': True,
                    'directory': 'queries',
                    'max_age_days': 36500,
                    'max_items_per_file': 25
                },
                'descriptions': {
                    'enabled': True,
                    'directory': 'descriptions',
                    'max_age_days': 36500,
                    'max_items_per_file': 25
                },
                'analysis': {
                    'enabled': True,
                    'directory': 'analysis',
                    'max_age_days': 180,
                    'max_items_per_file': 25
                },
                'similar': {
                    'enabled': True,
                    'directory': 'similar',
                    'max_age_days': 180,
                    'max_items_per_file': 50,
                    'max_items_per_data': 20
                },
                'failed_molecules': {
                    'enabled': True,
                    'directory': 'failed_molecules',
                    'max_age_days': 365,
                    'max_items_per_file': 1000
                }
            }
        }
        
        # Merge config if provided
        if config and 'cache' in config:
            cache_config = config['cache'].copy()
            
            # Ensure data_sources exists
            if 'data_sources' not in cache_config:
                cache_config['data_sources'] = default_config['data_sources']
            else:
                # Merge each data source config with defaults
                for source_name, default_source_config in default_config['data_sources'].items():
                    if source_name in cache_config['data_sources']:
                        # Merge provided config with defaults
                        merged_config = default_source_config.copy()
                        merged_config.update(cache_config['data_sources'][source_name])
                        cache_config['data_sources'][source_name] = merged_config
                    else:
                        # Use default if not provided
                        cache_config['data_sources'][source_name] = default_source_config
            
            logger.info("Loaded cache configuration from config file")
            return cache_config
        
        logger.info("Using default cache configuration")
        return default_config
    
    def save_query_compound_mapping(self, query_text: str, compounds: List[Dict]):
        """Save query-compound mapping to cache."""
        self.queries.save_query_compound_mapping(query_text, compounds)
    
    def save_compound_description(self, compound_name: str, description: str):
        """Save compound description to cache."""
        self.descriptions.save_compound_description(compound_name, description)
    
    def save_pubchem_data(self, compound_name: str, detailed_info: DetailedMoleculeInfo, cid: int):
        """Save PubChem data to cache."""
        self.pubchem.save_cached_molecule_data(compound_name, detailed_info, cid)
    
    def save_analysis_result(self, compound_name: str, analysis_text: str):
        """Save analysis result to cache."""
        self.analysis.save_analysis_result(compound_name, analysis_text)
    
    def save_similar_molecules(self, compound_name: str, similar_molecules: List[Dict]):
        """Save similar molecules to cache."""
        self.similar.save_similar_molecules(compound_name, similar_molecules)
    
    def save_name_mapping(self, name_jp: str, name_en: str):
        """Save name mapping to cache."""
        self.name_mappings.save_mapping(normalize_compound_name(name_en), name_jp, name_en)

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
        self.cache_manager = CacheManager(config.get('cache.base_directory', 'cache'), config.config)
        self.execution_log = {
            'session_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': datetime.now().isoformat(),
            'total_iterations': 0,
            'successful_iterations': 0,
            'failed_iterations': 0,
            'iterations': [],
            'errors': []
        }
    
    def select_random_query(self, queries: List[str]) -> str:
        """Select a random query from the list."""
        return random.choice(queries)
    
    def process_single_query(self, query: str, iteration: int) -> Dict[str, Any]:
        """Process a single query through the complete pipeline."""
        start_time = time.time()
        iteration_data = {
            'iteration': iteration,
            'selected_query': query,
            'timestamp': datetime.now().isoformat(),
            'status': 'processing'
        }
        
        try:
            # Step 1: Gemini API call
            logger.info(f"Iteration {iteration}: Processing query: {query}")
            prompt = AIPrompts.MOLECULAR_SEARCH.format(user_input=query)
            gemini_response = self.gemini_client.call_api(prompt)
            
            if not gemini_response:
                iteration_data.update({
                    'status': 'failed',
                    'error_type': 'gemini_api_error',
                    'error_message': 'No response from Gemini API'
                })
                return iteration_data
            
            # Parse Gemini response
            parsed_data = parse_json_response(gemini_response)
            if not parsed_data or is_no_result_response(gemini_response):
                iteration_data.update({
                    'status': 'failed',
                    'error_type': 'no_result',
                    'error_message': 'No valid compound found'
                })
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
                return iteration_data
            
            iteration_data['xyz_available'] = True
            
            # Step 4: Save to cache
            logger.info(f"Iteration {iteration}: Saving to cache: {english_name}")
            compounds = [{"compound_name": english_name, "timestamp": datetime.now().isoformat()}]
            self.cache_manager.save_query_compound_mapping(query, compounds)
            self.cache_manager.save_compound_description(english_name, description)
            self.cache_manager.save_pubchem_data(english_name, detailed_info, cid)
            self.cache_manager.save_name_mapping(japanese_name, english_name)
            
            # Step 5: Molecular analysis
            logger.info(f"Iteration {iteration}: Performing molecular analysis for: {english_name}")
            analysis_result = self.perform_molecular_analysis(detailed_info, english_name)
            if analysis_result:
                iteration_data['analysis_performed'] = True
            
            # Step 6: Similar molecule search
            logger.info(f"Iteration {iteration}: Finding similar molecules for: {english_name}")
            similar_result = self.find_similar_molecules(english_name)
            if similar_result:
                iteration_data['similar_search_performed'] = True
            
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
            logger.error(f"Iteration {iteration}: Error processing query '{query}': {e}")
            return iteration_data
    
    def perform_molecular_analysis(self, detailed_info: DetailedMoleculeInfo, molecule_name: str) -> Optional[str]:
        """Perform molecular analysis using Gemini."""
        try:
            # Prepare properties text
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
            
            response_text = self.gemini_client.call_api(prompt)
            
            if response_text:
                analysis_result = response_text.strip()
                self.cache_manager.save_analysis_result(molecule_name, analysis_result)
                logger.info(f"Analysis completed for: {molecule_name}")
                return analysis_result
            else:
                logger.warning(f"No analysis response for: {molecule_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error in molecular analysis for {molecule_name}: {e}")
            return None
    
    def find_similar_molecules(self, molecule_name: str) -> Optional[str]:
        """Find similar molecules using Gemini."""
        try:
            similar_prompt = AIPrompts.SIMILAR_MOLECULE_SEARCH.format(molecule_name=molecule_name)
            response_text = self.gemini_client.call_api(similar_prompt)
            
            if response_text and not is_no_result_response(response_text):
                parsed_data = parse_json_response(response_text)
                if parsed_data:
                    # Save similar molecules
                    current_timestamp = datetime.now().isoformat()
                    similar_compounds = [{
                        "compound_name": parsed_data.get("name_en", ""),
                        "timestamp": current_timestamp,
                        "descriptions": [{
                            "description": parsed_data.get("description", ""),
                            "timestamp": current_timestamp
                        }]
                    }]
                    self.cache_manager.save_similar_molecules(molecule_name, similar_compounds)
                    logger.info(f"Similar molecules found for: {molecule_name}")
                    return response_text
            
            logger.info(f"No similar molecules found for: {molecule_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error finding similar molecules for {molecule_name}: {e}")
            return None
    
    def run_batch_processing(self, queries: List[str]) -> Dict[str, Any]:
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
                    'query': selected_query,
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
