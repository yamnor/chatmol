#!/usr/bin/env python3
# Unified cache manager for both Streamlit app and batch processing
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Import unified cache utilities from core
from core.cache_utils import normalize_compound_name, NameMappingCacheManager
from core.cache_managers import (
    PubChemCacheManager, QueryCacheManager, DescriptionCacheManager,
    SimilarMoleculesCacheManager, AnalysisCacheManager, FailedMoleculesCacheManager
)

logger = logging.getLogger(__name__)


class UnifiedCacheManager:
    """Unified cache manager for both Streamlit app and batch processing."""
    
    def __init__(self, cache_base_dir: str = "cache", config: Optional[Dict] = None):
        """Initialize unified cache manager with flexible configuration support.
        
        Args:
            cache_base_dir: Base cache directory path
            config: Configuration dict. Can be:
                - Config.CACHE from main.py (Streamlit app)
                - BatchConfig cache section (batch processing)
                - None (use defaults)
        """
        self.cache_base_dir = cache_base_dir
        
        # Load cache configuration from config or use defaults
        cache_config = self._load_cache_config(config)
        
        # Initialize all cache managers
        self.name_mappings = NameMappingCacheManager(cache_base_dir)
        self.pubchem = PubChemCacheManager(cache_config['data_sources']['pubchem'], cache_base_dir)
        self.queries = QueryCacheManager(cache_config['data_sources']['queries'], cache_base_dir)
        self.descriptions = DescriptionCacheManager(cache_config['data_sources']['descriptions'], cache_base_dir)
        self.analysis = AnalysisCacheManager(cache_config['data_sources']['analysis'], cache_base_dir)
        self.similar = SimilarMoleculesCacheManager(cache_config['data_sources']['similar'], cache_base_dir)
        self.failed_molecules = FailedMoleculesCacheManager(cache_config['data_sources']['failed_molecules'], cache_base_dir)
        
        logger.info(f"UnifiedCacheManager initialized with base directory: {cache_base_dir}")
    
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
                    'max_items_per_data': 25
                },
                'failed_molecules': {
                    'enabled': True,
                    'directory': 'failed_molecules',
                    'max_age_days': 365,
                    'max_items_per_file': 1000
                }
            }
        }
        
        # Handle different config formats
        if config is None:
            logger.info("Using default cache configuration")
            return default_config
        
        # Handle main.py Config.CACHE format
        if 'data_sources' in config:
            logger.info("Using main.py Config.CACHE format")
            return config
        
        # Handle batch processor config format (nested under 'cache' key)
        if 'cache' in config and 'data_sources' in config['cache']:
            logger.info("Using batch processor config format")
            cache_config = config['cache'].copy()
            
            # Ensure data_sources exists and merge with defaults
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
            
            return cache_config
        
        # Handle direct cache config format
        logger.info("Using direct cache config format")
        return config
    
    # =============================================================================
    # Main.py CacheManager methods (Streamlit app specific)
    # =============================================================================
    
    def save_all_caches(self, name_jp: str, name_en: str, detailed_info, cid: int, user_query: str, description: str):
        """Save all cache types when xyz_data is successfully obtained."""
        try:
            # 1. Save name mapping
            self.name_mappings.save_mapping(normalize_compound_name(name_en), name_jp, name_en)
            
            # 2. PubChemキャッシュ保存
            self.pubchem.save_cached_molecule_data(name_en, detailed_info, cid)
            
            # 3. 質問-化合物マッピング保存
            if user_query:
                compounds = [{"compound_name": name_en, "timestamp": datetime.now().isoformat()}]
                self.queries.save_query_compound_mapping(
                    user_query,
                    compounds,
                    increment_count=False
                )
            
            # 4. 化合物-説明マッピング保存
            if description:
                self.descriptions.save_compound_description(name_en, description)
            
            logger.info(f"All caches saved successfully for {name_en}")
        except Exception as e:
            logger.error(f"Error saving caches for {name_en}: {e}")
    
    def get_compound_names_for_display(self, compound_name: str) -> Tuple[str, str]:
        """Get Japanese and English names for display purposes."""
        return self.name_mappings.get_names_for_display(normalize_compound_name(compound_name))
    
    def get_fallback_molecule_data(self, user_query: str = "") -> Optional[Tuple[str, str, str]]:
        """
        Get fallback molecule data when PubChem XYZ data is not available.
        Returns: (compound_name, description, xyz_data) or None
        """
        try:
            logger.info("Attempting to get fallback molecule data from cache")
            
            # Strategy 1: Try to get from queries cache first
            if user_query:
                random_compound = self.queries.get_random_compound_from_query(user_query)
                if random_compound:
                    logger.info(f"Found random compound from queries cache: {random_compound}")
                    # Get description and XYZ data for this compound
                    description = self.descriptions.get_random_description(random_compound)
                    cached_data = self.pubchem.get_cached_molecule_data(random_compound)
                    
                    if description and cached_data:
                        detailed_info, cid = cached_data
                        if detailed_info and detailed_info.xyz_data:
                            logger.info(f"Successfully got fallback data for {random_compound}")
                            return random_compound, description, detailed_info.xyz_data
            
            # Strategy 2: Try to get from any queries cache
            random_compound = self.queries.get_any_random_compound_from_queries()
            if random_compound:
                logger.info(f"Found random compound from any queries cache: {random_compound}")
                description = self.descriptions.get_random_description(random_compound)
                cached_data = self.pubchem.get_cached_molecule_data(random_compound)
                
                if description and cached_data:
                    detailed_info, cid = cached_data
                    if detailed_info and detailed_info.xyz_data:
                        logger.info(f"Successfully got fallback data for {random_compound}")
                        return random_compound, description, detailed_info.xyz_data
            
            # Strategy 3: Try to get from similar cache
            random_result = self.similar.get_any_random_similar_compound()
            if random_result:
                random_compound, description = random_result
                logger.info(f"Found random compound from similar cache: {random_compound}")
                cached_data = self.pubchem.get_cached_molecule_data(random_compound)
                
                if cached_data:
                    detailed_info, cid = cached_data
                    if detailed_info and detailed_info.xyz_data:
                        logger.info(f"Successfully got fallback data for {random_compound}")
                        return random_compound, description, detailed_info.xyz_data
            
            logger.warning("No fallback molecule data available from any cache")
            return None
            
        except Exception as e:
            logger.error(f"Error getting fallback molecule data: {e}")
            return None
    
    def clear_all_cache(self):
        """Clear all cache directories."""
        try:
            for manager in [self.name_mappings, self.pubchem, self.queries, self.descriptions, self.analysis, self.similar, self.failed_molecules]:
                if hasattr(manager, 'clear_cache'):
                    manager.clear_cache()
            logger.info("All cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing all cache: {e}")
    
    def get_all_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for all data sources."""
        try:
            all_stats = {}
            total_count = 0
            total_size = 0
            
            for manager_name, manager in [
                ('name_mappings', self.name_mappings), 
                ('pubchem', self.pubchem), 
                ('queries', self.queries), 
                ('descriptions', self.descriptions),
                ('analysis', self.analysis),
                ('similar', self.similar),
                ('failed_molecules', self.failed_molecules)
            ]:
                cache_dir = manager.cache_dir if hasattr(manager, 'cache_dir') else manager._get_source_cache_directory()
                if cache_dir and os.path.exists(cache_dir):
                    files = []
                    source_size = 0
                    
                    for filename in os.listdir(cache_dir):
                        if filename.endswith('.json'):
                            file_path = os.path.join(cache_dir, filename)
                            file_size = os.path.getsize(file_path)
                            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                            
                            files.append({
                                'name': filename,
                                'size_bytes': file_size,
                                'modified': file_time.isoformat()
                            })
                            source_size += file_size
                    
                    all_stats[manager_name] = {
                        'count': len(files),
                        'size_mb': round(source_size / (1024 * 1024), 2),
                        'files': files
                    }
                    total_count += len(files)
                    total_size += source_size
            
            all_stats['total'] = {
                'count': total_count,
                'size_mb': round(total_size / (1024 * 1024), 2)
                }
                
            return all_stats
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'count': 0, 'size_mb': 0, 'files': []}
    
    # =============================================================================
    # Batch processor CacheManager methods
    # =============================================================================
    
    def save_query_compound_mapping(self, query_text: str, compounds: List[Dict]):
        """Save query-compound mapping to cache."""
        self.queries.save_query_compound_mapping(query_text, compounds)
    
    def save_compound_description(self, compound_name: str, description: str):
        """Save compound description to cache."""
        self.descriptions.save_compound_description(compound_name, description)
    
    def save_pubchem_data(self, compound_name: str, detailed_info, cid: int):
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
