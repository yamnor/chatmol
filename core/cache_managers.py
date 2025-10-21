#!/usr/bin/env python3
# Updated cache managers using unified normalization
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from core.cache_utils import BaseCacheManager, normalize_compound_name

from core.models import DetailedMoleculeInfo

logger = logging.getLogger(__name__)


class PubChemCacheManager(BaseCacheManager):
    """Manages PubChem-specific cache operations with unified normalization."""
    
    def __init__(self, config: Dict, cache_base_directory: str = "cache"):
        """Initialize PubChem cache manager."""
        super().__init__('pubchem', config, cache_base_directory)
    
    def get_cached_molecule_data(self, compound_name: str) -> Optional[Tuple[Optional[Any], Optional[int]]]:
        """Get cached molecule data for compound."""
        cache_file_path = self._get_source_cache_file_path(compound_name)
        
        if not cache_file_path or not self._validate_cache_file(cache_file_path):
            return None
        
        cache_data = self._read_cache_file(cache_file_path)
        if not cache_data:
            return None
        
        try:
            # Reconstruct DetailedMoleculeInfo object
            detailed_info = DetailedMoleculeInfo(**cache_data['detailed_info'])
            cid = cache_data.get('cid')
            
            logger.info(f"Cache hit for compound: {compound_name}")
            return detailed_info, cid
            
        except Exception as e:
            logger.error(f"Error reconstructing molecule data for {compound_name}: {e}")
            return None
    
    def save_cached_molecule_data(self, compound_name: str, detailed_info: Any, cid: int):
        """Save molecule data to cache (only if xyz_data is available)."""
        # Only save if xyz_data is available
        if not detailed_info.xyz_data:
            logger.warning(f"Skipping cache save for {compound_name}: No xyz_data available")
            return
        
        cache_file_path = self._get_source_cache_file_path(compound_name)
        
        if not cache_file_path:
            logger.warning(f"Cannot save cache for {compound_name}: data source disabled")
            return
        
        try:
            # Convert DetailedMoleculeInfo to dictionary
            cache_data = {
                'compound_name': normalize_compound_name(compound_name),
                'data_source': self.data_source_name,
                'timestamp': datetime.now().isoformat(),
                'cid': cid,
                'detailed_info': {
                    'molecular_formula': detailed_info.molecular_formula,
                    'molecular_weight': detailed_info.molecular_weight,
                    'iupac_name': detailed_info.iupac_name,
                    'synonyms': detailed_info.synonyms,
                    'inchi': detailed_info.inchi,
                    'xlogp': detailed_info.xlogp,
                    'tpsa': detailed_info.tpsa,
                    'complexity': detailed_info.complexity,
                    'rotatable_bond_count': detailed_info.rotatable_bond_count,
                    'heavy_atom_count': detailed_info.heavy_atom_count,
                    'hbond_donor_count': detailed_info.hbond_donor_count,
                    'hbond_acceptor_count': detailed_info.hbond_acceptor_count,
                    'charge': detailed_info.charge,
                    'xyz_data': detailed_info.xyz_data,
                }
            }

            self._create_cache_file(cache_file_path, cache_data)
            logger.info(f"Cached molecule data for compound: {compound_name}")
            
        except Exception as e:
            logger.error(f"Error saving molecule cache for {compound_name}: {e}")


class QueryCacheManager(BaseCacheManager):
    """Manages query-compound mapping cache operations with unified normalization."""
    
    def __init__(self, config: Dict, cache_base_directory: str = "cache"):
        """Initialize query cache manager."""
        super().__init__('queries', config, cache_base_directory)
    
    def save_query_compound_mapping(self, query_text: str, compounds: List[Dict], increment_count: bool = False):
        """Save query-compound mapping to cache."""
        cache_file_path = self._get_source_cache_file_path(query_text)
        
        if not cache_file_path:
            logger.warning(f"Cannot save query cache for {query_text}: data source disabled")
            return
        
        # Read existing data if available
        existing_data = self._read_cache_file(cache_file_path) if os.path.exists(cache_file_path) else None
        
        if existing_data:
            # Update existing data
            existing_compounds = existing_data.get('compounds', [])
            existing_compound_names = [c.get('compound_name', '') for c in existing_compounds]
            
            # Add new compounds (avoid duplicates)
            for compound in compounds:
                normalized_name = normalize_compound_name(compound.get('compound_name', ''))
                if normalized_name not in existing_compound_names:
                    existing_compounds.append({
                        'compound_name': normalized_name,
                        'timestamp': compound.get('timestamp', datetime.now().isoformat())
                    })
            
            # Apply general item limit using the base class method
            existing_compounds = self._apply_item_limit(existing_compounds)
            
            cache_data = {
                'query_text': query_text,
                'compounds': existing_compounds,
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Create new data with normalized compound names
            normalized_compounds = []
            for compound in compounds:
                normalized_compounds.append({
                    'compound_name': normalize_compound_name(compound.get('compound_name', '')),
                    'timestamp': compound.get('timestamp', datetime.now().isoformat())
                })
            
            cache_data = {
                'query_text': query_text,
                'compounds': normalized_compounds,
                'timestamp': datetime.now().isoformat()
            }
        
        self._create_cache_file(cache_file_path, cache_data)
        logger.info(f"Cached query-compound mapping for: {query_text}")
    
    def get_query_compound_mapping(self, query_text: str) -> Optional[Dict]:
        """Get query-compound mapping from cache."""
        cache_file_path = self._get_source_cache_file_path(query_text)
        
        if not cache_file_path or not self._validate_cache_file(cache_file_path):
            return None
        
        return self._read_cache_file(cache_file_path)
    
    def get_random_compound_from_query(self, query_text: str) -> Optional[str]:
        """Get a random compound name from cached query results."""
        cache_data = self.get_query_compound_mapping(query_text)
        if not cache_data:
            return None
        
        compounds = cache_data.get('compounds', [])
        if not compounds:
            return None
        
        # Select a random compound
        import random
        random_compound = random.choice(compounds)
        return random_compound.get('compound_name')
    
    def get_any_random_compound_from_queries(self) -> Optional[str]:
        """Get a random compound name from any cached query file."""
        cache_dir = self._get_source_cache_directory()
        if not cache_dir or not os.path.exists(cache_dir):
            return None
        
        # Get all cache files
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
        if not cache_files:
            return None
        
        # Try each cache file until we find one with compounds
        import random
        for cache_file in cache_files:
            cache_file_path = os.path.join(cache_dir, cache_file)
            if not self._validate_cache_file(cache_file_path):
                continue
            
            cache_data = self._read_cache_file(cache_file_path)
            if not cache_data:
                continue
            
            compounds = cache_data.get('compounds', [])
            if compounds:
                # Select a random compound
                random_compound = random.choice(compounds)
                return random_compound.get('compound_name')
        
        return None


class DescriptionCacheManager(BaseCacheManager):
    """Manages compound-description mapping cache operations with unified normalization."""
    
    def __init__(self, config: Dict, cache_base_directory: str = "cache"):
        """Initialize description cache manager."""
        super().__init__('descriptions', config, cache_base_directory)
    
    def save_compound_description(self, compound_name: str, description: str):
        """Save compound description to cache with automatic item limit."""
        cache_file_path = self._get_source_cache_file_path(compound_name)
        
        if not cache_file_path:
            logger.warning(f"Cannot save description cache for {compound_name}: data source disabled")
            return
        
        # Read existing data if available
        existing_data = self._read_cache_file(cache_file_path) if os.path.exists(cache_file_path) else None
        
        if existing_data:
            # Add new description to existing list
            descriptions = existing_data.get('descriptions', [])
            descriptions.append({
                'description': description,
                'timestamp': datetime.now().isoformat()
            })
            
            # Apply general item limit using the base class method
            descriptions = self._apply_item_limit(descriptions)
            
            cache_data = {
                'compound_name': normalize_compound_name(compound_name),
                'descriptions': descriptions,
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Create new data
            cache_data = {
                'compound_name': normalize_compound_name(compound_name),
                'descriptions': [{
                    'description': description,
                    'timestamp': datetime.now().isoformat()
                }],
                'timestamp': datetime.now().isoformat()
            }
        
        self._create_cache_file(cache_file_path, cache_data)
        logger.info(f"Cached description for compound: {compound_name}")
    
    def get_compound_descriptions(self, compound_name: str) -> List[Dict]:
        """Get compound descriptions from cache."""
        cache_file_path = self._get_source_cache_file_path(compound_name)
        
        if not cache_file_path or not self._validate_cache_file(cache_file_path):
            return []
        
        cache_data = self._read_cache_file(cache_file_path)
        if cache_data:
            return cache_data.get('descriptions', [])
        
        return []
    
    def get_random_description(self, compound_name: str) -> Optional[str]:
        """Get a random description from cached descriptions."""
        cache_data = self.get_compound_descriptions(compound_name)
        if not cache_data:
            return None
        
        # Select a random description
        import random
        random_description = random.choice(cache_data)
        return random_description.get('description')


class SimilarMoleculesCacheManager(BaseCacheManager):
    """Manages similar molecules cache operations with unified normalization."""
    
    def __init__(self, config: Dict, cache_base_directory: str = "cache"):
        """Initialize similar molecules cache manager."""
        super().__init__('similar', config, cache_base_directory)
    
    def save_similar_molecules(self, compound_name: str, similar_molecules: List[Dict]):
        """Save similar molecules to cache with multiple descriptions per molecule."""
        cache_file_path = self._get_source_cache_file_path(compound_name)
        
        if not cache_file_path:
            logger.warning(f"Cannot save similar molecules cache for {compound_name}: data source disabled")
            return
        
        # Read existing data if available
        existing_data = self._read_cache_file(cache_file_path) if os.path.exists(cache_file_path) else None
        
        if existing_data:
            # Add new similar compounds to existing list
            existing_compounds = existing_data.get('similar_compounds', [])
            existing_compound_names = [m.get('compound_name', '') for m in existing_compounds]
            
            # Get max descriptions per compound from config
            max_descriptions = self.data_source_config.get('max_items_per_data', 20)
            
            for compound_data in similar_molecules:
                compound_name_inner = compound_data.get('compound_name', '')
                normalized_name = normalize_compound_name(compound_name_inner)
                descriptions = compound_data.get('descriptions', [])
                compound_timestamp = compound_data.get('timestamp', datetime.now().isoformat())
                
                if normalized_name in existing_compound_names:
                    # Existing compound: add descriptions
                    for existing_compound in existing_compounds:
                        if existing_compound.get('compound_name') == normalized_name:
                            existing_compound['descriptions'].extend(descriptions)
                            # Apply description limit (keep latest max_descriptions)
                            existing_compound['descriptions'] = existing_compound['descriptions'][-max_descriptions:]
                            # Update compound timestamp if newer
                            if compound_timestamp > existing_compound.get('timestamp', ''):
                                existing_compound['timestamp'] = compound_timestamp
                            break
                else:
                    # New compound: add to list with timestamp
                    compound_data['compound_name'] = normalized_name
                    compound_data['timestamp'] = compound_timestamp
                    existing_compounds.append(compound_data)
            
            # Apply compound limit
            existing_compounds = self._apply_item_limit(existing_compounds)
            
            cache_data = {
                'compound_name': normalize_compound_name(compound_name),
                'similar_compounds': existing_compounds,
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Create new data with normalized compound names
            normalized_similar_molecules = []
            for compound_data in similar_molecules:
                normalized_compound_data = compound_data.copy()
                normalized_compound_data['compound_name'] = normalize_compound_name(compound_data.get('compound_name', ''))
                normalized_similar_molecules.append(normalized_compound_data)
            
            cache_data = {
                'compound_name': normalize_compound_name(compound_name),
                'similar_compounds': normalized_similar_molecules,
                'timestamp': datetime.now().isoformat()
            }
        
        self._create_cache_file(cache_file_path, cache_data)
        logger.info(f"Cached similar molecules for compound: {compound_name}")
    
    def get_similar_molecules(self, compound_name: str) -> List[Dict]:
        """Get similar molecules from cache."""
        cache_file_path = self._get_source_cache_file_path(compound_name)
        
        if not cache_file_path or not self._validate_cache_file(cache_file_path):
            return []
        
        cache_data = self._read_cache_file(cache_file_path)
        if cache_data:
            return cache_data.get('similar_compounds', [])
        
        return []
    
    def get_random_similar_compound(self, compound_name: str) -> Optional[Tuple[str, str]]:
        """Get a random similar compound name and description from cache."""
        similar_compounds = self.get_similar_molecules(compound_name)
        if not similar_compounds:
            return None
        
        # Select a random similar compound
        import random
        random_compound = random.choice(similar_compounds)
        compound_name_result = random_compound.get('compound_name')
        
        # Get a random description from the compound's descriptions
        descriptions = random_compound.get('descriptions', [])
        if descriptions:
            random_description = random.choice(descriptions)
            description_text = random_description.get('description', '')
            return compound_name_result, description_text
        
        return None
    
    def get_any_random_similar_compound(self) -> Optional[Tuple[str, str]]:
        """Get a random similar compound from any cached similar compound file."""
        cache_dir = self._get_source_cache_directory()
        if not cache_dir or not os.path.exists(cache_dir):
            return None
        
        # Get all cache files
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
        if not cache_files:
            return None
        
        # Try each cache file until we find one with similar compounds
        import random
        for cache_file in cache_files:
            cache_file_path = os.path.join(cache_dir, cache_file)
            if not self._validate_cache_file(cache_file_path):
                continue
            
            cache_data = self._read_cache_file(cache_file_path)
            if not cache_data:
                continue
            
            similar_compounds = cache_data.get('similar_compounds', [])
            if similar_compounds:
                # Select a random similar compound
                random_compound = random.choice(similar_compounds)
                compound_name_result = random_compound.get('compound_name')
                
                # Get a random description from the compound's descriptions
                descriptions = random_compound.get('descriptions', [])
                if descriptions:
                    random_description = random.choice(descriptions)
                    description_text = random_description.get('description', '')
                    return compound_name_result, description_text
        
        return None


class AnalysisCacheManager(BaseCacheManager):
    """Manages detailed analysis cache operations with unified normalization."""
    
    def __init__(self, config: Dict, cache_base_directory: str = "cache"):
        """Initialize analysis cache manager."""
        super().__init__('analysis', config, cache_base_directory)
    
    def save_analysis_result(self, compound_name: str, analysis_text: str):
        """Save detailed analysis result to cache."""
        cache_file_path = self._get_source_cache_file_path(compound_name)
        
        if not cache_file_path:
            logger.warning(f"Cannot save analysis cache for {compound_name}: data source disabled")
            return
        
        # Read existing data if available
        existing_data = self._read_cache_file(cache_file_path) if os.path.exists(cache_file_path) else None
        
        if existing_data:
            # Add new analysis to existing list
            descriptions = existing_data.get('descriptions', [])
            descriptions.append({
                'description': analysis_text,
                'timestamp': datetime.now().isoformat()
            })
            
            # Apply item limit
            descriptions = self._apply_item_limit(descriptions)
            
            cache_data = {
                'compound_name': normalize_compound_name(compound_name),
                'descriptions': descriptions,
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Create new data
            cache_data = {
                'compound_name': normalize_compound_name(compound_name),
                'descriptions': [{
                    'description': analysis_text,
                    'timestamp': datetime.now().isoformat()
                }],
                'timestamp': datetime.now().isoformat()
            }
        
        self._create_cache_file(cache_file_path, cache_data)
        logger.info(f"Cached analysis result for compound: {compound_name}")
    
    def get_analysis_results(self, compound_name: str) -> List[Dict]:
        """Get analysis results from cache."""
        cache_file_path = self._get_source_cache_file_path(compound_name)
        
        if not cache_file_path or not self._validate_cache_file(cache_file_path):
            return []
        
        cache_data = self._read_cache_file(cache_file_path)
        if cache_data:
            return cache_data.get('descriptions', [])
        
        return []


class FailedMoleculesCacheManager(BaseCacheManager):
    """Manages failed molecule names cache (molecules without XYZ data) with unified normalization."""
    
    def __init__(self, config: Dict, cache_base_directory: str = "cache"):
        """Initialize failed molecules cache manager."""
        super().__init__('failed_molecules', config, cache_base_directory)
    
    def add_failed_molecule(self, compound_name: str):
        """Add a compound name to the failed list."""
        cache_file_path = self._get_source_cache_file_path('failed_list')
        
        if not cache_file_path:
            logger.warning(f"Cannot save failed molecule cache: data source disabled")
            return
        
        # Read existing data if available
        existing_data = self._read_cache_file(cache_file_path) if os.path.exists(cache_file_path) else None
        
        normalized_name = normalize_compound_name(compound_name)
        
        if existing_data:
            # Add new failed molecule to existing list
            failed_molecules = existing_data.get('failed_molecules', [])
            if normalized_name not in failed_molecules:
                failed_molecules.append(normalized_name)
                logger.info(f"Added {normalized_name} to failed molecules list")
            else:
                logger.info(f"{normalized_name} already in failed molecules list")
            
            cache_data = {
                'failed_molecules': failed_molecules,
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Create new data
            cache_data = {
                'failed_molecules': [normalized_name],
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"Created new failed molecules list with {normalized_name}")
        
        self._create_cache_file(cache_file_path, cache_data)
        logger.info(f"Cached failed molecule: {normalized_name}")
    
    def is_molecule_failed(self, compound_name: str) -> bool:
        """Check if a compound is in the failed list."""
        cache_file_path = self._get_source_cache_file_path('failed_list')
        
        if not cache_file_path or not self._validate_cache_file(cache_file_path):
            return False
        
        cache_data = self._read_cache_file(cache_file_path)
        if not cache_data:
            return False
        
        failed_molecules = cache_data.get('failed_molecules', [])
        normalized_name = normalize_compound_name(compound_name)
        return normalized_name in failed_molecules
    
    def get_failed_molecules(self) -> List[str]:
        """Get list of all failed molecule names."""
        cache_file_path = self._get_source_cache_file_path('failed_list')
        
        if not cache_file_path or not self._validate_cache_file(cache_file_path):
            return []
        
        cache_data = self._read_cache_file(cache_file_path)
        if cache_data:
            return cache_data.get('failed_molecules', [])
        
        return []
    
    def remove_failed_molecule(self, compound_name: str):
        """Remove a compound name from the failed list (if XYZ data becomes available)."""
        cache_file_path = self._get_source_cache_file_path('failed_list')
        
        if not cache_file_path or not os.path.exists(cache_file_path):
            return
        
        cache_data = self._read_cache_file(cache_file_path)
        if not cache_data:
            return
        
        failed_molecules = cache_data.get('failed_molecules', [])
        normalized_name = normalize_compound_name(compound_name)
        if normalized_name in failed_molecules:
            failed_molecules.remove(normalized_name)
            logger.info(f"Removed {normalized_name} from failed molecules list")
            
            cache_data = {
                'failed_molecules': failed_molecules,
                'timestamp': datetime.now().isoformat()
            }
            
            self._create_cache_file(cache_file_path, cache_data)
