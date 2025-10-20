#!/usr/bin/env python3
# Unified cache utilities for ChatMOL
import os
import json
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

def normalize_compound_name(name: str) -> str:
    """
    Normalize compound name for consistent PubChem search and cache operations.
    
    This function applies the same normalization logic used in PubChem search
    to ensure consistency between cache keys and PubChem queries.
    
    Args:
        name: Compound name to normalize
        
    Returns:
        Normalized compound name suitable for cache keys and PubChem search
    """
    if not name:
        return ""
    
    # Apply same normalization as PubChem search variations
    normalized = name.lower().strip()
    normalized = normalized.replace(" acid", "").replace(" salt", "")
    normalized = normalized.replace(" ", "")
    
    # Create safe filename by replacing special characters
    safe_key = re.sub(r'[^\w\-_]', '_', normalized)
    
    return safe_key


class NameMappingCacheManager:
    """Manages compound name mappings (Japanese <-> English) cache operations."""
    
    def __init__(self, cache_base_dir: str = "cache"):
        """Initialize name mapping cache manager."""
        self.cache_base_dir = cache_base_dir
        self.cache_dir = os.path.join(cache_base_dir, "name_mappings")
        self.max_age_days = 36500  # Long retention for name mappings
        self._ensure_cache_directory()
    
    def _ensure_cache_directory(self):
        """Ensure cache directory exists."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"Created name mappings cache directory: {self.cache_dir}")
    
    def _get_cache_file_path(self, compound_name: str) -> str:
        """Get cache file path for compound name."""
        cache_key = normalize_compound_name(compound_name)
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _validate_cache_file(self, cache_file_path: str) -> bool:
        """Check if cache file is valid (exists and not expired)."""
        if not os.path.exists(cache_file_path):
            return False
        
        # Check age
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file_path))
        age = datetime.now() - file_time
        if age > timedelta(days=self.max_age_days):
            logger.info(f"Name mapping cache expired for {cache_file_path}")
            return False
        
        return True
    
    def save_mapping(self, compound_name: str, name_jp: str, name_en: str):
        """Save compound name mapping to cache."""
        cache_file_path = self._get_cache_file_path(compound_name)
        
        try:
            cache_data = {
                "compound_name": compound_name,
                "name_jp": name_jp,
                "name_en": name_en,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(cache_file_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Cached name mapping: {compound_name} -> {name_jp} ({name_en})")
            
        except Exception as e:
            logger.error(f"Error saving name mapping cache for {compound_name}: {e}")
    
    def get_mapping(self, compound_name: str) -> Optional[Dict[str, str]]:
        """Get compound name mapping from cache."""
        cache_file_path = self._get_cache_file_path(compound_name)
        
        if not self._validate_cache_file(cache_file_path):
            return None
        
        try:
            with open(cache_file_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            logger.info(f"Cache hit for name mapping: {compound_name}")
            return {
                "name_jp": cache_data.get("name_jp", compound_name),
                "name_en": cache_data.get("name_en", compound_name)
            }
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Invalid name mapping cache data in {cache_file_path}: {e}")
            # Remove invalid cache file
            try:
                os.remove(cache_file_path)
            except OSError:
                pass
            return None
        except Exception as e:
            logger.error(f"Error reading name mapping cache file {cache_file_path}: {e}")
            return None
    
    def get_names_for_display(self, compound_name: str) -> Tuple[str, str]:
        """Get Japanese and English names for display purposes."""
        mapping = self.get_mapping(compound_name)
        if mapping:
            return mapping["name_jp"], mapping["name_en"]
        else:
            # Fallback to compound_name if no mapping found
            return compound_name, compound_name
    
    def clear_cache(self):
        """Clear all name mapping cache files."""
        try:
            if os.path.exists(self.cache_dir):
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.json'):
                        file_path = os.path.join(self.cache_dir, filename)
                        os.remove(file_path)
                logger.info("Name mapping cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing name mapping cache: {e}")


class BaseCacheManager:
    """Base cache manager with common functionality using unified normalization."""
    
    def __init__(self, data_source: str, config: Dict, cache_base_directory: str = "cache"):
        """Initialize base cache manager."""
        self.data_source_name = data_source
        self.data_source_config = config
        self.cache_base_directory = cache_base_directory
        self.cache_expiration_days = config.get('max_age_days', 360)
        self._ensure_cache_directory()
    
    def _ensure_cache_directory(self):
        """Ensure cache directory exists for this data source."""
        if not self.data_source_config.get('enabled', True):
            return
        
        cache_dir = self._get_source_cache_directory()
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            logger.info(f"Created cache directory for {self.data_source_name}: {cache_dir}")
    
    def _get_source_cache_directory(self) -> Optional[str]:
        """Get cache directory for this data source."""
        if not self.data_source_config.get('enabled', True):
            return None
        return os.path.join(self.cache_base_directory, self.data_source_config['directory'])
    
    def _get_source_cache_file_path(self, compound_name: str) -> Optional[str]:
        """Get cache file path for given compound name."""
        cache_dir = self._get_source_cache_directory()
        if not cache_dir:
            return None
        normalized_name = normalize_compound_name(compound_name)
        return os.path.join(cache_dir, f"{normalized_name}.json")
    
    def _validate_cache_file(self, cache_file_path: str) -> bool:
        """Check if cache file is valid (exists and not expired)."""
        if not os.path.exists(cache_file_path):
            return False
        
        # Check age
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file_path))
        age = datetime.now() - file_time
        if age > timedelta(days=self.cache_expiration_days):
            logger.info(f"Cache expired for {cache_file_path}")
            return False
        
        return True
    
    def _create_cache_file(self, cache_file_path: str, data: Dict) -> bool:
        """Create cache file with data."""
        try:
            with open(cache_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error creating cache file {cache_file_path}: {e}")
            return False
    
    def _read_cache_file(self, cache_file_path: str) -> Optional[Dict]:
        """Read cache file and return data."""
        try:
            with open(cache_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Invalid cache data in {cache_file_path}: {e}")
            # Remove invalid cache file
            try:
                os.remove(cache_file_path)
            except OSError:
                pass
            return None
        except Exception as e:
            logger.error(f"Error reading cache file {cache_file_path}: {e}")
            return None
    
    def _apply_item_limit(self, items: List[Dict], item_key: str = 'items') -> List[Dict]:
        """Apply item limit to a list of items, keeping only the latest entries."""
        max_items = self.data_source_config.get('max_items_per_file')
        if max_items is None or len(items) <= max_items:
            return items
        
        # Keep only the latest max_items entries
        limited_items = items[-max_items:]
        logger.info(f"Trimmed {self.data_source_name} cache to latest {max_items} entries (was {len(items)})")
        return limited_items
