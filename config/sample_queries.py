#!/usr/bin/env python3
# Sample queries configuration for ChatMOL
import json
import os
from typing import List, Dict

def load_sample_queries(config_file: str = "config/sample_queries.json") -> List[Dict[str, str]]:
    """Load sample queries from JSON file."""
    if not os.path.exists(config_file):
        return []
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        queries = []
        for category_name, category_data in data.get('categories', {}).items():
            queries.extend(category_data.get('queries', []))
        
        return queries
    except Exception as e:
        print(f"Error loading sample queries from {config_file}: {e}")
        return []

def get_sample_queries_by_category(config_file: str = "config/sample_queries.json") -> Dict[str, Dict]:
    """Load sample queries organized by category from JSON file."""
    if not os.path.exists(config_file):
        return {}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data.get('categories', {})
    except Exception as e:
        print(f"Error loading sample queries by category from {config_file}: {e}")
        return {}

# Load all queries for backward compatibility
SAMPLE_QUERIES: List[Dict[str, str]] = load_sample_queries()
