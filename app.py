# Standard library imports
import random
import json
import logging
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Third-party imports
import streamlit as st
import streamlit.components.v1 as components

from google import genai
from google.genai import types

import py3Dmol
import pubchempy as pcp

from st_screen_stats import WindowQueryHelper

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

# Import shared models from core
from core.models import DetailedMoleculeInfo

# Import common utilities from core
from core.utils import execute_with_timeout
from core.gemini_client import call_gemini_api

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Import shared prompts from core
from core.prompts import AIPrompts

# Import sample queries from config
from config.sample_queries import SAMPLE_QUERIES

# Import settings from config
from config.settings import Config, ANNOUNCEMENT_MESSAGE, MENU_ITEMS_ABOUT

# =============================================================================
# CACHE MODE CONFIGURATION
# =============================================================================

def get_cache_mode():
    """secrets.tomlからキャッシュモードを取得"""
    try:
        return st.secrets.get("CACHE_MODE", "fallback_only")
    except Exception as e:
        logger.warning(f"Failed to read CACHE_MODE from secrets: {e}")
        return "fallback_only"  # デフォルト値

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging() -> logging.Logger:
    """Setup application logging."""
    logger = logging.getLogger("chatmol")
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# =============================================================================
# ERROR HANDLING
# =============================================================================

# ErrorHandler is now imported from core.error_handler
# Add Streamlit-specific error display method
def show_error(message: str) -> None:
    """Show error message using Streamlit."""
    st.error(f"⚠️ {message}")


def has_similar_molecules_in_cache(english_name: str) -> bool:
    """Check if similar molecules exist in cache for the given English name."""
    try:
        if not english_name:
            return False
        
        # Normalize the English name for cache lookup
        normalized_name = english_name.lower()
        
        # Check if similar molecules cache exists for this compound
        similar_result = cache_manager.similar.get_random_similar_compound(normalized_name)
        return similar_result is not None
    except Exception as e:
        logger.warning(f"Error checking similar molecules cache for {english_name}: {e}")
        return False

def show_action_buttons(key_prefix: str = "action") -> None:
    """Show standardized action button set: 詳しく知りたい, 関連する分子は？, 他の分子を探す."""
    col1, col2, col3 = st.columns(3)
    
    # Check data availability
    current_data = st.session_state.get("current_molecule_data", None)
    has_cid = current_data and current_data.get("cid") is not None
    has_name = current_data and current_data.get("name") is not None
    
    # For cache_only mode, also check if similar molecules exist in cache
    has_similar_molecules = True  # Default to True for fallback_only mode
    if get_cache_mode() == 'cache_only':
        english_name = current_data.get("name_en") if current_data else None
        has_similar_molecules = has_similar_molecules_in_cache(english_name) if english_name else False
        logger.info(f"Cache only mode: similar molecules available for {english_name}: {has_similar_molecules}")
    
    with col1:
        if st.button("詳しく知りたい", key=f"{key_prefix}_detail", use_container_width=True, icon="🧪", disabled=not has_cid):
            if has_cid:
                # Log user action
                log_user_action("detail_view")
                # Reset analysis execution flag and clear cache to allow new analysis
                st.session_state.detail_analysis_executed = False
                st.session_state.cached_analysis_result = ""
                st.session_state.screen = "detail_response"
                st.rerun()
    
    with col2:
        # Disable similar button if no similar molecules available in cache_only mode
        similar_button_disabled = not has_name or not has_similar_molecules
        if st.button("関連する分子は？", key=f"{key_prefix}_similar", use_container_width=True, icon="🔍", disabled=similar_button_disabled):
            if has_name and has_similar_molecules:
                # Log user action
                log_user_action("similar_search")
                # Reset search execution flag to allow new search
                st.session_state.similar_search_executed = False
                st.session_state.screen = "similar_response"
                st.rerun()
    
    with col3:
        if st.button("他の分子を探す", key=f"{key_prefix}_new", use_container_width=True, icon="😀"):
            reset_to_initial_state()

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def reset_to_initial_state():
    """Reset the application to initial state."""
    # End current session before resetting
    end_current_session()
    
    st.session_state.screen = "initial"
    st.session_state.user_query = ""
    st.session_state.selected_sample = ""
    st.session_state.current_molecule_data = None
    st.session_state.gemini_output = None
    st.session_state.random_queries = generate_random_queries()
    st.session_state.similar_search_executed = False
    st.session_state.detail_analysis_executed = False
    st.session_state.cached_analysis_result = ""
    # Reset new fields
    st.session_state.detailed_info = None
    
    st.rerun()


def generate_random_queries_from_cache() -> List[Dict[str, str]]:
    """Cache Onlyモード用: cache/queries/からクエリを取得"""
    cache_dir = "cache/queries"
    if not os.path.exists(cache_dir):
        logger.warning(f"Cache directory not found: {cache_dir}")
        return []
    
    # 全クエリファイルを取得
    query_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
    if not query_files:
        logger.warning(f"No query files found in: {cache_dir}")
        return []
    
    # 各ファイルからクエリ情報を読み取り
    queries = []
    for file in query_files:
        file_path = os.path.join(cache_dir, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                query_text = data.get('query_text', '')
                icon = data.get('icon', '🔍')
                if query_text:
                    queries.append({
                        'text': query_text,
                        'icon': icon
                    })
        except Exception as e:
            logger.warning(f"Failed to read query file {file}: {e}")
    
    # 指定された最大数までランダム選択
    max_count = Config.RANDOM_QUERY['count']
    if len(queries) <= max_count:
        return queries
    else:
        return random.sample(queries, max_count)


def generate_random_queries() -> List[Dict[str, str]]:
    """Generate random samples from all available queries."""
    cache_mode = get_cache_mode()
    
    if cache_mode == 'cache_only':
        return generate_random_queries_from_cache()
    else:
        # 既存の実装
        if SAMPLE_QUERIES:
            return random.sample(SAMPLE_QUERIES, min(Config.RANDOM_QUERY['count'], len(SAMPLE_QUERIES)))
        else:
            return []


def get_compound_from_query_cache(query_text: str) -> Optional[str]:
    """選択されたクエリからランダムに化合物を選択"""
    # QueryCacheManagerのメソッドを使用して正規化されたファイルパスを取得
    cache_file_path = cache_manager.queries._get_source_cache_file_path(query_text)
    if not cache_file_path or not os.path.exists(cache_file_path):
        logger.warning(f"Query cache file not found for: {query_text}")
        return None
    
    try:
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        compounds = data.get('compounds', [])
        if not compounds:
            logger.warning(f"No compounds found in query cache: {query_text}")
            return None
        
        # ランダムに1つ選択
        random_compound = random.choice(compounds)
        compound_name = random_compound.get('compound_name')
        logger.info(f"Selected compound from query cache: {compound_name}")
        return compound_name
    except Exception as e:
        logger.error(f"Failed to read compound from query cache: {e}")
        return None


def build_molecule_data_from_cache(compound_name: str) -> Optional[Dict[str, Any]]:
    """キャッシュのみから分子データを構築"""
    try:
        logger.info(f"Building molecule data from cache for: {compound_name}")
        
        # 1. 日本語名を取得
        name_jp, name_en = cache_manager.get_compound_names_for_display(compound_name)
        logger.info(f"Retrieved names: JP={name_jp}, EN={name_en}")
        
        # 2. 説明を取得
        description = cache_manager.descriptions.get_random_description(compound_name)
        if not description:
            logger.warning(f"No description found for: {compound_name}")
            return None
        logger.info(f"Retrieved description for: {compound_name}")
        
        # 3. 立体構造データを取得
        cached_data = cache_manager.pubchem.get_cached_molecule_data(compound_name)
        if not cached_data:
            logger.warning(f"No PubChem data found for: {compound_name}")
            return None
        
        detailed_info, cid = cached_data
        if not detailed_info or not detailed_info.xyz_data:
            logger.warning(f"No XYZ data found for: {compound_name}")
            return None
        
        logger.info(f"Retrieved XYZ data for: {compound_name}")
        
        # 4. 統合された分子データを構築
        molecule_data = {
            'name': name_jp,
            'name_en': name_en,
            'cid': cid,
            'xyz_data': detailed_info.xyz_data,
            'memo': description,
            'molecular_formula': detailed_info.molecular_formula,
            'molecular_weight': detailed_info.molecular_weight,
            'canonical_smiles': getattr(detailed_info, 'canonical_smiles', None),  # 安全にアクセス
            'detailed_info': detailed_info  # detailed_infoオブジェクトを保存
        }
        
        logger.info(f"Successfully built molecule data for: {compound_name}")
        return molecule_data
    except Exception as e:
        logger.error(f"Failed to build molecule data from cache: {e}")
        return None


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

import os
import hashlib
from datetime import datetime, timedelta

# Import unified cache utilities from core
from core.cache_utils import normalize_compound_name, NameMappingCacheManager, BaseCacheManager
from core.cache_managers import (
    PubChemCacheManager, QueryCacheManager, DescriptionCacheManager,
    SimilarMoleculesCacheManager, AnalysisCacheManager, FailedMoleculesCacheManager
)
from core.cache import UnifiedCacheManager

# Initialize cache manager using UnifiedCacheManager
cache_manager = UnifiedCacheManager(cache_base_dir=Config.CACHE['base_directory'], config=Config.CACHE)

# =============================================================================
# QUERY ANALYTICS FUNCTIONS
# =============================================================================

def get_or_create_session_id():
    """Get existing session ID or create new one."""
    if 'analytics_session_id' not in st.session_state:
        st.session_state.analytics_session_id = str(uuid.uuid4())
    return st.session_state.analytics_session_id

def end_current_session():
    """End current session and prepare for new session."""
    try:
        # Log session end action before clearing session ID
        if 'analytics_session_id' in st.session_state:
            log_user_action("session_end")
            logger.info(f"Session ended: {st.session_state.analytics_session_id}")
        
        # Clear current session ID
        if 'analytics_session_id' in st.session_state:
            del st.session_state.analytics_session_id
            
    except Exception as e:
        logger.error(f"Error ending session: {e}")

def log_user_action(action_type: str):
    """Log user action to analytics."""
    try:
        analytics_file = os.path.join(Config.CACHE['base_directory'], 'analytics', 'query_log.json')
        
        # Ensure analytics directory exists
        os.makedirs(os.path.dirname(analytics_file), exist_ok=True)
        
        # Load existing data
        if os.path.exists(analytics_file):
            with open(analytics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"sessions": []}
        
        # Get current session ID
        session_id = get_or_create_session_id()
        current_time = datetime.now().isoformat()
        
        # Find existing session or create new one
        session_found = False
        for session in data["sessions"]:
            if session["session_id"] == session_id:
                session["actions"].append({
                    "action_type": action_type,
                    "timestamp": current_time
                })
                session_found = True
                break
        
        # If no existing session found, create new one
        if not session_found:
            # Get initial query from session state
            initial_query = st.session_state.get("user_query", "")
            data["sessions"].append({
                "session_id": session_id,
                "initial_query": initial_query,
                "initial_timestamp": current_time,
                "actions": [{
                    "action_type": action_type,
                    "timestamp": current_time
                }]
            })
        
        # Save updated data
        with open(analytics_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Logged action: {action_type} for session {session_id} at {current_time}")
        
    except Exception as e:
        logger.error(f"Error logging user action: {e}")


def save_query_selection(query_text: str):
    """Save query selection to analytics log using new session-based structure."""
    # Log the initial query as an action
    log_user_action("initial_query")

# =============================================================================
# AI AND MOLECULAR PROCESSING FUNCTIONS
# =============================================================================


def search_molecule_by_query_cache_only(user_input_text: str) -> Optional[Dict[str, Any]]:
    """Cache Onlyモード用: キャッシュのみを使用する検索関数"""
    logger.info(f"Cache Only mode: Processing query: {user_input_text[:50]}...")
    
    # 1. クエリから化合物を選択
    compound_name = get_compound_from_query_cache(user_input_text)
    if not compound_name:
        logger.info(f"No cached compounds found for query: {user_input_text[:50]}...")
        return None
    
    # 2. 分子データを構築
    molecule_data = build_molecule_data_from_cache(compound_name)
    if not molecule_data:
        logger.info(f"Failed to build molecule data for: {compound_name}")
        return None
    
    logger.info(f"Successfully built molecule data for: {compound_name}")
    return molecule_data


def search_molecule_by_query(user_input_text: str) -> Optional[str]:
    """Search and recommend molecules based on user query."""
    cache_mode = get_cache_mode()
    logger.info(f"Processing user query: {user_input_text[:50]}... (Cache mode: {cache_mode})")
    
    if cache_mode == 'cache_only':
        # Cache Only: キャッシュのみ使用
        logger.info("Using cache-only mode")
        molecule_data = search_molecule_by_query_cache_only(user_input_text)
        if molecule_data:
            # 分子データをセッションに保存
            st.session_state.current_molecule_data = molecule_data
            st.session_state.gemini_output = molecule_data
            logger.info(f"Successfully processed query in cache-only mode: {user_input_text[:50]}...")
            return "success"  # 成功を示すマーカー
        else:
            logger.info(f"No cached data available for query: {user_input_text[:50]}...")
            return None  # キャッシュがない場合は何も返さない
    else:
        # Fallback Only: 既存の実装
        logger.info("Using fallback-only mode")
        prompt = AIPrompts.MOLECULAR_SEARCH.format(user_input=user_input_text)
        
        with st.spinner(f"AI (`{model_name}`) に問い合わせ中...", show_time=True):
            response = call_gemini_api(
                prompt=prompt,
                client=client,
                model_name=model_name,
                query_type="molecular_search"
            )
            # Handle response (backward compatibility)
            response_text = response if isinstance(response, str) else response['text'] if isinstance(response, dict) else None
        
        # Check if response indicates no results
        if is_no_result_response(response_text):
            logger.info(f"No results from Gemini for query: {user_input_text[:50]}...")
            
            # Try to get a random compound from cache
            random_compound = cache_manager.queries.get_random_compound_from_query(user_input_text)
            if random_compound:
                logger.info(f"Using cached compound as fallback: {random_compound}")
                # Create a fallback response with the cached compound
                fallback_response = f'{{"name_jp": "{random_compound}", "name_en": "{random_compound}", "description": "過去の検索結果から選んだ分子だよ！✨ この分子について詳しく調べてみよう！"}}'
                return fallback_response
            else:
                logger.info(f"No cached compounds found for query: {user_input_text[:50]}...")
                return response_text  # Return original "該当なし" response
        
        return response_text

def find_similar_molecules_cache_only(molecule_name: str) -> Optional[str]:
    """Cache Onlyモード用: キャッシュのみを使用する類似分子検索関数"""
    logger.info(f"Cache Only mode: Finding similar molecules for: {molecule_name}")
    
    current_data = st.session_state.get("current_molecule_data", None)
    if not current_data:
        logger.warning("No current_molecule_data available")
        return None
    
    # 英語名を取得（name_enフィールドから）
    english_name = current_data.get("name_en")
    if not english_name:
        logger.warning(f"No English name found in current_molecule_data: {current_data}")
        return None
    
    # 英語名を小文字に正規化してキャッシュを検索
    normalized_english_name = english_name.lower()
    logger.info(f"Searching similar compounds for normalized name: {normalized_english_name}")
    
    random_result = cache_manager.similar.get_random_similar_compound(normalized_english_name)
    if random_result:
        logger.info(f"Using cached similar compound for: {normalized_english_name}")
        # ランダムに1つ選択
        random_compound_name, random_description = random_result
        
        # 日本語名と英語名を適切に取得
        name_jp, name_en = cache_manager.get_compound_names_for_display(random_compound_name)
        
        # 日本語名が取得できない場合は英語名を使用
        if not name_jp:
            name_jp = name_en
        
        similar_response = f'{{"name_jp": "{name_jp}", "name_en": "{name_en}", "description": "{random_description}"}}'
        return similar_response
    
    logger.info(f"No cached similar compounds available for: {normalized_english_name}")
    return None


def find_similar_molecules_with_cache(molecule_name: str) -> Optional[str]:
    """Find similar molecules using common function with cache support."""
    cache_mode = get_cache_mode()
    logger.info(f"Searching for similar molecules to: {molecule_name} (Cache mode: {cache_mode})")
    
    if cache_mode == 'cache_only':
        # Cache Only: キャッシュのみ使用
        logger.info("Using cache-only mode")
        return find_similar_molecules_cache_only(molecule_name)
    else:
        # Fallback Only: 既存の実装
        logger.info("Using fallback-only mode")
        # Get English name from current data for cache key
        current_data = st.session_state.get("current_molecule_data", None)
        
        # Create a wrapper function that matches the expected signature
        def gemini_client_wrapper(prompt: str, use_google_search: bool = True) -> Optional[str]:
            response = call_gemini_api(
                prompt=prompt,
                client=client,
                model_name=model_name,
                query_type="similar_molecule_search"
            )
            # Handle response (backward compatibility)
            return response if isinstance(response, str) else response['text'] if isinstance(response, dict) else None
        
        # Use common function for similar molecule search
        return find_similar_molecules(molecule_name, gemini_client_wrapper, cache_manager, current_data)

# Import PubChem client functions from core
from core.pubchem import (
    get_compounds_by_name,
    get_3d_coordinates_by_cid,
    convert_pubchem_to_xyz,
    execute_with_timeout,
    get_comprehensive_molecule_data
)

def get_comprehensive_molecule_data_cache_only(english_name: str) -> Tuple[bool, Optional[DetailedMoleculeInfo], Optional[int], Optional[str]]:
    """Cache Onlyモード用: キャッシュのみを使用する分子データ取得関数"""
    logger.info(f"Cache Only mode: Getting data for: {english_name}")
    
    cached_data = cache_manager.pubchem.get_cached_molecule_data(english_name)
    if cached_data:
        detailed_info, cid = cached_data
        logger.info(f"Using cached data for: {english_name}")
        return True, detailed_info, cid, None
    else:
        logger.info(f"No cached data available for: {english_name}")
        return False, None, None, "No cached data available"


def get_comprehensive_molecule_data_with_cache(english_name: str) -> Tuple[bool, Optional[DetailedMoleculeInfo], Optional[int], Optional[str]]:
    """Get comprehensive molecule data from PubChem using English name with cache support."""
    cache_mode = get_cache_mode()
    logger.info(f"Getting comprehensive data for: {english_name} (Cache mode: {cache_mode})")
    
    if cache_mode == 'cache_only':
        # Cache Only: キャッシュのみ使用
        logger.info("Using cache-only mode")
        return get_comprehensive_molecule_data_cache_only(english_name)
    else:
        # Fallback Only: 既存の実装
        logger.info("Using fallback-only mode")
        # Check cache first
        cached_data = cache_manager.pubchem.get_cached_molecule_data(english_name)
        if cached_data:
            detailed_info, cid = cached_data
            logger.info(f"Using cached data for: {english_name}")
            return True, detailed_info, cid, None
        
        with st.spinner("分子データを取得中...", show_time=True):
            # Use the shared function from core
            success, detailed_info, cid, error_msg = get_comprehensive_molecule_data(english_name)
            
            if success and detailed_info:
                # Save to cache only if xyz_data is available
                if detailed_info.xyz_data:
                    cache_manager.pubchem.save_cached_molecule_data(english_name, detailed_info, cid)
                else:
                    logger.warning(f"Skipping cache save for {english_name}: No xyz_data available")
                    # Add to failed molecules list when XYZ data is not available
                    cache_manager.failed_molecules.add_failed_molecule(english_name)
            
            return success, detailed_info, cid, error_msg

def analyze_molecule_properties_cache_only(detailed_info: DetailedMoleculeInfo, molecule_name: str) -> Optional[str]:
    """Cache Onlyモード用: キャッシュのみを使用する分析関数"""
    logger.info(f"Cache Only mode: Getting analysis for: {molecule_name}")
    
    current_data = st.session_state.get("current_molecule_data", None)
    english_name = current_data.get("name_en") if current_data else None
    
    if english_name:
        cached_analyses = cache_manager.analysis.get_analysis_results(english_name)
        if cached_analyses:
            logger.info(f"Using cached analysis for: {english_name}")
            # ランダムに1つ選択
            random_analysis = random.choice(cached_analyses)
            return random_analysis.get('description')
    
    logger.info(f"No cached analysis available for: {molecule_name}")
    return None


def analyze_molecule_properties_with_cache(detailed_info: DetailedMoleculeInfo, molecule_name: str) -> Optional[str]:
    """Analyze molecular properties using common function with cache support."""
    cache_mode = get_cache_mode()
    logger.info(f"Getting Gemini analysis for molecule: {molecule_name} (Cache mode: {cache_mode})")
    
    if cache_mode == 'cache_only':
        # Cache Only: キャッシュのみ使用
        logger.info("Using cache-only mode")
        return analyze_molecule_properties_cache_only(detailed_info, molecule_name)
    else:
        # Fallback Only: 既存の実装
        logger.info("Using fallback-only mode")
        # Check cache first using English name
        current_data = st.session_state.get("current_molecule_data", None)
        english_name = current_data.get("name_en") if current_data else None
        
        if english_name:
            cached_analyses = cache_manager.analysis.get_analysis_results(english_name)
            if cached_analyses:
                logger.info(f"Using cached analysis for: {english_name}")
                # Return the most recent analysis
                return cached_analyses[-1]['description']
        
        # Create a wrapper function that matches the expected signature
        def gemini_client_wrapper(prompt: str, use_google_search: bool = False) -> Optional[str]:
            response = call_gemini_api(
                prompt=prompt,
                client=client,
                model_name=model_name,
                query_type="molecular_analysis"
            )
            # Handle response (backward compatibility)
            return response if isinstance(response, str) else response['text'] if isinstance(response, dict) else None
        
        # Use common function for analysis
        return analyze_molecule_properties(detailed_info, molecule_name, gemini_client_wrapper, cache_manager)

# =============================================================================
# APPLICATION INITIALIZATION
# =============================================================================

# Configure Streamlit page settings
# These settings control the overall appearance and behavior of the app
st.set_page_config(
    page_title="ChatMOL",
    page_icon="images/favicon.png",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'About': MENU_ITEMS_ABOUT,
    }
)

# Configure molecule viewer size based on window size
if WindowQueryHelper().minimum_window_size(min_width=Config.VIEWER['width_pc'])["status"]:
    # PC size
    MOLECULE_VIEWER_WIDTH = Config.VIEWER['width_pc']
    MOLECULE_VIEWER_HEIGHT = Config.VIEWER['height_pc']
else:
    # Mobile size
    MOLECULE_VIEWER_WIDTH = Config.VIEWER['width_mobile']
    MOLECULE_VIEWER_HEIGHT = Config.VIEWER['height_mobile']

# Initialize Gemini AI API with comprehensive error handling
# This ensures the app fails gracefully if API configuration is missing
try:
    # Configure API key from Streamlit secrets
    client = genai.Client(api_key=st.secrets["api_key"])
    
    # Use fixed model name
    model_name = Config.DEFAULT_MODEL_NAME
    
    # Configure cache settings from secrets.toml (optional)
    if "cache_enabled" in st.secrets:
        Config.CACHE['enabled'] = st.secrets["cache_enabled"]
        logger.info(f"Cache enabled from secrets.toml: {Config.CACHE['enabled']}")

except KeyError as e:
    if str(e) == "'api_key'":
        st.error("api_key が設定されていません。Streamlit の Secrets で設定してください。")
    else:
        st.error(f"設定が見つかりません: {e}")
    st.stop()
except Exception as e:
    st.error(f"Gemini API の初期化に失敗しました: {e}")
    st.stop()

# =============================================================================
# RESPONSE PARSING AND VISUALIZATION FUNCTIONS
# =============================================================================

# Import error handling utilities from core
from core.error_handler import (
    ErrorHandler,
    is_no_result_response,
    parse_json_response
)

# Import common molecular analysis functions from core
from core.analysis import (
    analyze_molecule_properties,
    find_similar_molecules
)

def create_default_molecule_data() -> Dict[str, Union[str, None, Any]]:
    """Create default molecule data structure."""
    return {
        "name": "分子が見つかりませんでした",
        "name_jp": "分子が見つかりませんでした",
        "name_en": "Molecule not found",
        "memo": "ご要望に合う分子を見つけることができませんでした。もう少し具体的な情報を教えていただけますか？",
        "properties": None,
        "cid": None,
        "detailed_info": None,
        "xyz_data": None
    }

def parse_gemini_response(response_text: str, save_to_query_cache: bool = True, is_similar_search: bool = False) -> Dict[str, Union[str, None, Any]]:
    """Parse Gemini's JSON response and fetch comprehensive data from PubChem.
    
    Args:
        response_text: Gemini API response text
        save_to_query_cache: Whether to save to query cache (default: True)
    """
    data = create_default_molecule_data()
    
    if not response_text:
        logger.warning("Empty response text received")
        return data
    
    logger.info(f"Parsing Gemini response: {response_text[:200]}...")
    
    try:
        # Extract JSON from response text
        json_data = parse_json_response(response_text)
        
        if json_data:
            logger.info(f"Parsed JSON data: {json_data}")
            # Handle new format (name_jp, name_en, description)
            molecule_name_jp = json_data.get("name_jp", "").strip()
            molecule_name_en = json_data.get("name_en", "").strip()
            description = json_data.get("description", "").strip()
            
            logger.info(f"Extracted: name_jp='{molecule_name_jp}', name_en='{molecule_name_en}', description='{description[:50]}...'")
            
            if molecule_name_jp and molecule_name_jp != "該当なし" and molecule_name_en:
                # For similar search, use basic data without PubChem lookup
                if is_similar_search:
                    logger.info(f"Similar search mode: using basic data for {molecule_name_en}")
                    data["name"] = molecule_name_jp
                    data["name_jp"] = molecule_name_jp
                    data["name_en"] = molecule_name_en
                    data["memo"] = description if description else "関連する分子の情報です。"
                    # For similar molecules, we don't need xyz_data or detailed_info
                    data["xyz_data"] = None
                    data["detailed_info"] = None
                    data["cid"] = None
                    return data
                
                # Check if molecule is in failed list first
                if cache_manager.failed_molecules.is_molecule_failed(molecule_name_en):
                    logger.info(f"Molecule {molecule_name_en} is in failed list, using fallback directly")
                    # Use fallback directly without trying PubChem
                    user_query = st.session_state.get("user_query", "")
                    fallback_data = cache_manager.get_fallback_molecule_data(user_query)
                    
                    if fallback_data:
                        fallback_compound_name, fallback_description, fallback_xyz_data = fallback_data
                        logger.info(f"Using fallback data: {fallback_compound_name}")
                        
                        # Replace Gemini's response with fallback data
                        data["name"] = fallback_compound_name
                        data["name_jp"] = fallback_compound_name
                        data["name_en"] = fallback_compound_name
                        data["memo"] = fallback_description
                        data["xyz_data"] = fallback_xyz_data
                        
                        # Get cached detailed info for the fallback compound
                        cached_data = cache_manager.pubchem.get_cached_molecule_data(fallback_compound_name)
                        if cached_data:
                            detailed_info, cid = cached_data
                            data["detailed_info"] = detailed_info
                            data["cid"] = cid
                    else:
                        data["memo"] = f"分子「{molecule_name_en}」は過去にXYZデータの取得に失敗しており、フォールバックデータも見つかりませんでした。"
                else:
                    # Set basic data
                    data["name"] = molecule_name_jp
                    data["name_jp"] = molecule_name_jp
                    data["name_en"] = molecule_name_en
                    data["memo"] = description if description else "分子の詳細情報を取得中..."
                    
                    logger.info(f"Attempting to get comprehensive data for: {molecule_name_en}")
                    # Get comprehensive data from PubChem
                    success, detailed_info, cid, error_msg = get_comprehensive_molecule_data_with_cache(molecule_name_en)
                
                    if success and detailed_info:
                        logger.info(f"Successfully got comprehensive data for {molecule_name_en}")
                        data["detailed_info"] = detailed_info
                        data["xyz_data"] = detailed_info.xyz_data
                        data["cid"] = cid
                        
                        # 説明キャッシュをここで保存（xyz_dataが存在する場合のみ）
                        if detailed_info.xyz_data and description:
                            if save_to_query_cache:
                                user_query = st.session_state.get("user_query", "")
                                cache_manager.save_all_caches(molecule_name_jp, molecule_name_en, detailed_info, cid, user_query, description)
                            else:
                                # 関連分子処理時はqueriesキャッシュをスキップし、他のキャッシュのみ保存
                                cache_manager.name_mappings.save_mapping(normalize_compound_name(molecule_name_en), molecule_name_jp, molecule_name_en)
                                cache_manager.pubchem.save_cached_molecule_data(molecule_name_en, detailed_info, cid)
                                cache_manager.descriptions.save_compound_description(molecule_name_en, description)
                    else:
                        logger.warning(f"Failed to get comprehensive data: {error_msg}")
                        
                        # Try fallback from cache when XYZ data is not available
                        user_query = st.session_state.get("user_query", "")
                        fallback_data = cache_manager.get_fallback_molecule_data(user_query)
                        
                        if fallback_data:
                            fallback_compound_name, fallback_description, fallback_xyz_data = fallback_data
                            logger.info(f"Using fallback data: {fallback_compound_name}")
                            
                            # Replace Gemini's response with fallback data
                            data["name"] = fallback_compound_name
                            data["name_jp"] = fallback_compound_name
                            data["name_en"] = fallback_compound_name
                            data["memo"] = fallback_description
                            data["xyz_data"] = fallback_xyz_data
                            
                            # Get cached detailed info for the fallback compound
                            cached_data = cache_manager.pubchem.get_cached_molecule_data(fallback_compound_name)
                            if cached_data:
                                detailed_info, cid = cached_data
                                data["detailed_info"] = detailed_info
                                data["cid"] = cid
                        else:
                            data["memo"] = f"PubChemから分子データを取得できませんでした（{error_msg}）。"
            else:
                logger.warning(f"Invalid molecule data: name_jp='{molecule_name_jp}', name_en='{molecule_name_en}'")
                data["memo"] = Config.ERROR_MESSAGES['invalid_data']
        else:
            logger.warning("Failed to parse JSON from response")
            data["memo"] = Config.ERROR_MESSAGES['parse_error']
            
    except Exception as e:
        logger.error(f"Exception in parse_gemini_response: {e}")
        data["memo"] = f"{Config.ERROR_MESSAGES['parse_error']}: {e}"
    
    logger.info(f"Final parsed data: name='{data['name']}', memo='{data['memo'][:50]}...'")
    return data

# =============================================================================
# HELPER FUNCTIONS FOR SCREEN DISPLAY
# =============================================================================

def validate_molecule_data() -> bool:
    """Validate that molecule data exists and has required fields."""
    current_data = st.session_state.get("current_molecule_data", None)
    return (current_data and 
            current_data.get("cid") is not None and 
            current_data.get("detailed_info") is not None)

def get_molecule_name() -> str:
    """Get molecule name from current data."""
    current_data = st.session_state.get("current_molecule_data", None)
    return current_data.get("name", "分子") if current_data else "分子"

def ensure_random_queries():
    """Ensure random samples are generated and synced to session state."""
    if not st.session_state.get("random_queries", []):
        st.session_state.random_queries = generate_random_queries()

def create_error_molecule_data(error_message: str) -> Dict[str, Union[str, None, Any]]:
    """Create error molecule data structure."""
    return {
        "name": "エラーが発生しました",
        "memo": error_message,
        "properties": None,
        "cid": None
    }

def handle_error_and_show_buttons(error_message: str, button_key: str):
    """Handle error case and show appropriate buttons."""
    with st.chat_message("assistant"):
        st.write(error_message)
    show_action_buttons(button_key)

def process_molecule_query():
    """Process AI query and update molecule data."""
    try:
        user_query = st.session_state.get("user_query", "")
        
        # user_queryが空の場合は処理をスキップ
        if not user_query:
            logger.info("Skipping process_molecule_query: user_query is empty")
            return
        
        # Save query selection to analytics
        save_query_selection(user_query)
        
        # Log query response action
        log_user_action("query_response")
        
        cache_mode = get_cache_mode()
        if cache_mode == 'cache_only':
            # Cache Onlyモードでもsearch_molecule_by_query()を呼び出してデータをセッションに保存
            logger.info("Cache Only mode: Calling search_molecule_by_query to save data to session")
            response_text = search_molecule_by_query(user_query)
            if not response_text:
                # データが取得できなかった場合のエラーハンドリング
                error_data = create_error_molecule_data(
                    "キャッシュからデータを取得できませんでした。"
                )
                st.session_state.gemini_output = error_data
            return
        else:
            # Fallback Onlyモードの既存処理
            response_text = search_molecule_by_query(user_query)
            if response_text:
                # 通常のクエリ処理時はqueriesキャッシュに保存する
                parsed_output = parse_gemini_response(response_text, save_to_query_cache=True)
                st.session_state.gemini_output = parsed_output
            else:
                error_data = create_error_molecule_data(
                    "AIからの応答を取得できませんでした。"
                )
                st.session_state.gemini_output = error_data
    except Exception as e:
        error_data = create_error_molecule_data(
            f"予期しないエラーが発生しました: {e}"
        )
        st.session_state.gemini_output = error_data

def get_molecule_analysis() -> str:
    """Get molecule analysis result using saved detailed info."""
    try:
        current_data = st.session_state.get("current_molecule_data", None)
        
        if not current_data or not current_data.get("detailed_info"):
            logger.warning("No detailed_info available in current_molecule_data")
            return Config.ERROR_MESSAGES['no_data']
        
        detailed_info = current_data["detailed_info"]
        molecule_name = get_molecule_name()
        
        # Check cache first using English name
        english_name = current_data.get("name_en")
        if english_name:
            cached_analyses = cache_manager.analysis.get_analysis_results(english_name)
            if cached_analyses:
                logger.info(f"Using cached analysis for: {english_name}")
                # Return the most recent analysis
                return cached_analyses[-1]['description']
        
        # Use saved detailed info for analysis
        logger.info(f"Generating analysis result for: {molecule_name}")
        analysis_result = analyze_molecule_properties_with_cache(detailed_info, molecule_name)
        
        if analysis_result:
            return analysis_result
        else:
            logger.warning(f"No analysis result generated for: {molecule_name}")
            return Config.ERROR_MESSAGES['display_error']
    except Exception as e:
        logger.error(f"Error in get_molecule_analysis: {e}")
        return f"{Config.ERROR_MESSAGES['display_error']}: {e}"

def find_and_process_similar_molecule() -> Optional[Dict]:
    """Find and process similar molecule data."""
    try:
        similar_response = find_similar_molecules_with_cache(get_molecule_name())
        if similar_response:
            # Parse the basic response first
            basic_data = parse_gemini_response(similar_response, save_to_query_cache=False, is_similar_search=True)
            
            if basic_data:
                # Get the English name for cache lookup
                english_name = basic_data.get("name_en")
                if english_name:
                    logger.info(f"Looking up PubChem data for similar molecule: {english_name}")
                    
                    # Try to get comprehensive data from cache
                    success, detailed_info, cid, error_msg = get_comprehensive_molecule_data_with_cache(english_name)
                    
                    if success and detailed_info:
                        logger.info(f"Found PubChem data for similar molecule: {english_name}")
                        # Update the basic data with comprehensive information
                        basic_data["detailed_info"] = detailed_info
                        basic_data["xyz_data"] = detailed_info.xyz_data
                        basic_data["cid"] = cid
                        basic_data["molecular_formula"] = detailed_info.molecular_formula
                        basic_data["molecular_weight"] = detailed_info.molecular_weight
                        
                        # Get Japanese name from name mapping if available
                        name_jp, name_en = cache_manager.get_compound_names_for_display(english_name)
                        if name_jp:
                            basic_data["name"] = name_jp
                            basic_data["name_jp"] = name_jp
                        
                        logger.info(f"Successfully enhanced similar molecule data for: {english_name}")
                    else:
                        logger.warning(f"No PubChem data found for similar molecule: {english_name}")
                
                return basic_data
        return None
    except Exception as e:
        logger.error(f"Error finding similar molecules: {e}")
        return None

def show_initial_screen():
    """Display initial screen with greeting and random samples."""
    ensure_random_queries()

    with st.chat_message("assistant"):
        st.write("何かお手伝いできますか？")
    
    # display sample buttons
    random_queries = st.session_state.get("random_queries", [])
    if random_queries:
        cols = st.columns(Config.RANDOM_QUERY['columns'])
        for i, query in enumerate(random_queries):
            col_idx = i % Config.RANDOM_QUERY['columns']
            with cols[col_idx]:
                button_text = f"{query['icon']} {query['text']}"
                if st.button(button_text, key=f"random_query_{query['text']}", width="stretch"):
                    logger.info(f"User selected sample: {query['text']}")
                    st.session_state.selected_sample = query['text']
                    st.rerun()

def handle_cache_only_errors():
    """Cache Onlyモード用のエラーメッセージ"""
    return "キャッシュモードでは新しいデータを取得できません。既存のキャッシュから選択してください。"


def show_query_response_screen():
    """Display query response screen."""
    user_query = st.session_state.get("user_query", "")
    
    # user_queryが空の場合はinitial画面にリダイレクト
    if not user_query:
        st.session_state.screen = "initial"
        st.rerun()
        return
    
    with st.chat_message("user"):
        st.write(user_query)
    
    gemini_output = st.session_state.get("gemini_output", None)
    if not gemini_output or gemini_output.get("xyz_data") is None:
        process_molecule_query()
    
    gemini_output = st.session_state.get("gemini_output", None)
    
    if gemini_output and gemini_output.get("xyz_data"):
        output_data = gemini_output
        
        if output_data["xyz_data"] is None:
            st.write(output_data["memo"])
            show_action_buttons("no_molecule_found")
        else:
            st.session_state.current_molecule_data = output_data

            message = f"あなたにオススメする分子は「 **[{output_data['name']}](https://pubchem.ncbi.nlm.nih.gov/compound/{output_data['cid']})** 」だよ。{output_data['memo']}"
            with st.chat_message("assistant"):
                st.write(message)
                            
            display_molecule_3d(output_data)
            show_action_buttons("main_action")
    else:
        if gemini_output:
            st.write(gemini_output["memo"])
        else:
            # Cache Onlyモードでエラーの場合
            cache_mode = get_cache_mode()
            if cache_mode == 'cache_only':
                st.write(handle_cache_only_errors())
            else:
                st.write("エラーが発生しました。")
        show_action_buttons("error_main")

def show_detail_response_screen():
    """Display detail response screen."""
    if not validate_molecule_data():
        handle_error_and_show_buttons(Config.ERROR_MESSAGES['no_data'], "no_data_error")
        return

    with st.chat_message("user"):
        st.write(f"「 **{get_molecule_name()}** 」について、詳しく教えて")

    # Execute analysis only once per screen transition
    if not st.session_state.get("detail_analysis_executed", False):
        analysis_result = get_molecule_analysis()
        st.session_state.cached_analysis_result = analysis_result
        st.session_state.detail_analysis_executed = True
    
    # Display cached analysis result
    cached_result = st.session_state.get("cached_analysis_result", "")
    if cached_result:
        with st.chat_message("assistant"):

            # Display molecular properties metrics before analysis
            current_data = st.session_state.get("current_molecule_data", None)
            if current_data and current_data.get("detailed_info"):
                detailed_info = current_data["detailed_info"]
                
                # Display metrics in 3 columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if detailed_info.molecular_formula:
                        st.metric("分子式", detailed_info.molecular_formula)
                    if detailed_info.xlogp is not None:
                        st.metric("LogP", f"{detailed_info.xlogp:.2f}")
                    if detailed_info.hbond_donor_count is not None:
                        st.metric("水素結合供与体数", f"{detailed_info.hbond_donor_count}")
                
                with col2:
                    if detailed_info.molecular_weight:
                        st.metric("分子量（g/mol）", f"{detailed_info.molecular_weight:.2f}")
                    if detailed_info.tpsa:
                        st.metric("TPSA（Å²）", f"{detailed_info.tpsa:.1f}")
                    if detailed_info.hbond_acceptor_count is not None:
                        st.metric("水素結合受容体数", f"{detailed_info.hbond_acceptor_count}")
                
                with col3:
                    if detailed_info.heavy_atom_count is not None:
                        st.metric("重原子数", f"{detailed_info.heavy_atom_count}")
                    if detailed_info.complexity:
                        st.metric("分子複雑度", f"{detailed_info.complexity:.1f}")
                    if detailed_info.rotatable_bond_count is not None:
                        st.metric("回転可能結合数", f"{detailed_info.rotatable_bond_count}")
            
            st.write(cached_result)

    current_data = st.session_state.get("current_molecule_data", None)
    display_molecule_3d(current_data)

    show_action_buttons("detail_action")

def show_similar_response_screen():
    """Display similar molecule response screen."""
    current_data = st.session_state.get("current_molecule_data", None)
    if not current_data:
        handle_error_and_show_buttons(Config.ERROR_MESSAGES['no_data'], "no_data_error")
        return
    
    with st.chat_message("user"):
        st.write(f"「 {get_molecule_name()} 」に関連する分子は？")
    
    # Execute search only once per screen transition
    if not st.session_state.get("similar_search_executed", False):
        similar_data = find_and_process_similar_molecule()
        
        if similar_data:
            st.session_state.current_molecule_data = similar_data
        else:
            error_message = Config.ERROR_MESSAGES['molecule_not_found']
            handle_error_and_show_buttons(error_message, "similar_error_none")
            return
        
        st.session_state.similar_search_executed = True
    
    # Display current molecule data
    current_data = st.session_state.get("current_molecule_data", None)
    if current_data:
        # For similar molecules, we may not have xyz_data or cid
        if current_data.get("cid"):
            message = f"あなたにオススメする分子は「 **[{current_data['name']}](https://pubchem.ncbi.nlm.nih.gov/compound/{current_data['cid']})** 」だよ。{current_data['memo']}"
        else:
            message = f"あなたにオススメする分子は「 **{current_data['name']}** 」だよ。{current_data['memo']}"
        
        with st.chat_message("assistant"):
            st.write(message)
        
        # Only display 3D structure if xyz_data is available
        if current_data.get("xyz_data"):
            display_molecule_3d(current_data)
    
    show_action_buttons("similar_main_action")


# =============================================================================
# MAIN APPLICATION LOGIC
# =============================================================================

def initialize_session_state():
    """Initialize session state with default values."""
    # Initialize defaults if not present
    defaults = {
        "user_query": "",
        "gemini_output": None,
        "selected_sample": "",
        "random_queries": [],
        "screen": "initial",
        "current_molecule_data": None,
        "similar_search_executed": False,
        "detail_analysis_executed": False,
        "cached_analysis_result": "",
        # New fields for enhanced data structure
        "detailed_info": None,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Validate and fix state consistency
    validate_session_state()

def validate_session_state():
    """Validate session state consistency and fix any inconsistencies."""
    # If screen is initial but we have data, clear it
    if st.session_state.screen == "initial":
        if st.session_state.user_query:
            st.session_state.user_query = ""
        if st.session_state.gemini_output:
            st.session_state.gemini_output = None
        if st.session_state.current_molecule_data:
            st.session_state.current_molecule_data = None

# Initialize session state
initialize_session_state()

# Create sidebar with logo and promotion message only
with st.sidebar:
    st.logo("images/logo.png", size="large")
    
    # Promotion message
    st.write(ANNOUNCEMENT_MESSAGE)

# =============================================================================
# CONVERSATION FLOW IMPLEMENTATION
# =============================================================================

def display_molecule_3d(molecule_data: Dict) -> bool:
    """Display 3D molecule structure using XYZ coordinate data."""
    try:
        # Use saved XYZ data from PubChem
        xyz_string = molecule_data.get("xyz_data")
        
        if xyz_string:
            # Create 3D molecular viewer with explicit clearing
            viewer = py3Dmol.view(width=MOLECULE_VIEWER_WIDTH, height=MOLECULE_VIEWER_HEIGHT)
            
            # Clear any existing models first
            viewer.clear()
            
            # Add the new model
            viewer.addModel(xyz_string, 'xyz')
            viewer.setStyle({'stick': {}})  # Stick representation
            viewer.setZoomLimits(Config.VIEWER['zoom_min'], Config.VIEWER['zoom_max'])  # Set zoom limits
            viewer.zoomTo()  # Auto-fit molecule
            viewer.spin('y', Config.VIEWER['rotation_speed'])  # Auto-rotate around Y-axis
            
            # Use components.html to display the viewer
            components.html(viewer._make_html(), height=MOLECULE_VIEWER_HEIGHT)
            return True
        else:
            st.error("立体構造データが見つかりませんでした。")
            return False
    except Exception as e:
        st.error(f"立体構造の準備中にエラーが発生しました: {e}")
        return False

# Handle sample selection transition
if st.session_state.selected_sample:
    st.session_state.user_query = st.session_state.selected_sample
    st.session_state.selected_sample = ""  # Reset selection to prevent reuse
    st.session_state.screen = "query_response"
    logger.info(f"Transitioning to query_response screen with query: {st.session_state.user_query}")
    st.rerun()

# Main conversation flow - using simplified screen structure
current_screen = st.session_state.get("screen", "initial")
if current_screen == "initial":
    show_initial_screen()
elif current_screen == "query_response":
    show_query_response_screen()
elif current_screen == "detail_response":
    show_detail_response_screen()
elif current_screen == "similar_response":
    show_similar_response_screen()

