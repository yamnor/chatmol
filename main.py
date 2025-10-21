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

# Import shared models from tools
from tools.utils.shared_models import DetailedMoleculeInfo

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

class Config:
    """Application configuration constants."""
    
    # Timeout settings for preventing freezes
    TIMEOUTS = {
        'api': 10,  # Gemini API timeout
        'pubchem_3d': 10,  # PubChem 3D record fetch timeout
    }
    
    # Random sample configuration
    RANDOM_QUERY = {
        'count': 30,  # Number of random samples to display
        'columns': 2,  # Number of columns for random samples
    }
    
    # Cache configuration
    CACHE = {
        'enabled': False,  # Enable/disable cache functionality (can be overridden by secrets.toml)
        'base_directory': 'cache',  # Base cache directory name
        'max_size_mb': 100,  # Maximum cache size in MB
        'max_age_days': 360,  # Maximum age of cache entries in days
        'data_sources': {
            'pubchem': {
                'enabled': True,
                'directory': 'pubchem',
                'max_age_days': 36500,
            },
            'queries': {
                'enabled': True,
                'directory': 'queries',
                'max_age_days': 36500,
                'max_items_per_file': 25,
            },
            'descriptions': {
                'enabled': True,
                'directory': 'descriptions',
                'max_age_days': 36500,
                'max_items_per_file': 25,
            },
            'analysis': {
                'enabled': True,
                'directory': 'analysis',
                'max_age_days': 180,
                'max_items_per_file': 25,
            },
            'similar': {
                'enabled': True,
                'directory': 'similar',
                'max_age_days': 180,  #
                'max_items_per_file': 50,
                'max_items_per_data': 25,
            },
            'failed_molecules': {
                'enabled': True,
                'directory': 'failed_molecules',
                'max_age_days': 365,
                'max_items_per_file': 1000,
            },
        }
    }
        
    # 3D Molecular Viewer Configuration
    # Responsive viewer size based on window size
    VIEWER = {
        'width_pc': 700,
        'height_pc': 550,
        'width_mobile': 340,
        'height_mobile': 200,
        'zoom_min': 0.1,
        'zoom_max': 50,
        'rotation_speed': 1,
    }
    
    # Default AI Model Configuration
    DEFAULT_MODEL_NAME = "gemini-2.5-flash-lite"
    
    # Error messages - simplified to essential ones only
    ERROR_MESSAGES = {
        # API related errors
        'api_error': "API接続エラーが発生しました。しばらく待ってから再試行してください。",
        'timeout': "操作がタイムアウトしました。",
        
        # Data retrieval errors
        'molecule_not_found': "分子データが見つかりませんでした。",
        'invalid_data': "無効なデータが返されました。",
        
        # Molecular processing errors
        'processing_error': "分子データの処理中にエラーが発生しました。",
        
        # General errors
        'parse_error': "データの解析に失敗しました。",
        'display_error': "表示中にエラーが発生しました。",
        'no_data': "データが見つかりません。最初からやり直してください。",
        'general_error': "予期しないエラーが発生しました。",
    }

# Announcement Configuration
ANNOUNCEMENT_MESSAGE: str = """
[![サイエンスアゴラ2025](https://i.gyazo.com/208ecdf2f06260f4d90d58ae291f0104.png)](https://yamlab.jp/sciago2025)

10/25, 26 の サイエンスアゴラ で、分子を作る / 動かす/ 感じる体験 & 展示を出展。詳細は **[【こちら】](https://yamlab.jp/sciago2025)**
"""

MENU_ITEMS_ABOUT: str = '''
**ChatMOL** was created by [yamnor](https://yamnor.me),
a chemist 🧪 specializing in molecular simulation 🖥️ living in Japan 🇯🇵.

If you have any questions, thoughts, or comments,
feel free to [contact me](https://letterbird.co/yamnor) ✉️
or find me on [X (Twitter)](https://x.com/yamnor) 🐦.

GitHub: [yamnor/chatmol](https://github.com/yamnor/chatmol)
'''

# Import shared prompts from tools
from tools.utils.shared_prompts import AIPrompts

# Import sample queries from config
from config.sample_queries import SAMPLE_QUERIES

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

# ErrorHandler is now imported from tools.utils.error_handler
# Add Streamlit-specific error display method
def show_error(message: str) -> None:
    """Show error message using Streamlit."""
    st.error(f"⚠️ {message}")


def show_action_buttons(key_prefix: str = "action") -> None:
    """Show standardized action button set: 詳しく知りたい, 関連する分子は？, 他の分子を探す."""
    col1, col2, col3 = st.columns(3)
    
    # Check data availability
    current_data = st.session_state.get("current_molecule_data", None)
    has_cid = current_data and current_data.get("cid") is not None
    has_name = current_data and current_data.get("name") is not None
    
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
        if st.button("関連する分子は？", key=f"{key_prefix}_similar", use_container_width=True, icon="🔍", disabled=not has_name):
            if has_name:
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

def execute_with_timeout(func, timeout_seconds: int, error_type: str = "timeout"):
    """Execute a function with timeout control using ThreadPoolExecutor."""
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            return future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        show_error(ErrorHandler.handle_error(Exception("Timeout"), error_type))
        return None
    except Exception as e:
        show_error(ErrorHandler.handle_error(e, "general_error"))
        return None

def generate_random_queries() -> List[Dict[str, str]]:
    """Generate random samples from all available queries."""
    if SAMPLE_QUERIES:
        return random.sample(SAMPLE_QUERIES, min(Config.RANDOM_QUERY['count'], len(SAMPLE_QUERIES)))
    else:
        return []


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

import os
import hashlib
from datetime import datetime, timedelta

# Import unified cache utilities
from tools.utils.cache_utils import normalize_compound_name, NameMappingCacheManager, BaseCacheManager
from tools.utils.updated_cache_managers import (
    PubChemCacheManager, QueryCacheManager, DescriptionCacheManager,
    SimilarMoleculesCacheManager, AnalysisCacheManager, FailedMoleculesCacheManager
)
from tools.utils.unified_cache_manager import UnifiedCacheManager

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
        
        # Migrate old data structure if needed
        if "selections" in data and "sessions" not in data:
            data = migrate_old_analytics_data(data)
        
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

def migrate_old_analytics_data(data: Dict) -> Dict:
    """Migrate old analytics data structure to new session-based structure."""
    try:
        sessions = []
        
        # Convert old selections to sessions
        if "selections" in data:
            for query_text, timestamps in data["selections"].items():
                for i, timestamp in enumerate(timestamps):
                    session_id = str(uuid.uuid4())
                    sessions.append({
                        "session_id": session_id,
                        "initial_query": query_text,
                        "initial_timestamp": timestamp,
                        "actions": [
                            {
                                "action_type": "initial_query",
                                "timestamp": timestamp
                            }
                        ]
                    })
        
        # Create new data structure
        new_data = {"sessions": sessions}
        
        # Backup old data
        backup_file = os.path.join(Config.CACHE['base_directory'], 'analytics', 'query_log_backup.json')
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Migrated old analytics data. Backup saved to {backup_file}")
        return new_data
        
    except Exception as e:
        logger.error(f"Error migrating analytics data: {e}")
        return {"sessions": []}

def save_query_selection(query_text: str):
    """Save query selection to analytics log using new session-based structure."""
    # Log the initial query as an action
    log_user_action("initial_query")

# =============================================================================
# AI AND MOLECULAR PROCESSING FUNCTIONS
# =============================================================================

def call_gemini_api(prompt: str, use_google_search: bool = True) -> Optional[str]:
    """Common function to call Gemini API with configurable options."""
    logger.info(f"Calling Gemini API with prompt length: {len(prompt)}")
    
    def api_call():
        """Execute API call with optional Google Search tool."""
        config = types.GenerateContentConfig()
        
        # Add Google Search tool if requested
        if use_google_search:
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
        
    with st.spinner(f"AI (`{model_name}`) に問い合わせ中...", show_time=True):
        response = execute_with_timeout(
            api_call, 
            Config.TIMEOUTS['api'], 
            "api_error"
        )
    
    if response is None:
        logger.warning("No response received from Gemini API")
        return None
    
    try:
        logger.info("Successfully received response from Gemini API")
        return response.text
    except Exception as e:
        show_error(ErrorHandler.handle_error(e, "api_error"))
        return None

def search_molecule_by_query(user_input_text: str) -> Optional[str]:
    """Search and recommend molecules based on user query."""
    logger.info(f"Processing user query: {user_input_text[:50]}...")
    
    prompt = AIPrompts.MOLECULAR_SEARCH.format(user_input=user_input_text)
    
    response_text = call_gemini_api(
        prompt=prompt,
        use_google_search=True
    )
    
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

def find_similar_molecules_with_cache(molecule_name: str) -> Optional[str]:
    """Find similar molecules using common function with cache support."""
    logger.info(f"Searching for similar molecules to: {molecule_name}")
    
    # Get English name from current data for cache key
    current_data = st.session_state.get("current_molecule_data", None)
    
    # Use common function for similar molecule search
    return find_similar_molecules(molecule_name, call_gemini_api, cache_manager, current_data)

# Import PubChem client functions from tools
from tools.utils.pubchem_client import (
    get_compounds_by_name,
    get_3d_coordinates_by_cid,
    convert_pubchem_to_xyz,
    execute_with_timeout,
    get_comprehensive_molecule_data
)

def get_comprehensive_molecule_data_with_cache(english_name: str) -> Tuple[bool, Optional[DetailedMoleculeInfo], Optional[int], Optional[str]]:
    """Get comprehensive molecule data from PubChem using English name with cache support."""
    logger.info(f"Getting comprehensive data for: {english_name}")
    
    # Check cache first
    cached_data = cache_manager.pubchem.get_cached_molecule_data(english_name)
    if cached_data:
        detailed_info, cid = cached_data
        logger.info(f"Using cached data for: {english_name}")
        return True, detailed_info, cid, None
    
    with st.spinner("分子データを取得中...", show_time=True):
        # Use the shared function from tools
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

def analyze_molecule_properties_with_cache(detailed_info: DetailedMoleculeInfo, molecule_name: str) -> Optional[str]:
    """Analyze molecular properties using common function with cache support."""
    logger.info(f"Getting Gemini analysis for molecule: {molecule_name}")
    
    # Check cache first using English name
    current_data = st.session_state.get("current_molecule_data", None)
    english_name = current_data.get("name_en") if current_data else None
    
    if english_name:
        cached_analyses = cache_manager.analysis.get_analysis_results(english_name)
        if cached_analyses:
            logger.info(f"Using cached analysis for: {english_name}")
            # Return the most recent analysis
            return cached_analyses[-1]['description']
    
    # Use common function for analysis
    return analyze_molecule_properties(detailed_info, molecule_name, call_gemini_api, cache_manager)

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

# Import error handling utilities from tools
from tools.utils.error_handler import (
    ErrorHandler,
    is_no_result_response,
    parse_json_response
)

# Import common molecular analysis functions
from tools.utils.molecular_analysis import (
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

def parse_gemini_response(response_text: str, save_to_query_cache: bool = True) -> Dict[str, Union[str, None, Any]]:
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
    return current_data and current_data.get("cid") is not None

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
            return Config.ERROR_MESSAGES['display_error']
    except Exception as e:
        return f"{Config.ERROR_MESSAGES['display_error']}: {e}"

def find_and_process_similar_molecule() -> Optional[Dict]:
    """Find and process similar molecule data."""
    try:
        similar_response = find_similar_molecules_with_cache(get_molecule_name())
        if similar_response:
            # 関連分子処理時はqueriesキャッシュに保存しない
            return parse_gemini_response(similar_response, save_to_query_cache=False)
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
        
        if similar_data and similar_data.get("xyz_data"):
            st.session_state.current_molecule_data = similar_data
        else:
            error_message = Config.ERROR_MESSAGES['molecule_not_found']
            handle_error_and_show_buttons(error_message, "similar_error_none")
            return
        
        st.session_state.similar_search_executed = True
    
    # Display current molecule data
    current_data = st.session_state.get("current_molecule_data", None)
    if current_data and current_data.get("xyz_data"):
        message = f"あなたにオススメする分子は「 **[{current_data['name']}](https://pubchem.ncbi.nlm.nih.gov/compound/{current_data['cid']})** 」だよ。{current_data['memo']}"
        with st.chat_message("assistant"):
            st.write(message)
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

