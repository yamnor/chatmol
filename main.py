# Standard library imports
import random
import json
import re
import logging
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass

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

@dataclass
class DetailedMoleculeInfo:
    """Detailed molecule information from PubChem."""
    molecular_formula: Optional[str]
    molecular_weight: Optional[float]
    iupac_name: Optional[str]
    synonyms: List[str]
    inchi: Optional[str]
    # Chemical properties
    xlogp: Optional[float]  # LogP (calculated)
    tpsa: Optional[float]  # Topological polar surface area
    complexity: Optional[float]  # Molecular complexity
    rotatable_bond_count: Optional[int]  # Number of rotatable bonds
    heavy_atom_count: Optional[int]  # Number of heavy atoms
    hbond_donor_count: Optional[int]  # Number of H-bond donors
    hbond_acceptor_count: Optional[int]  # Number of H-bond acceptors
    charge: Optional[int]  # Total charge
    xyz_data: Optional[str]  # XYZ coordinate data for 3D visualization

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
        'enabled': True,  # Enable/disable cache functionality (can be overridden by secrets.toml)
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
                'max_age_days': 36500,  # Longer retention for query statistics
                'max_items_per_file': 25,  # Maximum number of query compounds to keep per query file
            },
            'descriptions': {
                'enabled': True,
                'directory': 'descriptions',
                'max_age_days': 36500,  # Medium retention for descriptions
                'max_items_per_file': 25,  # Maximum number of items to keep per cache file
            },
            'analysis': {
                'enabled': True,
                'directory': 'analysis',
                'max_age_days': 180,  # Medium retention for AI-generated analysis
                'max_items_per_file': 25,  # Maximum number of analysis results to keep per cache file
            },
            'similar': {
                'enabled': True,
                'directory': 'similar',
                'max_age_days': 180,  # Medium retention for AI-generated similar molecules
                'max_items_per_file': 50,  # Maximum number of similar molecules to keep per cache file
                'max_items_per_data': 25,  # Maximum number of descriptions per molecule
            },
            'failed_molecules': {
                'enabled': True,
                'directory': 'failed_molecules',
                'max_age_days': 365,  # Long retention for failed molecules
                'max_items_per_file': 1000,  # Maximum number of failed molecules to keep
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

# AI Prompts Configuration
class AIPrompts:
    """AI prompts for different molecular operations."""
    
    # Molecular search prompt
    MOLECULAR_SEARCH: str = """
# SYSTEM
あなたは「分子コンシェルジュ」です。
ユーザーが求める効能・イメージ・用途・ニーズなどを説明してもらったら、
(1) それに最も関連する・関係がありそうだ・適していると考えられる候補の分子を「1 個」のみ、
(2) その分子の日本語での名称（name_jp）、英語での名称（name_en）、説明（description）を、
(3) 以下のルールに厳密に従い、JSON形式でのみ出力してください：

- **重要**: 英語名（name_en）は、PubChemで検索できる具体的な分子名を選んでください。一般的な分類名（例：「脂肪酸塩」「アルカロイド」）ではなく、具体的な化合物名（例：「ステアリン酸ナトリウム」「カフェイン」）を選んでください
- もし全く思いつかないときには、科学的な根拠のない「こじつけ」「無理矢理」「適当」でもオーケーですが、その場合は必ず、その旨を「こじつけ」「無理矢理」「適当」と明記してください
- 該当する分子を思いつかなかった場合は、name_jpに「該当なし」とのみ出力します
- 説明は、小学生にもわかるように、「その分子を選んだ理由」と「その分子の性質や特徴」を「２文」で、絵文字も用いてフレンドリーに表現してください
- できれば、その分子に関連して、小学生が笑ってしまうような、ギャグ・面白い一言も追加してください

# USER
{user_input}

```json
{{
  "name_jp": "<分子名>（見つかった分子の日本語での名称）",
  "name_en": "<分子名>（見つかった分子の英語での名称）",
  "description": "<説明> （その分子を選んだ理由とその分子の性質や特徴を２文で説明）"
}}
```
"""

    # Similar molecule search prompt
    SIMILAR_MOLECULE_SEARCH: str = """
# SYSTEM
あなたは「分子コンシェルジュ」です。
ユーザーが指定した分子「{molecule_name}」について、
(1) それに最も関連する・関係がありそうだ・適していると考えられる候補の分子を「1 個」のみ、
(2) その分子の日本語での名称（name_jp）、英語での名称（name_en）、説明（description）を、
(3) 以下のルールに厳密に従い、JSON形式でのみ出力してください：

- **重要**: 英語名（name_en）は、PubChemで検索できる具体的な分子名を選んでください。一般的な分類名（例：「脂肪酸塩」「アルカロイド」）ではなく、具体的な化合物名（例：「ステアリン酸ナトリウム」「カフェイン」）を選んでください
- **重要**: 必ず指定された分子とは異なる分子を提案してください
- 関連性については、下記の関連性の観点から、関連性の高い分子を選んでください
    1. **構造的類似性**: 同じ官能基、骨格構造、分子サイズ
    2. **機能的類似性**: 同じ作用機序、生体活性、薬理効果
    3. **用途的類似性**: 同じ分野での利用、同じ目的での使用
    4. **化学的類似性**: 同じ化学反応性、物理化学的性質
    5. **生物学的類似性**: 同じ代謝経路、同じ受容体への結合
    6. **歴史的関連性**: 同じ発見者、同じ研究グループ、同じ時代
    7. **対照的関連性**: 相反する作用、拮抗作用、補完的効果
    8. **進化的関連性**: 同じ生物種由来、同じ進化系統
- もし全く思いつかないときには、科学的な根拠のない「こじつけ」「無理矢理」「適当」でもオーケーですが、その場合は必ず、その旨を「こじつけ」「無理矢理」「適当」と明記してください
- 該当する分子を思いつかなかった場合は、name_jpに「該当なし」とのみ出力します
- 説明は、小学生にもわかるように、「どの観点で関連しているか」と「その分子を選んだ理由や性質の特徴」を２行で、絵文字も用いてフレンドリーに表現してください
- できれば、その分子に関連して、小学生が笑ってしまうような、ギャグ・面白い一言も追加してください

```json
{{
  "name_jp": "<分子名>（見つかった分子の日本語での名称）",
  "name_en": "<分子名>（見つかった分子の英語での名称）",
  "description": "<説明> （どの観点で関連しているかと、その分子を選んだ理由や性質の特徴を２文で説明）"
}}
```
"""

    # Molecular analysis prompt
    MOLECULAR_ANALYSIS: str = """
# SYSTEM
あなたは「分子コンシェルジュ」です。
以下の分子「{molecule_name}」の化学的性質データを基に、この分子の特徴・性質・用途・効果などを分析してください。

# 化学的性質データ
{properties_str}

# 分析指示
上記の化学的性質データから、ケモインフォマティクスの観点で以下のように分析してください：

1. **物理化学的性質**: LogP、TPSA、分子量などから推測される溶解性、膜透過性、薬物動態
2. **構造的特徴**: 分子複雑度、回転可能結合数から推測される立体構造の柔軟性、受容体選択性
3. **水素結合特性**: 水素結合供与体数と水素結合受容体数から推測される分子間相互作用、溶解性、膜透過性、分子標的への結合への影響
4. **分子メカニズム**: 上記の性質から推測される生体内での作用メカニズムや分子標的への結合様式

# 出力形式
- ケモインフォマティクスの観点から科学的に分析してください
- 分子データの具体的な数値を示しながら、3-5文程度の簡潔な説明にまとめてください
- 「〜があるよ」「〜だよ」「〜だよね」など、親しみやすい口調で説明してください
- 溶解性、膜透過性、薬物動態などの分子メカニズムを、分かりやすい比喩や表現で説明してください
- 推測であることを明記してください（「〜と考えられるよ」「〜の可能性があるよ」など）
- 分子量、重原子数、LogP、TPSA、分子複雑度、水素結合供与体数、水素結合受容体数、回転可能結合数などの文字は **太字** で表示してください
- 数値は、必ず、`数値` の形式で表示してください
- 絵文字も使って、親しみやすい口調で説明してください
- heading は使わないでください
- 分析結果のみを出力してください。他の説明や補足は不要です

# 出力例
**カフェイン** は **分子量** `194.19` の小さな分子で、**LogP** が `-0.07` と水に溶けやすい性質があるよ。
**TPSA** が `58.4` と比較的高いから、体内での吸収が良くて、脳に届きやすいんだよね。
**分子複雑度** が `62.3` と中程度で、**回転可能結合** が `0` 個だから構造がしっかりしていて、特定の受容体にピンポイントで結合できるんだよね。
**水素結合供与体数** が `0` 個で、**水素結合受容体数** が `3` 個だから、水素結合による分子間相互作用が弱く、水に溶けやすい性質があるんだよね。
この分子は、体内での吸収が良くて、脳に届きやすいんだよね。
"""

# Sample queries organized by category for readability
SAMPLE_QUERIES: List[Dict[str, str]] = [
    # 🌸 香り
    {"icon": "🌸", "text": "良い香りのする成分は？"},
    {"icon": "🍯", "text": "甘い香りのする成分は？"},
    {"icon": "🌿", "text": "フレッシュな香りが欲しい"},
    {"icon": "🕯️", "text": "落ち着く香りを探している"},
    {"icon": "🌶️", "text": "スパイシーな香りが欲しい"},
    
    # 🍋 食べ物・飲み物
    {"icon": "🍋", "text": "レモンの成分は？"},
    {"icon": "🍦", "text": "バニラの成分は？"},
    {"icon": "☕", "text": "コーヒーの成分は？"},
    {"icon": "🍫", "text": "チョコレートの成分は？"},
    {"icon": "🌿", "text": "ミントの成分は？"},
    
    # 🌸 花・植物
    {"icon": "🌹", "text": "バラの香り成分は？"},
    {"icon": "🌸", "text": "桜の香り成分は？"},
    {"icon": "💜", "text": "ラベンダーの香り成分は？"},
    {"icon": "🌼", "text": "ジャスミンの香り成分は？"},
    {"icon": "🌺", "text": "金木犀の香り成分は？"},
    
    # 🎨 色・染料
    {"icon": "🍎", "text": "リンゴの赤色の成分は？"},
    {"icon": "🫐", "text": "ベリーの青色の成分は？"},
    {"icon": "🍋", "text": "レモンの黄色の成分は？"},
    {"icon": "🍇", "text": "ぶどうの紫色の成分は？"},
    {"icon": "👖", "text": "デニムの青色の成分は？"},
    
    # 👅 味覚
    {"icon": "🍯", "text": "甘い味の成分は？"},
    {"icon": "🍋", "text": "酸っぱい味の成分は？"},
    {"icon": "☕", "text": "苦い味の成分は？"},
    {"icon": "🌶️", "text": "辛い味の成分は？"},
    {"icon": "🍄", "text": "うま味の成分は？"},
    
    # 💊 医薬品
    {"icon": "🤧", "text": "風邪薬の成分は？"},
    {"icon": "🤕", "text": "頭痛薬の成分を教えて"},
    {"icon": "🤢", "text": "胃薬の成分は？"},
    {"icon": "🦠", "text": "インフル治療薬の成分は？"},
    {"icon": "💉", "text": "抗生物質の成分は？"},
    
    # 🌲 自然・環境
    {"icon": "🌲", "text": "森の香り成分は？"},
    {"icon": "🌊", "text": "海の香り成分は？"},
    {"icon": "🌱", "text": "土の匂い成分は？"},
    {"icon": "🌳", "text": "木の香り成分は？"},
    {"icon": "🌿", "text": "草の香り成分は？"},
    
    # 🧴 日用品
    {"icon": "🧽", "text": "洗剤の成分は？"},
    {"icon": "🧴", "text": "シャンプーの成分は？"},
    {"icon": "🧼", "text": "石鹸の成分は？"},
    {"icon": "👕", "text": "柔軟剤の成分は？"},
    
    # 💪 スポーツ・運動
    {"icon": "💪", "text": "筋肉を鍛えたい"},
    {"icon": "🔄", "text": "疲労を回復させたい"},
    {"icon": "🏃", "text": "持久力をアップさせたい"},
    {"icon": "⚡", "text": "瞬発力をアップさせたい"},
    {"icon": "⚡", "text": "エネルギーを補給したい"},
    
    # 💚 健康・体調
    {"icon": "😊", "text": "気分をすっきりさせたい"},
    {"icon": "😴", "text": "疲れを取りたい"},
    {"icon": "🌅", "text": "目覚めを良くしたい"},
    {"icon": "🛡️", "text": "免疫力を高めたい"},
    {"icon": "❤️", "text": "血行を良くしたい"},
    
    # 😴 リラックス・睡眠
    {"icon": "🧘", "text": "リラックスしたい"},
    {"icon": "🕊️", "text": "心を落ち着かせたい"},
    {"icon": "😌", "text": "ゆっくり休みたい"},
    {"icon": "🌙", "text": "ストレスを和らげたい"},
    {"icon": "😊", "text": "幸福感を感じたい"},
    
    # 🧠 集中・学習
    {"icon": "🎯", "text": "集中力を高めたい"},
    {"icon": "📚", "text": "勉強に集中したい"},
    {"icon": "💡", "text": "思考力を高めたい"},
    {"icon": "🧠", "text": "脳を活性化したい"},
    
    # ✨ 美容・スキンケア
    {"icon": "✨", "text": "肌を美しく保ちたい"},
    {"icon": "🌟", "text": "若々しさを維持したい"},
    {"icon": "💇", "text": "髪の毛を健康にしたい"},
    {"icon": "🛡️", "text": "シミを防ぎたい"},
    {"icon": "💧", "text": "肌の潤いを保ちたい"},

    {"icon": "🪲", "text": "ホタルが光るのはなぜ？"},
]

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

class ErrorHandler:
    """Simplified error handling for the application."""
    
    @staticmethod
    def handle_error(e: Exception, error_type: str = "general_error") -> str:
        """Handle all types of errors with simplified messaging."""
        logger.error(f"Error ({error_type}): {str(e)}")
        
        # Map error types to appropriate messages
        error_messages = {
            'api_error': Config.ERROR_MESSAGES['api_error'],
            'timeout': Config.ERROR_MESSAGES['timeout'],
            'molecule_not_found': Config.ERROR_MESSAGES['molecule_not_found'],
            'invalid_data': Config.ERROR_MESSAGES['invalid_data'],
            'processing_error': Config.ERROR_MESSAGES['processing_error'],
            'parse_error': Config.ERROR_MESSAGES['parse_error'],
            'display_error': Config.ERROR_MESSAGES['display_error'],
            'no_data': Config.ERROR_MESSAGES['no_data'],
            'general_error': Config.ERROR_MESSAGES['general_error'],
        }
        
        return error_messages.get(error_type, Config.ERROR_MESSAGES['general_error'])
    
    @staticmethod
    def show_error(message: str) -> None:
        """Show error message."""
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
        ErrorHandler.show_error(ErrorHandler.handle_error(Exception("Timeout"), error_type))
        return None
    except Exception as e:
        ErrorHandler.show_error(ErrorHandler.handle_error(e, "general_error"))
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

class BaseCacheManager:
    """Base cache manager with common functionality."""
    
    def __init__(self, data_source: str, config: Dict):
        """Initialize base cache manager."""
        self.data_source_name = data_source
        self.data_source_config = config
        self.cache_base_directory = Config.CACHE['base_directory']
        self.cache_expiration_days = config.get('max_age_days', Config.CACHE['max_age_days'])
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
    
    def _normalize_cache_key(self, key: str) -> str:
        """Normalize key for cache filename."""
        # Convert to lowercase, strip whitespace, remove common suffixes
        normalized = key.lower().strip()
        normalized = normalized.replace(" acid", "").replace(" salt", "")
        # Create a safe filename by replacing special characters
        safe_key = re.sub(r'[^\w\-_]', '_', normalized)
        return safe_key
    
    def _get_source_cache_file_path(self, cache_key: str) -> Optional[str]:
        """Get cache file path for given key."""
        cache_dir = self._get_source_cache_directory()
        if not cache_dir:
            return None
        return os.path.join(cache_dir, f"{cache_key}.json")
    
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
        """
        Apply item limit to a list of items, keeping only the latest entries.
        
        This is a general-purpose method that can be used by any cache manager
        to limit the number of items stored in cache files. The limit is
        configured via 'max_items_per_file' in the data source configuration.
        
        Args:
            items: List of items to limit
            item_key: Key name for logging purposes (optional)
            
        Returns:
            Limited list of items (latest entries only)
        """
        max_items = self.data_source_config.get('max_items_per_file')
        if max_items is None or len(items) <= max_items:
            return items
        
        # Keep only the latest max_items entries
        limited_items = items[-max_items:]
        logger.info(f"Trimmed {self.data_source_name} cache to latest {max_items} entries (was {len(items)})")
        return limited_items
    
    def _get_item_timestamp(self, item: Dict) -> str:
        """Extract timestamp from an item for sorting purposes."""
        # Try different common timestamp field names
        for timestamp_field in ['timestamp', 'created_at', 'updated_at', 'date']:
            if timestamp_field in item:
                return item[timestamp_field]
        
        # If no timestamp found, return current time
        return datetime.now().isoformat()


class PubChemCacheManager(BaseCacheManager):
    """Manages PubChem-specific cache operations."""
    
    def __init__(self):
        """Initialize PubChem cache manager."""
        super().__init__('pubchem', Config.CACHE['data_sources']['pubchem'])
    
    def get_cached_molecule_data(self, compound_name: str) -> Optional[Tuple[Optional[DetailedMoleculeInfo], Optional[int]]]:
        """Get cached molecule data for compound."""
        if not Config.CACHE['enabled']:
            return None
        
        cache_key = self._normalize_cache_key(compound_name)
        cache_file_path = self._get_source_cache_file_path(cache_key)
        
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
    
    def save_cached_molecule_data(self, compound_name: str, detailed_info: DetailedMoleculeInfo, cid: int):
        """Save molecule data to cache (only if xyz_data is available)."""
        if not Config.CACHE['enabled']:
            return
        
        # Only save if xyz_data is available
        if not detailed_info.xyz_data:
            logger.warning(f"Skipping cache save for {compound_name}: No xyz_data available")
            return
        
        cache_key = self._normalize_cache_key(compound_name)
        cache_file_path = self._get_source_cache_file_path(cache_key)
        
        if not cache_file_path:
            logger.warning(f"Cannot save cache for {compound_name}: data source disabled")
            return
        
        try:
            # Convert DetailedMoleculeInfo to dictionary
            cache_data = {
                'compound_name': compound_name,
                'cache_key': cache_key,
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
    """Manages query-compound mapping cache operations."""
    
    def __init__(self):
        """Initialize query cache manager."""
        super().__init__('queries', Config.CACHE['data_sources']['queries'])
    
    def save_query_compound_mapping(self, query_text: str, compounds: List[Dict], increment_count: bool = False):
        """Save query-compound mapping to cache."""
        if not Config.CACHE['enabled']:
            return
        
        cache_key = self._normalize_cache_key(query_text)
        cache_file_path = self._get_source_cache_file_path(cache_key)
        
        if not cache_file_path:
            logger.warning(f"Cannot save query cache for {query_text}: data source disabled")
            return
        
        # Read existing data if available
        existing_data = self._read_cache_file(cache_file_path) if os.path.exists(cache_file_path) else None
        
        if existing_data:
            # Update existing data
            existing_compounds = existing_data.get('compounds', [])
            existing_compounds_names = [c.get('compound_name', '') for c in existing_compounds]
            
            # Add new compounds (avoid duplicates)
            for compound in compounds:
                if compound.get('compound_name', '') not in existing_compounds_names:
                    existing_compounds.append(compound)
            
            # Apply general item limit using the base class method
            existing_compounds = self._apply_item_limit(existing_compounds)
            
            cache_data = {
                'query_text': query_text,
                'compounds': existing_compounds,
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Create new data
            cache_data = {
                'query_text': query_text,
                'compounds': compounds,
                'timestamp': datetime.now().isoformat()
            }
        
        self._create_cache_file(cache_file_path, cache_data)
        logger.info(f"Cached query-compound mapping for: {query_text}")
    
    def get_query_compound_mapping(self, query_text: str) -> Optional[Dict]:
        """Get query-compound mapping from cache."""
        if not Config.CACHE['enabled']:
            return None
        
        cache_key = self._normalize_cache_key(query_text)
        cache_file_path = self._get_source_cache_file_path(cache_key)
        
        if not cache_file_path or not self._validate_cache_file(cache_file_path):
            return None
        
        return self._read_cache_file(cache_file_path)
    
    def get_random_compound_from_query(self, query_text: str) -> Optional[str]:
        """Get a random compound name from cached query results."""
        if not Config.CACHE['enabled']:
            return None
        
        cache_key = self._normalize_cache_key(query_text)
        cache_file_path = self._get_source_cache_file_path(cache_key)
        
        if not cache_file_path or not self._validate_cache_file(cache_file_path):
            return None
        
        cache_data = self._read_cache_file(cache_file_path)
        if not cache_data:
            return None
        
        compounds = cache_data.get('compounds', [])
        if not compounds:
            return None
        
        # Select a random compound
        random_compound = random.choice(compounds)
        return random_compound.get('compound_name')
    
    def get_any_random_compound_from_queries(self) -> Optional[str]:
        """Get a random compound name from any cached query file."""
        if not Config.CACHE['enabled']:
            return None
        
        cache_dir = self._get_source_cache_directory()
        if not cache_dir or not os.path.exists(cache_dir):
            return None
        
        # Get all cache files
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
        if not cache_files:
            return None
        
        # Try each cache file until we find one with compounds
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
    """Manages compound-description mapping cache operations."""
    
    def __init__(self):
        """Initialize description cache manager."""
        super().__init__('descriptions', Config.CACHE['data_sources']['descriptions'])
    
    def save_compound_description(self, compound_name: str, description: str):
        """Save compound description to cache with automatic item limit."""
        if not Config.CACHE['enabled']:
            return
        
        cache_key = self._normalize_cache_key(compound_name)
        cache_file_path = self._get_source_cache_file_path(cache_key)
        
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
                'compound_name': compound_name,
                'descriptions': descriptions,
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Create new data
            cache_data = {
                'compound_name': compound_name,
                'descriptions': [{
                    'description': description,
                    'timestamp': datetime.now().isoformat()
                }],
                'timestamp': datetime.now().isoformat()
            }
        
        self._create_cache_file(cache_file_path, cache_data)
        logger.info(f"Cached description for compound: {compound_name} (total descriptions: {len(cache_data['descriptions'])})")
    
    def get_compound_descriptions(self, compound_name: str) -> List[Dict]:
        """Get compound descriptions from cache."""
        if not Config.CACHE['enabled']:
            return []
        
        cache_key = self._normalize_cache_key(compound_name)
        cache_file_path = self._get_source_cache_file_path(cache_key)
        
        if not cache_file_path or not self._validate_cache_file(cache_file_path):
            return []
        
        cache_data = self._read_cache_file(cache_file_path)
        if cache_data:
            return cache_data.get('descriptions', [])
        
        return []
    
    def get_random_description(self, compound_name: str) -> Optional[str]:
        """Get a random description from cached descriptions."""
        if not Config.CACHE['enabled']:
            return None
        
        cache_key = self._normalize_cache_key(compound_name)
        cache_file_path = self._get_source_cache_file_path(cache_key)
        
        if not cache_file_path or not self._validate_cache_file(cache_file_path):
            return None
        
        cache_data = self._read_cache_file(cache_file_path)
        if not cache_data:
            return None
        
        descriptions = cache_data.get('descriptions', [])
        if not descriptions:
            return None
        
        # Select a random description
        random_description = random.choice(descriptions)
        return random_description.get('description')


class AnalysisCacheManager(BaseCacheManager):
    """Manages detailed analysis cache operations."""
    
    def __init__(self):
        """Initialize analysis cache manager."""
        super().__init__('analysis', Config.CACHE['data_sources']['analysis'])
    
    def save_analysis_result(self, compound_name: str, analysis_text: str):
        """Save detailed analysis result to cache."""
        if not Config.CACHE['enabled']:
            return
        
        cache_key = self._normalize_cache_key(compound_name)
        cache_file_path = self._get_source_cache_file_path(cache_key)
        
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
                'compound_name': compound_name,
                'descriptions': descriptions,
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Create new data
            cache_data = {
                'compound_name': compound_name,
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
        if not Config.CACHE['enabled']:
            return []
        
        cache_key = self._normalize_cache_key(compound_name)
        cache_file_path = self._get_source_cache_file_path(cache_key)
        
        if not cache_file_path or not self._validate_cache_file(cache_file_path):
            return []
        
        cache_data = self._read_cache_file(cache_file_path)
        if cache_data:
            return cache_data.get('descriptions', [])
        
        return []


class SimilarMoleculesCacheManager(BaseCacheManager):
    """Manages similar molecules cache operations."""
    
    def __init__(self):
        """Initialize similar molecules cache manager."""
        super().__init__('similar', Config.CACHE['data_sources']['similar'])
    
    def save_similar_molecules(self, compound_name: str, similar_molecules: List[Dict]):
        """Save similar molecules to cache with multiple descriptions per molecule."""
        if not Config.CACHE['enabled']:
            return
        
        cache_key = self._normalize_cache_key(compound_name)
        cache_file_path = self._get_source_cache_file_path(cache_key)
        
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
                descriptions = compound_data.get('descriptions', [])
                compound_timestamp = compound_data.get('timestamp', datetime.now().isoformat())
                
                if compound_name_inner in existing_compound_names:
                    # Existing compound: add descriptions
                    for existing_compound in existing_compounds:
                        if existing_compound.get('compound_name') == compound_name_inner:
                            existing_compound['descriptions'].extend(descriptions)
                            # Apply description limit (keep latest max_descriptions)
                            existing_compound['descriptions'] = existing_compound['descriptions'][-max_descriptions:]
                            # Update compound timestamp if newer
                            if compound_timestamp > existing_compound.get('timestamp', ''):
                                existing_compound['timestamp'] = compound_timestamp
                            break
                else:
                    # New compound: add to list with timestamp
                    compound_data['timestamp'] = compound_timestamp
                    existing_compounds.append(compound_data)
            
            # Apply compound limit
            existing_compounds = self._apply_item_limit(existing_compounds)
            
            cache_data = {
                'compound_name': compound_name,
                'similar_compounds': existing_compounds,
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Create new data
            cache_data = {
                'compound_name': compound_name,
                'similar_compounds': similar_molecules,
                'timestamp': datetime.now().isoformat()
            }
        
        self._create_cache_file(cache_file_path, cache_data)
        logger.info(f"Cached similar molecules for compound: {compound_name}")
    
    def get_similar_molecules(self, compound_name: str) -> List[Dict]:
        """Get similar molecules from cache."""
        if not Config.CACHE['enabled']:
            return []
        
        cache_key = self._normalize_cache_key(compound_name)
        cache_file_path = self._get_source_cache_file_path(cache_key)
        
        if not cache_file_path or not self._validate_cache_file(cache_file_path):
            return []
        
        cache_data = self._read_cache_file(cache_file_path)
        if cache_data:
            return cache_data.get('similar_compounds', [])
        
        return []
    
    def get_random_similar_compound(self, compound_name: str) -> Optional[Tuple[str, str]]:
        """Get a random similar compound name and description from cache."""
        if not Config.CACHE['enabled']:
            return None
        
        cache_key = self._normalize_cache_key(compound_name)
        cache_file_path = self._get_source_cache_file_path(cache_key)
        
        if not cache_file_path or not self._validate_cache_file(cache_file_path):
            return None
        
        cache_data = self._read_cache_file(cache_file_path)
        if not cache_data:
            return None
        
        similar_compounds = cache_data.get('similar_compounds', [])
        if not similar_compounds:
            return None
        
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
    
    def get_any_random_similar_compound(self) -> Optional[Tuple[str, str]]:
        """Get a random similar compound from any cached similar compound file."""
        if not Config.CACHE['enabled']:
            return None
        
        cache_dir = self._get_source_cache_directory()
        if not cache_dir or not os.path.exists(cache_dir):
            return None
        
        # Get all cache files
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
        if not cache_files:
            return None
        
        # Try each cache file until we find one with similar compounds
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


class FailedMoleculesCacheManager(BaseCacheManager):
    """Manages failed molecule names cache (molecules without XYZ data)."""
    
    def __init__(self):
        """Initialize failed molecules cache manager."""
        super().__init__('failed_molecules', Config.CACHE['data_sources']['failed_molecules'])
    
    def add_failed_molecule(self, compound_name: str):
        """Add a compound name to the failed list."""
        if not Config.CACHE['enabled']:
            return
        
        cache_file_path = self._get_source_cache_file_path('failed_list')
        
        if not cache_file_path:
            logger.warning(f"Cannot save failed molecule cache: data source disabled")
            return
        
        # Read existing data if available
        existing_data = self._read_cache_file(cache_file_path) if os.path.exists(cache_file_path) else None
        
        if existing_data:
            # Add new failed molecule to existing list
            failed_molecules = existing_data.get('failed_molecules', [])
            if compound_name not in failed_molecules:
                failed_molecules.append(compound_name)
                logger.info(f"Added {compound_name} to failed molecules list")
            else:
                logger.info(f"{compound_name} already in failed molecules list")
            
            cache_data = {
                'failed_molecules': failed_molecules,
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Create new data
            cache_data = {
                'failed_molecules': [compound_name],
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"Created new failed molecules list with {compound_name}")
        
        self._create_cache_file(cache_file_path, cache_data)
        logger.info(f"Cached failed molecule: {compound_name}")
    
    def is_molecule_failed(self, compound_name: str) -> bool:
        """Check if a compound is in the failed list."""
        if not Config.CACHE['enabled']:
            return False
        
        cache_file_path = self._get_source_cache_file_path('failed_list')
        
        if not cache_file_path or not self._validate_cache_file(cache_file_path):
            return False
        
        cache_data = self._read_cache_file(cache_file_path)
        if not cache_data:
            return False
        
        failed_molecules = cache_data.get('failed_molecules', [])
        return compound_name in failed_molecules
    
    def get_failed_molecules(self) -> List[str]:
        """Get list of all failed molecule names."""
        if not Config.CACHE['enabled']:
            return []
        
        cache_file_path = self._get_source_cache_file_path('failed_list')
        
        if not cache_file_path or not self._validate_cache_file(cache_file_path):
            return []
        
        cache_data = self._read_cache_file(cache_file_path)
        if cache_data:
            return cache_data.get('failed_molecules', [])
        
        return []
    
    def remove_failed_molecule(self, compound_name: str):
        """Remove a compound name from the failed list (if XYZ data becomes available)."""
        if not Config.CACHE['enabled']:
            return
        
        cache_file_path = self._get_source_cache_file_path('failed_list')
        
        if not cache_file_path or not os.path.exists(cache_file_path):
            return
        
        cache_data = self._read_cache_file(cache_file_path)
        if not cache_data:
            return
        
        failed_molecules = cache_data.get('failed_molecules', [])
        if compound_name in failed_molecules:
            failed_molecules.remove(compound_name)
            logger.info(f"Removed {compound_name} from failed molecules list")
            
            cache_data = {
                'failed_molecules': failed_molecules,
                'timestamp': datetime.now().isoformat()
            }
            
            self._create_cache_file(cache_file_path, cache_data)


class CacheManager:
    """Unified cache manager coordinating all cache operations."""
    
    def __init__(self):
        """Initialize unified cache manager."""
        self.pubchem = PubChemCacheManager()
        self.queries = QueryCacheManager()
        self.descriptions = DescriptionCacheManager()
        self.analysis = AnalysisCacheManager()
        self.similar = SimilarMoleculesCacheManager()
        self.failed_molecules = FailedMoleculesCacheManager()
    
    def save_all_caches(self, english_name: str, detailed_info: DetailedMoleculeInfo, cid: int, user_query: str, description: str):
        """Save all cache types when xyz_data is successfully obtained."""
        try:
            # 1. PubChemキャッシュ保存
            self.pubchem.save_cached_molecule_data(english_name, detailed_info, cid)
            
            # 2. 質問-化合物マッピング保存
            if user_query:
                compounds = [{"compound_name": english_name, "timestamp": datetime.now().isoformat()}]  # 英語名とタイムスタンプを保存
                self.queries.save_query_compound_mapping(
                    user_query,
                    compounds,
                    increment_count=False  # 回数カウントは別途実行
                )
            
            # 3. 化合物-説明マッピング保存
            if description:
                self.descriptions.save_compound_description(english_name, description)
            
            logger.info(f"All caches saved successfully for {english_name}")
        except Exception as e:
            logger.error(f"Error saving caches for {english_name}: {e}")
    
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
            for manager in [self.pubchem, self.queries, self.descriptions]:
                cache_dir = manager._get_source_cache_directory()
                if cache_dir and os.path.exists(cache_dir):
                    for filename in os.listdir(cache_dir):
                        if filename.endswith('.json'):
                            file_path = os.path.join(cache_dir, filename)
                            os.remove(file_path)
            logger.info("All cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing all cache: {e}")
    
    def get_all_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for all data sources."""
        try:
            all_stats = {}
            total_count = 0
            total_size = 0
            
            for manager_name, manager in [('pubchem', self.pubchem), ('queries', self.queries), ('descriptions', self.descriptions)]:
                cache_dir = manager._get_source_cache_directory()
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
    
    def clear_cache(self, data_source: str = None):
        """Clear cache files for specific data source or all sources."""
        try:
            if data_source:
                # Clear specific data source
                cache_dir = self._get_cache_directory(data_source)
                if cache_dir and os.path.exists(cache_dir):
                    for filename in os.listdir(cache_dir):
                        if filename.endswith('.json'):
                            file_path = os.path.join(cache_dir, filename)
                            os.remove(file_path)
                    logger.info(f"Cache cleared for data source: {data_source}")
            else:
                # Clear all cache directories
                if os.path.exists(self.base_cache_dir):
                    for root, dirs, files in os.walk(self.base_cache_dir):
                        for filename in files:
                            if filename.endswith('.json'):
                                file_path = os.path.join(root, filename)
                                os.remove(file_path)
                    logger.info("All cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_cache_stats(self, data_source: str = None) -> Dict[str, Any]:
        """Get cache statistics for specific data source or all sources."""
        try:
            if data_source:
                # Stats for specific data source
                cache_dir = self._get_cache_directory(data_source)
                if not cache_dir or not os.path.exists(cache_dir):
                    return {'count': 0, 'size_mb': 0, 'files': []}
                
                files = []
                total_size = 0
                
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
                        total_size += file_size
                
                return {
                    'data_source': data_source,
                    'count': len(files),
                    'size_mb': round(total_size / (1024 * 1024), 2),
                    'files': files
                }
            else:
                # Stats for all data sources
                all_stats = {}
                total_count = 0
                total_size = 0
                
                for source_name, source_config in self.data_sources.items():
                    if source_config['enabled']:
                        source_stats = self.get_cache_stats(source_name)
                        all_stats[source_name] = source_stats
                        total_count += source_stats['count']
                        total_size += source_stats['size_mb']
                
                all_stats['total'] = {
                    'count': total_count,
                    'size_mb': round(total_size, 2)
                }
                
                return all_stats
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'count': 0, 'size_mb': 0, 'files': []}

# Initialize cache manager
cache_manager = CacheManager()

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
        ErrorHandler.show_error(ErrorHandler.handle_error(e, "api_error"))
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

def find_similar_molecules(molecule_name: str) -> Optional[str]:
    """Find molecules similar to the specified molecule."""
    logger.info(f"Searching for similar molecules to: {molecule_name}")
    
    # Get English name from current data for cache key
    current_data = st.session_state.get("current_molecule_data", None)
    english_name = current_data.get("name_en") if current_data else None
    
    # Always call Gemini API for similar molecules (don't use cache for direct response)
    similar_prompt = AIPrompts.SIMILAR_MOLECULE_SEARCH.format(molecule_name=molecule_name)
    
    response_text = call_gemini_api(
        prompt=similar_prompt,
        use_google_search=True
    )
    
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
            cache_manager.similar.save_similar_molecules(english_name, similar_compounds)
    
    return response_text

def get_compounds_by_name(english_name: str) -> Optional[Any]:
    """Get compound from PubChem using English name with timeout protection."""
    logger.info(f"Searching PubChem for compound: {english_name}")
    
    def search_pubchem():
        """Execute PubChem search by name."""
        try:
            compounds = pcp.get_compounds(english_name, 'name')
            if compounds:
                compound = compounds[0]  # Get first result
                logger.info(f"Found compound: {compound.cid}")
                return compound
            else:
                logger.warning(f"No compounds found for name: {english_name}")
                return None
        except Exception as e:
            logger.warning(f"PubChem search error for {english_name}: {type(e).__name__}: {str(e)}")
            return None
    
    return execute_with_timeout(
        search_pubchem, 
        Config.TIMEOUTS['pubchem_3d'], 
        "timeout"
    )

def get_3d_coordinates_by_cid(cid: int) -> Optional[str]:
    """Get 3D coordinates from PubChem using CID with timeout protection."""
    logger.info(f"Fetching 3D coordinates for CID: {cid}")
    
    def fetch_3d_coords():
        """Execute 3D coordinates fetch."""
        try:
            # Get 3D compound data from PubChem
            compounds_3d = pcp.get_compounds(cid, record_type='3d')
            if compounds_3d and len(compounds_3d) > 0:
                compound_3d = compounds_3d[0]
                
                # Extract coordinates and convert to XYZ format
                xyz_data = convert_pubchem_to_xyz(compound_3d)
                if xyz_data:
                    logger.info(f"Successfully fetched 3D coordinates for CID {cid}")
                    return xyz_data
                else:
                    logger.warning(f"Failed to convert 3D coordinates for CID {cid}")
                    return None
            else:
                logger.warning(f"No 3D coordinates available for CID {cid}")
                return None
        except Exception as e:
            logger.warning(f"3D coordinates fetch error for CID {cid}: {type(e).__name__}: {str(e)}")
            return None
    
    return execute_with_timeout(
        fetch_3d_coords, 
        Config.TIMEOUTS['pubchem_3d'], 
        "timeout"
    )

def convert_pubchem_to_xyz(compound_3d) -> Optional[str]:
    """Convert PubChem 3D compound data to XYZ format."""
    try:
        # Get atom coordinates and symbols
        atoms = compound_3d.atoms
        if not atoms:
            return None
        
        # Count atoms
        num_atoms = len(atoms)
        
        # Create XYZ header
        xyz_lines = [str(num_atoms)]
        xyz_lines.append(f"PubChem 3D coordinates for CID {compound_3d.cid}")
        
        # Add atom coordinates
        for atom in atoms:
            # Get atom symbol and coordinates
            symbol = atom.element
            x = atom.x
            y = atom.y
            z = atom.z
            
            # Format: symbol x y z
            xyz_lines.append(f"{symbol:2s} {x:12.6f} {y:12.6f} {z:12.6f}")
        
        return "\n".join(xyz_lines)
        
    except Exception as e:
        logger.warning(f"Error converting PubChem data to XYZ: {type(e).__name__}: {str(e)}")
        return None

def get_comprehensive_molecule_data(english_name: str) -> Tuple[bool, Optional[DetailedMoleculeInfo], Optional[int], Optional[str]]:
    """Get comprehensive molecule data from PubChem using English name with cache support."""
    logger.info(f"Getting comprehensive data for: {english_name}")
    
    # Check cache first
    cached_data = cache_manager.pubchem.get_cached_molecule_data(english_name)
    if cached_data:
        detailed_info, cid = cached_data
        logger.info(f"Using cached data for: {english_name}")
        
        # PubChemキャッシュのみ使用（説明キャッシュは別途保存）
        
        return True, detailed_info, cid, None
    
    with st.spinner("分子データを取得中...", show_time=True):

        # Try multiple search strategies
        compound = None
        
        # Strategy 1: Direct name search
        compound = get_compounds_by_name(english_name)

        # Strategy 2: If direct search fails, try common variations
        if not compound:
            logger.info(f"Direct search failed for '{english_name}', trying variations...")
            variations = [
                english_name.lower(),
                english_name.replace(" ", ""),
                english_name.replace(" acid", ""),
                english_name.replace(" salt", ""),
            ]
            
            for variation in variations:
                if variation != english_name:
                    logger.info(f"Trying variation: '{variation}'")
                    compound = get_compounds_by_name(variation)
                    if compound:
                        logger.info(f"Found compound with variation: '{variation}'")
                        break
        
        if not compound:
            logger.warning(f"No compound found for '{english_name}' with any search strategy")
            # Add to failed molecules list
            cache_manager.failed_molecules.add_failed_molecule(english_name)
            return False, None, None, Config.ERROR_MESSAGES['molecule_not_found']
        
        try:
            # Safely extract detailed information with proper error handling
            def safe_get_attr(obj, attr_name, default=None):
                """Safely get attribute from compound object."""
                try:
                    value = getattr(obj, attr_name, default)
                    return value if value is not None else default
                except (AttributeError, TypeError):
                    return default
            
            def safe_get_numeric_attr(obj, attr_name, default=None):
                """Safely get numeric attribute from compound object."""
                try:
                    value = getattr(obj, attr_name, default)
                    if value is not None:
                        return float(value) if isinstance(value, (int, float, str)) else default
                    return default
                except (AttributeError, TypeError, ValueError):
                    return default
            
            def safe_get_int_attr(obj, attr_name, default=None):
                """Safely get integer attribute from compound object."""
                try:
                    value = getattr(obj, attr_name, default)
                    if value is not None:
                        return int(value) if isinstance(value, (int, float, str)) else default
                    return default
                except (AttributeError, TypeError, ValueError):
                    return default

            # Extract basic information
            molecular_formula = safe_get_attr(compound, 'molecular_formula')
            molecular_weight = safe_get_attr(compound, 'molecular_weight')
            
            # Convert molecular_weight to float if it's a string
            if molecular_weight and isinstance(molecular_weight, str):
                try:
                    molecular_weight = float(molecular_weight)
                except (ValueError, TypeError):
                    molecular_weight = None
            elif molecular_weight and not isinstance(molecular_weight, (int, float)):
                try:
                    molecular_weight = float(molecular_weight)
                except (ValueError, TypeError):
                    molecular_weight = None
            
            # Get 3D coordinates data
            xyz_data = get_3d_coordinates_by_cid(compound.cid)
            
            # Create detailed info object
            detailed_info = DetailedMoleculeInfo(
                molecular_formula=molecular_formula,
                molecular_weight=molecular_weight,
                iupac_name=safe_get_attr(compound, 'iupac_name'),
                synonyms=safe_get_attr(compound, 'synonyms', [])[:5] if safe_get_attr(compound, 'synonyms') else [],
                inchi=safe_get_attr(compound, 'inchi'),
                # Chemical properties
                xlogp=safe_get_numeric_attr(compound, 'xlogp'),
                tpsa=safe_get_numeric_attr(compound, 'tpsa'),
                complexity=safe_get_numeric_attr(compound, 'complexity'),
                rotatable_bond_count=safe_get_int_attr(compound, 'rotatable_bond_count'),
                heavy_atom_count=safe_get_int_attr(compound, 'heavy_atom_count'),
                hbond_donor_count=safe_get_int_attr(compound, 'h_bond_donor_count'),
                hbond_acceptor_count=safe_get_int_attr(compound, 'h_bond_acceptor_count'),
                charge=safe_get_int_attr(compound, 'charge'),
                # XYZ coordinate data
                xyz_data=xyz_data,
            )
            
            logger.info(f"Successfully created detailed info for {english_name}")
            
            # Save to cache only if xyz_data is available
            if detailed_info.xyz_data:
                # PubChemキャッシュのみ保存（説明キャッシュは別途保存）
                cache_manager.pubchem.save_cached_molecule_data(english_name, detailed_info, compound.cid)
            else:
                logger.warning(f"Skipping cache save for {english_name}: No xyz_data available")
                # Add to failed molecules list when XYZ data is not available
                cache_manager.failed_molecules.add_failed_molecule(english_name)
            
            return True, detailed_info, compound.cid, None
            
        except Exception as e:
            logger.error(f"Error creating detailed info for {english_name}: {type(e).__name__}: {str(e)}")
            return False, None, None, Config.ERROR_MESSAGES['processing_error']

def analyze_molecule_properties(detailed_info: DetailedMoleculeInfo, molecule_name: str) -> Optional[str]:
    """Analyze molecular properties and generate human-readable explanation."""
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
    
    response_text = call_gemini_api(
        prompt=prompt,
        use_google_search=False
    )
    
    if response_text:
        analysis_result = response_text.strip()
        
        # Save analysis result to cache using English name
        current_data = st.session_state.get("current_molecule_data", None)
        if current_data and current_data.get("name_en"):
            cache_manager.analysis.save_analysis_result(current_data["name_en"], analysis_result)
        
        return analysis_result
    else:
        logger.warning("No response received from Gemini API for molecular analysis")
        return None

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

def parse_json_response(response_text: str) -> Optional[Dict]:
    """Parse JSON data from Gemini response text."""
    if not response_text:
        return None
    
    # Define JSON patterns in order of preference
    patterns = [
        r'```json\s*(\{.*?\})\s*```',  # JSON code blocks
        r'(\{[^{}]*"name_jp"[^{}]*"name_en"[^{}]*"description"[^{}]*\})',  # New format
        r'(\{[^{}]*"name_jp"[^{}]*\})',  # Any JSON with name_jp
        r'(\{[^{}]*"name"[^{}]*"id"[^{}]*"description"[^{}]*\})',  # Legacy format
        r'(\{[^{}]*"name"[^{}]*\})',  # Any JSON with name
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for json_str in matches:
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and ("name_jp" in parsed or "name" in parsed):
                    return parsed
            except json.JSONDecodeError:
                continue
    
    return None

def is_no_result_response(response_text: str) -> bool:
    """Check if the response indicates no results found."""
    if not response_text:
        return True
    
    # Check for "該当なし" in the response
    if "該当なし" in response_text:
        return True
    
    # Also check parsed JSON for "該当なし" in name_jp field
    try:
        json_data = parse_json_response(response_text)
        if json_data and json_data.get("name_jp") == "該当なし":
            return True
    except:
        pass
    
    return False

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
                    success, detailed_info, cid, error_msg = get_comprehensive_molecule_data(molecule_name_en)
                
                    if success and detailed_info:
                        logger.info(f"Successfully got comprehensive data for {molecule_name_en}")
                        data["detailed_info"] = detailed_info
                        data["xyz_data"] = detailed_info.xyz_data
                        data["cid"] = cid
                        
                        # 説明キャッシュをここで保存（xyz_dataが存在する場合のみ）
                        if detailed_info.xyz_data and description:
                            if save_to_query_cache:
                                user_query = st.session_state.get("user_query", "")
                                cache_manager.save_all_caches(molecule_name_en, detailed_info, cid, user_query, description)
                            else:
                                # 関連分子処理時はqueriesキャッシュをスキップし、他のキャッシュのみ保存
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
        analysis_result = analyze_molecule_properties(detailed_info, molecule_name)
        
        if analysis_result:
            return analysis_result
        else:
            return Config.ERROR_MESSAGES['display_error']
    except Exception as e:
        return f"{Config.ERROR_MESSAGES['display_error']}: {e}"

def find_and_process_similar_molecule() -> Optional[Dict]:
    """Find and process similar molecule data."""
    try:
        similar_response = find_similar_molecules(get_molecule_name())
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

