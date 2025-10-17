# Standard library imports
import random
import json
import re
import logging
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
    description: Optional[str]
    inchi: Optional[str]
    inchi_key: Optional[str]
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
                'max_age_days': 360,
            },
            'psi4': {
                'enabled': True,
                'directory': 'psi4',
                'max_age_days': 360,  # Shorter for AI responses
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
                # Reset analysis execution flag and clear cache to allow new analysis
                st.session_state.detail_analysis_executed = False
                st.session_state.cached_analysis_result = ""
                st.session_state.screen = "detail_response"
                st.rerun()
    
    with col2:
        if st.button("関連する分子は？", key=f"{key_prefix}_similar", use_container_width=True, icon="🔍", disabled=not has_name):
            if has_name:
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

class CacheManager:
    """Manages local cache for multiple data sources."""
    
    def __init__(self):
        """Initialize cache manager."""
        self.base_cache_dir = Config.CACHE['base_directory']
        self.max_size_mb = Config.CACHE['max_size_mb']
        self.max_age_days = Config.CACHE['max_age_days']
        self.data_sources = Config.CACHE['data_sources']
        self._ensure_cache_directories()
    
    def _ensure_cache_directories(self):
        """Ensure cache directories exist for all data sources."""
        if not os.path.exists(self.base_cache_dir):
            os.makedirs(self.base_cache_dir)
            logger.info(f"Created base cache directory: {self.base_cache_dir}")
        
        # Create subdirectories for each data source
        for source_name, source_config in self.data_sources.items():
            if source_config['enabled']:
                source_dir = os.path.join(self.base_cache_dir, source_config['directory'])
                if not os.path.exists(source_dir):
                    os.makedirs(source_dir)
                    logger.info(f"Created cache directory for {source_name}: {source_dir}")
    
    def _get_data_source_config(self, data_source: str) -> Optional[Dict]:
        """Get configuration for specific data source."""
        return self.data_sources.get(data_source)
    
    def _get_cache_directory(self, data_source: str) -> Optional[str]:
        """Get cache directory for specific data source."""
        config = self._get_data_source_config(data_source)
        if not config or not config['enabled']:
            return None
        return os.path.join(self.base_cache_dir, config['directory'])
    
    def _normalize_cache_key(self, compound_name: str) -> str:
        """Normalize compound name for cache key."""
        # Convert to lowercase, strip whitespace, remove common suffixes
        normalized = compound_name.lower().strip()
        normalized = normalized.replace(" acid", "").replace(" salt", "")
        # Create a safe filename by replacing special characters
        safe_key = re.sub(r'[^\w\-_]', '_', normalized)
        return safe_key
    
    def _get_cache_file_path(self, data_source: str, cache_key: str) -> Optional[str]:
        """Get cache file path for given data source and key."""
        cache_dir = self._get_cache_directory(data_source)
        if not cache_dir:
            return None
        return os.path.join(cache_dir, f"{cache_key}.json")
    
    def _is_cache_valid(self, cache_file_path: str, data_source: str) -> bool:
        """Check if cache file is valid (exists and not expired)."""
        if not os.path.exists(cache_file_path):
            return False
        
        # Get data source specific max age
        config = self._get_data_source_config(data_source)
        max_age_days = config.get('max_age_days', self.max_age_days) if config else self.max_age_days
        
        # Check age
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file_path))
        age = datetime.now() - file_time
        if age > timedelta(days=max_age_days):
            logger.info(f"Cache expired for {cache_file_path}")
            return False
        
        return True
    
    def get_cached_data(self, compound_name: str, data_source: str = 'pubchem') -> Optional[Tuple[Optional[DetailedMoleculeInfo], Optional[int]]]:
        """Get cached data for compound from specific data source."""
        if not Config.CACHE['enabled']:
            return None
        
        cache_key = self._normalize_cache_key(compound_name)
        cache_file_path = self._get_cache_file_path(data_source, cache_key)
        
        if not cache_file_path or not self._is_cache_valid(cache_file_path, data_source):
            return None
        
        try:
            with open(cache_file_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Reconstruct DetailedMoleculeInfo object
            detailed_info = DetailedMoleculeInfo(**cache_data['detailed_info'])
            cid = cache_data.get('cid')
            
            logger.info(f"Cache hit for compound: {compound_name} (source: {data_source})")
            return detailed_info, cid
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Invalid cache data for {compound_name}: {e}")
            # Remove invalid cache file
            try:
                os.remove(cache_file_path)
            except OSError:
                pass
            return None
        except Exception as e:
            logger.error(f"Error reading cache for {compound_name}: {e}")
            return None
    
    def save_cached_data(self, compound_name: str, detailed_info: DetailedMoleculeInfo, cid: int, data_source: str = 'pubchem'):
        """Save data to cache for specific data source."""
        if not Config.CACHE['enabled']:
            return
        
        cache_key = self._normalize_cache_key(compound_name)
        cache_file_path = self._get_cache_file_path(data_source, cache_key)
        
        if not cache_file_path:
            logger.warning(f"Cannot save cache for {data_source}: data source disabled")
            return
        
        try:
            # Convert DetailedMoleculeInfo to dictionary
            cache_data = {
                'compound_name': compound_name,
                'cache_key': cache_key,
                'data_source': data_source,
                'timestamp': datetime.now().isoformat(),
                'cid': cid,
                'detailed_info': {
                    'molecular_formula': detailed_info.molecular_formula,
                    'molecular_weight': detailed_info.molecular_weight,
                    'iupac_name': detailed_info.iupac_name,
                    'synonyms': detailed_info.synonyms,
                    'description': detailed_info.description,
                    'inchi': detailed_info.inchi,
                    'inchi_key': detailed_info.inchi_key,
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
            
            with open(cache_file_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Cached data for compound: {compound_name} (source: {data_source})")
            
        except Exception as e:
            logger.error(f"Error saving cache for {compound_name}: {e}")
    
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
    
    return call_gemini_api(
        prompt=prompt,
        use_google_search=True
    )

def find_similar_molecules(molecule_name: str) -> Optional[str]:
    """Find molecules similar to the specified molecule."""
    logger.info(f"Searching for similar molecules to: {molecule_name}")
    
    similar_prompt = AIPrompts.SIMILAR_MOLECULE_SEARCH.format(molecule_name=molecule_name)
    
    return call_gemini_api(
        prompt=similar_prompt,
        use_google_search=True
    )

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
    cached_data = cache_manager.get_cached_data(english_name)
    if cached_data:
        detailed_info, cid = cached_data
        logger.info(f"Using cached data for: {english_name}")
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
                description=safe_get_attr(compound, 'description'),
                inchi=safe_get_attr(compound, 'inchi'),
                inchi_key=safe_get_attr(compound, 'inchi_key'),
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
            
            # Save to cache
            cache_manager.save_cached_data(english_name, detailed_info, compound.cid)
            
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
        return response_text.strip()
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

def parse_gemini_response(response_text: str) -> Dict[str, Union[str, None, Any]]:
    """Parse Gemini's JSON response and fetch comprehensive data from PubChem."""
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
                else:
                    logger.warning(f"Failed to get comprehensive data: {error_msg}")
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
        response_text = search_molecule_by_query(user_query)
        if response_text:
            parsed_output = parse_gemini_response(response_text)
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
            return parse_gemini_response(similar_response)
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

