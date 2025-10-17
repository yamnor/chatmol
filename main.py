# Standard library imports
import random
import json
import re
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime

# Third-party imports
import streamlit as st
import streamlit.components.v1 as components

from google import genai
from google.genai import types

import py3Dmol
import pubchempy as pcp

from st_screen_stats import WindowQueryHelper

from rdkit import Chem
from rdkit.Chem import AllChem

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
    canonical_smiles: Optional[str]
    isomeric_smiles: Optional[str]
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

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

class Config:
    """Application configuration constants."""
    
    # Timeout settings for preventing freezes
    TIMEOUTS = {
        'api': 30,  # Gemini API timeout
        'structure_generation': 15,  # 3D structure generation timeout
        'pubchem_3d': 10,  # PubChem 3D record fetch timeout
        'pubchem_smiles': 10,  # PubChem SMILES fetch timeout
    }
    
    # Random sample configuration
    RANDOM_QUERY = {
        'count': 20,  # Number of random samples to display
        'columns': 2,  # Number of columns for random samples
    }
    
    # Molecular Size Limits
    MOLECULE_LIMITS = {
        'max_atoms_3d_display': 100,
        'max_atoms_3d_generation': 100,
    }
    
    # 3D Molecular Viewer Configuration
    # Responsive viewer size based on window size
    VIEWER = {
        'width_pc': 700,
        'height_pc': 600,
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
        'processing_error': "分子の処理中にエラーが発生しました。",
        'structure_error': "3D構造の生成に失敗しました。",
        'molecule_too_large': "分子が大きすぎます（原子数: {num_atoms}）。",
        
        # General errors
        'parse_error': "データの解析に失敗しました。",
        'display_error': "表示中にエラーが発生しました。",
        'no_data': "データが見つかりません。最初からやり直してください。",
        'general_error': "予期しないエラーが発生しました。",
    }

# Legacy constant for backward compatibility (only PUBCHEM_3D_TIMEOUT_SECONDS is still used)
PUBCHEM_3D_TIMEOUT_SECONDS = Config.TIMEOUTS['pubchem_3d']

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
 ユーザーが求める効能・イメージ・用途・ニーズなどを 1 文でもらったら、
 (1) それに最も関連すると考える複数の候補分子を優先度の高い順に PubChem で検索して、
 (2) 最初に見つかった分子のみについて、その分子の日本語での名称（name）、一言の説明（description）、PubChem CID (id) を、以下のルールに厳密に従い、JSON 形式でのみ出力してください。

- 分子の検索は、必ず、「 Google Search 」を用いて、PubChem のページ「 https://pubchem.ncbi.nlm.nih.gov/compound/<分子名（英語名称）> 」で行ってください
- 分子名は、必ず、英語名称で検索してください。日本語名称では検索できません。
- PubChem で分子が見つからなかった、または PubChem CID データを取得できなかった場合は、次の優先度の分子を検索します
- 該当する分子を思いつかなかった、または優先度順のすべての分子が PubChem で見つからなかった場合は、「該当なし」とのみ出力します
- ひとこと理由は、小学生にもわかるように、1 行でフレンドリーに表現してください

# USER
{user_input}

```json
{{
  "name": "<分子名>（見つかった分子の日本語での名称）",
  "id": "<PubChem CID>（整数値）",
  "description": "<一言の説明> （その分子を選んだ理由や性質の特徴を１行で説明）"
}}
```
"""

    # Similar molecule search prompt
    SIMILAR_MOLECULE_SEARCH: str = """
# SYSTEM
あなたは「分子コンシェルジュ」です。
ユーザーが指定した分子「{molecule_name}」に関連する分子を探してください。
以下の多様な観点から関連する分子を検討し、必ず指定された分子とは異なる分子を提案してください：

## 関連性の観点
1. **構造的類似性**: 同じ官能基、骨格構造、分子サイズ
2. **機能的類似性**: 同じ作用機序、生体活性、薬理効果
3. **用途的類似性**: 同じ分野での利用、同じ目的での使用
4. **化学的類似性**: 同じ化学反応性、物理化学的性質
5. **生物学的類似性**: 同じ代謝経路、同じ受容体への結合
6. **歴史的関連性**: 同じ発見者、同じ研究グループ、同じ時代
7. **対照的関連性**: 相反する作用、拮抗作用、補完的効果
8. **進化的関連性**: 同じ生物種由来、同じ進化系統

## 検索手順
(1) 上記の観点から複数の候補分子を優先度の高い順に考え、
(2) 最初に見つかった分子のみについて、その分子の日本語での名称（name）、説明（description）、PubChem CID (id) を、以下のルールに厳密に従い、JSON 形式でのみ出力してください。

## 必須ルール
- **必ず指定された分子とは異なる分子を提案してください**
- 分子の検索は、必ず、「 Google Search 」を用いて、PubChem のページ「 https://pubchem.ncbi.nlm.nih.gov/compound/<分子名（英語名称）> 」で行ってください
- 分子名は、必ず、英語名称で検索してください。日本語名称では検索できません。
- PubChem で分子が見つからなかった、または PubChem CID データを取得できなかった場合は、次の優先度の分子を検索します
- 該当する分子を思いつかなかった、または優先度順のすべての分子が PubChem で見つからなかった場合は、「該当なし」とのみ出力します
- 説明は、小学生にもわかるように、「どの観点で関連しているか」と「その分子を選んだ理由や性質の特徴」を２行でフレンドリーに表現してください
- どの観点で関連しているかを説明に含めてください

```json
{{
  "name": "<分子名>（見つかった分子の日本語での名称）",
  "id": "<PubChem CID>（整数値）",
  "description": "<一言の説明> （どの観点で関連しているかと、その分子を選んだ理由や性質の特徴を２行で説明）"
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
3. **分子メカニズム**: 上記の性質から推測される生体内での作用メカニズムや分子標的への結合様式

# 出力形式
- ケモインフォマティクスの観点から科学的に分析してください
- 分子データの具体的な数値を示しながら、3-5文程度の簡潔な説明にまとめてください
- 「〜があるよ」「〜だよ」「〜だよね」など、親しみやすい口調で説明してください
- 溶解性、膜透過性、薬物動態などの分子メカニズムを、分かりやすい比喩や表現で説明してください
- 推測であることを明記してください（「〜と考えられるよ」「〜の可能性があるよ」など）

# 出力例
以下は出力例です。このような形式で分析結果を出力してください：

**カフェイン**は分子量194.19の小さな分子で、LogPが-0.07と水に溶けやすい性質があるよ。
TPSAが58.4と比較的高いから、体内での吸収が良くて、脳に届きやすいんだよね。
分子複雑度が62.3と中程度で、回転可能結合が0個だから構造がしっかりしていて、特定の受容体にピンポイントで結合できるんだよね。

分析結果のみを出力してください。他の説明や補足は不要です。
"""


# Sample queries organized by category for readability
SAMPLE_QUERIES: List[str] = [
    # 🌸 香り
    "良い香りのする成分は？",
    "甘い香りのする成分は？",
    "フレッシュな香りが欲しい",
    "落ち着く香りを探している",
    "スパイシーな香りが欲しい",
    
    # 🍋 食べ物・飲み物
    "レモンの成分は？",
    "バニラの成分は？",
    "コーヒーの成分は？",
    "チョコレートの成分は？",
    "ミントの成分は？",
    
    # 🌸 花・植物
    "バラの香り成分は？",
    "桜の香り成分は？",
    "ラベンダーの香り成分は？",
    "ジャスミンの香り成分は？",
    "金木犀の香り成分は？",
    
    # 🎨 色・染料
    "リンゴの赤色の成分は？",
    "ベリーの青色の成分は？",
    "レモンの黄色の成分は？",
    "ぶどうの紫色の成分は？",
    "デニムの青色の成分は？",
    
    # 👅 味覚
    "甘い味の成分は？",
    "酸っぱい味の成分は？",
    "苦い味の成分は？",
    "辛い味の成分は？",
    "うま味の成分は？",
    
    # 💊 医薬品
    "風邪薬の成分は？",
    "頭痛薬の成分を教えて",
    "胃薬の成分は？",
    "インフル治療薬の成分は？",
    "抗生物質の成分は？",
    
    # 🌲 自然・環境
    "森の香り成分は？",
    "海の香り成分は？",
    "土の匂い成分は？",
    "木の香り成分は？",
    "草の香り成分は？",
    
    # 🧴 日用品
    "洗剤の成分は？",
    "シャンプーの成分は？",
    "石鹸の成分は？",
    "柔軟剤の成分は？",
    "消臭剤の成分は？",
    
    # 💪 スポーツ・運動
    "筋肉を鍛えたい",
    "疲労を回復させたい",
    "持久力をアップさせたい",
    "瞬発力をアップさせたい",
    "エネルギーを補給したい",
    
    # 💚 健康・体調
    "気分をすっきりさせたい",
    "疲れを取りたい",
    "目覚めを良くしたい",
    "免疫力を高めたい",
    "血行を良くしたい",
    
    # 😴 リラックス・睡眠
    "リラックスしたい",
    "心を落ち着かせたい",
    "ゆっくり休みたい",
    "ストレスを和らげたい",
    "幸福感を感じたい",
    
    # 🧠 集中・学習
    "集中力を高めたい",
    "勉強に集中したい",
    "思考力を高めたい",
    "脳を活性化したい",
    
    # ✨ 美容・スキンケア
    "肌を美しく保ちたい",
    "若々しさを維持したい",
    "髪の毛を健康にしたい",
    "シミを防ぎたい",
    "肌の潤いを保ちたい"
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
            'structure_error': Config.ERROR_MESSAGES['structure_error'],
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
    
    @staticmethod
    def show_error_with_retry(message: str) -> None:
        """Show error message with retry button."""
        ErrorHandler.show_error(message)
        st.write("---")
        if st.button("他の分子を探す", key="error_retry_button", use_container_width=True):
            reset_to_initial_state()
            st.rerun()


def show_action_buttons(key_prefix: str = "action") -> None:
    """Show standardized action button set: 詳しく知りたい, 関連する分子は？, 他の分子を探す."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("詳しく知りたい", key=f"{key_prefix}_detail", use_container_width=True):
            current_data = st.session_state.get("current_molecule_data", None)
            if current_data and current_data.get("cid"):
                # Reset analysis execution flag and clear cache to allow new analysis
                st.session_state.detail_analysis_executed = False
                st.session_state.cached_analysis_result = ""
                st.session_state.screen = "detail_response"
                st.rerun()
            else:
                st.warning("分子データがありません。最初からやり直してください。")
    
    with col2:
        if st.button("関連する分子は？", key=f"{key_prefix}_similar", use_container_width=True):
            current_data = st.session_state.get("current_molecule_data", None)
            if current_data and current_data.get("name"):
                # Reset search execution flag to allow new search
                st.session_state.similar_search_executed = False
                st.session_state.screen = "similar_response"
                st.rerun()
            else:
                st.warning("分子データがありません。最初からやり直してください。")
    
    with col3:
        if st.button("他の分子を探す", key=f"{key_prefix}_new", use_container_width=True):
            reset_to_initial_state()

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def reset_to_initial_state():
    """Reset the application to initial state."""
    st.session_state.screen = "initial"
    st.session_state.user_query = ""
    st.session_state.selected_sample = ""
    st.session_state.smiles_error_occurred = False
    st.session_state.current_molecule_data = None
    st.session_state.gemini_output = None
    st.session_state.random_queries = generate_random_queries()
    st.session_state.similar_search_executed = False
    st.session_state.detail_analysis_executed = False
    st.session_state.cached_analysis_result = ""
    
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

def generate_random_queries() -> List[str]:
    """Generate random samples from all available queries."""
    if SAMPLE_QUERIES:
        return random.sample(SAMPLE_QUERIES, min(Config.RANDOM_QUERY['count'], len(SAMPLE_QUERIES)))
    else:
        return []


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

def get_smiles_from_pubchem(cid: int) -> Tuple[bool, Optional[str], Optional[str]]:
    """Get SMILES string from PubChem using CID with timeout protection."""
    logger.info(f"Fetching SMILES from PubChem for CID: {cid}")
    
    def fetch_from_pubchem():
        """Execute PubChem API call."""
        try:
            compound = pcp.get_compounds(cid, 'cid')[0]
            logger.info(f"Successfully fetched compound from PubChem: {compound.canonical_smiles[:50]}...")
            return compound.canonical_smiles
        except (IndexError, Exception) as e:
            logger.warning(f"Error fetching compound from PubChem for CID {cid}: {str(e)}")
            return None
    
    smiles = execute_with_timeout(
        fetch_from_pubchem, 
        Config.TIMEOUTS['pubchem_smiles'], 
        "timeout"
    )
    
    if smiles:
        return True, smiles, None
    else:
        return False, None, Config.ERROR_MESSAGES['molecule_not_found']

def analyze_molecule_properties(detailed_info: DetailedMoleculeInfo, molecule_name: str) -> Optional[str]:
    """Analyze molecular properties and generate human-readable explanation."""
    logger.info(f"Getting Gemini analysis for molecule: {molecule_name}")
    
    # PubChemの詳細情報を整理
    properties_text = []
    if detailed_info.molecular_formula:
        properties_text.append(f"分子式: {detailed_info.molecular_formula}")
    if detailed_info.molecular_weight:
        properties_text.append(f"分子量: {detailed_info.molecular_weight:.2f}")
    if detailed_info.xlogp is not None:
        properties_text.append(f"LogP: {detailed_info.xlogp:.2f}")
    if detailed_info.tpsa is not None:
        properties_text.append(f"TPSA: {detailed_info.tpsa:.1f} Å²")
    if detailed_info.complexity is not None:
        properties_text.append(f"分子複雑度: {detailed_info.complexity:.1f}")
    if detailed_info.hbond_donor_count is not None:
        properties_text.append(f"H結合供与体数: {detailed_info.hbond_donor_count}")
    if detailed_info.hbond_acceptor_count is not None:
        properties_text.append(f"H結合受容体数: {detailed_info.hbond_acceptor_count}")
    if detailed_info.rotatable_bond_count is not None:
        properties_text.append(f"回転可能結合数: {detailed_info.rotatable_bond_count}")
    if detailed_info.heavy_atom_count is not None:
        properties_text.append(f"重原子数: {detailed_info.heavy_atom_count}")
    
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

def get_detailed_molecule_info(cid: int) -> DetailedMoleculeInfo:
    """Get detailed molecule information from PubChem using CID."""
    def fetch_detailed_info():
        """Execute PubChem API call for detailed information."""
        try:
            compounds = pcp.get_compounds(cid, 'cid')
            if not compounds:
                logger.warning(f"No compounds found for CID: {cid}")
                return None
            
            compound = compounds[0]
            
            # Safely extract detailed information with proper error handling
            def safe_get_attr(obj, attr_name, default=None):
                """Safely get attribute from compound object."""
                try:
                    value = getattr(obj, attr_name, default)
                    return value if value is not None else default
                except (AttributeError, TypeError):
                    return default
            
            # Extract detailed information with safe attribute access
            molecular_formula = safe_get_attr(compound, 'molecular_formula')
            molecular_weight = safe_get_attr(compound, 'molecular_weight')
            
            # Convert molecular_weight to float if it's a string
            if molecular_weight and isinstance(molecular_weight, str):
                try:
                    molecular_weight = float(molecular_weight)
                except (ValueError, TypeError):
                    molecular_weight = None
            
            # Convert molecular_weight to float if it's a number
            elif molecular_weight and not isinstance(molecular_weight, (int, float)):
                try:
                    molecular_weight = float(molecular_weight)
                except (ValueError, TypeError):
                    molecular_weight = None
            
            # Extract chemical properties
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
            
            detailed_info = DetailedMoleculeInfo(
                molecular_formula=molecular_formula,
                molecular_weight=molecular_weight,
                iupac_name=safe_get_attr(compound, 'iupac_name'),
                synonyms=safe_get_attr(compound, 'synonyms', [])[:5] if safe_get_attr(compound, 'synonyms') else [],
                description=safe_get_attr(compound, 'description'),
                canonical_smiles=safe_get_attr(compound, 'canonical_smiles'),
                isomeric_smiles=safe_get_attr(compound, 'isomeric_smiles'),
                inchi=safe_get_attr(compound, 'inchi'),
                inchi_key=safe_get_attr(compound, 'inchi_key'),
                # Chemical properties
                xlogp=safe_get_numeric_attr(compound, 'xlogp'),
                tpsa=safe_get_numeric_attr(compound, 'tpsa'),
                complexity=safe_get_numeric_attr(compound, 'complexity'),
                rotatable_bond_count=safe_get_int_attr(compound, 'rotatable_bond_count'),
                heavy_atom_count=safe_get_int_attr(compound, 'heavy_atom_count'),
                hbond_donor_count=safe_get_int_attr(compound, 'hbond_donor_count'),
                hbond_acceptor_count=safe_get_int_attr(compound, 'hbond_acceptor_count'),
                charge=safe_get_int_attr(compound, 'charge'),
            )
            
            return detailed_info
        except Exception as e:
            logger.error(f"Error fetching detailed info for CID {cid}: {e}")
            return None
    
    detailed_info = execute_with_timeout(
        fetch_detailed_info, 
        Config.TIMEOUTS['pubchem_smiles'], 
        "timeout"
    )
    
    if detailed_info:
        return detailed_info
    else:
        return DetailedMoleculeInfo(
            molecular_formula=None,
            molecular_weight=None,
            iupac_name=None,
            synonyms=[],
            description=None,
            canonical_smiles=None,
            isomeric_smiles=None,
            inchi=None,
            inchi_key=None,
            # Chemical properties
            xlogp=None,
            tpsa=None,
            complexity=None,
            rotatable_bond_count=None,
            heavy_atom_count=None,
            hbond_donor_count=None,
            hbond_acceptor_count=None,
            charge=None,
        )

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
    
    # Get model name from Streamlit secrets with fallback
    try:
        model_name = st.secrets["model_name"]
    except KeyError:
        # Fallback to default model if not specified in secrets
        model_name = Config.DEFAULT_MODEL_NAME

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
        r'(\{[^{}]*"name"[^{}]*"id"[^{}]*"description"[^{}]*\})',  # New format
        r'(\{[^{}]*"name"[^{}]*\})',  # Any JSON with name
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for json_str in matches:
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and "name" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
    
    return None

def create_default_molecule_data() -> Dict[str, Union[str, None, Any]]:
    """Create default molecule data structure."""
    return {
        "name": "分子が見つかりませんでした",
        "smiles": None,
        "memo": "申し訳ありません。ご要望に合う分子を見つけることができませんでした。もう少し具体的な情報を教えていただけますか？",
        "mol": None,
        "mol_with_h": None,
        "properties": None,
        "cid": None
    }

def process_molecule_cid(cid_value) -> Tuple[bool, Optional[int], Optional[str]]:
    """Process and validate PubChem CID."""
    if cid_value is None:
        return False, None, Config.ERROR_MESSAGES['invalid_data']
    
    try:
        # Convert to integer if it's a string
        if isinstance(cid_value, str):
            cid = int(cid_value.strip())
        else:
            cid = int(cid_value)
        return True, cid, None
    except (ValueError, TypeError):
        return False, None, Config.ERROR_MESSAGES['invalid_data']

def fetch_and_process_molecule_data(cid: int, molecule_name: str, description: str) -> Dict[str, Union[str, None, Any]]:
    """Fetch molecule data from PubChem and create molecular objects."""
    data = create_default_molecule_data()
    data["name"] = molecule_name
    data["memo"] = description if description else "分子の詳細情報を取得中..."
    data["cid"] = cid
    
    success, pubchem_smiles, error_msg = get_smiles_from_pubchem(cid)
    
    if success and pubchem_smiles:
        data["smiles"] = pubchem_smiles
        _create_molecular_objects(pubchem_smiles, data)
    else:
        data["memo"] = f"申し訳ありません。PubChemから分子データを取得できませんでした（{error_msg}）。"
    
    return data

def parse_gemini_response(response_text: str) -> Dict[str, Union[str, None, Any]]:
    """Parse Gemini's JSON response and fetch SMILES from PubChem."""
    data = create_default_molecule_data()
    
    if not response_text:
        return data
    
    try:
        # Extract JSON from response text
        json_data = parse_json_response(response_text)
        
        if json_data:
            # Handle current format (name, id, description)
            molecule_name = json_data.get("name", "").strip()
            cid_value = json_data.get("id")
            description = json_data.get("description", "").strip()
            
            if molecule_name:
                # Process CID
                cid_valid, cid, cid_error = process_molecule_cid(cid_value)
                
                if cid_valid and cid is not None:
                    # Fetch and process molecule data
                    data = fetch_and_process_molecule_data(cid, molecule_name, description)
                else:
                    data["memo"] = cid_error if cid_error else Config.ERROR_MESSAGES['invalid_data']
            else:
                data["memo"] = Config.ERROR_MESSAGES['invalid_data']
        else:
            data["memo"] = Config.ERROR_MESSAGES['parse_error']
            
    except Exception as e:
        data["memo"] = f"{Config.ERROR_MESSAGES['parse_error']}: {e}"
    
    return data


def _create_molecular_objects(canonical_smiles: str, data: Dict[str, Union[str, None, Any]]) -> None:
    """Create molecular objects and calculate properties with enhanced error handling."""
    try:
        # Create molecular object (PubChem SMILESは既に検証済み)
        data["mol"] = Chem.MolFromSmiles(canonical_smiles)
        if data["mol"] is None:
            raise ValueError(Config.ERROR_MESSAGES['processing_error'])
        
        # Check molecule complexity before adding hydrogens
        num_atoms = data["mol"].GetNumAtoms()
        if num_atoms > Config.MOLECULE_LIMITS['max_atoms_3d_display']:
            st.warning(Config.ERROR_MESSAGES['molecule_too_large'].format(num_atoms=num_atoms))
        
        # Add hydrogens
        data["mol_with_h"] = Chem.AddHs(data["mol"])
        
        # Set properties to None since we're not calculating them
        data["properties"] = None
    except Exception as e:
        # Clear all molecular data and set error state
        data["mol"] = None
        data["mol_with_h"] = None
        data["properties"] = None
        data["smiles"] = None
        data["memo"] = f"{Config.ERROR_MESSAGES['processing_error']}（{str(e)}）。別の分子をお探ししましょうか？"
        
        # Set error state to prevent further processing
        st.session_state.smiles_error_occurred = True

def get_pubchem_3d_sdf_by_cid(cid: Optional[int]) -> Optional[str]:
    """Fetch 3D SDF from PubChem by CID with timeout protection.

    Returns SDF string if available; otherwise returns None.
    """
    if cid is None:
        return None

    def fetch_sdf():
        try:
            import urllib.request
            import urllib.error
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/{cid}/record/SDF/?record_type=3d"
            req = urllib.request.Request(url, headers={"User-Agent": "chatmol/1.0"})
            with urllib.request.urlopen(req, timeout=PUBCHEM_3D_TIMEOUT_SECONDS) as resp:
                data_bytes = resp.read()
                sdf_text = data_bytes.decode("utf-8", errors="ignore")
                # Basic sanity check for SDF content
                if sdf_text and "M  END" in sdf_text:
                    return sdf_text
                return None
        except Exception:
            return None

    return execute_with_timeout(
        fetch_sdf, 
        Config.TIMEOUTS['pubchem_3d'] + 2, 
        "timeout"
    )

def get_molecule_structure_3d_sdf(mol_with_h) -> Optional[str]:
    """Generate 3D molecular structure from molecular object with timeout protection."""
    if not mol_with_h:
        return None
    
    def generate_3d_structure():
        """Execute 3D structure generation."""
        return _generate_3d_structure(mol_with_h)
    
    return execute_with_timeout(
        generate_3d_structure, 
        Config.TIMEOUTS['structure_generation'], 
        "structure_error"
    )

def _embed_molecule_3d(mol_copy) -> bool:
    """Embed 3D coordinates into molecule using multiple methods."""
    embed_methods = [
        AllChem.ETKDG(),
        AllChem.ETKDGv2(), 
        AllChem.ETKDGv3()
    ]
    
    for method in embed_methods:
        try:
            embed_result = AllChem.EmbedMolecule(mol_copy, method)
            if embed_result == 0:
                return True
        except Exception:
            continue
    return False

def _optimize_molecule_3d(mol_copy) -> None:
    """Optimize 3D structure while preserving stereochemistry."""
    try:
        AllChem.MMFFOptimizeMolecule(mol_copy)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol_copy)
        except Exception:
            pass  # Continue without optimization

def _generate_3d_structure(mol_with_h) -> str:
    """Generate 3D structure and convert to SDF format."""
    try:
        # Create a copy to avoid modifying the original molecule
        mol_copy = Chem.Mol(mol_with_h)
        if mol_copy is None:
            raise ValueError("分子オブジェクトのコピーに失敗しました")
        
        # Check molecule complexity before embedding
        num_atoms = mol_copy.GetNumAtoms()
        if num_atoms > Config.MOLECULE_LIMITS['max_atoms_3d_generation']:
            raise ValueError(Config.ERROR_MESSAGES['molecule_too_large'].format(num_atoms=num_atoms))
        
        # Preserve stereochemistry information before embedding
        stereo_info = {}
        for atom in mol_copy.GetAtoms():
            if atom.HasProp('_CIPCode'):
                stereo_info[atom.GetIdx()] = atom.GetProp('_CIPCode')
        
        # Embed 3D coordinates
        if not _embed_molecule_3d(mol_copy):
            raise ValueError(Config.ERROR_MESSAGES['structure_error'])
        
        # Restore stereochemistry information after embedding
        try:
            Chem.AssignStereochemistry(mol_copy, force=True, cleanIt=True)
            
            # Restore original CIP codes if they were preserved
            for atom_idx, cip_code in stereo_info.items():
                if atom_idx < mol_copy.GetNumAtoms():
                    atom = mol_copy.GetAtomWithIdx(atom_idx)
                    if atom.HasProp('_CIPCode'):
                        atom.SetProp('_CIPCode', cip_code)
        except Exception:
            pass  # If stereochemistry restoration fails, continue without it
        
        # Optimize the 3D structure
        _optimize_molecule_3d(mol_copy)
        
        # Convert to SDF format
        sdf_string = Chem.MolToMolBlock(mol_copy)
        if not sdf_string:
            raise ValueError(Config.ERROR_MESSAGES['structure_error'])
        
        return sdf_string
        
    except Exception as e:
        raise ValueError(f"{Config.ERROR_MESSAGES['structure_error']}: {str(e)}")

# =============================================================================
# CHAT DISPLAY FUNCTIONS
# =============================================================================

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
        "smiles": None,
        "memo": error_message,
        "mol": None,
        "mol_with_h": None,
        "properties": None,
        "cid": None
    }

def handle_error_and_show_buttons(error_message: str, button_key: str):
    """Handle error case and show appropriate buttons."""
    add_chat_message("assistant", error_message)
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
                "申し訳ありません。AIからの応答を取得できませんでした。"
            )
            st.session_state.gemini_output = error_data
            st.session_state.smiles_error_occurred = True
    except Exception as e:
        error_data = create_error_molecule_data(
            f"申し訳ありません。予期しないエラーが発生しました: {e}"
        )
        st.session_state.gemini_output = error_data
        st.session_state.smiles_error_occurred = True

def get_molecule_analysis() -> str:
    """Get molecule analysis result - always generate new analysis."""
    try:
        current_data = st.session_state.get("current_molecule_data", None)
        
        # Always generate new analysis to provide variety
        logger.info(f"Generating new analysis result for CID: {current_data['cid']}")
        detailed_info = get_detailed_molecule_info(current_data["cid"])
        if detailed_info and detailed_info.molecular_formula:
            analysis_result = analyze_molecule_properties(detailed_info, get_molecule_name())
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



def display_molecule_message(molecule_data: Dict[str, Union[str, None, Any]], with_link: bool = True) -> None:
    """Display standardized molecule recommendation message."""
    if with_link and molecule_data.get('cid'):
        message = f"あなたにオススメする分子は「 [{molecule_data['name']}](https://pubchem.ncbi.nlm.nih.gov/compound/{molecule_data['cid']}) 」だよ。{molecule_data['memo']}"
    else:
        message = f"あなたにオススメする分子は「 **{molecule_data['name']}** 」だよ。{molecule_data['memo']}"
    
    with st.chat_message("assistant"):
        st.write(message)

def add_chat_message(role: str, content: str):
    """Display a chat message without adding to history."""
    with st.chat_message(role):
        st.write(content)

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
                if st.button(query, key=f"random_query_{query}", width="stretch"):
                    logger.info(f"User selected sample: {query}")
                    st.session_state.selected_sample = query
                    st.rerun()

def show_query_response_screen():
    """Display query response screen."""
    user_query = st.session_state.get("user_query", "")
    
    with st.chat_message("user"):
        st.write(user_query)
    
    gemini_output = st.session_state.get("gemini_output", None)
    if not gemini_output or gemini_output.get("smiles") is None:
        process_molecule_query()
    
    gemini_output = st.session_state.get("gemini_output", None)
    error_occurred = st.session_state.get("smiles_error_occurred", False)
    
    if gemini_output and not error_occurred:
        output_data = gemini_output
        
        if output_data["smiles"] is None:
            st.write(output_data["memo"])
            show_action_buttons("no_molecule_found")
        else:
            st.session_state.current_molecule_data = output_data

            display_molecule_message(output_data)
                            
            if display_molecule_3d(output_data):
                show_action_buttons("main_action")
    else:
        if gemini_output:
            st.write(gemini_output["memo"])
        else:
            st.write("申し訳ありません。エラーが発生しました。")
        show_action_buttons("error_main")

def show_detail_response_screen():
    """Display detail response screen."""
    if not validate_molecule_data():
        handle_error_and_show_buttons(Config.ERROR_MESSAGES['no_data'], "no_data_error")
        return

    with st.chat_message("user"):
        st.write(f"「{get_molecule_name()}」についてもっと詳しく")

    current_data = st.session_state.get("current_molecule_data", None)
    display_molecule_3d(current_data)

    # Check if we already processed analysis for this screen
    # This prevents chain analysis when Streamlit reruns
    if not st.session_state.get("detail_analysis_executed", False):
        # Execute analysis only once per screen transition
        analysis_result = get_molecule_analysis()
        # Cache the analysis result for display on reruns
        st.session_state.cached_analysis_result = analysis_result
        with st.chat_message("assistant"):
            st.write(analysis_result)
        
        # Mark analysis as executed to prevent re-execution
        st.session_state.detail_analysis_executed = True
    else:
        # Already processed, show cached analysis result
        cached_result = st.session_state.get("cached_analysis_result", "")
        if cached_result:
            with st.chat_message("assistant"):
                st.write(cached_result)

    show_action_buttons("detail_action")

def show_similar_response_screen():
    """Display similar molecule response screen."""
    current_data = st.session_state.get("current_molecule_data", None)
    if not current_data:
        handle_error_and_show_buttons(Config.ERROR_MESSAGES['no_data'], "no_data_error")
        return
    
    with st.chat_message("user"):
        st.write(f"「{get_molecule_name()}」に関連する分子は？")
    
    # Check if we already processed similar molecules for this screen
    # This prevents chain searching when Streamlit reruns
    if not st.session_state.get("similar_search_executed", False):
        # Execute search only once per screen transition
        similar_data = find_and_process_similar_molecule()
        
        if similar_data and similar_data.get("smiles"):
            display_molecule_message(similar_data)
            
            st.session_state.current_molecule_data = similar_data
            
            if display_molecule_3d(similar_data):
                show_action_buttons("similar_main_action")
        else:
            error_message = Config.ERROR_MESSAGES['molecule_not_found']
            handle_error_and_show_buttons(error_message, "similar_error_none")
        
        # Mark search as executed to prevent re-execution
        st.session_state.similar_search_executed = True
    else:
        # Already processed, just show the current molecule
        current_data = st.session_state.get("current_molecule_data", None)
        if current_data and current_data.get("smiles"):
            display_molecule_message(current_data)
            
            if display_molecule_3d(current_data):
                show_action_buttons("similar_main_action")

# =============================================================================
# CONVERSATION FLOW HANDLERS (LEGACY - TO BE REMOVED)
# =============================================================================

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
        "smiles_error_occurred": False,
        "random_queries": [],
        "screen": "initial",
        "current_molecule_data": None,
        "similar_search_executed": False,
        "detail_analysis_executed": False,
        "cached_analysis_result": "",
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
    """Display 3D molecule structure and return True if successful."""
    try:
        # Prefer PubChem-provided 3D SDF when available
        sdf_string = None
        if molecule_data.get("cid") is not None:
            sdf_string = get_pubchem_3d_sdf_by_cid(molecule_data.get("cid"))
        # Fallback to RDKit 3D embedding
        if not sdf_string:
            sdf_string = get_molecule_structure_3d_sdf(molecule_data["mol_with_h"])
        
        if sdf_string:
            # Create 3D molecular viewer
            viewer = py3Dmol.view(width=MOLECULE_VIEWER_WIDTH, height=MOLECULE_VIEWER_HEIGHT)
            viewer.addModel(sdf_string, 'sdf')
            viewer.setStyle({'stick': {}})  # Stick representation
            viewer.setZoomLimits(Config.VIEWER['zoom_min'], Config.VIEWER['zoom_max'])  # Set zoom limits
            viewer.zoomTo()  # Auto-fit molecule
            viewer.spin('y', Config.VIEWER['rotation_speed'])  # Auto-rotate around Y-axis
            components.html(viewer._make_html(), height=MOLECULE_VIEWER_HEIGHT)
            return True
        else:
            st.write("立体構造の生成に失敗しました。")
            return False
    except Exception as e:
        st.write(f"立体構造の準備中にエラーが発生しました: {e}")
        return False

# Handle sample selection transition
if st.session_state.selected_sample:
    st.session_state.user_query = st.session_state.selected_sample
    st.session_state.selected_sample = ""  # Reset selection to prevent reuse
    st.session_state.smiles_error_occurred = False  # Reset error state
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

