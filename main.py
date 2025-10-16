# Standard library imports
import random
import json
import re
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

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
    RANDOM_SAMPLE = {
        'count': 6,  # Number of random samples to display
        'columns': 3,  # Number of columns for random samples
    }
    
    # Molecular Size Limits
    MOLECULE_LIMITS = {
        'max_atoms_3d_display': 100,
        'max_atoms_3d_generation': 100,
    }
    
    # 3D Molecular Viewer Configuration
    # Responsive viewer size based on window size
    VIEWER = {
        'width_pc': 632,
        'height_pc': 400,
        'width_mobile': 280,
        'height_mobile': 200,
        'zoom_min': 0.1,
        'zoom_max': 50,
        'rotation_speed': 1,
    }
    
    # Default AI Model Configuration
    DEFAULT_MODEL_NAME = "gemini-2.5-flash-lite"
    
    # Error messages
    ERROR_MESSAGES = {
        'api_limit': "API利用制限に達しました。しばらく待ってから再試行してください。",
        'api_timeout': "API応答タイムアウト",
        'similar_search_timeout': "類似分子検索API応答タイムアウト",
        'pubchem_timeout': "PubChem API タイムアウト",
        'pubchem_3d_timeout': "PubChem 3Dデータ取得タイムアウト",
        'structure_generation_timeout': "3D構造生成タイムアウト",
        'pubchem_detailed_info_timeout': "PubChem詳細情報取得タイムアウト",
        'general_timeout': "操作がタイムアウトしました",
        'molecule_not_found': "PubChemで分子が見つかりませんでした",
        'invalid_cid': "申し訳ありません。無効なPubChem CIDが返されました。",
        'no_cid': "申し訳ありません。PubChem CIDを取得できませんでした。",
        'no_molecule_name': "申し訳ありません。分子名を取得できませんでした。",
        'parse_error': "申し訳ありません。AIからの応答を解析できませんでした。",
        'response_error': "申し訳ありません。応答の解析中にエラーが発生しました",
        'molecule_processing_error': "申し訳ありません。分子の処理中にエラーが発生しました",
        'smiles_error': "SMILESから分子オブジェクトの作成に失敗しました",
        'molecule_too_large': "分子が大きすぎます（原子数: {num_atoms}）。3D表示をスキップする可能性があります。",
        'molecule_too_large_generation': "分子が大きすぎます（原子数: {num_atoms}）。シンプルな分子を提案してください。",
        'embedding_failed': "すべての3D構造埋め込み方法が失敗しました",
        'sdf_conversion_failed': "SDF形式への変換に失敗しました",
        'structure_generation_error': "3D構造生成エラー",
        'detailed_info_error': "詳細情報を取得できませんでした。",
        'display_error': "詳細情報の表示中にエラーが発生しました",
        'similar_molecule_not_found': "申し訳ありません。類似分子を見つけることができませんでした。",
        'similar_search_error': "申し訳ありません。類似分子検索中にエラーが発生しました。",
        'no_molecule_data': "分子データが見つかりません。最初からやり直してください。",
    }

# Legacy constants for backward compatibility (will be removed after migration)
API_TIMEOUT_SECONDS = Config.TIMEOUTS['api']
STRUCTURE_GENERATION_TIMEOUT_SECONDS = Config.TIMEOUTS['structure_generation']
PUBCHEM_3D_TIMEOUT_SECONDS = Config.TIMEOUTS['pubchem_3d']
PUBCHEM_SMILES_TIMEOUT_SECONDS = Config.TIMEOUTS['pubchem_smiles']
RANDOM_SAMPLE_COUNT = Config.RANDOM_SAMPLE['count']
RANDOM_SAMPLE_COLUMNS = Config.RANDOM_SAMPLE['columns']
MAX_ATOMS_FOR_3D_DISPLAY = Config.MOLECULE_LIMITS['max_atoms_3d_display']
MAX_ATOMS_FOR_3D_GENERATION = Config.MOLECULE_LIMITS['max_atoms_3d_generation']
MOLECULE_VIEWER_WIDTH_PC = Config.VIEWER['width_pc']
MOLECULE_VIEWER_HEIGHT_PC = Config.VIEWER['height_pc']
MOLECULE_VIEWER_WIDTH_MOBILE = Config.VIEWER['width_mobile']
MOLECULE_VIEWER_HEIGHT_MOBILE = Config.VIEWER['height_mobile']
MOLECULE_VIEWER_ZOOM_MIN = Config.VIEWER['zoom_min']
MOLECULE_VIEWER_ZOOM_MAX = Config.VIEWER['zoom_max']
MOLECULE_VIEWER_ROTATION_SPEED = Config.VIEWER['rotation_speed']
DEFAULT_MODEL_NAME = Config.DEFAULT_MODEL_NAME

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

SYSTEM_PROMPT: str = """
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

```json
{
  "name": "<分子名>（見つかった分子の日本語での名称）",
  "id": "<PubChem CID>（整数値）",
  "description": "<一言の説明> （その分子を選んだ理由や性質の特徴を１行で説明）"
}
```
"""

SAMPLE_QUERIES: Dict[str, List[str]] = {
    "🎲 ランダム": [],
    "🌸 香り": [
        "良い香りのする成分は？",
        "甘い香りのする成分は？",
        "フレッシュな香りが欲しい",
        "落ち着く香りを探している",
        "スパイシーな香りが欲しい"
    ],
    "🍋 食べ物・飲み物": [
        "レモンの成分は？",
        "バニラの成分は？",
        "コーヒーの成分は？",
        "チョコレートの成分は？",
        "ミントの成分は？"
    ],
    "🌸 花・植物": [
        "バラの香り成分は？",
        "桜の香り成分は？",
        "ラベンダーの香り成分は？",
        "ジャスミンの香り成分は？",
        "金木犀の香り成分は？"
    ],
    "🎨 色・染料": [
        "リンゴの赤色の成分は？",
        "ベリーの青色の成分は？",
        "レモンの黄色の成分は？",
        "ぶどうの紫色の成分は？",
        "デニムの青色の成分は？"
    ],
    "👅 味覚": [
        "甘い味の成分は？",
        "酸っぱい味の成分は？",
        "苦い味の成分は？",
        "辛い味の成分は？",
        "うま味の成分は？"
    ],
    "💊 医薬品": [
        "風邪薬の成分は？",
        "頭痛薬の成分を教えて",
        "胃薬の成分は？",
        "インフル治療薬の成分は？",
        "抗生物質の成分は？"
    ],
    "🌲 自然・環境": [
        "森の香り成分は？",
        "海の香り成分は？",
        "土の匂い成分は？",
        "木の香り成分は？",
        "草の香り成分は？"
    ],
    "🧴 日用品": [
        "洗剤の成分は？",
        "シャンプーの成分は？",
        "石鹸の成分は？",
        "柔軟剤の成分は？",
        "消臭剤の成分は？"
    ],
    "💪 スポーツ・運動": [
        "筋肉を鍛えたい",
        "疲労を回復させたい",
        "持久力をアップさせたい",
        "瞬発力をアップさせたい",
        "エネルギーを補給したい"
    ],
    "💚 健康・体調": [
        "気分をすっきりさせたい",
        "疲れを取りたい",
        "目覚めを良くしたい",
        "免疫力を高めたい",
        "血行を良くしたい"
    ],
    "😴 リラックス・睡眠": [
        "リラックスしたい",
        "心を落ち着かせたい",
        "ゆっくり休みたい",
        "ストレスを和らげたい",
        "幸福感を感じたい"
    ],
    "🧠 集中・学習": [
        "集中力を高めたい",
        "勉強に集中したい",
        "思考力を高めたい",
        "脳を活性化したい"
    ],
    "✨ 美容・スキンケア": [
        "肌を美しく保ちたい",
        "若々しさを維持したい",
        "髪の毛を健康にしたい",
        "シミを防ぎたい",
        "肌の潤いを保ちたい"
    ]
}

# =============================================================================
# ERROR HANDLING
# =============================================================================

class ErrorHandler:
    """Unified error handling for the application."""
    
    @staticmethod
    def handle_api_error(e: Exception, operation: str = "API操作") -> str:
        """Handle API-related errors with consistent messaging."""
        error_str = str(e)
        
        if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
            return Config.ERROR_MESSAGES['api_limit']
        elif "timeout" in error_str.lower():
            return f"{operation}がタイムアウトしました。"
        else:
            return f"{operation}中にエラーが発生しました: {e}"
    
    @staticmethod
    def handle_timeout_error(timeout_seconds: int, operation: str = "操作") -> str:
        """Handle timeout errors with consistent messaging."""
        return f"⏰ {operation}がタイムアウトしました（{timeout_seconds}秒）"
    
    @staticmethod
    def handle_general_error(e: Exception, operation: str = "操作") -> str:
        """Handle general errors with consistent messaging."""
        return f"⚠️ {operation}中にエラーが発生しました: {e}"
    
    @staticmethod
    def show_error_message(message: str, error_type: str = "error") -> None:
        """Show standardized error messages."""
        if error_type == "warning":
            st.warning(f"⚠️ {message}")
        else:
            st.error(f"⚠️ {message}")
    
    @staticmethod
    def show_error_with_retry_button(message: str, error_type: str = "error") -> None:
        """Show error message with retry button."""
        ErrorHandler.show_error_message(message, error_type)
        
        # Add retry button
        st.write("---")
        if st.button("他の分子を探す", key="error_retry_button", use_container_width=True):
            reset_to_initial_state()
            st.rerun()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def execute_with_timeout(func, timeout_seconds: int, error_message: str = None):
    """Execute a function with timeout control using ThreadPoolExecutor."""
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            return future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        if error_message is None:
            error_message = Config.ERROR_MESSAGES['general_timeout']
        ErrorHandler.show_error_message(ErrorHandler.handle_timeout_error(timeout_seconds, error_message))
        return None
    except Exception as e:
        ErrorHandler.show_error_message(ErrorHandler.handle_general_error(e))
        return None

def generate_random_samples() -> List[str]:
    """Generate random samples from all categories except random category."""
    all_samples = []
    for category_name, category_samples in SAMPLE_QUERIES.items():
        if category_name != "🎲 ランダム" and category_samples:  # Skip random category and empty categories
            all_samples.extend(category_samples)
    
    if all_samples:
        return random.sample(all_samples, min(RANDOM_SAMPLE_COUNT, len(all_samples)))
    else:
        return []


# =============================================================================
# AI AND MOLECULAR PROCESSING FUNCTIONS
# =============================================================================

def get_gemini_response(user_input_text: str) -> Optional[str]:
    """Send user input to Gemini AI and retrieve molecular recommendation response."""
    prompt = f"{SYSTEM_PROMPT}\n\n# USER\n{user_input_text}"
    
    def api_call():
        """Execute API call."""
        # Google Searchツールを設定
        search_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        
        config = types.GenerateContentConfig(
            tools=[search_tool]
        )

        # モデルにツールを渡してコンテンツを生成
        return client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config
        )
    
    response = execute_with_timeout(
        api_call, 
        Config.TIMEOUTS['api'], 
        Config.ERROR_MESSAGES['api_timeout']
    )
    
    if response is None:
        return None
    
    try:
        return response.text
    except Exception as e:
        ErrorHandler.show_error_message(ErrorHandler.handle_api_error(e, "Gemini API へのリクエスト"))
        return None

def search_similar_molecules(molecule_name: str) -> Optional[str]:
    """Search for similar molecules using Gemini AI."""
    similar_prompt = f"""
# SYSTEM
あなたは「分子コンシェルジュ」です。
ユーザーが指定した分子「{molecule_name}」に似た分子を探してください。
(1) 指定された分子と類似した性質・構造・用途を持つ複数の候補分子を優先度の高い順に PubChem で検索して、
(2) 最初に見つかった分子のみについて、その分子の日本語での名称（name）、一言の説明（description）、PubChem CID (id) を、以下のルールに厳密に従い、JSON 形式でのみ出力してください。

- 分子の検索は、必ず、「 Google Search 」を用いて、PubChem のページ「 https://pubchem.ncbi.nlm.nih.gov/compound/<分子名（英語名称）> 」で行ってください
- 分子名は、必ず、英語名称で検索してください。日本語名称では検索できません。
- PubChem で分子が見つからなかった、または PubChem CID データを取得できなかった場合は、次の優先度の分子を検索します
- 該当する分子を思いつかなかった、または優先度順のすべての分子が PubChem で見つからなかった場合は、「該当なし」とのみ出力します
- ひとこと理由は、小学生にもわかるように、1 行でフレンドリーに表現してください

```json
{{
  "name": "<分子名>（見つかった分子の日本語での名称）",
  "id": "<PubChem CID>（整数値）",
  "description": "<一言の説明> （その分子を選んだ理由や性質の特徴を１行で説明）"
}}
```
"""
    
    def api_call():
        """Execute API call for similar molecule search."""
        # Google Searchツールを設定
        search_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        
        config = types.GenerateContentConfig(
            tools=[search_tool]
        )

        # モデルにツールを渡してコンテンツを生成
        return client.models.generate_content(
            model=model_name,
            contents=similar_prompt,
            config=config
        )
    
    response = execute_with_timeout(
        api_call, 
        Config.TIMEOUTS['api'], 
        Config.ERROR_MESSAGES['similar_search_timeout']
    )
    
    if response is None:
        return None
    
    try:
        return response.text
    except Exception as e:
        ErrorHandler.show_error_message(ErrorHandler.handle_api_error(e, "類似分子検索"))
        return None

def get_smiles_from_pubchem(cid: int) -> Tuple[bool, Optional[str], Optional[str]]:
    """Get SMILES string from PubChem using CID with timeout protection."""
    def fetch_from_pubchem():
        """Execute PubChem API call."""
        try:
            compound = pcp.get_compounds(cid, 'cid')[0]
            return compound.canonical_smiles
        except IndexError:
            return None
        except Exception as e:
            return None
    
    smiles = execute_with_timeout(
        fetch_from_pubchem, 
        Config.TIMEOUTS['pubchem_smiles'], 
        Config.ERROR_MESSAGES['pubchem_timeout']
    )
    
    if smiles:
        return True, smiles, None
    else:
        return False, None, Config.ERROR_MESSAGES['molecule_not_found']

def get_detailed_molecule_info(cid: int) -> Dict[str, Union[str, None]]:
    """Get detailed molecule information from PubChem using CID."""
    def fetch_detailed_info():
        """Execute PubChem API call for detailed information."""
        try:
            compound = pcp.get_compounds(cid, 'cid')[0]
            
            # Extract detailed information
            detailed_info = {
                "molecular_formula": compound.molecular_formula,
                "molecular_weight": compound.molecular_weight,
                "iupac_name": compound.iupac_name,
                "synonyms": compound.synonyms[:5] if compound.synonyms else [],  # Limit to 5 synonyms
                "description": compound.description,
                "canonical_smiles": compound.canonical_smiles,
                "isomeric_smiles": compound.isomeric_smiles,
                "inchi": compound.inchi,
                "inchi_key": compound.inchi_key,
            }
            
            return detailed_info
        except IndexError:
            return None
        except Exception as e:
            return None
    
    detailed_info = execute_with_timeout(
        fetch_detailed_info, 
        Config.TIMEOUTS['pubchem_smiles'], 
        Config.ERROR_MESSAGES['pubchem_detailed_info_timeout']
    )
    
    if detailed_info:
        return detailed_info
    else:
        return {
            "molecular_formula": None,
            "molecular_weight": None,
            "iupac_name": None,
            "synonyms": [],
            "description": None,
            "canonical_smiles": None,
            "isomeric_smiles": None,
            "inchi": None,
            "inchi_key": None,
        }


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
if WindowQueryHelper().minimum_window_size(min_width=MOLECULE_VIEWER_WIDTH_PC)["status"]:
    # PC size
    MOLECULE_VIEWER_WIDTH = MOLECULE_VIEWER_WIDTH_PC
    MOLECULE_VIEWER_HEIGHT = MOLECULE_VIEWER_HEIGHT_PC
else:
    # Mobile size
    MOLECULE_VIEWER_WIDTH = MOLECULE_VIEWER_WIDTH_MOBILE
    MOLECULE_VIEWER_HEIGHT = MOLECULE_VIEWER_HEIGHT_MOBILE

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
        model_name = DEFAULT_MODEL_NAME

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

def parse_gemini_response(response_text: str) -> Dict[str, Union[str, None]]:
    """Parse Gemini's JSON response and fetch SMILES from PubChem."""
    data = {
        "name": "分子が見つかりませんでした",
        "smiles": None,
        "memo": "申し訳ありません。ご要望に合う分子を見つけることができませんでした。もう少し具体的な情報を教えていただけますか？",
        "mol": None,
        "mol_with_h": None,
        "properties": None,
        "cid": None
    }
    
    if not response_text:
        return data
    
    try:
        # Extract JSON from response text
        json_data = _extract_json_from_response(response_text)
        
        if json_data:
            # Handle current format (name, id, description)
            molecule_name = json_data.get("name", "").strip()
            cid_value = json_data.get("id")
            description = json_data.get("description", "").strip()
            
            if molecule_name:
                data["name"] = molecule_name
                data["memo"] = description if description else "分子の詳細情報を取得中..."
                
                # If we have a CID, fetch SMILES from PubChem
                if cid_value is not None:
                    try:
                        # Convert to integer if it's a string
                        if isinstance(cid_value, str):
                            cid = int(cid_value.strip())
                        else:
                            cid = int(cid_value)
                        # Store CID for downstream 3D fetch
                        data["cid"] = cid
                        
                        success, pubchem_smiles, error_msg = get_smiles_from_pubchem(cid)
                        
                        if success and pubchem_smiles:
                            data["smiles"] = pubchem_smiles
                            _create_molecular_objects(pubchem_smiles, data)
                        else:
                            data["memo"] = f"申し訳ありません。PubChemから分子データを取得できませんでした（{error_msg}）。"
                            
                    except (ValueError, TypeError):
                        data["memo"] = Config.ERROR_MESSAGES['invalid_cid']
                else:
                    data["memo"] = Config.ERROR_MESSAGES['no_cid']
            else:
                data["memo"] = Config.ERROR_MESSAGES['no_molecule_name']
        else:
            data["memo"] = Config.ERROR_MESSAGES['parse_error']
            
    except Exception as e:
        data["memo"] = f"{Config.ERROR_MESSAGES['response_error']}: {e}"
    
    return data

def _extract_json_from_response(response_text: str) -> Optional[Dict]:
    """Extract JSON data from Gemini response text."""
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


def _create_molecular_objects(canonical_smiles: str, data: Dict[str, Union[str, None]]) -> None:
    """Create molecular objects and calculate properties with enhanced error handling."""
    try:
        # Create molecular object (PubChem SMILESは既に検証済み)
        data["mol"] = Chem.MolFromSmiles(canonical_smiles)
        if data["mol"] is None:
            raise ValueError(Config.ERROR_MESSAGES['smiles_error'])
        
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
        data["memo"] = f"{Config.ERROR_MESSAGES['molecule_processing_error']}（{str(e)}）。別の分子をお探ししましょうか？"
        
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
        Config.ERROR_MESSAGES['pubchem_3d_timeout']
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
        Config.ERROR_MESSAGES['structure_generation_timeout']
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
            raise ValueError(Config.ERROR_MESSAGES['molecule_too_large_generation'].format(num_atoms=num_atoms))
        
        # Preserve stereochemistry information before embedding
        stereo_info = {}
        for atom in mol_copy.GetAtoms():
            if atom.HasProp('_CIPCode'):
                stereo_info[atom.GetIdx()] = atom.GetProp('_CIPCode')
        
        # Embed 3D coordinates
        if not _embed_molecule_3d(mol_copy):
            raise ValueError(Config.ERROR_MESSAGES['embedding_failed'])
        
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
            raise ValueError(Config.ERROR_MESSAGES['sdf_conversion_failed'])
        
        return sdf_string
        
    except Exception as e:
        raise ValueError(f"{Config.ERROR_MESSAGES['structure_generation_error']}: {str(e)}")

# =============================================================================
# MAIN APPLICATION LOGIC
# =============================================================================

# Initialize session state variables for maintaining app state across reruns
# This ensures the app remembers user interactions and AI responses
def initialize_session_state():
    """Initialize all session state variables in one place for better maintainability."""
    defaults = {
        "user_query": "",
        "gemini_output": None,
        "selected_sample": "",
        "smiles_error_occurred": False,
        "random_samples": [],
        # New conversation flow state management
        "conversation_state": "initial",  # "initial", "molecule_displayed", "detail_view", "similar_search"
        "current_molecule_data": None,  # Store current molecule data for detailed view
        "similar_molecules": [],  # Store similar molecule search results
        "chat_history": [],  # Store chat messages for conversation flow
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Validate state consistency
    validate_session_state()

def validate_session_state():
    """Validate session state consistency and fix any inconsistencies."""
    # If conversation_state is initial but we have a user_query, clear it
    if st.session_state.conversation_state == "initial" and st.session_state.user_query:
        st.session_state.user_query = ""
    
    # If conversation_state is initial but we have gemini_output, clear it
    if st.session_state.conversation_state == "initial" and st.session_state.gemini_output:
        st.session_state.gemini_output = None
    
    # If conversation_state is initial but we have current_molecule_data, clear it
    if st.session_state.conversation_state == "initial" and st.session_state.current_molecule_data:
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

def reset_to_initial_state():
    """Reset the application to initial state."""
    st.session_state.user_query = ""
    # Keep gemini_output to prevent unnecessary re-processing
    # st.session_state.gemini_output = None  # Commented out to prevent re-processing
    st.session_state.selected_sample = ""
    st.session_state.smiles_error_occurred = False
    st.session_state.conversation_state = "initial"
    st.session_state.current_molecule_data = None
    st.session_state.similar_molecules = []
    st.session_state.chat_history = []
    st.session_state.random_samples = generate_random_samples()

def display_molecule_3d(molecule_data: Dict) -> bool:
    """Display 3D molecule structure and return True if successful."""
    try:
        with st.spinner("3D構造を生成中..."):
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
            viewer.setZoomLimits(MOLECULE_VIEWER_ZOOM_MIN, MOLECULE_VIEWER_ZOOM_MAX)  # Set zoom limits
            viewer.zoomTo()  # Auto-fit molecule
            viewer.spin('y', MOLECULE_VIEWER_ROTATION_SPEED)  # Auto-rotate around Y-axis
            components.html(viewer._make_html(), height=MOLECULE_VIEWER_HEIGHT)
            return True
        else:
            st.write("3D立体構造の生成に失敗しました。分子構造が複雑すぎるか、立体配座の生成ができませんでした。")
            return False
    except Exception as e:
        st.write(f"3D構造生成中にエラーが発生しました: {e}")
        return False

def display_detailed_info(cid: int):
    """Display detailed molecule information."""
    try:
        with st.spinner("詳細情報を取得中..."):
            detailed_info = get_detailed_molecule_info(cid)
        
        if detailed_info and detailed_info.get("molecular_formula"):
            st.write("### 📊 詳細情報")
            
            # Basic information
            col1, col2 = st.columns(2)
            with col1:
                if detailed_info.get("molecular_formula"):
                    st.write(f"**分子式:** {detailed_info['molecular_formula']}")
                if detailed_info.get("molecular_weight"):
                    st.write(f"**分子量:** {detailed_info['molecular_weight']:.2f}")
            
            with col2:
                if detailed_info.get("iupac_name"):
                    st.write(f"**IUPAC名:** {detailed_info['iupac_name']}")
            
            # Synonyms
            if detailed_info.get("synonyms"):
                st.write("**別名:**")
                for synonym in detailed_info["synonyms"][:5]:  # Show first 5 synonyms
                    st.write(f"- {synonym}")
            
            # Description
            if detailed_info.get("description"):
                st.write("**説明:**")
                st.write(detailed_info["description"])
            
            # Chemical identifiers
            st.write("### 🧪 化学識別子")
            col1, col2 = st.columns(2)
            with col1:
                if detailed_info.get("canonical_smiles"):
                    st.write(f"**SMILES:** `{detailed_info['canonical_smiles']}`")
            with col2:
                if detailed_info.get("inchi_key"):
                    st.write(f"**InChI Key:** `{detailed_info['inchi_key']}`")
        else:
            st.write(Config.ERROR_MESSAGES['detailed_info_error'])
    except Exception as e:
        st.write(f"{Config.ERROR_MESSAGES['display_error']}: {e}")
        st.write(Config.ERROR_MESSAGES['detailed_info_error'])

# Main conversation flow
if st.session_state.conversation_state == "initial":
    # Initial state: Show sample queries
    with st.chat_message("assistant"):
        st.write("何かお手伝いできますか？")
    
    if not st.session_state.random_samples:
        st.session_state.random_samples = generate_random_samples()
    
    # Display random samples in 3 columns
    if st.session_state.random_samples:
        cols = st.columns(RANDOM_SAMPLE_COLUMNS)
        for i, sample in enumerate(st.session_state.random_samples):
            col_idx = i % RANDOM_SAMPLE_COLUMNS
            with cols[col_idx]:
                if st.button(sample, key=f"random_sample_{sample}", width="stretch"):
                    st.session_state.selected_sample = sample
                    st.rerun()
    
    # Refresh button
    if st.button("", key="new_random_samples", width="stretch", icon=":material/refresh:", type="tertiary"):
        st.session_state.random_samples = generate_random_samples()
        st.rerun()
    
    # Handle sample selection
    if st.session_state.selected_sample:
        st.session_state.user_query = st.session_state.selected_sample
        st.session_state.selected_sample = ""  # Reset selection to prevent reuse
        st.session_state.smiles_error_occurred = False  # Reset error state
        st.session_state.conversation_state = "molecule_displayed"
        st.rerun()

elif st.session_state.conversation_state == "molecule_displayed":
    # Handle error case first
    if st.session_state.smiles_error_occurred:
        with st.chat_message("user"):
            st.write(st.session_state.user_query)
        
        with st.chat_message("assistant"):
            if st.session_state.gemini_output:
                st.write(st.session_state.gemini_output["memo"])
            else:
                st.write("申し訳ありません。エラーが発生しました。")
            
            # Show retry button for error cases
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("他の分子を探す", key="error_retry_main", use_container_width=True):
                    reset_to_initial_state()
                    st.rerun()
    
    # Display user query and get AI response
    elif st.session_state.user_query and not st.session_state.smiles_error_occurred:
        # Only process if we don't already have a valid response
        if not st.session_state.gemini_output or st.session_state.gemini_output.get("smiles") is None:
            with st.chat_message("user"):
                st.write(st.session_state.user_query)
            
            with st.spinner(f"AI (`{model_name}`) に問い合わせ中..."):
                try:
                    response_text = get_gemini_response(st.session_state.user_query)
                    if response_text:
                        # Parse and store successful response
                        st.session_state.gemini_output = parse_gemini_response(response_text)
                    else:
                        # Handle error case gracefully
                        st.session_state.gemini_output = {
                            "name": "エラーが発生しました",
                            "smiles": None,
                            "memo": "申し訳ありません。AIからの応答を取得できませんでした。",
                            "mol": None,
                            "mol_with_h": None,
                            "properties": None,
                            "cid": None
                        }
                        st.session_state.smiles_error_occurred = True
                        
                except Exception as e:
                    st.session_state.gemini_output = {
                        "name": "エラーが発生しました",
                        "smiles": None,
                        "memo": f"申し訳ありません。予期しないエラーが発生しました: {e}",
                        "mol": None,
                        "mol_with_h": None,
                        "properties": None,
                        "cid": None
                    }
                    st.session_state.smiles_error_occurred = True
        else:
            # Display the user query if we already have a response
            with st.chat_message("user"):
                st.write(st.session_state.user_query)
        
        # Process AI response
        if st.session_state.gemini_output and not st.session_state.smiles_error_occurred:
            output_data = st.session_state.gemini_output
            
            with st.chat_message("assistant"):
                if output_data["smiles"] is None:
                    st.write(output_data["memo"])
                    # Add "Search Another Molecule" button when no molecule is found
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("他の分子を探す", key="search_another_when_none_found", use_container_width=True):
                            reset_to_initial_state()
                            st.rerun()
                else:
                    st.write(f"あなたにオススメする分子は「 **{output_data['name']}** 」だよ。{output_data['memo']}")
                    
                    # Store current molecule data
                    st.session_state.current_molecule_data = output_data
                    
                    # Display 3D structure
                    if display_molecule_3d(output_data):
                        # Action buttons after molecule display
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button("詳しく知りたい", key="detail_button", use_container_width=True):
                                st.session_state.conversation_state = "detail_view"
                                st.rerun()
                                                
                        with col2:
                            if st.button("似た分子を探す", key="similar_button", use_container_width=True):
                                st.session_state.conversation_state = "similar_search"
                                st.rerun()

                        with col3:
                            if st.button("他の分子を探す", key="new_molecule_button", use_container_width=True):
                                reset_to_initial_state()
                                st.rerun()

elif st.session_state.conversation_state == "detail_view":
    # Display detailed information
    if st.session_state.current_molecule_data and st.session_state.current_molecule_data.get("cid"):
        with st.chat_message("assistant"):
            display_detailed_info(st.session_state.current_molecule_data["cid"])

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("他の分子を探す", key="new_molecule_from_detail", use_container_width=True):
                    reset_to_initial_state()
                    st.rerun()

elif st.session_state.conversation_state == "similar_search":
    # Search for similar molecules
    if st.session_state.current_molecule_data:
        molecule_name = st.session_state.current_molecule_data.get("name", "")
        
        with st.chat_message("assistant"):
            st.write(f"「{molecule_name}」に似た分子を探しています...")
        
        with st.spinner("類似分子を検索中..."):
            try:
                similar_response = search_similar_molecules(molecule_name)
                if similar_response:
                    # Parse similar molecule response
                    similar_data = parse_gemini_response(similar_response)
                    if similar_data and similar_data.get("smiles"):
                        st.session_state.current_molecule_data = similar_data
                        st.session_state.conversation_state = "molecule_displayed"
                        st.rerun()
                    else:
                        st.write(Config.ERROR_MESSAGES['similar_molecule_not_found'])
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("他の分子を探す", key="error_retry_similar_none", use_container_width=True):
                                reset_to_initial_state()
                                st.rerun()
                else:
                    st.write(Config.ERROR_MESSAGES['similar_search_error'])
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("他の分子を探す", key="error_retry_similar_error", use_container_width=True):
                            reset_to_initial_state()
                            st.rerun()
            except Exception as e:
                st.write(f"{Config.ERROR_MESSAGES['similar_search_error']}: {e}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("他の分子を探す", key="error_retry_similar", use_container_width=True):
                        reset_to_initial_state()
                        st.rerun()
    else:
        # Handle case where current_molecule_data is None
        st.write(Config.ERROR_MESSAGES['no_molecule_data'])
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("他の分子を探す", key="error_retry_no_data", use_container_width=True):
                reset_to_initial_state()
                st.rerun()

