# Standard library imports
import time
import random
import json
import re
from typing import Dict, List, Optional, Tuple, Union, Generator
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Third-party imports
import streamlit as st
import streamlit.components.v1 as components

# import google.generativeai as genai
# from google.genai import types

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

# Timeout settings for preventing freezes
API_TIMEOUT_SECONDS = 30  # Gemini API timeout
STRUCTURE_GENERATION_TIMEOUT_SECONDS = 15  # 3D structure generation timeout

# Molecular Size Limits
MAX_ATOMS_FOR_3D_DISPLAY = 100
MAX_ATOMS_FOR_3D_GENERATION = 100
MAX_SMILES_LENGTH = 200

# 3D Molecular Viewer Configuration
# Responsive viewer size based on window size
MOLECULE_VIEWER_WIDTH_PC = 632
MOLECULE_VIEWER_HEIGHT_PC = 400
MOLECULE_VIEWER_WIDTH_MOBILE = 280
MOLECULE_VIEWER_HEIGHT_MOBILE = 200
MOLECULE_VIEWER_ZOOM_MIN = 0.1
MOLECULE_VIEWER_ZOOM_MAX = 50
MOLECULE_VIEWER_ROTATION_SPEED = 1

# User Input Configuration
CHAT_INPUT_PLACEHOLDER = "どんな分子を探しているの？"
CHAT_INPUT_MAX_CHARS = 25

# Default AI Model Configuration
DEFAULT_MODEL_NAME = "gemini-2.5-flash-lite"

ABOUT_MESSAGE: str = """
「チョコレートの成分は？」「肌を美しく保ちたい」「スパイシーな香りが欲しい」、そんな質問・疑問・要望に応えてくれる AI 分子コンシェルジェだよ。

AI と対話しながら、分子の世界を探索してみよう！

:material/warning: 注意： 出力される分子の情報は、正しくない・間違っていることがあります。
"""

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
 ❶ それに最も関連すると考える複数の候補分子を優先度の高い順に PubChem で検索して、
 ❷ 最初に見つかった分子のみについて、その分子の日本語での名称（name）、一言の説明（description）、PubChem CID (id) を、以下のルールに厳密に従い、JSON 形式でのみ出力してください。

- 分子の検索は、必ず、「 Google Search 」を用いて「 PubChem 」で行ってください。
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
# UTILITY FUNCTIONS
# =============================================================================

def generate_random_samples() -> List[str]:
    """Generate random samples from all categories except random category."""
    all_samples = []
    for category_name, category_samples in SAMPLE_QUERIES.items():
        if category_name != "🎲 ランダム" and category_samples:  # Skip random category and empty categories
            all_samples.extend(category_samples)
    
    if all_samples:
        return random.sample(all_samples, min(5, len(all_samples)))
    else:
        return []

def stream_text(text: str) -> Generator[str, None, None]:
    """Stream text character by character with adaptive delay."""
    # Adaptive delay based on text length
    delay = 0.01 if len(text) < 100 else 0.005 if len(text) < 500 else 0.002
    
    for char in text:
        yield char
        time.sleep(delay)

# =============================================================================
# AI AND MOLECULAR PROCESSING FUNCTIONS
# =============================================================================

def get_gemini_response(user_input_text: str) -> Optional[str]:
    """Send user input to Gemini AI and retrieve molecular recommendation response."""
    prompt = f"{SYSTEM_PROMPT}\n\n# USER\n{user_input_text}"
    
    def api_call():
        """Execute API call in separate thread for timeout control."""
        # Google Searchツールを設定
        search_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        
        config = types.GenerateContentConfig(
            tools=[search_tool]
        )

        # モデルにツールを渡してコンテンツを生成
        return client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt,
            config=config
        )
    
    try:
        # Use ThreadPoolExecutor for timeout control
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(api_call)
            response = future.result(timeout=API_TIMEOUT_SECONDS)
            return response.text
            
    except FutureTimeoutError:
        st.error(f"⏰ API応答タイムアウト（{API_TIMEOUT_SECONDS}秒）")
        return None
        
    except Exception as e:
        error_str = str(e)
        
        # Check for rate limit error (429)
        if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
            st.error("⏰ API利用制限に達しました。しばらく待ってから再試行してください。")
        else:
            st.error(f"Gemini API へのリクエスト中にエラーが発生しました: {e}")
        return None

def get_smiles_from_pubchem(cid: int) -> Tuple[bool, Optional[str], Optional[str]]:
    """Get SMILES string from PubChem using CID with timeout protection."""
    def fetch_from_pubchem():
        """Execute PubChem API call in separate thread for timeout control."""
        try:
            compound = pcp.get_compounds(cid, 'cid')[0]
            return compound.canonical_smiles
        except IndexError:
            return None
        except Exception as e:
            return None
    
    try:
        # Use ThreadPoolExecutor for timeout control
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fetch_from_pubchem)
            smiles = future.result(timeout=10)  # 10秒タイムアウト
            
            if smiles:
                return True, smiles, None
            else:
                return False, None, "PubChemで分子が見つかりませんでした"
                
    except FutureTimeoutError:
        return False, None, "PubChem API タイムアウト（10秒）"
    except Exception as e:
        return False, None, f"PubChem API エラー: {str(e)}"

def validate_and_normalize_smiles(smiles: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """簡素化されたSMILES検証（PubChemから取得したSMILESは基本的に有効）"""
    if not smiles:
        return False, None, "SMILESが空です"
    
    # 基本的な長さチェックのみ
    if len(smiles) > MAX_SMILES_LENGTH:
        return False, None, f"SMILES文字列が長すぎます（{len(smiles)}文字）"
    
    # PubChemから取得したSMILESは信頼性が高いため、基本的なRDKit検証のみ
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, None, "無効なSMILES形式です"
        
        # PubChemから取得したSMILESは既に正規化されているため、そのまま使用
        return True, smiles, None
        
    except Exception as e:
        return False, None, f"SMILES検証エラー: {str(e)}"

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
        "properties": None
    }
    
    if not response_text:
        return data
    
    try:
        # Extract JSON from response text
        json_data = _extract_json_from_response(response_text)
        
        if json_data:
            # Handle both new format (name, id, description) and old format (molecule_name, smiles, memo)
            molecule_name = json_data.get("name") or json_data.get("molecule_name", "").strip()
            cid_value = json_data.get("id")
            description = json_data.get("description") or json_data.get("memo", "").strip()
            smiles = json_data.get("smiles", "").strip()
            
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
                        
                        success, pubchem_smiles, error_msg = get_smiles_from_pubchem(cid)
                        
                        if success and pubchem_smiles:
                            data["smiles"] = pubchem_smiles
                            _create_molecular_objects(pubchem_smiles, data)
                        else:
                            data["memo"] = f"申し訳ありません。PubChemから分子データを取得できませんでした（{error_msg}）。"
                            
                    except (ValueError, TypeError):
                        data["memo"] = "申し訳ありません。無効なPubChem CIDが返されました。"
                
                # Fallback to direct SMILES if available (for backward compatibility)
                elif smiles:
                    _process_smiles_data(smiles, data)
                else:
                    data["memo"] = "申し訳ありません。PubChem CIDまたはSMILESを取得できませんでした。"
            else:
                data["memo"] = "申し訳ありません。分子名を取得できませんでした。"
        else:
            # Fallback to text parsing if JSON extraction fails
            _fallback_text_parsing(response_text, data)
            
    except Exception as e:
        st.warning(f"応答の解析中にエラーが発生しました: {e}")
        # Try fallback text parsing
        _fallback_text_parsing(response_text, data)
    
    return data

def _extract_json_from_response(response_text: str) -> Optional[Dict]:
    """Extract JSON data from Gemini response text."""
    try:
        # Look for JSON code blocks first
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response_text, re.DOTALL)
        
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        
        # Look for JSON with new format (name, id, description)
        json_pattern = r'(\{[^{}]*"name"[^{}]*"id"[^{}]*"description"[^{}]*\})'
        match = re.search(json_pattern, response_text, re.DOTALL)
        
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        
        # Try to find any JSON object containing name
        json_pattern = r'(\{[^{}]*"name"[^{}]*\})'
        match = re.search(json_pattern, response_text, re.DOTALL)
        
        if match:
            json_str = match.group(1)
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and "name" in parsed:
                return parsed
        
        # Try to find any JSON object in the response
        json_pattern = r'(\{[^{}]*\})'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        for json_str in matches:
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and ("name" in parsed or "molecule_name" in parsed):
                    return parsed
            except json.JSONDecodeError:
                continue
            
        return None
        
    except json.JSONDecodeError as e:
        st.warning(f"JSON解析エラー: {e}")
        return None
    except Exception as e:
        st.warning(f"JSON抽出エラー: {e}")
        return None

def _fallback_text_parsing(response_text: str, data: Dict[str, Union[str, None]]) -> None:
    """Fallback to original text parsing method if JSON parsing fails."""
    try:
        _parse_response_lines(response_text, data)
    except Exception as e:
        st.warning(f"フォールバック解析中にエラーが発生しました: {e}")

def _parse_response_lines(response_text: str, data: Dict[str, Union[str, None]]) -> None:
    """Parse individual lines of the response (fallback method)."""
    for line in response_text.split('\n'):
        if line.startswith("【分子】:"):
            data["name"] = line.split(":", 1)[1].strip()
        elif line.startswith("【SMILES】:"):
            raw_smiles = line.split(":", 1)[1].strip()
            _process_smiles_data(raw_smiles, data)
        elif line.startswith("【メモ】:"):
            if data["smiles"] is not None:
                data["memo"] = line.split(":", 1)[1].strip()

def _process_smiles_data(smiles: str, data: Dict[str, Union[str, None]]) -> None:
    """Process SMILES data and create molecular objects."""
    is_valid, canonical_smiles, error_msg = validate_and_normalize_smiles(smiles)
    
    if is_valid:
        data["smiles"] = canonical_smiles
        _create_molecular_objects(canonical_smiles, data)
    else:
        # Clear all molecular data to prevent further processing
        data["smiles"] = None
        data["mol"] = None
        data["mol_with_h"] = None
        data["properties"] = None
        data["memo"] = f"申し訳ありません。分子データの処理に問題がありました（{error_msg}）。別の分子をお探ししましょうか？"
        
        # Set session state to prevent further processing
        if "smiles_error_occurred" not in st.session_state:
            st.session_state.smiles_error_occurred = True

def _create_molecular_objects(canonical_smiles: str, data: Dict[str, Union[str, None]]) -> None:
    """Create molecular objects and calculate properties with enhanced error handling."""
    try:
        # Create molecular object (PubChem SMILESは既に検証済み)
        data["mol"] = Chem.MolFromSmiles(canonical_smiles)
        if data["mol"] is None:
            raise ValueError("SMILESから分子オブジェクトの作成に失敗しました")
        
        # Check molecule complexity before adding hydrogens
        num_atoms = data["mol"].GetNumAtoms()
        if num_atoms > MAX_ATOMS_FOR_3D_DISPLAY:
            st.warning(f"分子が大きすぎます（原子数: {num_atoms}）。3D表示をスキップする可能性があります。")
        
        # Add hydrogens
        data["mol_with_h"] = Chem.AddHs(data["mol"])
        
        # Set properties to None since we're not calculating them
        data["properties"] = None
    except Exception as e:
        # Clear all molecular data and set error state
        st.error(f"⚠️ 分子オブジェクトの作成に失敗しました: {e}")
        data["mol"] = None
        data["mol_with_h"] = None
        data["properties"] = None
        data["smiles"] = None
        data["memo"] = f"申し訳ありません。分子の処理中にエラーが発生しました（{str(e)}）。別の分子をお探ししましょうか？"
        
        # Set error state to prevent further processing
        st.session_state.smiles_error_occurred = True

def get_molecule_structure_3d_sdf(mol_with_h) -> Optional[str]:
    """Generate 3D molecular structure from molecular object with timeout protection."""
    if not mol_with_h:
        return None
    
    def generate_3d_structure():
        """Execute 3D structure generation in separate thread for timeout control."""
        return _generate_3d_structure(mol_with_h)
    
    try:
        # Use ThreadPoolExecutor for timeout control
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(generate_3d_structure)
            return future.result(timeout=STRUCTURE_GENERATION_TIMEOUT_SECONDS)
            
    except FutureTimeoutError:
        st.error(f"⏰ 3D構造生成タイムアウト（{STRUCTURE_GENERATION_TIMEOUT_SECONDS}秒）")
        return None
        
    except Exception as e:
        st.error(f"⚠️ 3D立体構造の生成に失敗しました: {e}")
        return None

def _generate_3d_structure(mol_with_h) -> str:
    """Generate 3D structure and convert to SDF format."""
    try:
        # Create a copy to avoid modifying the original molecule
        mol_copy = Chem.Mol(mol_with_h)
        if mol_copy is None:
            raise ValueError("分子オブジェクトのコピーに失敗しました")
        
        # Check molecule complexity before embedding
        num_atoms = mol_copy.GetNumAtoms()
        if num_atoms > MAX_ATOMS_FOR_3D_GENERATION:
            raise ValueError(f"分子が大きすぎます（原子数: {num_atoms}）。シンプルな分子を提案してください。")
        
        # Preserve stereochemistry information before embedding
        stereo_info = {}
        for atom in mol_copy.GetAtoms():
            if atom.HasProp('_CIPCode'):
                stereo_info[atom.GetIdx()] = atom.GetProp('_CIPCode')
        
        # Try multiple embedding methods with stereochemistry preservation
        embed_methods = [
            (AllChem.ETKDG(), "ETKDG"),
            (AllChem.ETKDGv2(), "ETKDGv2"),
            (AllChem.ETKDGv3(), "ETKDGv3"),
            (AllChem.UFFOptimizeMolecule, "UFF")
        ]
        
        embed_success = False
        for method, method_name in embed_methods:
            try:
                if method_name == "UFF":
                    # UFF is an optimization method, not embedding
                    continue
                
                embed_result = AllChem.EmbedMolecule(mol_copy, method)
                if embed_result == 0:
                    embed_success = True
                    break
            except Exception:
                continue
        
        if not embed_success:
            raise ValueError("すべての3D構造埋め込み方法が失敗しました")
        
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
            # If stereochemistry restoration fails, continue without it
            pass
        
        # Optimize the 3D structure while preserving stereochemistry
        try:
            AllChem.MMFFOptimizeMolecule(mol_copy)
        except Exception:
            # If MMFF optimization fails, try UFF
            try:
                AllChem.UFFOptimizeMolecule(mol_copy)
            except Exception:
                # If both optimizations fail, continue without optimization
                pass
        
        # Convert to SDF format with stereochemistry information
        sdf_string = Chem.MolToMolBlock(mol_copy)
        if not sdf_string:
            raise ValueError("SDF形式への変換に失敗しました")
        
        return sdf_string
        
    except Exception as e:
        raise ValueError(f"3D構造生成エラー: {str(e)}")

# =============================================================================
# MAIN APPLICATION LOGIC
# =============================================================================

# Initialize session state variables for maintaining app state across reruns
# This ensures the app remembers user interactions and AI responses
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "gemini_output" not in st.session_state:
    st.session_state.gemini_output = None
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = ""
if "smiles_error_occurred" not in st.session_state:
    st.session_state.smiles_error_occurred = False
if "random_samples" not in st.session_state:
    st.session_state.random_samples = []
if "current_category" not in st.session_state:
    st.session_state.current_category = ""
if "announcement_visible" not in st.session_state:
    st.session_state.announcement_visible = True

# Create sidebar with sample input examples
# This provides users with inspiration and common use cases
with st.sidebar:
    st.logo("images/logo.png", size="large")
    st.header("入力例")
        
    # Category selection with selectbox for organized sample queries
    selected_category = st.selectbox(
        "カテゴリー",
        options=list(SAMPLE_QUERIES.keys()),
        key="category_selector"
    )
    
    # Check if category has changed and generate new random samples if needed
    if selected_category == "🎲 ランダム":
        if (st.session_state.current_category != selected_category or 
            not st.session_state.random_samples):
            # Generate new random samples when switching to random category
            st.session_state.random_samples = generate_random_samples()
        
        # Update current category
        st.session_state.current_category = selected_category
        
        # Display the stored random samples
        for sample in st.session_state.random_samples:
            # Create clickable sample buttons with consistent styling
            if st.button(sample, key=f"random_sample_{sample}", width="stretch"):
                st.session_state.selected_sample = sample
                st.rerun()  # Trigger app rerun to process the sample query
        
        # Add button to generate new random samples
        if st.button("", key="new_random_samples", width="stretch", icon=":material/refresh:", type="tertiary"):
            # Generate new random samples
            st.session_state.random_samples = generate_random_samples()
            st.rerun()

    else:
        # For other categories, clear random samples and display samples normally
        if st.session_state.current_category == "🎲 ランダム":
            st.session_state.random_samples = []
        
        # Update current category
        st.session_state.current_category = selected_category
        
        for sample in SAMPLE_QUERIES[selected_category]:
            # Create clickable sample buttons with consistent styling
            if st.button(sample, key=f"sample_{sample}", width="stretch"):
                st.session_state.selected_sample = sample
                st.rerun()  # Trigger app rerun to process the sample query

    # Promotion message
    st.divider()
    if st.checkbox("お知らせを表示", value=st.session_state.announcement_visible, key="announcement_checkbox") and ANNOUNCEMENT_MESSAGE:
        st.session_state.announcement_visible = True
        st.write(ANNOUNCEMENT_MESSAGE)
    else:
        st.session_state.announcement_visible = False


# Display chat input field for user queries
# This is the primary interface for user interaction
user_input = st.chat_input(CHAT_INPUT_PLACEHOLDER, max_chars=CHAT_INPUT_MAX_CHARS)

# Display promotional toast notifications (first time only)
# This ensures users see important announcements without being intrusive
if "first_time_shown" not in st.session_state:
    # Show welcome message with streaming effect for better UX
    st.chat_message("user").write("ChatMOLとは？")
    st.chat_message("assistant").write_stream(stream_text(ABOUT_MESSAGE))

    # Mark as shown to prevent repeated display
    st.session_state.first_time_shown = True

# Handle user input: either from sample selection or direct input
# This logic determines which input source to use and processes accordingly
if st.session_state.selected_sample:
    # Use selected sample query from sidebar
    user_input = st.session_state.selected_sample
    st.session_state.user_query = user_input
    st.session_state.selected_sample = ""  # Reset selection to prevent reuse
    st.session_state.smiles_error_occurred = False  # Reset error state
elif user_input:
    # Use direct user input from chat interface
    st.session_state.user_query = user_input
    st.session_state.smiles_error_occurred = False  # Reset error state

# Process user input and get AI response
# This is the core functionality of the application
if user_input and not st.session_state.smiles_error_occurred:
    # Display user message in chat interface
    with st.chat_message("user"):
        st.write(user_input)

    # Get AI response with loading spinner
    with st.spinner(f"AI (`{model_name}`) に問い合わせ中..."):
        try:
            response_text = get_gemini_response(user_input)
            if response_text:
                # Parse and store successful response
                st.session_state.gemini_output = parse_gemini_response(response_text)
            else:
                # Handle error case gracefully
                st.session_state.gemini_output = None
                
        except Exception as e:
            st.error(f"予期しないエラーが発生しました: {e}")
            st.session_state.gemini_output = None

# Display AI response and molecular visualization
if st.session_state.gemini_output and not st.session_state.smiles_error_occurred:
    output_data = st.session_state.gemini_output

    with st.chat_message("assistant"):
        if output_data["smiles"] is None:
            # Display error message when no molecule found
            st.write(output_data["memo"])
        else:
            # Display molecular recommendation
            st.write(f"あなたにオススメする分子は「 **{output_data['name']}** 」だよ。{output_data['memo']}")

            # Generate and display 3D molecular structure
            with st.spinner("3D構造を生成中..."):
                try:
                    sdf_string = get_molecule_structure_3d_sdf(output_data["mol_with_h"])
                except Exception as e:
                    st.error(f"3D構造生成中にエラーが発生しました: {e}")
                    sdf_string = None
            
            if sdf_string:
                # Create 3D molecular viewer
                viewer = py3Dmol.view(width=MOLECULE_VIEWER_WIDTH, height=MOLECULE_VIEWER_HEIGHT)
                viewer.addModel(sdf_string, 'sdf')
                viewer.setStyle({'stick': {}})  # Stick representation
                viewer.setZoomLimits(MOLECULE_VIEWER_ZOOM_MIN, MOLECULE_VIEWER_ZOOM_MAX)  # Set zoom limits
                viewer.zoomTo()  # Auto-fit molecule
                viewer.spin('y', MOLECULE_VIEWER_ROTATION_SPEED)  # Auto-rotate around Y-axis
                components.html(viewer._make_html(), height=MOLECULE_VIEWER_HEIGHT)
            else:
                st.error("⚠️ 3D立体構造の生成に失敗しました。分子構造が複雑すぎるか、立体配座の生成ができませんでした。")

