# Standard library imports
import time
import signal
import threading
import random
from typing import Dict, List, Optional, Tuple, Union, Generator
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Third-party imports
import streamlit as st
import streamlit.components.v1 as components

import google.generativeai as genai
import py3Dmol

from st_screen_stats import WindowQueryHelper

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

PROMOTION_MESSAGES: List[Dict[str, str]] = [
    { "message": "デモモード。サービス全体で可能なリクエスト数は「15 回 / 分」まで。", "icon": ":material/timer:", "duration": "short" },
    { "message": "出力される分子の情報や構造について、正しくないことがあります。", "icon": ":material/warning:", "duration": "short" },
    { "message": "10/25~26開催の「サイエンスアゴラ」に出展するよ。詳細は **[こちら](https://yamlab.jp/sciago2025)**", "icon": ":material/festival:", "duration": "infinite" },
]

# Gemini AI Configuration
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"
GEMINI_API_KEY_SECRET = "api_key"

# Timeout settings for preventing freezes
API_TIMEOUT_SECONDS = 30  # Gemini API timeout
STRUCTURE_GENERATION_TIMEOUT_SECONDS = 15  # 3D structure generation timeout
SMILES_VALIDATION_TIMEOUT_SECONDS = 5  # SMILES validation timeout
MOLECULAR_PROPERTY_CALCULATION_TIMEOUT_SECONDS = 10  # Property calculation timeout
MOLECULAR_OBJECT_CREATION_TIMEOUT_SECONDS = 15  # Molecular object creation timeout

# Molecular Size Limits
MAX_ATOMS_FOR_SIMPLE_MOLECULE = 75
MAX_ATOMS_FOR_PROPERTY_CALCULATION = 200
MAX_ATOMS_FOR_3D_DISPLAY = 100
MAX_ATOMS_FOR_3D_GENERATION = 100
MAX_MOLECULAR_WEIGHT = 1000
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
CHAT_INPUT_PLACEHOLDER = "分子のイメージや求める効果を教えて"
CHAT_INPUT_MAX_CHARS = 25

# Error Messages
API_TIMEOUT_ERROR_MESSAGE = """
⏰ **API応答タイムアウト**

Gemini APIからの応答が{timeout_seconds}秒以内に得られませんでした。

**対処法：**
- ネットワーク接続を確認してください
- しばらく待ってから再度お試しください
- より短い質問に変更してみてください

ご不便をおかけして申し訳ありません 🙏
"""

API_RATE_LIMIT_ERROR_MESSAGE = """
⏰ **APIの利用制限に達しました**

現在、APIの利用制限（15回/分）に達しているため、しばらくお待ちください。

**対処法：**
- 約10秒〜1分程度お待ちいただいてから、再度お試しください
- デモモードでは1分間に15回までリクエスト可能です

ご不便をおかけして申し訳ありません 🙏
"""

STRUCTURE_GENERATION_TIMEOUT_ERROR_MESSAGE = """
⏰ **3D構造生成タイムアウト**

3D立体構造の生成が{timeout_seconds}秒以内に完了しませんでした。

**対処法：**
- よりシンプルな分子を試してみてください
- 分子が複雑すぎる可能性があります
- ページを再読み込みして再度お試しください

ご不便をおかけして申し訳ありません 🙏
"""

ABOUT_MESSAGE: str = """
「バラの香りってどんな分子？」そんな素朴な疑問に、AI が答えてくれるよ。

普段なにげなく感じている色・香り・味。

実はそれぞれに対応する分子があって、分子の化学的な性質が、私たちのさまざまな感覚を生み出しているんだ。

このアプリでは、AI と対話しながら様々な分子を探索して、その分子の立体的な形を眺めることができるよ。

分子の世界の面白さを体験してみよう！
"""

MENU_ITEMS: Dict[str, str] = {
    'About' : f'''
            **ChatMOL** was created by [yamnor](https://yamnor.me),
            a chemist 🧪 specializing in molecular simulation 🖥️ living in Japan 🇯🇵.

            If you have any questions, thoughts, or comments,
            feel free to [contact me](https://letterbird.co/yamnor) ✉️
            or find me on [X (Twitter)](https://x.com/yamnor) 🐦.
            ''',
    'Issues' : 'https://github.com/yamnor/chatmol/issues',
}

SYSTEM_PROMPT: str = """
# SYSTEM
あなたは「分子コンシェルジュ」です。
ユーザーが求める効能・イメージ・用途・ニーズなどを 1 文でもらったら、  
❶ それに最も関連すると考えられる既知の分子を 1 つ選び、
❷ 分子名、SMILES 文字列、ひとこと理由 を返してください。

## 重要なルール
- **必ず実在する化学物質**のみを提案してください
- SMILESは**標準的な形式（canonical SMILES）**で正確に記述してください
- **立体化学情報を含む場合は、正確な立体化学記述子（@, @@, /, \）を使用してください**
- **SMILESは必ず短く、シンプルな構造の分子のみ**を提案してください（原子数50以下を推奨）
- **複雑な高分子や長い鎖状構造は避けてください**
- 不確実な場合や適切な分子が見当たらない場合は、正直にその旨を伝えてください
- ひとこと理由は、小学生にもわかるように、1 行でフレンドリーに表現してください
- 薬理作用・香り・色など科学的根拠が薄い場合は「伝統的に～とされる」等と表現し、医学的助言は行わないでください
- SMILESは必ず化学的に正しい構造を表すものにしてください（不確実なら提案しない）
- **立体異性体が存在する場合は、最も一般的な立体異性体を提案してください**

出力フォーマット：

【分子】: <分子名>  
【SMILES】: <SMILES 文字列>  
【メモ】: <選んだ理由を 1 行で>

# EXAMPLES
ユーザー: 気分をすっきりさせたい  
アシスタント:  
【分子】: カフェイン  
【SMILES】: CN1C=NC2=C1C(=O)N(C(=O)N2C)C
【メモ】: 中枢神経を刺激して覚醒感を高める代表的なアルカロイドだよ。

ユーザー: リラックスして眠りやすくなるものは？  
アシスタント:  
【分子】: リナロール  
【SMILES】: CC(O)(C=C)CCC=C(C)C
【メモ】: ラベンダーの香気成分で、アロマテラピーで鎮静が期待されるよ。

ユーザー: バラの香りってどんな分子？
アシスタント:  
【分子】: ゲラニオール  
【SMILES】: CC(C)=CCC/C(C)=C/CO
【メモ】: バラの香りの主成分で、甘くフローラルな香りが特徴だよ。

ユーザー: レモンの香り成分は？
アシスタント:  
【分子】: リモネン  
【SMILES】: CC1=CCC(CC1)C(=C)C
【メモ】: 柑橘類の皮に豊富に含まれる爽やかな香りの成分だよ。

ユーザー: 甘い味の分子は？
アシスタント:  
【分子】: スクロース
【SMILES】: O1[C@H](CO)[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O[C@@]2(O[C@@H]([C@@H](O)[C@@H]2O)CO)CO
【メモ】: 私たちが毎日使っているお砂糖の主成分で、強い甘味があるよ。

ユーザー：疲労回復に良い分子は？
アシスタント:  
【分子】: クエン酸
【SMILES】: OC(=O)CC(O)(C(=O)O)CC(=O)O
【メモ】: レモンなどの柑橘類に多く含まれていて、疲労回復に効果的だよ。

# END OF SYSTEM
"""

SAMPLE_QUERIES: Dict[str, List[str]] = {
    "🎲 ランダム": [],
    "🌸 香り": [
        "良い香りのする分子は？",
        "甘い香りのする分子は？",
        "フレッシュな香りが欲しい",
        "落ち着く香りを探している",
        "スパイシーな香り成分は？"
    ],
    "🍋 食べ物・飲み物": [
        "レモンの香り成分は？",
        "バニラの香り分子を教えて",
        "コーヒーの香り成分は？",
        "チョコレートの香り分子は？",
        "ミントの香り成分は？"
    ],
    "🌸 花・植物": [
        "バラの香り成分は？",
        "桜の香り分子を教えて",
        "ラベンダーの香り成分は？",
        "ジャスミンの香り分子は？",
        "キンモクセイの香り成分は？"
    ],
    "🎨 色・染料": [
        "リンゴの赤い色の分子は？",
        "ブルーベリーの青い色の分子は？",
        "レモンの黄色い色の分子は？",
        "ぶどうの紫色の分子は？",
        "デニムの青い色の分子は？"
    ],
    "👅 味覚": [
        "甘い味の分子は？",
        "酸っぱい味の分子を教えて",
        "苦い味の分子は？",
        "辛い味の分子を教えて",
        "うま味の分子は？"
    ],
    "💊 医薬品": [
        "風邪薬の成分は？",
        "頭痛薬の分子を教えて",
        "胃薬の成分は？",
        "インフル治療薬の成分は？",
        "抗生物質の成分は？"
    ],
    "🌲 自然・環境": [
        "森の香り成分は？",
        "海の香り分子を教えて",
        "土の匂い成分は？",
        "木の香り分子は？",
        "草の香り成分は？"
    ],
    "🧴 日用品": [
        "洗剤の香り成分は？",
        "シャンプーの香り分子は？",
        "石鹸の香り成分は？",
        "柔軟剤の香り分子は？",
        "消臭剤の成分は？"
    ],
    "💪 スポーツ・運動": [
        "筋肉に良い分子は？",
        "疲労回復の成分を教えて",
        "持久力アップの分子は？",
        "運動後の回復に良い成分は？",
        "エネルギー補給の分子は？"
    ],
    "💚 健康・体調": [
        "気分をすっきりさせたい",
        "疲れを取って元気になりたい",
        "朝の目覚めを良くしたい",
        "免疫力を高めたい",
        "血行を良くしたい"
    ],
    "😴 リラックス・睡眠": [
        "リラックスして眠りたい",
        "心を落ち着かせたい",
        "ゆっくり休みたい",
        "ストレスを和らげたい",
        "幸福感を感じる分子は？"
    ],
    "🧠 集中・学習": [
        "集中力を高めたい",
        "勉強に集中したい",
        "記憶力を良くしたい",
        "エナジードリンクの成分は？",
        "スッキリした香りの分子は？"
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

def stream_text(text: str) -> Generator[str, None, None]:
    """
    Stream text character by character with optimized delay for better user experience.
    
    Args:
        text: The text to stream character by character
        
    Yields:
        str: Individual characters from the input text
        
    Note:
        Uses adaptive delay based on text length to prevent long waits
    """
    # Adaptive delay based on text length
    delay = 0.01 if len(text) < 100 else 0.005 if len(text) < 500 else 0.002
    
    for char in text:
        yield char
        time.sleep(delay)

def safe_calculate(calculation_func, default_value=None, error_message: Optional[str] = None):
    """
    Safely execute a calculation function with comprehensive error handling.
    
    Args:
        calculation_func: Function to execute safely
        default_value: Value to return if calculation fails (default: None)
        error_message: Optional error message to display to user
        
    Returns:
        Result of calculation_func() or default_value if exception occurs
        
    Note:
        Displays warning message to user if error_message is provided
    """
    try:
        return calculation_func()
    except Exception as e:
        if error_message:
            st.warning(f"{error_message}: {e}")
        return default_value

def safe_descriptor_calculation(mol, descriptor_func, default_value: Union[int, float] = 0) -> Union[int, float]:
    """
    Safely calculate RDKit molecular descriptor with fallback value.
    
    Args:
        mol: RDKit molecule object
        descriptor_func: RDKit descriptor function to apply
        default_value: Value to return if calculation fails (default: 0)
        
    Returns:
        Calculated descriptor value or default_value if exception occurs
        
    Note:
        Silently handles exceptions to prevent application crashes
    """
    try:
        return descriptor_func(mol)
    except Exception:
        return default_value

# =============================================================================
# AI AND MOLECULAR PROCESSING FUNCTIONS
# =============================================================================

def get_gemini_response(user_input_text: str) -> Optional[str]:
    """
    Send user input to Gemini AI and retrieve molecular recommendation response with timeout protection.
    
    This function constructs a prompt using the system prompt and user input,
    then sends it to the Gemini AI model for processing with timeout protection.
    
    Args:
        user_input_text: User's request for molecular properties/effects
        
    Returns:
        Gemini's response text containing molecular information, or None if error
        
    Raises:
        Displays error message to user if API request fails or times out
        
    Example:
        >>> response = get_gemini_response("甘い香りの分子は？")
        >>> print(response)
        【分子】: バニリン
        【SMILES】: COc1ccc(C=O)cc1O
        【メモ】: バニラの香りの主成分で、甘く温かい香りが特徴だよ。
    """
    prompt = f"{SYSTEM_PROMPT}\nユーザー: {user_input_text}\nアシスタント:"
    
    def api_call():
        """Execute API call in separate thread for timeout control."""
        return model.generate_content(prompt)
    
    try:
        # Use ThreadPoolExecutor for timeout control
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(api_call)
            response = future.result(timeout=API_TIMEOUT_SECONDS)
            return response.text
            
    except FutureTimeoutError:
        st.error(API_TIMEOUT_ERROR_MESSAGE.format(timeout_seconds=API_TIMEOUT_SECONDS))
        return None
        
    except Exception as e:
        error_str = str(e)
        
        # Check for rate limit error (429)
        if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
            st.error(API_RATE_LIMIT_ERROR_MESSAGE)
        else:
            st.error(f"Gemini API へのリクエスト中にエラーが発生しました: {e}")
        return None

def validate_and_normalize_smiles(smiles: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate and normalize SMILES string using RDKit with comprehensive checks and timeout protection.
    
    This function performs multiple validation steps:
    1. Basic syntax validation using RDKit
    2. Molecular size checks (atom count, molecular weight)
    3. Stereochemistry validation and analysis
    4. Canonicalization to standard format with stereochemistry preservation
    
    Args:
        smiles: SMILES notation string to validate
        
    Returns:
        Tuple containing:
        - is_valid (bool): Whether SMILES is valid
        - canonical_smiles (str|None): Canonicalized SMILES or None if invalid
        - error_message (str|None): Error description or None if valid
        
    Example:
        >>> is_valid, canonical, error = validate_and_normalize_smiles("CCO")
        >>> print(is_valid, canonical)
        True CCO
        
        >>> is_valid, canonical, error = validate_and_normalize_smiles("invalid")
        >>> print(is_valid, error)
        False 無効なSMILES形式です
    """
    if not smiles:
        return False, None, "SMILESが空です"
    
    # Pre-validation: Check SMILES length to prevent extremely long strings
    if len(smiles) > MAX_SMILES_LENGTH:
        return False, None, f"SMILES文字列が長すぎます（{len(smiles)}文字）。シンプルな分子を提案してください。"
    
    def validate_smiles():
        """Execute SMILES validation in separate thread for timeout control."""
        try:
            # Try to parse SMILES with timeout protection
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False, None, "無効なSMILES形式です"
            
            # Basic sanity checks
            num_atoms = mol.GetNumAtoms()
            if num_atoms == 0:
                return False, None, "原子が含まれていません"
            if num_atoms > MAX_ATOMS_FOR_SIMPLE_MOLECULE:
                return False, None, f"分子が大きすぎます（原子数: {num_atoms}）。シンプルな分子を提案してください。"
            
            # Check molecular weight
            mol_weight = Chem.Descriptors.MolWt(mol)
            if mol_weight > MAX_MOLECULAR_WEIGHT:
                return False, None, f"分子量が大きすぎます（{mol_weight:.1f}）。シンプルな分子を提案してください。"
            
            # Stereochemistry validation and analysis
            try:
                # Check for stereochemistry information
                stereo_centers = Descriptors.NumStereocenters(mol) if hasattr(Descriptors, 'NumStereocenters') else 0
                stereo_bonds = sum(1 for bond in mol.GetBonds() if bond.GetStereo() != Chem.BondStereo.STEREONONE)
                
                # Validate stereochemistry if present
                if stereo_centers > 0 or stereo_bonds > 0:
                    # Try to assign stereochemistry to validate it
                    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
                    
                    # Check if stereochemistry assignment was successful
                    assigned_stereo = sum(1 for atom in mol.GetAtoms() 
                                        if atom.HasProp('_CIPCode') and atom.GetProp('_CIPCode') != '')
                    
                    if stereo_centers > 0 and assigned_stereo == 0:
                        return False, None, f"立体中心の立体化学情報が不完全です（{stereo_centers}個の立体中心が検出されましたが、立体化学が指定されていません）"
            except Exception as stereo_error:
                return False, None, f"立体化学検証エラー: {str(stereo_error)}"
            
            # Canonicalize SMILES with stereochemistry preservation
            try:
                # Use MolToSmiles with stereochemistry flags for better preservation
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                if not canonical_smiles:
                    return False, None, "SMILESの正規化に失敗しました"
            except Exception as canon_error:
                return False, None, f"SMILES正規化エラー: {str(canon_error)}"
            
            return True, canonical_smiles, None
            
        except Exception as e:
            # Catch all RDKit parsing errors
            error_msg = str(e)
            if "extra open parentheses" in error_msg:
                return False, None, "SMILESの括弧の対応が取れていません。複雑すぎる分子です。"
            elif "parsing" in error_msg.lower():
                return False, None, f"SMILES解析エラー: {error_msg}"
            else:
                return False, None, f"SMILES検証中にエラー: {error_msg}"
    
    try:
        # Use ThreadPoolExecutor for timeout control
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(validate_smiles)
            return future.result(timeout=SMILES_VALIDATION_TIMEOUT_SECONDS)
            
    except FutureTimeoutError:
        return False, None, f"SMILES検証が{SMILES_VALIDATION_TIMEOUT_SECONDS}秒以内に完了しませんでした。複雑すぎる分子の可能性があります。"

# =============================================================================
# APPLICATION INITIALIZATION
# =============================================================================

# Configure Streamlit page settings
# These settings control the overall appearance and behavior of the app
st.set_page_config(
    page_title="ChatMOL",
    page_icon=":material/smart_toy:",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'About': MENU_ITEMS['About'],
        'Report a bug': MENU_ITEMS['Issues']
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

# Add Plausible analytics tracking
# This enables traffic analytics for the application
plausible_domain = st.secrets.get("plausible_domain", "")
if plausible_domain:
    st.html(f"""
    <script defer data-domain="{plausible_domain}" src="https://plausible.io/js/script.js"></script>
    """)

# Initialize Gemini AI API with comprehensive error handling
# This ensures the app fails gracefully if API configuration is missing
try:
    # Configure API key from Streamlit secrets
    genai.configure(api_key=st.secrets[GEMINI_API_KEY_SECRET])
    # Initialize the Gemini model with latest version
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
except KeyError:
    st.error("GEMINI_API_KEY が設定されていません。Streamlit の Secrets で設定してください。")
    st.stop()
except Exception as e:
    st.error(f"Gemini API の初期化に失敗しました: {e}")
    st.stop()

# =============================================================================
# MOLECULAR PROPERTY CALCULATION FUNCTIONS
# =============================================================================

def calculate_basic_properties(mol, mol_with_h) -> Optional[Dict[str, Union[str, int, float]]]:
    """Calculate basic molecular properties with optimized error handling."""
    if not mol or not mol_with_h:
        return None
    
    # Use safe calculation functions for all properties
    properties = {
        "formula": safe_calculate(
            lambda: rdMolDescriptors.CalcMolFormula(mol),
            "Unknown",
            "分子式の計算に失敗"
        ),
        "num_atoms": mol_with_h.GetNumAtoms(),
        "num_bonds": mol_with_h.GetNumBonds(),
        "mol_weight": safe_descriptor_calculation(mol, Descriptors.MolWt, 0.0),
        "logp": safe_descriptor_calculation(mol, Crippen.MolLogP, 0.0),
        "tpsa": safe_descriptor_calculation(mol, Descriptors.TPSA, 0.0),
        "hbd": safe_descriptor_calculation(mol, Descriptors.NumHDonors, 0),
        "hba": safe_descriptor_calculation(mol, Descriptors.NumHAcceptors, 0),
        "aromatic_rings": safe_descriptor_calculation(mol, Descriptors.NumAromaticRings, 0),
        "rotatable_bonds": safe_descriptor_calculation(mol, Descriptors.NumRotatableBonds, 0),
    }
    
    # Calculate stereo centers with optimized fallback
    properties["stereo_centers"] = safe_calculate(
        lambda: Descriptors.NumStereocenters(mol) if hasattr(Descriptors, 'NumStereocenters') 
                else sum(1 for atom in mol.GetAtoms() if atom.HasProp('_ChiralityPossible')),
        0,
        "立体中心数の計算に失敗"
    )
    
    return properties

def calculate_fraction_csp3(mol) -> float:
    """Calculate sp3 carbon ratio with optimized error handling."""
    return safe_calculate(
        lambda: (
            rdMolDescriptors.FractionCsp3(mol) if hasattr(rdMolDescriptors, 'FractionCsp3')
            else Descriptors.FractionCsp3(mol) if hasattr(Descriptors, 'FractionCsp3')
            else _manual_fraction_csp3(mol)
        ),
        0,
        "sp3炭素比の計算に失敗"
    )

def _manual_fraction_csp3(mol) -> float:
    """Manual calculation of sp3 carbon ratio."""
    csp3_count = 0
    total_carbons = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:  # Carbon
            total_carbons += 1
            if atom.GetHybridization() == Chem.HybridizationType.SP3:
                csp3_count += 1
    return csp3_count / total_carbons if total_carbons > 0 else 0

def calculate_derived_properties(properties: Dict[str, Union[str, int, float]]) -> None:
    """Calculate derived properties like solubility and drug-likeness."""
    mw = properties["mol_weight"]
    logp = properties["logp"]
    hbd = properties["hbd"]
    hba = properties["hba"]
    tpsa = properties["tpsa"]
    
    # Solubility estimation
    if logp < 0:
        properties["solubility"] = "💧💧💧"
    elif logp < 2:
        properties["solubility"] = "💧💧"
    elif logp < 4:
        properties["solubility"] = "💧"
    else:
        properties["solubility"] = "❌"
    
    # Drug-likeness score
    drug_score = 0
    if mw <= 500: drug_score += 1
    if logp <= 5: drug_score += 1
    if hbd <= 5: drug_score += 1
    if hba <= 10: drug_score += 1
    if tpsa <= 140: drug_score += 1
    
    if drug_score >= 4:
        properties["drug_likeness"] = "💊💊💊"
    elif drug_score >= 3:
        properties["drug_likeness"] = "💊💊"
    else:
        properties["drug_likeness"] = "💊"
    
    # Bioavailability score
    if mw <= 500 and logp <= 5 and tpsa <= 140:
        properties["bioavailability"] = "🍪🍪🍪"
    elif mw <= 600 and logp <= 6 and tpsa <= 160:
        properties["bioavailability"] = "🍪🍪"
    else:
        properties["bioavailability"] = "🍪"

def calculate_molecular_properties(mol, mol_with_h) -> Optional[Dict[str, Union[str, int, float]]]:
    """
    Calculate and cache molecular properties with optimized error handling and memory management.
    
    Args:
        mol: RDKit molecule object
        mol_with_h: RDKit molecule object with hydrogens
        
    Returns:
        Cached molecular properties or None if calculation fails
        
    Note:
        Includes memory usage checks and timeout protection for large molecules
    """
    if not mol or not mol_with_h:
        return None
    
    # Check molecule size to prevent memory issues
    num_atoms = mol_with_h.GetNumAtoms()
    if num_atoms > MAX_ATOMS_FOR_PROPERTY_CALCULATION:
        st.warning(f"分子が大きすぎます（原子数: {num_atoms}）。プロパティ計算をスキップします。")
        return None
    
    try:
        # Calculate basic properties with timeout protection
        def calculate_properties():
            properties = calculate_basic_properties(mol, mol_with_h)
            if not properties:
                return None
            
            # Add fraction_csp3 and derived properties
            properties["fraction_csp3"] = calculate_fraction_csp3(mol)
            calculate_derived_properties(properties)
            
            return properties
        
        # Use ThreadPoolExecutor for timeout control
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(calculate_properties)
            return future.result(timeout=MOLECULAR_PROPERTY_CALCULATION_TIMEOUT_SECONDS)
            
    except FutureTimeoutError:
        st.warning("分子プロパティの計算がタイムアウトしました。分子が複雑すぎる可能性があります。")
        return None
    except Exception as e:
        st.warning(f"分子プロパティの計算中にエラーが発生しました: {e}")
        return None

# =============================================================================
# RESPONSE PARSING AND VISUALIZATION FUNCTIONS
# =============================================================================

def parse_gemini_response(response_text: str) -> Dict[str, Union[str, None]]:
    """
    Parse Gemini's response text to extract molecular information with optimized error handling.
    
    Args:
        response_text: Raw response from Gemini AI
        
    Returns:
        Parsed data containing name, SMILES, memo, and molecular object
    """
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
        _parse_response_lines(response_text, data)
    except Exception as e:
        st.warning(f"応答の解析中にエラーが発生しました: {e}")
    
    return data

def _parse_response_lines(response_text: str, data: Dict[str, Union[str, None]]) -> None:
    """Parse individual lines of the response."""
    for line in response_text.split('\n'):
        if line.startswith("【分子】:"):
            data["name"] = line.split(":", 1)[1].strip()
        elif line.startswith("【SMILES】:"):
            _process_smiles_line(line, data)
        elif line.startswith("【メモ】:"):
            if data["smiles"] is not None:
                data["memo"] = line.split(":", 1)[1].strip()

def _process_smiles_line(line: str, data: Dict[str, Union[str, None]]) -> None:
    """Process SMILES line and create molecular objects."""
    raw_smiles = line.split(":", 1)[1].strip()
    is_valid, canonical_smiles, error_msg = validate_and_normalize_smiles(raw_smiles)
    
    if is_valid:
        data["smiles"] = canonical_smiles
        _create_molecular_objects(canonical_smiles, data)
    else:
        # Clear all molecular data to prevent further processing
        data["smiles"] = None
        data["mol"] = None
        data["mol_with_h"] = None
        data["properties"] = None
        data["memo"] = f"申し訳ありません。提案された分子のSMILESに問題がありました（{error_msg}）。別の分子をお探ししましょうか？"
        
        # Show error message and stop processing
        st.error(f"⚠️ SMILES検証エラー: {error_msg}")
        st.error(f"無効なSMILES: {raw_smiles[:100]}{'...' if len(raw_smiles) > 100 else ''}")
        
        # Set session state to prevent further processing
        if "smiles_error_occurred" not in st.session_state:
            st.session_state.smiles_error_occurred = True

def _create_molecular_objects(canonical_smiles: str, data: Dict[str, Union[str, None]]) -> None:
    """Create molecular objects and calculate properties with enhanced error handling."""
    try:
        # Additional validation before creating molecular objects
        if not canonical_smiles or len(canonical_smiles) > MAX_SMILES_LENGTH:
            raise ValueError("SMILES文字列が無効または長すぎます")
        
        # Create molecular object with additional error handling
        data["mol"] = Chem.MolFromSmiles(canonical_smiles)
        if data["mol"] is None:
            raise ValueError("SMILESから分子オブジェクトの作成に失敗しました")
        
        # Check molecule complexity before adding hydrogens
        num_atoms = data["mol"].GetNumAtoms()
        if num_atoms > MAX_ATOMS_FOR_3D_DISPLAY:
            st.warning(f"分子が大きすぎます（原子数: {num_atoms}）。3D表示をスキップする可能性があります。")
        
        # Add hydrogens and calculate properties
        data["mol_with_h"] = Chem.AddHs(data["mol"])
        
        # Calculate properties with timeout protection
        def calculate_props():
            return calculate_molecular_properties(data["mol"], data["mol_with_h"])
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(calculate_props)
            data["properties"] = future.result(timeout=MOLECULAR_OBJECT_CREATION_TIMEOUT_SECONDS)
        
    except FutureTimeoutError:
        st.warning("分子プロパティの計算がタイムアウトしました。基本的な情報のみ表示します。")
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
    """
    Generate 3D molecular structure from molecular object with timeout protection and optimized error handling.
    
    Args:
        mol_with_h: RDKit molecule object with hydrogens
        
    Returns:
        SDF format string for 3D visualization or None if failed
        
    Note:
        Uses timeout protection to prevent freezes during 3D structure generation
    """
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
        st.error(STRUCTURE_GENERATION_TIMEOUT_ERROR_MESSAGE.format(timeout_seconds=STRUCTURE_GENERATION_TIMEOUT_SECONDS))
        return None
        
    except Exception as e:
        st.error(f"⚠️ 3D立体構造の生成に失敗しました: {e}")
        return None

def _generate_3d_structure(mol_with_h) -> str:
    """Generate 3D structure and convert to SDF format with enhanced error handling and stereochemistry preservation."""
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

# Create sidebar with sample input examples
# This provides users with inspiration and common use cases
with st.sidebar:
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
            all_samples = []
            for category_name, category_samples in SAMPLE_QUERIES.items():
                if category_name != "🎲 ランダム" and category_samples:  # Skip random category and empty categories
                    all_samples.extend(category_samples)
            
            # Select 5 random samples and store in session state
            if all_samples:
                st.session_state.random_samples = random.sample(all_samples, min(5, len(all_samples)))
            else:
                st.session_state.random_samples = []
        
        # Update current category
        st.session_state.current_category = selected_category
        
        # Display the stored random samples
        for sample in st.session_state.random_samples:
            # Create clickable sample buttons with consistent styling
            if st.button(sample, key=f"random_sample_{sample}", width="content"):
                st.session_state.selected_sample = sample
                st.rerun()  # Trigger app rerun to process the sample query
        
        # Add button to generate new random samples
        if st.button("", key="new_random_samples", width="stretch", icon=":material/refresh:", type="tertiary"):
            # Generate new random samples
            all_samples = []
            for category_name, category_samples in SAMPLE_QUERIES.items():
                if category_name != "🎲 ランダム" and category_samples:  # Skip random category and empty categories
                    all_samples.extend(category_samples)
            
            if all_samples:
                st.session_state.random_samples = random.sample(all_samples, min(5, len(all_samples)))
            else:
                st.session_state.random_samples = []
            st.rerun()
    else:
        # For other categories, clear random samples and display samples normally
        if st.session_state.current_category == "🎲 ランダム":
            st.session_state.random_samples = []
        
        # Update current category
        st.session_state.current_category = selected_category
        
        for sample in SAMPLE_QUERIES[selected_category]:
            # Create clickable sample buttons with consistent styling
            if st.button(sample, key=f"sample_{sample}", width="content"):
                st.session_state.selected_sample = sample
                st.rerun()  # Trigger app rerun to process the sample query

# Display chat input field for user queries
# This is the primary interface for user interaction
user_input = st.chat_input(CHAT_INPUT_PLACEHOLDER, max_chars=CHAT_INPUT_MAX_CHARS)

# Display promotional toast notifications (first time only)
# This ensures users see important announcements without being intrusive
if "first_time_shown" not in st.session_state:
    # Display all promotional messages with individual icons
    for promotion in PROMOTION_MESSAGES:
        st.toast(promotion["message"], icon=promotion["icon"], duration = promotion["duration"])
    
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
    with st.spinner("AI (Gemini) に問い合わせ中..."):
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

    # Display detailed molecular properties with expander (outside chat_message)
    if st.session_state.gemini_output and st.session_state.gemini_output["smiles"] is not None and not st.session_state.smiles_error_occurred:
        with st.popover("", icon=":material/info:", width="stretch"):
            try:
                properties = output_data["properties"]
                if properties:

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        # Molecular formula
                        st.caption("分子式")
                        st.code(properties["formula"], language=None)
                    with col2:
                        # SMILES notation
                        st.caption("SMILES 記法")
                        st.code(f"{output_data['smiles']}", language=None)


                    # Basic molecular information
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("原子数", f"{properties['num_atoms']}")
                    with col2:
                        st.metric("分子量（g/mol）", f"{properties['mol_weight']:.2f}")
                    with col3:
                        st.metric("結合数", f"{properties['num_bonds']}")
                                            
                    # Physical and chemical properties
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("LogP", f"{properties['logp']:.2f}")
                    with col2:
                        st.metric("tPSA", f"{properties['tpsa']:.1f}")
                    with col3:
                        st.metric("sp³炭素比", f"{properties['fraction_csp3']:.2f}")

                    # Structural features
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("芳香環数", f"{properties['aromatic_rings']}")
                    with col2:
                        st.metric("回転可能結合", f"{properties['rotatable_bonds']}")
                    with col3:
                        st.metric("立体中心数", f"{properties['stereo_centers']}")
                    
                    # Solubility and drug-likeness
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("水溶性", properties["solubility"])
                    with col2:
                        st.metric("薬物類似性", properties["drug_likeness"])
                    with col3:
                        st.metric("生物学的利用能", properties["bioavailability"])
                    
                else:
                    st.warning("分子プロパティの計算に失敗しました。")
                                            
            except Exception as e:
                st.warning(f"分子情報の取得に失敗しました: {e}")
