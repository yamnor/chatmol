# Standard library imports
import time
from typing import Dict, List, Optional, Tuple, Union, Generator

# Third-party imports
import streamlit as st
import google.generativeai as genai
import py3Dmol

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors

# Streamlit molecular visualization
from stmol import showmol

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

PROMOTION_MESSAGES: List[Dict[str, str]] = [
    {
        "message": "10/25~26開催の「サイエンスアゴラ」に出展するよ。詳細は [こちら](https://peatix.com/event/4534946/)",
        "icon": "🎪"
    },
    {
        "message": "ChatMOLの最新情報は [GitHub](https://github.com/yamnor/ChatMOL) でチェック！",
        "icon": "📚"
    },
]

ABOUT_MESSAGE: str = """
「バラの香りってどんな分子？」そんな素朴な疑問に、AI が答えてくれるよ。

普段なにげなく感じている色・香り・味。

実はそれぞれに対応する分子があって、分子の化学的な性質が、私たちのさまざまな感覚を生み出しているんだ。

このアプリでは、AI と対話しながら様々な分子を探索して、その分子の立体的な形を眺めることができるよ。

分子の世界の面白さを体験してみよう！
"""

SYSTEM_PROMPT: str = """
# SYSTEM
あなたは「分子コンシェルジュ」です。
ユーザーが求める効能・イメージ・用途・ニーズなどを 1 文でもらったら、  
❶ それに最も関連すると考えられる既知の分子を 1 つ選び、
❷ 分子名、SMILES 文字列、ひとこと理由 を返してください。

## 重要なルール
- **必ず実在する化学物質**のみを提案してください
- SMILESは**標準的な形式（canonical SMILES）**で正確に記述してください
- 不確実な場合や適切な分子が見当たらない場合は、正直にその旨を伝えてください
- ひとこと理由は、小学生にもわかるように、1 行でフレンドリーに表現してください
- 薬理作用・香り・色など科学的根拠が薄い場合は「伝統的に～とされる」等と表現し、医学的助言は行わないでください
- SMILESは必ず化学的に正しい構造を表すものにしてください（不確実なら提案しない）

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
【SMILES】: CC(=CCCC(C)(C=C)O)C
【メモ】: ラベンダーの香気成分で、アロマテラピーで鎮静が期待されるよ。

ユーザー: バラの香りってどんな分子？
アシスタント:  
【分子】: ゲラニオール  
【SMILES】: CC(=CCCC(=CCO)C)C
【メモ】: バラの香りの主成分で、甘くフローラルな香りが特徴だよ。

ユーザー: レモンの香り成分は？
アシスタント:  
【分子】: リモネン  
【SMILES】: CC(=C)C1CCC(=CC1)C
【メモ】: 柑橘類の皮に豊富に含まれる爽やかな香りの成分だよ。

ユーザー: 甘い味の分子は？
アシスタント:  
【分子】: スクロース（砂糖）  
【SMILES】: C(C1C(C(C(C(O1)OC2(C(C(C(O2)CO)O)O)CO)O)O)O)O
【メモ】: 私たちが毎日使っているお砂糖の主成分で、強い甘味があるよ。

# END OF SYSTEM
"""

SAMPLE_QUERIES: Dict[str, List[str]] = {
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
        "赤い色の分子は？",
        "青い色の分子を教えて",
        "黄色い色の分子は？",
        "緑色の分子を教えて",
        "紫色の分子は？"
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
        "目薬の分子を教えて",
        "鎮痛剤の成分は？"
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
        "日焼け止めの成分は？"
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
        "ツンとくる刺激臭の分子は？"
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
    Stream text character by character with delay for better user experience.
    
    Args:
        text: The text to stream character by character
        
    Yields:
        str: Individual characters from the input text
        
    Note:
        Adds a 0.01 second delay between characters for visual effect
    """
    for char in text:
        yield char
        time.sleep(0.01)

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
    Send user input to Gemini AI and retrieve molecular recommendation response.
    
    This function constructs a prompt using the system prompt and user input,
    then sends it to the Gemini AI model for processing.
    
    Args:
        user_input_text: User's request for molecular properties/effects
        
    Returns:
        Gemini's response text containing molecular information, or None if error
        
    Raises:
        Displays error message to user if API request fails
        
    Example:
        >>> response = get_gemini_response("甘い香りの分子は？")
        >>> print(response)
        【分子】: バニリン
        【SMILES】: COc1ccc(C=O)cc1O
        【メモ】: バニラの香りの主成分で、甘く温かい香りが特徴だよ。
    """
    prompt = f"{SYSTEM_PROMPT}\nユーザー: {user_input_text}\nアシスタント:"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini API へのリクエスト中にエラーが発生しました: {e}")
        return None

def validate_and_normalize_smiles(smiles: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate and normalize SMILES string using RDKit with comprehensive checks.
    
    This function performs multiple validation steps:
    1. Basic syntax validation using RDKit
    2. Molecular size checks (atom count, molecular weight)
    3. Canonicalization to standard format
    
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
    
    try:
        # Try to parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, None, "無効なSMILES形式です"
        
        # Basic sanity checks
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return False, None, "原子が含まれていません"
        if num_atoms > 200:
            return False, None, f"分子が大きすぎます（原子数: {num_atoms}）"
        
        # Check molecular weight
        mol_weight = Chem.Descriptors.MolWt(mol)
        if mol_weight > 2000:
            return False, None, f"分子量が大きすぎます（{mol_weight:.1f}）"
        
        # Canonicalize SMILES
        canonical_smiles = Chem.CanonSmiles(smiles)
        
        return True, canonical_smiles, None
        
    except Exception as e:
        return False, None, f"SMILES検証中にエラー: {str(e)}"


# =============================================================================
# APPLICATION INITIALIZATION
# =============================================================================

# Display promotional toast notifications (first time only)
# This ensures users see important announcements without being intrusive
if "first_time_shown" not in st.session_state:
    # Display all promotional messages with individual icons
    for promotion in PROMOTION_MESSAGES:
        st.toast(promotion["message"], icon=promotion["icon"], duration = "infinite")
    
    # Show welcome message with streaming effect for better UX
    st.chat_message("user").write("ChatMOLとは？")
    st.chat_message("assistant").write_stream(stream_text(ABOUT_MESSAGE))

    # Mark as shown to prevent repeated display
    st.session_state.first_time_shown = True

# Configure Streamlit page settings
# These settings control the overall appearance and behavior of the app
st.set_page_config(
    page_title="ChatMOL",
    page_icon=":material/smart_toy:",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'about': "https://github.com/yamnor/ChatMOL"
    }
)

# Initialize Gemini AI API with comprehensive error handling
# This ensures the app fails gracefully if API configuration is missing
try:
    # Configure API key from Streamlit secrets
    genai.configure(api_key=st.secrets["api_key"])
    # Initialize the Gemini model with latest version
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
except KeyError:
    st.error("GEMINI_API_KEY が設定されていません。Streamlit の Secrets で設定してください。")
    st.stop()
except Exception as e:
    st.error(f"Gemini API の初期化に失敗しました: {e}")
    st.stop()

# --- Function Definitions ---

def get_gemini_response(user_input_text):
    """
    Send user input to Gemini AI and get molecular recommendation response.
    
    Args:
        user_input_text (str): User's request for molecular properties/effects
        
    Returns:
        str: Gemini's response text containing molecular information
    """

    prompt = f"{SYSTEM_PROMPT}\nユーザー: {user_input_text}\nアシスタント:"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini API へのリクエスト中にエラーが発生しました: {e}")
        return None

def validate_and_normalize_smiles(smiles):
    """
    Validate and normalize SMILES string using RDKit.
    
    Args:
        smiles (str): SMILES notation to validate
        
    Returns:
        tuple: (is_valid: bool, canonical_smiles: str or None, error_message: str or None)
    """
    if not smiles:
        return False, None, "SMILESが空です"
    
    try:
        # Try to parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, None, "無効なSMILES形式です"
        
        # Basic sanity checks
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return False, None, "原子が含まれていません"
        if num_atoms > 200:
            return False, None, f"分子が大きすぎます（原子数: {num_atoms}）"
        
        # Check molecular weight
        mol_weight = Chem.Descriptors.MolWt(mol)
        if mol_weight > 2000:
            return False, None, f"分子量が大きすぎます（{mol_weight:.1f}）"
        
        # Canonicalize SMILES
        canonical_smiles = Chem.CanonSmiles(smiles)
        
        return True, canonical_smiles, None
        
    except Exception as e:
        return False, None, f"SMILES検証中にエラー: {str(e)}"

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
        properties["solubility"] = "高い"
    elif logp < 2:
        properties["solubility"] = "中程度"
    elif logp < 4:
        properties["solubility"] = "低い"
    else:
        properties["solubility"] = "非常に低い"
    
    # Drug-likeness score
    drug_score = 0
    if mw <= 500: drug_score += 1
    if logp <= 5: drug_score += 1
    if hbd <= 5: drug_score += 1
    if hba <= 10: drug_score += 1
    if tpsa <= 140: drug_score += 1
    
    if drug_score >= 4:
        properties["drug_likeness"] = "高い"
    elif drug_score >= 3:
        properties["drug_likeness"] = "中程度"
    else:
        properties["drug_likeness"] = "低い"
    
    # Bioavailability score
    if mw <= 500 and logp <= 5 and tpsa <= 140:
        properties["bioavailability"] = "良好"
    elif mw <= 600 and logp <= 6 and tpsa <= 160:
        properties["bioavailability"] = "中程度"
    else:
        properties["bioavailability"] = "低い"

def calculate_molecular_properties(mol, mol_with_h) -> Optional[Dict[str, Union[str, int, float]]]:
    """
    Calculate and cache molecular properties with optimized error handling.
    
    Args:
        mol: RDKit molecule object
        mol_with_h: RDKit molecule object with hydrogens
        
    Returns:
        Cached molecular properties or None if calculation fails
    """
    if not mol or not mol_with_h:
        return None
    
    # Calculate basic properties
    properties = calculate_basic_properties(mol, mol_with_h)
    if not properties:
        return None
    
    # Add fraction_csp3 and derived properties
    properties["fraction_csp3"] = calculate_fraction_csp3(mol)
    calculate_derived_properties(properties)
    
    return properties

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
        data["smiles"] = None
        data["memo"] = f"申し訳ありません。提案された分子のSMILESに問題がありました（{error_msg}）。別の分子をお探ししましょうか？"
        st.warning(f"⚠️ SMILES検証エラー: {error_msg} (入力: {raw_smiles})")

def _create_molecular_objects(canonical_smiles: str, data: Dict[str, Union[str, None]]) -> None:
    """Create molecular objects and calculate properties."""
    try:
        data["mol"] = Chem.MolFromSmiles(canonical_smiles)
        data["mol_with_h"] = Chem.AddHs(data["mol"]) if data["mol"] else None
        data["properties"] = calculate_molecular_properties(data["mol"], data["mol_with_h"])
    except Exception as e:
        st.warning(f"分子オブジェクトの作成に失敗しました: {e}")
        data["mol"] = None
        data["mol_with_h"] = None
        data["properties"] = None

def get_molecule_structure_3d_sdf(mol_with_h) -> Optional[str]:
    """
    Generate 3D molecular structure from molecular object with optimized error handling.
    
    Args:
        mol_with_h: RDKit molecule object with hydrogens
        
    Returns:
        SDF format string for 3D visualization or None if failed
    """
    if not mol_with_h:
        return None
    
    return safe_calculate(
        lambda: _generate_3d_structure(mol_with_h),
        None,
        "3D立体構造の生成に失敗"
    )

def _generate_3d_structure(mol_with_h) -> str:
    """Generate 3D structure and convert to SDF format."""
    mol_copy = Chem.Mol(mol_with_h)
    AllChem.EmbedMolecule(mol_copy, AllChem.ETKDG())
    return Chem.MolToMolBlock(mol_copy)

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

# Create sidebar with sample input examples
# This provides users with inspiration and common use cases
with st.sidebar:
    st.header(":material/chat: 入力例")
        
    # Category selection with selectbox for organized sample queries
    selected_category = st.selectbox(
        "カテゴリー",
        options=list(SAMPLE_QUERIES.keys()),
        key="category_selector"
    )
    
    # Display sample queries as buttons for selected category
    # Each button triggers a sample query when clicked
    for sample in SAMPLE_QUERIES[selected_category]:
        # Create clickable sample buttons with consistent styling
        if st.button(sample, key=f"sample_{sample}", width="content", icon=":material/face:"):
            st.session_state.selected_sample = sample
            st.rerun()  # Trigger app rerun to process the sample query

# Display chat input field for user queries
# This is the primary interface for user interaction
user_input = st.chat_input("分子のイメージや求める効果を教えて", max_chars=50)

# Handle user input: either from sample selection or direct input
# This logic determines which input source to use and processes accordingly
if st.session_state.selected_sample:
    # Use selected sample query from sidebar
    user_input = st.session_state.selected_sample
    st.session_state.user_query = user_input
    st.session_state.selected_sample = ""  # Reset selection to prevent reuse
elif user_input:
    # Use direct user input from chat interface
    st.session_state.user_query = user_input

# Process user input and get AI response
# This is the core functionality of the application
if user_input:
    # Display user message in chat interface
    with st.chat_message("user"):
        st.write(user_input)

    # Get AI response with loading spinner for better UX
    with st.spinner("AI (Gemini) に問い合わせ中..."):
        response_text = get_gemini_response(user_input)
        if response_text:
            # Parse and store successful response
            st.session_state.gemini_output = parse_gemini_response(response_text)
        else:
            # Handle error case gracefully
            st.session_state.gemini_output = None

# Display AI response and molecular visualization
if st.session_state.gemini_output:
    output_data = st.session_state.gemini_output

    with st.chat_message("assistant"):
        if output_data["smiles"] is None:
            # Display error message when no molecule found
            st.write(output_data["memo"])
        else:
            # Display molecular recommendation
            st.write(f"あなたにオススメする分子は「**{output_data['name']}**」だよ。{output_data['memo']}")

            # Generate and display 3D molecular structure
            with st.spinner("3D構造を生成中..."):
                sdf_string = get_molecule_structure_3d_sdf(output_data["mol_with_h"])
            
            if sdf_string:
                # Create 3D molecular viewer
                viewer = py3Dmol.view(width=600, height=450)
                viewer.addModel(sdf_string, 'sdf')
                viewer.setStyle({'stick': {}})  # Stick representation
                viewer.setZoomLimits(0.1,100)   # Set zoom limits
                viewer.zoomTo()                 # Auto-fit molecule
                viewer.spin('y', 1)            # Auto-rotate around Y-axis
                showmol(viewer, width=600, height=450)
            else:
                st.error("⚠️ 3D立体構造の生成に失敗しました。分子構造が複雑すぎるか、立体配座の生成ができませんでした。")
            
            # Display detailed molecular properties with expander (after 3D structure)
            with st.expander("この分子の性質・特徴は？", icon=":material/info:"):
                try:
                    properties = output_data["properties"]
                    if properties:
                        # Molecular formula
                        st.write("分子式")
                        st.code(properties["formula"], language=None)

                        # Basic molecular information
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("原子数", f"{properties['num_atoms']}")
                        with col2:
                            st.metric("分子量（g/mol）", f"{properties['mol_weight']:.2f}")
                        with col3:
                            st.metric("結合数", f"{properties['num_bonds']}")
                        with col4:
                            st.metric("立体中心数", f"{properties['stereo_centers']}")
                                                
                        # Physical and chemical properties
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("脂溶性指標（LogP）", f"{properties['logp']:.2f}")
                        with col2:
                            st.metric("極性表面積（TPSA）", f"{properties['tpsa']:.1f}")
                        with col3:
                            st.metric("水素結合供与体", f"{properties['hbd']}")
                        with col4:
                            st.metric("水素結合受容体", f"{properties['hba']}")

                        # Structural features
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("芳香環数", f"{properties['aromatic_rings']}")
                        with col2:
                            st.metric("回転可能結合", f"{properties['rotatable_bonds']}")
                        with col3:
                            st.metric("立体中心数", f"{properties['stereo_centers']}")
                        with col4:
                            st.metric("sp³炭素比", f"{properties['fraction_csp3']:.2f}")
                        
                        # Solubility and drug-likeness
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("水溶性", properties["solubility"])
                        with col2:
                            st.metric("薬物類似性", properties["drug_likeness"])
                        with col3:
                            st.metric("bioavailability", properties["bioavailability"])
                        
                        # SMILES notation
                        st.write("SMILES 記法")
                        st.code(f"{output_data['smiles']}", language=None)
                    else:
                        st.warning("分子プロパティの計算に失敗しました。")
                                                
                except Exception as e:
                    st.warning(f"分子情報の取得に失敗しました: {e}")
