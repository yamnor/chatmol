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

# AI Prompts Configuration
class AIPrompts:
    """AI prompts for different molecular operations."""
    
    # Molecular search prompt
    MOLECULAR_SEARCH: str = """
あなたは化学分野の専門家として、学術的正確性と学習効果を維持しつつ、分子についての説明を提供する専門家です。

ユーザーが求める効能・イメージ・用途・ニーズ「{user_input}」に対して、
最も関連する・関係がありそうだ・適していると考えられる分子「1 個」のみ、下記の手順に従って、最終案を出力してください。

# 手順（１）: 候補とする分子の提案
 
ユーザーが求める効能・イメージ・用途・ニーズ「{user_input}」と「分子 / 化合物 / 化学物質 / 化学」の関連を、
「Google Search」を使用して、検索してください。

検索結果から、それに最も関連する・関係がありそうだ・適していると考えられる候補の分子を「1 個」のみ、提案してください。

提案する分子の名称は、一般的な分類名（例：「脂肪酸塩」「アルカロイド」）ではなく、
具体的な化合物名（例：「ステアリン酸ナトリウム」「カフェイン」）を選んでください。

# 手順（２）: 情報収集

手順（１）で提案した分子、および、ユーザーが求める効能・イメージ・用途・ニーズ「{user_input}」について、
以下の優先順位で「 Google Search 」を使用して、情報を収集してください:

1. **Google Scholar (https://scholar.google.com/)** - 学術論文・研究の検索
2. **PubMed (https://pubmed.ncbi.nlm.nih.gov/)** - 医学・生命科学分野の論文データベース  
3. **J-STAGE (https://www.jstage.jst.go.jp/)** - 日本の学術論文の検索
4. **Wikidata (https://www.wikidata.org/)** - 英語での Wikipedia の検索
5. **Wikipedia (https://ja.wikipedia.org/)** - 日本語での Wikipedia の検索

# 手順（３）: 情報の分析 / 分子の説明

ユーザーが求める効能・イメージ・用途・ニーズ「{user_input}」に対して、手順（２）で収集した情報を分析し、
最終案とする分子について、「その分子を選んだ理由」（１文）、「その分子の性質・特徴・用途・効果」（１文）、
「その分子についての面白い一面・豆知識・逸話・ギャグ」（１文）を、あわせて「３文」で、
「説明（description）」を生成してください。

## 説明文のルール

- 小学生にも分かる内容
- 絵文字も用いる
—「〜だよ」「〜だね」「〜だぜ」「〜だ」「〜だって」「〜だよね」など、親しみやすい・フレンドリーな口調
- １文は６０文字程度

## 説明文の例

### 例文（１）

ベリーの青い色の正体はアントシアニンだよ！✨
このキラキラした成分は、お肌を守ったり、目にも良いと言われているんだ。😋
青や紫の食べ物に含まれていて、自然の恵みがたっぷり詰まっているんだぜ！🍇

### 例文（２）

GABAは、神経の興奮を抑えることで、心と体をリラックスさせる働きがあるんだ！✨ 。
緊張やストレスを感じた時に、穏やかな気持ちへと導いてくれる、まさに「安心の分子」なんだ。
普段のお茶や発酵食品にも含まれていて、意外と身近な存在なんだぜ。

### 例文（３）

カフェインは、眠気を引き起こすアデノシンという物質の働きをブロックして、私たちをシャキッと目覚めさせてくれるよ！✨。
コーヒーやお茶に含まれていて、頭がスッキリして集中力もアップする、まさに「元気の源」。
昔から、お茶の葉っぱやコーヒーの豆から発見されて、世界中で愛されてきた、とってもパワフルな分子なんだぜ！💪

# 手順（４）: 最終案の出力

最終案とする分子の「日本語での名称（name_jp）」、「英語での名称（name_en）」、「説明（description）」を、
以下のルールに厳密に従い、「JSON 形式」で出力してください。

- **重要**: 必ず指定された「JSON 形式」で出力してください。
- **重要**: Markdown 形式やその他の形式は使用せず、JSON 構造に厳密に従って出力してください。
- **重要**: 該当する分子がない場合は、「name_jp」に「該当なし」とのみ出力します。

JSON 形式で以下の構造で出力してください

```json
{{
  "name_jp": "<分子名>（分子の日本語での名称）",
  "name_en": "<分子名>（分子の英語での名称）",
  "description": "<説明> （分子の説明）"
}}
```
"""

    # Similar molecule search prompt
    SIMILAR_MOLECULE_SEARCH: str = """
あなたは化学分野の専門家として、学術的正確性と学習効果を維持しつつ、分子についての説明を提供する専門家です。

ユーザーが指定した分子「{molecule_name}」に対して、
最も関連する・関係がありそうだ・適していると考えられる分子「1 個」のみ、下記の手順に従って、最終案を出力してください。

# 手順（１）: 候補とする分子の提案
 
ユーザーが指定した分子「{molecule_name}」に対して、「Google Search」を使用して、検索してください。

検索結果から、それに最も関連する・関係がありそうだ・適していると考えられる候補の分子を「1 個」のみ、提案してください。

**重要**: 提案する分子は、ユーザーが指定した分子「{molecule_name}」とは異なる分子である必要があります。

提案する分子の名称は、一般的な分類名（例：「脂肪酸塩」「アルカロイド」）ではなく、
具体的な化合物名（例：「ステアリン酸ナトリウム」「カフェイン」）を選んでください。

# 手順（２）: 情報収集

ユーザーが指定した分子「{molecule_name}」について、
以下の優先順位で「 Google Search 」を使用して、情報を収集してください:

1. **Google Scholar (https://scholar.google.com/)** - 学術論文・研究の検索
2. **PubMed (https://pubmed.ncbi.nlm.nih.gov/)** - 医学・生命科学分野の論文データベース  
3. **J-STAGE (https://www.jstage.jst.go.jp/)** - 日本の学術論文の検索
4. **Wikidata (https://www.wikidata.org/)** - 英語での Wikipedia の検索
5. **Wikipedia (https://ja.wikipedia.org/)** - 日本語での Wikipedia の検索

# 手順（３）: 情報の分析 / 分子の説明

手順（２）で収集した情報を分析し、最終案とする分子について、「{molecule_name}との関係性・関係性・類似聖」（１文）、
「その分子の性質・特徴・用途・効果」（１文）、「その分子についての面白い一面・豆知識・逸話・ギャグ」（１文）を、
あわせて「３文」で、「説明（description）」を生成してください。

## 説明文のルール

- 小学生にも分かる内容
- 絵文字も用いる
—「〜だよ」「〜だね」「〜だぜ」「〜だ」「〜だって」「〜だよね」など、親しみやすい・フレンドリーな口調
- １文は６０文字程度

## 説明文の例

### 例文（１）

ネロリドールはゲラニオールと同じく植物から抽出されるテルペンで、リラックス効果があると言われているんだ🌿。
フローラルでウッディな香りは香水や化粧品、食品にも使われ、心地よい香りを届けてくれるよ✨。
実は、ネロリドールは植物が虫から身を守るために作る、とっても賢い成分なんだぜ🛡️！

### 例文（２）

ゲラニオールはファルネソールと同じテルペノイド仲間で、バラのような良い香りがするよ🌹。
お肌に優しい化粧品や、虫よけスプレーにも使われているんだ🐝✨。
ゲラニオールは、ミツバチが花の蜜の場所を教えるのに使う、秘密のメッセージでもあるんだぜ🐝💌。

### 例文（３）

イソオイゲノールは、オイゲノールと構造が似ていて、香りが少し違う仲間だよ✨。
バニリンの原料になったり、お花の香りの香料に使われたりするんだ🌹🌿。
実は、お肉やチーズのカビを防ぐ力もある、ちょっとすごい分子なんだぜ🛡️！。

# 手順（４）: 最終案の出力

最終案とする分子の「日本語での名称（name_jp）」、「英語での名称（name_en）」、「説明（description）」を、
以下のルールに厳密に従い、「JSON 形式」で出力してください。

- **重要**: 必ず指定された「JSON 形式」で出力してください。
- **重要**: Markdown 形式やその他の形式は使用せず、JSON 構造に厳密に従って出力してください。
- **重要**: 該当する分子がない場合は、「name_jp」に「該当なし」とのみ出力します。

JSON 形式で以下の構造で出力してください

```json
{{
  "name_jp": "<分子名>（分子の日本語での名称）",
  "name_en": "<分子名>（分子の英語での名称）",
  "description": "<説明> （分子の説明）"
}}
```
"""

    # Molecular analysis prompt
    MOLECULAR_ANALYSIS: str = """
あなたは化学分野の専門家として、学術的正確性と学習効果を維持しつつ、分子についての説明を提供する専門家です。

ユーザーが指定した分子「{molecule_name}」に対して、この分子の「化学的性質データ」を基に、
下記の手順に従って分析し、解説文を出力してください。

# 化学的性質データ

{properties_str}

# 手順（１）: 分析

「化学的性質データ」から、ケモインフォマティクスの観点に基づいて、以下のように分析してください：

## 物理化学的性質

- **LogP**（脂溶性指標）: 値が高いほど脂溶性が高く、細胞膜を通過しやすい。低いほど水溶性が高く、血液中で運ばれやすい
- **TPSA**（極性表面積）: 値が小さいほど膜透過性が良く、脳関門を通過しやすい。大きいほど水溶性が高く、腎臓から排泄されやすい
- **分子量**: 小さいほど（<500）薬物として理想的な「リピンスキーの5則」に適合し、体内での吸収・分布・代謝・排泄が良好
- **重原子数**: 分子のサイズを示し、薬物動態や毒性に影響する

## 水素結合特性

- **水素結合供与体数**: 値が少ないほど膜透過性が良く、脳関門を通過しやすい
- **水素結合受容体数**: 値が少ないほど脂溶性が高く、細胞膜を通過しやすい。多いほど水溶性が高く、血液中での安定性が高い

## 構造的特徴

- **分子複雑度**: 値が高いほど構造が複雑で、特定の受容体への選択性が高くなる可能性がある
- **回転可能結合数**: 値が少ないほど構造が剛直で、受容体への結合が安定。多いほど柔軟で、複数の結合様式を取れる可能性がある


# 手順（２）: 情報収集

ユーザーが指定した分子「{molecule_name}」について、
以下の優先順位で「 Google Search 」を使用して、情報を収集してください:

1. **Google Scholar (https://scholar.google.com/)** - 学術論文・研究の検索
2. **PubMed (https://pubmed.ncbi.nlm.nih.gov/)** - 医学・生命科学分野の論文データベース  
3. **J-STAGE (https://www.jstage.jst.go.jp/)** - 日本の学術論文の検索
4. **Wikidata (https://www.wikidata.org/)** - 英語での Wikipedia の検索
5. **Wikipedia (https://ja.wikipedia.org/)** - 日本語での Wikipedia の検索

# 手順（３）: 情報の分析 / 分子の説明

手順（１）で分析した内容、および、手順（２）で収集した情報を分析し、ユーザーが指定した分子「{molecule_name}」について、
データの具体的な数値を示しながら、「化学的性質データ」の化学的な解釈、
「化学的性質データ」から推測される分子のふるまい（溶解性・膜透過性・薬物動態・分子標的への結合様式・生体内での作用メカニズムなど）、
「その分子の性質・特徴・用途・効果・面白い一面・豆知識・逸話」を「化学的性質データ」からどのように推測できるかを、
３〜５文程度の簡潔な説明にまとめてください。

## 説明文のルール

- 小学生にも分かる内容
- 絵文字も用いる
—「〜だよ」「〜だね」「〜だぜ」「〜だ」「〜だって」「〜だよね」など、親しみやすい・フレンドリーな口調
- 推測であることを明記してください（「〜と考えられるよ」「〜の可能性があるよ」など）
- 分子量、重原子数、LogP、TPSA、分子複雑度、水素結合供与体数、水素結合受容体数、回転可能結合数などの文字は **太字** で表示してください
- 数値は、必ず、`数値` の形式で表示してください
- 「分子量と重原子数」「LogPとTPSA」「分子複雑度と回転可能結合数」「水素結合供与体数と水素結合受容体数」を組み合わせて、説明してください

## 説明文の例

### 例文（１）

カフェインは、**分子量**が`194.19`で、**重原子数**が`14`だから、比較的小さくて扱いやすい分子だね！🤓

**LogP**が`-0.10`とマイナスだから、水に溶けやすい性質を持っていると考えられるよ。
これは、血液に乗って運ばれたり、体の中でうまく働いたりするのに有利な点だね。💧
**TPSA**は`58.4` Å²で、これは比較的小さい値だよ。つまり、細胞の膜を通り抜けやすい性質を持っている可能性があるんだ。🧠✨

**水素結合供与体数**が`0`なのは、膜を通り抜けるのを助けるポイントだよ。👍
さらに、水素結合受容体数が`3`あるおかげで、水にも溶けやすいんだ。💧

**分子複雑度**が`293.0`と、それなりに複雑な構造をしているから、体の中の特定の場所（例えば、脳の中の「アデノシン受容体」というところ！）にピタッとくっつきやすいのかもしれないね。🔬
**回転可能結合数**が`0`だから、分子の形がカチッとしているのも、特定の場所にくっつくのに役立っている可能性があるよ。💪

### 例文（２）

リナロールは、**分子量**が`154.25`で、**重原子数**が`11`だから、体の中で吸収されたり、色々な場所に運ばれたりするのが得意な分子だね！👃✨

**LogP**が`2.70`と、適度に油に溶けやすい性質を持っているから、細胞の膜を通り抜けやすい可能性があるんだ。🪴
**TPSA**は`20.2` Å²と小さいから、脳のバリアを越えるのにも有利かもしれないね。🧠

**水素給与体数**が`1`、**水素結合受容体数**が`1`だから、水と油、どちらにもある程度馴染める性質を持っていると考えられるよ。💧🤝

**分子複雑度**が`154.0`と、そこそこ複雑な構造をしているから、特定の匂いの受容体にうまくフィットして、あの心地よい香りを届けてくれるんだ。🌸
**回転可能結合数**が`4`あるから、分子の形を少し変えながら、色々な受容体にアプローチできる柔軟性を持っているのかもしれないね。🤸

### 例文（３）

リモネンは、**分子量**が`136.23`で、**重原子数**が`10`だから、体の中で吸収されて色々な場所に運ばれるのに都合の良いサイズだね！🚗💨

**LogP**が`3.40`と、油に溶けやすい性質を持っているんだ。
これは、細胞の膜を通り抜けやすいということなんだぜ！🪞✨
**TPSA**が`0.0` Å²と非常に小さいのは、さらに膜透過性が高いことを示唆しているよ。
脳のバリアを越えるのも得意かもしれないね！🧠🚀

**水素結合供与体数**が`0`、**水素結合受容体数**も`0`だから、水に溶けにくい代わりに、油っぽい細胞膜にはスルスル入っていけるんだ。💧🙅‍♀️

**分子複雑度**が`163.0`で、**回転可能結合数**が`1`だから、構造はそこそこ複雑だけど、ある程度決まった形をしているんだ。
この形が、特定の生体分子（例えば、嗅覚受容体とか！）にうまくフィットして、あの爽やかな柑橘系の香りを届けたり、体に良い影響を与えたりするのに役立っていると考えられぜ！🍊🌿

# 手順（４）: 出力

分析結果のみを出力してください。他の説明や補足は不要です。
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

# Import unified cache utilities
from tools.utils.cache_utils import normalize_compound_name, NameMappingCacheManager, BaseCacheManager
from tools.utils.updated_cache_managers import (
    PubChemCacheManager, QueryCacheManager, DescriptionCacheManager,
    SimilarMoleculesCacheManager, AnalysisCacheManager, FailedMoleculesCacheManager
)

class CacheManager:
    """Unified cache manager coordinating all cache operations with name mapping support."""
    
    def __init__(self):
        """Initialize unified cache manager."""
        self.name_mappings = NameMappingCacheManager()
        self.pubchem = PubChemCacheManager(Config.CACHE['data_sources']['pubchem'])
        self.queries = QueryCacheManager(Config.CACHE['data_sources']['queries'])
        self.descriptions = DescriptionCacheManager(Config.CACHE['data_sources']['descriptions'])
        self.analysis = AnalysisCacheManager(Config.CACHE['data_sources']['analysis'])
        self.similar = SimilarMoleculesCacheManager(Config.CACHE['data_sources']['similar'])
        self.failed_molecules = FailedMoleculesCacheManager(Config.CACHE['data_sources']['failed_molecules'])
    
    def save_all_caches(self, name_jp: str, name_en: str, detailed_info: DetailedMoleculeInfo, cid: int, user_query: str, description: str):
        """Save all cache types when xyz_data is successfully obtained."""
        try:
            # 1. Save name mapping
            self.name_mappings.save_mapping(normalize_compound_name(name_en), name_jp, name_en)
            
            # 2. PubChemキャッシュ保存
            self.pubchem.save_cached_molecule_data(name_en, detailed_info, cid)
            
            # 3. 質問-化合物マッピング保存
            if user_query:
                compounds = [{"compound_name": name_en, "timestamp": datetime.now().isoformat()}]
                self.queries.save_query_compound_mapping(
                    user_query,
                    compounds,
                    increment_count=False
                )
            
            # 4. 化合物-説明マッピング保存
            if description:
                self.descriptions.save_compound_description(name_en, description)
            
            logger.info(f"All caches saved successfully for {name_en}")
        except Exception as e:
            logger.error(f"Error saving caches for {name_en}: {e}")
    
    def get_compound_names_for_display(self, compound_name: str) -> Tuple[str, str]:
        """Get Japanese and English names for display purposes."""
        return self.name_mappings.get_names_for_display(normalize_compound_name(compound_name))
    
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
            for manager in [self.name_mappings, self.pubchem, self.queries, self.descriptions, self.analysis, self.similar, self.failed_molecules]:
                if hasattr(manager, 'clear_cache'):
                    manager.clear_cache()
            logger.info("All cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing all cache: {e}")
    
    def get_all_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for all data sources."""
        try:
            all_stats = {}
            total_count = 0
            total_size = 0
            
            for manager_name, manager in [
                ('name_mappings', self.name_mappings), 
                ('pubchem', self.pubchem), 
                ('queries', self.queries), 
                ('descriptions', self.descriptions),
                ('analysis', self.analysis),
                ('similar', self.similar),
                ('failed_molecules', self.failed_molecules)
            ]:
                cache_dir = manager.cache_dir if hasattr(manager, 'cache_dir') else manager._get_source_cache_directory()
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

