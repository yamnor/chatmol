<p align="center">
  <img src="https://i.gyazo.com/e6424c1c7f3d48c212fadc993e253481.png" alt="ChatMOL" width="200">
</p>

「バラの香りってどんな分子？」「青色を作る分子は？」そんな素朴な疑問に、AI が答えてくれるアプリだよ。

普段なにげなく感じている色・香り・味。実はそれぞれに対応する分子があって、その分子の化学的な性質が、私たちのさまざまな感覚を生み出しているんだ。このアプリでは、AIと対話しながら様々な分子を探索して、その分子の立体的な形を眺めることができるよ。分子の世界の面白さを体験してみよう！

## 🌟 できること

- 「甘い香りの分子は？」「解熱作用がある分子は？」など、自然な言葉で質問できるよ
- マウスで自由に動かせる立体的な分子モデルで、リアルな形を理解できるよ
- 分子式や分子量の他に、水への溶解性など、いろいろな化学的な性質を予測できるよ
- 香り・色・味・薬効など、さまざまな観点から分子の世界を探索できるよ

## ⚠️ 注意事項

出力される分子の情報や構造について、**正確ではない** ことがあります。

- `gemini-2.5-flash` モデルでは、不正確な情報が出力されることが多いです
- `gemini-2.5-pro` モデルにすると改善されますが、応答に時間がかかります
- デフォルトでは、応答が早い・コストの安い `gemini-2.5-flash-lite` を使っています

## 🚀 デモ

URL: [chatmol.yamlab.app](https://chatmol.yamlab.app)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatmol.yamlab.app/) 

<p align="center">
    <img src="https://i.gyazo.com/3b3cd1dda9191e5ca7c27a0fbe715c57.gif" alt="ChatMOL Demo">
</p>

> [!NOTE]
> 上記の動画では、「リナロール」の構造は正しく出力されていますが、「グルタミン酸」は正確ではありません。このように、出力される分子の情報は不正確なことがあります。

## 📦 自分のパソコンで動かしてみよう

### 必要なもの

- Python 3.8 以上（プログラミング環境）
- Google Gemini API キー（無料枠もあるよ）

### セットアップ手順

#### 1. このプロジェクトをダウンロード

```bash
git clone https://github.com/yamnor/chatmol.git
cd chatmol
```

#### 2. condaで仮想環境を作成してパッケージをインストール

```bash
# condaで仮想環境を作成
conda create -n chatmol python=3.13 -y

# 仮想環境をアクティベート
conda activate chatmol

# 必要なパッケージをインストール
pip install -r requirements.txt
```

#### 3. API キーを設定

1. [Google AI Studio](https://makersuite.google.com/app/apikey) で Gemini API キーを取得（無料枠もあるよ）
2. `.streamlit/secrets.toml` ファイルを作成して、取得した API キーを下記のように書き込む：

```toml
api_key = "ここに取得した API キーを貼り付け"
```

#### 4. アプリを起動

```bash
# 仮想環境をアクティベート（まだアクティベートしていない場合）
conda activate chatmol

# アプリを起動
streamlit run main.py
```

ブラウザが自動で開いて、アプリが使えるよ！

## 🛠️ 使用した技術

- [Google Gemini](https://ai.google.dev/) - Google が開発している生成 AI モデル
- [Streamlit](https://streamlit.io/) - Python ベースの Web フレームワーク
- [RDKit](https://www.rdkit.org/) - 分子構造の操作と物性計算のためのツールキット
- [py3Dmol](https://3dmol.csb.pitt.edu/) - インタラクティブな分子構造ビューア

## 👨‍💻 開発者

**yamnor** ([@yamnor](https://github.com/yamnor))

大学教員。専門は計算化学。化学の学びを身近にすることにも興味を持っています。お気軽に[こちら](https://letterbird.co/yamnor)からお声がけください。

- ラボ: [yamlab.jp](https://yamlab.jp/)
- Blog: [yamnor.me](https://yamnor.me/)
- X: [@yamnor](https://x.com/yamnor)

## 📄 ライセンス

このプロジェクトは MIT ライセンスで公開しているよ。自由に使ってもらって大丈夫！詳細は [LICENSE](LICENSE) ファイルを見てね。

## 💬 質問や提案があれば

「こんな機能があったらいいな」「うまく動かない」など、何かあれば [Issues](https://github.com/yamnor/ChatMOL/issues) で教えてね。
