# RAG Chatbot Demo

PDFドキュメントをアップロードして、その内容について質問できるStreamlitベースのRAG（Retrieval-Augmented Generation）チャットボットです。

## 特徴

- 📄 **PDF文書処理**: PyMuPDFを使用したPDFの読み込みと解析
- 🔍 **高精度検索**: Chromaベクトルデータベースによる文書検索
- 🤖 **AI対話**: OpenAI GPTモデルを使用した自然な会話
- 💬 **会話履歴**: ConversationBufferMemoryによる文脈の保持
- ⚙️ **柔軟な設定**: モデル選択、温度調整、チャンクサイズ設定
- 📊 **詳細ログ**: 包括的なログ機能

## 技術スタック

- **フロントエンド**: Streamlit
- **文書処理**: PyMuPDF, RecursiveCharacterTextSplitter
- **ベクトルDB**: Chroma
- **LLM統合**: LangChain + OpenAI
- **メモリ**: ConversationBufferMemory

## セットアップ

### 1. 依存関係のインストール

```bash
# uvを使用（推奨）
uv sync

# または pip を使用
pip install -r requirements.txt
```

### 2. 環境変数の設定

```bash
# .envファイルを作成
cp .env.example .env

# OpenAI APIキーを設定
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. アプリケーションの起動

```bash
# uvを使用
uv run streamlit run app.py

# または直接実行
streamlit run app.py
```

## 使い方

1. **APIキー設定**: サイドバーでOpenAI APIキーを入力
2. **PDF アップロード**: 質問したいPDFファイルをアップロード
3. **設定調整**（オプション）:
   - モデル選択: GPT-3.5-turbo, GPT-4, GPT-4o-mini
   - Temperature: 0.0-1.0（応答の創造性）
   - チャンクサイズ: 500-2000文字（文書分割サイズ）
4. **質問**: チャット欄で文書の内容について質問

## 設定オプション

### モデル選択
- `gpt-3.5-turbo`: 高速で費用効率的
- `gpt-4`: より高精度な回答
- `gpt-4o-mini`: バランスの取れた選択肢

### Temperature設定
- `0.0`: 決定論的で一貫した回答
- `1.0`: より創造的で多様な回答

### チャンクサイズ
- `500-1000`: 短い文書や正確な検索に適用
- `1500-2000`: 長い文書や文脈重視の場合に適用

## ログ機能

アプリケーションは詳細なログを記録します：
- ファイル位置: `logs/streamlit_rag_YYYYMMDD_HHMMSS.log`
- ログレベル: DEBUG（開発用）、INFO（コンソール表示）
- 対象コンポーネント: Streamlit, LangChain, OpenAI, ChromaDB

## プロジェクト構造

```
streamlit-demo/
├── app.py              # メインアプリケーション
├── pyproject.toml      # プロジェクト設定・依存関係
├── .env.example        # 環境変数テンプレート
├── .env               # 環境変数（非コミット）
├── logs/              # ログファイル
└── README.md          # このファイル
```

## 要件

- Python ≥ 3.11
- OpenAI API キー
- インターネット接続（OpenAI API使用のため）

## トラブルシューティング

### よくある問題

1. **APIキーエラー**: 有効なOpenAI APIキーが設定されているか確認
2. **PDF読み込みエラー**: PDFファイルが破損していないか確認
3. **メモリ不足**: 大きなPDFファイルの場合、チャンクサイズを小さく設定

### ログの確認

問題が発生した場合は、ログファイルを確認してください：
- アプリ内の「📄 ログ情報」エキスパンダーでファイル場所を確認
- エラーの詳細情報がログに記録されています

## ライセンス

[ライセンス情報をここに記載]

## 貢献

プルリクエストやイシューの報告を歓迎します。

---

🤖 **RAG Chatbot Demo** - Powered by LangChain & OpenAI