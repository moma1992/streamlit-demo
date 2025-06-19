import streamlit as st
import tempfile
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging():
    """Set up logging configuration to capture all logs to a file"""
    log_filename = f"logs/streamlit_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create file handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Configure specific loggers
    logging.getLogger('streamlit').setLevel(logging.DEBUG)
    logging.getLogger('langchain').setLevel(logging.DEBUG)
    logging.getLogger('openai').setLevel(logging.DEBUG)
    logging.getLogger('chromadb').setLevel(logging.DEBUG)
    
    return log_filename

# Initialize logging
log_file = setup_logging()
logger = logging.getLogger(__name__)
logger.info(f"Starting RAG Chatbot application - Log file: {log_file}")

# Page config
st.set_page_config(
    page_title="RAG Chatbot Demo",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Chatbot Demo")
st.markdown("PDFをアップロードして、その内容について質問できるチャットボットです。")

# Show log file info
with st.expander("📄 ログ情報"):
    st.info(f"ログファイル: {log_file}")
    st.markdown("全てのアプリケーションログがファイルに記録されています。")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ 設定")
    
    # API Key input
    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="OpenAI APIキーを入力してください"
    )
    
    # Model selection
    model_name = st.selectbox(
        "モデル選択",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"],
        index=0
    )
    
    # Temperature setting
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="応答の創造性を調整します"
    )
    
    # Chunk size setting
    chunk_size = st.slider(
        "チャンクサイズ",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100,
        help="文書分割のサイズを調整します"
    )

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None

# PDF upload
uploaded_file = st.file_uploader(
    "PDFファイルをアップロードしてください",
    type="pdf",
    help="質問したいPDFファイルを選択してください"
)

if uploaded_file and api_key:
    logger.info(f"Processing PDF file: {uploaded_file.name}")
    # Process PDF
    with st.spinner("PDFを処理中..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            logger.info(f"Loading PDF from: {tmp_file_path}")
            # Load PDF
            loader = PyMuPDFLoader(tmp_file_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from PDF")
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(documents)
            logger.info(f"Split documents into {len(texts)} chunks with size {chunk_size}")
            
            # Create embeddings and vectorstore
            logger.info("Creating embeddings and vectorstore")
            embeddings = OpenAIEmbeddings(api_key=api_key)
            vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=embeddings
            )
            st.session_state.vectorstore = vectorstore
            logger.info("Vectorstore created successfully")
            
            # Create LLM
            logger.info(f"Creating LLM with model: {model_name}, temperature: {temperature}")
            llm = ChatOpenAI(
                api_key=api_key,
                model_name=model_name,
                temperature=temperature
            )
            
            # Create memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            # Create chain
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory,
                return_source_documents=True
            )
            st.session_state.chain = chain
            logger.info("Conversational chain created successfully")
            
            st.success(f"PDFを処理しました！ {len(texts)}個のチャンクに分割されました。")
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
            st.error(f"PDFの処理中にエラーが発生しました: {e}")
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            logger.info("Temporary file cleaned up")

# Chat interface
if st.session_state.chain:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("質問を入力してください"):
        logger.info(f"User question received: {prompt}")
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("回答を生成中..."):
                try:
                    logger.info("Generating response with RAG chain")
                    response = st.session_state.chain({"question": prompt})
                    answer = response["answer"]
                    logger.info(f"Generated answer: {answer[:100]}...")
                    
                    st.markdown(answer)
                    
                    # Show source documents
                    if "source_documents" in response and response["source_documents"]:
                        logger.info(f"Found {len(response['source_documents'])} source documents")
                        with st.expander("参考文書"):
                            for i, doc in enumerate(response["source_documents"]):
                                st.markdown(f"**文書 {i+1}:**")
                                st.markdown(doc.page_content[:500] + "...")
                                st.markdown("---")
                    
                    # Add assistant message
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    logger.info("Response completed successfully")
                    
                except Exception as e:
                    logger.error(f"Error generating response: {str(e)}", exc_info=True)
                    error_msg = f"エラーが発生しました: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

else:
    if not api_key:
        st.warning("サイドバーでOpenAI APIキーを入力してください。")
    elif not uploaded_file:
        st.info("PDFファイルをアップロードしてください。")
    
    # Show sample questions
    st.markdown("### 💡 使い方")
    st.markdown("""
    1. サイドバーでOpenAI APIキーを入力
    2. PDFファイルをアップロード
    3. 文書の内容について質問を入力
    """)

# Clear chat button
if st.session_state.messages:
    if st.button("チャット履歴をクリア"):
        logger.info("Clearing chat history")
        st.session_state.messages = []
        if st.session_state.chain and hasattr(st.session_state.chain, 'memory'):
            st.session_state.chain.memory.clear()
        logger.info("Chat history cleared")
        st.rerun()