import streamlit as st
from dotenv import load_dotenv
from graph.workflow import app
from agent.state import AgentState
from rag.vectordb import initialize_or_get_db
from rag.embedding import create_sentence_embedding
from rag.ingestion import ingest_database
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
load_dotenv()

# ----------- INIT & CONFIG -----------
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("📚 RAG Chatbot: Ask Your PDFs")

# ----------- SESSION STATE INIT -----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_texts" not in st.session_state:
    st.session_state.pdf_texts = []

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = create_sentence_embedding()

if "milvus_client" not in st.session_state:
    embedding = st.session_state.embedding_model
    st.session_state.milvus_client = initialize_or_get_db(
        docs = "./doc"
        collection_name="docs",
        embedding_model=embedding
    )

# ----------- CLEAR BUTTON -----------
if st.button("🧹 Xoá lịch sử"):
    st.session_state.chat_history.clear()
    st.session_state.pdf_texts.clear()
    if "milvus_client" in st.session_state:
        del ssession_state.milvus_clientt
    st.rerun()

# ----------- HIỂN THỊ CHAT HISTORY -----------
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)

# ----------- NHẬN CÂU HỎI -----------
user_question = st.chat_input("💬 Nhập câu hỏi của bạn:")
if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)

    # ----------- XỬ LÝ TRUY VẤN RAG -----------
    embedding = st.session_state.embedding_model
    milvus_client = st.session_state.milvus_client

    input_state = {
        "question": user_question,
        "sql_query": "",
        "query_result": "",
        "query_rows": [],
        "attempts": 0,
        "relevance": "",
        "sql_error": False
    }
    state = AgentState(**input_state)
    config = {
        "configurable": {
            "milvus_client": milvus_client,
            "collection_name": "docs"
        }
    }

    final_answer = ""
    for output in app.stream(state, config=config):
        for key, value in output.items():
            final_answer = value.get("query_result", "")

    with st.chat_message("assistant"):
        st.markdown(final_answer)

    st.session_state.chat_history.append((user_question, final_answer))
