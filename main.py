import streamlit as st
from dotenv import load_dotenv
from graph.workflow import app
from agent.state import AgentState
from rag.vectordb import initialize_or_get_db
from rag.embedding import create_sentence_embedding
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os

load_dotenv()
# ----------- INIT & CONFIG -----------
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("📚 RAG Chatbot: Ask Your PDFs")

# ----------- SESSION STATE INIT -----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_texts" not in st.session_state:
    st.session_state.pdf_texts = []

# ----------- UPLOAD PDF -----------
uploaded_files = st.file_uploader("📎 Tải lên các file PDF", type=["pdf"], accept_multiple_files=True)

for file in uploaded_files:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path=tmp_file.name
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    st.session_state.pdf_texts.append(documents)
    os.unlink(tmp_file_path)

# ----------- CLEAR BUTTON -----------
if st.button("🧹 Xoá lịch sử"):
    st.session_state.chat_history.clear()
    st.session_state.pdf_texts.clear()
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
    embedding = create_sentence_embedding()
    milvus_client = initialize_or_get_db(
        db_path="rag/ai_courses.db",
        collection_name="docs",
        docs=st.session_state.pdf_texts,  # Trích từ PDF
        embedding_model=embedding,
        dimension=embedding.get_sentence_embedding_dimension()
    )

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
            final_answer = value["query_result"]  # lấy kết quả cuối cùng

    # ----------- HIỂN THỊ CÂU TRẢ LỜI -----------
    with st.chat_message("assistant"):
        st.markdown(final_answer)

    st.session_state.chat_history.append((user_question, final_answer))
