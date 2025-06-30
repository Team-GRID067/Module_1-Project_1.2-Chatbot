from dotenv import load_dotenv
import streamlit as st
from graph.workflow import app
from agent.state import AgentState
from rag.vectordb import initialize_or_get_db
from rag.embedding import create_sentence_embedding


load_dotenv()

def init_db():
    # Khởi tạo retriever config
    embedding = create_sentence_embedding()
    milvus_client = initialize_or_get_db(
        db_path="rag/ai_courses.db",
        collection_name="docs",
        docs=[],  # nếu collection đã tồn tại thì bỏ trống docs cũng được
        embedding_model=embedding,
        dimension=embedding.get_sentence_embedding_dimension()
    )
    return milvus_client
def setup_page(): 
    st.set_page_config(page_title="RAG Chatbot", layout="centered")
    st.title("📚 RAG Chatbot: Ask Your PDFs")

def main(): 
   

    input_state = {
        "question": "Show me top 10 customers by sales",
        "sql_query": "",
        "query_result": "",
        "query_rows": [],
        "attempts": 0,
        "relevance": "",
        "sql_error": False
    }

    state = AgentState(**input_state)

    milvus_client = init_db()

    config = {
        "configurable": {
            "milvus_client": milvus_client,
            "collection_name": "docs"
        }
    }

    for output in app.stream(state, config=config):
        for key, value in output.items():
            print(f"Node '{key}':")
            print(value["question"], value["sql_query"], value["query_result"], sep="\n")



if __name__ == "__main__":
    main()