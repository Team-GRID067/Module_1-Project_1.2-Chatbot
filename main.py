from dotenv import load_dotenv
from graph.workflow import app
from agent.state import AgentState
from rag.vectordb import initialize_or_get_db
from rag.embedding import create_sentence_embedding

load_dotenv()

if __name__ == "__main__":
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

    # Khởi tạo retriever config
    embedding = create_sentence_embedding()
    milvus_client = initialize_or_get_db(
        db_path="vector_db/vector_db",
        collection_name="docs",
        docs=[],  # nếu collection đã tồn tại thì bỏ trống docs cũng được
        embedding_model=embedding,
        dimension=embedding.get_sentence_embedding_dimension()
    )

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
