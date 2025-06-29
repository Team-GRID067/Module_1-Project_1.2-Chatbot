from rag.retriever import get_retriever 
from rag.embedding import create_sentence_embedding


def retrieve_context(state, config):
    query = state["question"]
    embedding_model = create_sentence_embedding()
    retriever = get_retriever(
        client=config.get("milvus_client"),     
        collection_name=config.get("collection_name", "docs"),
        embedding_model=embedding_model,
        top_k=3
    )
    
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    state["retrieved_context"] = context
    return state


