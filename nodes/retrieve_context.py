from rag.retriever import get_retriever 
from rag.embedding import create_sentence_embedding
from rag.reranker import rerank_documents

def retrieve_context(state, config):
    query = state["question"]
    embedding_model = create_sentence_embedding()
    retriever = get_retriever(
        client=config.get("milvus_client"),     
        collection_name=config.get("collection_name", "docs"),
        embedding_model=embedding_model,
        top_k=10
    )
    
    raw_docs  = retriever.get_relevant_documents(query)
    top_docs = rerank_documents(query, raw_docs, top_k=5)

    context = "\n\n".join([doc.page_content for doc in top_docs])
    
    state["retrieved_context"] = context
    return state


