from pymilvus import MilvusClient
from typing import List, Optional
from langchain.schema import Document

class Retriever:
    """
    A simple retriever using MilvusClient.
    """
    def __init__(
        self,
        client: MilvusClient,
        collection_name: str,
        embedding_model,
        top_k: int = 5,
        filter_expression: Optional[str] = None,
    ):
        self.client = client
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.filter = filter_expression

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_vector = self.embedding_model.encode([query])[0]

        search_params = {
            "collection_name": self.collection_name,
            "data": [query_vector],
            "limit": self.top_k,
            "output_fields": ["text", "source"],
        }
        if self.filter:
            search_params["filter"] = self.filter

        results = self.client.search(**search_params)
        docs: List[Document] = []
        for hit in results:
            text = hit.get("text", "")
            metadata = {"source": hit.get("source", "")} if "source" in hit else {}
            docs.append(Document(page_content=text, metadata=metadata))

        return docs

def get_retriever(
    client: MilvusClient,
    collection_name: str,
    embedding_model,
    top_k: int = 5,
    filter_expression: Optional[str] = None,
) -> Retriever:
    """
    Factory to create a Milvus Retriever.
    """
    return Retriever(
        client=client,
        collection_name=collection_name,
        embedding_model=embedding_model,
        top_k=top_k,
        filter_expression=filter_expression,
    )
