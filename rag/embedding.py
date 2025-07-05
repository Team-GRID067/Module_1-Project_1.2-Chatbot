from functools import lru_cache

from sentence_transformers import SentenceTransformer
from  langchain.schema import Document
from pyvi.ViTokenizer import tokenize
from langchain.embeddings import HuggingFaceEmbeddings

@lru_cache(maxsize=2)
def create_sentence_embedding(model="dangvantuan/vietnamese-embedding"):
    return HuggingFaceEmbeddings(
        model_name=model,
        encode_kwargs={"truncate": True, "max_length": 512}  # ✅ Tránh lỗi input dài
    )

def tokenize_docs(docs):
    return [
        Document(
            page_content=tokenize(doc.page_content),
            metadata=doc.metadata
        )
        for doc in docs
    ]
def get_embedding_dimension(model_name="dangvantuan/vietnamese-embedding"):
    model = SentenceTransformer(model_name)
    return model.get_sentence_embedding_dimension()