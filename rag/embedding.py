from functools import lru_cache

from sentence_transformers import SentenceTransformer
from  langchain.schema import Document
from pyvi.ViTokenizer import tokenize

@lru_cache(maxsize=2)
def create_sentence_embedding(model="dangvantuan/vietnamese-embedding"):
    return SentenceTransformer(model)

def tokenize_docs(docs):
    return [
        Document(
            page_content=tokenize(doc.page_content),
            metadata=doc.metadata
        )
        for doc in docs
    ]
