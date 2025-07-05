from functools import lru_cache
import torch                            
from sentence_transformers import SentenceTransformer
from  langchain.schema import Document
from pyvi.ViTokenizer import tokenize
from langchain.embeddings import HuggingFaceEmbeddings

@lru_cache(maxsize=2)
def create_sentence_embedding(model="dangvantuan/vietnamese-embedding"):
    # Determine device based on CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device},
        encode_kwargs={
            "device": device,
            "truncate": True,
            "max_length": 512
        }
    )


# In rag/ingestion.py, add document validation
def tokenize_docs(docs):
    tokenized = []
    for i, doc in enumerate(docs):
        # Check for empty content
        if not doc.page_content.strip():
            print(f"⚠️ Empty document skipped (index {i})")
            continue
            
        try:
            tokenized_content = tokenize(doc.page_content)
            tokenized.append(Document(
                page_content=tokenized_content,
                metadata=doc.metadata
            ))
        except Exception as e:
            print(f"❌ Tokenization failed for doc {i}: {str(e)}")
            print(f"Content snippet: {doc.page_content[:100]}...")
    return tokenized



def get_embedding_dimension(model_name="dangvantuan/vietnamese-embedding"):
    model = SentenceTransformer(model_name)
    return model.get_sentence_embedding_dimension()