from functools import lru_cache

from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize

@lru_cache(maxsize=2)
def create_sentence_embedding(model="dangvantuan/vietnamese-embedding"):
    return SentenceTransformer(model)
