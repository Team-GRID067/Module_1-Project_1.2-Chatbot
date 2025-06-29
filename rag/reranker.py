from FlagEmbedding import FlagReranker
from langchain.schema import Document
from typing import List

reranker = FlagReranker('namdp-ptit/ViRanker',
                        use_fp16=True)

def rerank_documents(query: str, docs: List[Document], top_k: int = 5):
    pairs = [[query,doc.page_content] for doc in docs]
    
    scores = reranker.compute_score(pairs,normalize=True)
    doc_score_pairs = list(zip(docs, scores))
    sorted_docs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in sorted_docs[:top_k]]
    return top_docs
