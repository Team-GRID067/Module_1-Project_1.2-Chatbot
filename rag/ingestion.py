import os

from langchain.document_loaders import UnstructuredPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

from rag.embedding import create_sentence_embedding, tokenize_docs
from rag.vectordb import initialize_or_get_db

def PDF_parser(pdf_folder):
    docs = []
    for pdf_path in os.listdir(pdf_folder):
        if pdf_path.endswith(".pdf"):
            loader = UnstructuredPDFLoader(pdf_path, strategy="auto")  
            doc = loader.load()
            docs.extend(doc)
    return docs

def get_text_spliter(embeddings):
    return SemanticChunker(
        embeddings=embeddings,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=500,
        add_start_index=True
    )

def ingest_database( collection_name,pdf_folder = "doc/", db_path="ai_courses.db"):
    documents = PDF_parser(pdf_folder)
    tokenized_docs = tokenize_docs(documents)
    embedding = create_sentence_embedding()
    sem_chunker = get_text_spliter(embedding)
    chunked_docs= sem_chunker.split_documents(tokenized_docs)
    
    client = initialize_or_get_db(
        db_path=db_path,
        collection_name=collection_name,
        docs=chunked_docs,
        embedding_model=embedding,
        dimension=embedding.get_sentence_embedding_dimension()
    )
    
    return client