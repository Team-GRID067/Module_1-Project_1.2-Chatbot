import os

from langchain.document_loaders import UnstructuredPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

from rag.embedding import create_sentence_embedding
from rag.vectordb import initialize_or_get_db
def PDF_parser(pdf_folder):
    docs = []
    for pdf_path in os.listdir(pdf_folder):
        if pdf_path.endswith(".pdf"):
            loader = UnstructuredPDFLoader(pdf_path, strategy="hi-res")  
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

def ingest_database(pdf_folder, db_path, collection_name):
    documents = PDF_parser(pdf_folder)
    embedding = create_sentence_embedding()
    sem_chunker = get_text_spliter(embedding)
    chunked_docs= sem_chunker.split_documents(documents)
    
    client = initialize_or_get_db(
        db_path=db_path,
        collection_name=collection_name,
        docs=chunked_docs,
        embedding_model=embedding,
        dimension=embedding.get_sentence_embedding_dimension()
    )
    
    return client