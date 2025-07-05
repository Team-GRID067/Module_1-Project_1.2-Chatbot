import os

from langchain.document_loaders import UnstructuredPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from rag.embedding import create_sentence_embedding, tokenize_docs,get_embedding_dimension
from rag.vectordb import initialize_or_get_db
from langchain_community.document_loaders import PyPDFLoader 
def PDF_parser(pdf_folder):
    docs = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.lower().endswith(".pdf"):
            full_path = os.path.join(pdf_folder, pdf_file)
            print(f"Đang load: {full_path}")
            try:
                loader = PyPDFLoader(full_path)
                pages = loader.load()  # mỗi trang là 1 Document
                docs.extend(pages)
            except Exception as e:
                print(f"❌ Lỗi khi xử lý {full_path}: {e}")
    return docs

def get_text_spliter(embeddings):
    return SemanticChunker(
        embeddings=embeddings,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=256,
        add_start_index=True
    )

def ingest_database( collection_name,pdf_folder = "doc/", db_path="ai_courses.db"):
    documents = PDF_parser(pdf_folder)
    
    tokenized_docs = tokenize_docs(documents)
    embedding = create_sentence_embedding()
    sem_chunker = get_text_spliter(embedding)
    chunked_docs= sem_chunker.split_documents(tokenized_docs)
    for index, chunk in enumerate(chunked_docs): 
      print(f"{index} THIS \n {chunk.page_content}")
      print('-'*40)
    dimension = get_embedding_dimension()

    client = initialize_or_get_db(
        db_path=db_path,
        collection_name=collection_name,
        docs=chunked_docs,
        embedding_model=embedding,
        dimension=dimension
    )
    
    return client