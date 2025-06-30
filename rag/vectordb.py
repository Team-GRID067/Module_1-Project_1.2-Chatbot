from pymilvus import MilvusClient

def initialize_or_get_db(
    
    collection_name: str,
    docs,
    embedding_model,
    db_path: str = "rag/ai_courses.db",
    dimension: int = 384
):
    client = MilvusClient(uri=db_path)

    if not client.has_collection(collection_name):
        client.create_collection(
            collection_name=collection_name,
            dimension=dimension
        )
        print(f"Created collection '{collection_name}' with dimension={dimension}")
    else:
        print(f"Collection '{collection_name}' already exists.")

    texts = [doc.page_content for doc in docs]
    vectors = embedding_model.encode(texts)

    data = [
        {
            "id": i,
            "vector": vectors[i],
            "text": texts[i],
            "source": docs[i].metadata.get("source", "unknown")
        }
        for i in range(len(docs))
    ]

    res = client.insert(
        collection_name=collection_name,
        data=data
    )
    print(f"Inserted {len(data)} documents into '{collection_name}'.")
    return client
