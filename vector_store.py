import chromadb
from embeddings import get_embedding

def init_vector_store(collection_name="video_chunks"):
    """
    Initialize persistent ChromaDB client + collection.
    Automatically wipes old data so you don't get mixed timestamps!
    """
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # NEW FIX: Delete the old polluted data before starting!
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass # If it doesn't exist yet, just continue
        
    # Create a fresh, clean collection
    collection = client.create_collection(
        name=collection_name, 
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def store_embeddings(collection, embedded_chunks):
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    for i, item in enumerate(embedded_chunks):
        ids.append(f"chunk_{i}")
        embeddings.append(item["embedding"])
        documents.append(item["metadata"]["text"])
        metadatas.append({
            "start": item["metadata"]["start"],
            "end": item["metadata"]["end"],
            "text": item["metadata"]["text"]
        })
        
    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Successfully stored {len(ids)} chunks in the vector store.")

def retrieve_similar(collection, query_text, n_results=3):
    query_embedding = get_embedding(query_text)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results

# import chromadb
# from embeddings import get_embedding

# def init_vector_store(collection_name="video_chunks"):
#     """
#     Initialize persistent ChromaDB client + collection.
#     Chroma uses an in-memory database with a local save path.
#     """
#     client = chromadb.PersistentClient(path="./chroma_db")
    
#     # We use cosine similarity to find the most relevant chunks
#     collection = client.get_or_create_collection(
#         name=collection_name, 
#         metadata={"hnsw:space": "cosine"}
#     )
#     return collection

# def store_embeddings(collection, embedded_chunks):
#     """
#     Store embedding vectors + metadata into Chroma.
#     """
#     ids = []
#     embeddings = []
#     documents = []
#     metadatas = []
    
#     for i, item in enumerate(embedded_chunks):
#         ids.append(f"chunk_{i}")
#         embeddings.append(item["embedding"])
        
#         # We need to store exact string values for documents as required by Chroma
#         documents.append(item["metadata"]["text"])
#         metadatas.append({
#             "start": item["metadata"]["start"],
#             "end": item["metadata"]["end"],
#             "text": item["metadata"]["text"]
#         })
        
#     if ids:
#         collection.add(
#             ids=ids,
#             embeddings=embeddings,
#             documents=documents,
#             metadatas=metadatas
#         )
#         print(f"Successfully stored {len(ids)} chunks in the vector store.")

# def retrieve_similar(collection, query_text, n_results=3):
#     """
#     Embed the query via SambaNova and return similar chunks + timestamp contexts.
#     """
#     # 1. Embed query
#     query_embedding = get_embedding(query_text)
    
#     # 2. Search Chroma
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=n_results
#     )
    
#     return results

