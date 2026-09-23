import chromadb.utils.embedding_functions as embedding_functions

# Load ChromaDB's built-in, lightweight embedder (No API keys required!)
default_ef = embedding_functions.DefaultEmbeddingFunction()

def get_embedding(text):
    """
    Generates a mathematical vector for the text block.
    """
    # default_ef takes a list of strings and returns a list of vectors
    return default_ef([text])[0]

def embed_chunks(chunks):
    """
    Loops through Team 2's chunks, filters out bad data, and embeds the rest.
    """
    embeddings = []
    
    print(f"   -> Embedding {len(chunks)} chunks locally (this is very fast)...")
    
    for chunk in chunks:
        # 1. Clean the text and skip empty chunks (which cause crashes!)
        text = chunk.get("text", "").strip()
        if not text:
            continue
            
        # 2. Generate the embedding vector
        emb = get_embedding(text)
        
        # 3. Store it exactly how Team 3's vector_store.py expects it
        embeddings.append({
            "embedding": emb,
            "metadata": chunk
        })
    
    return embeddings


