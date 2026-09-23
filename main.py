import json
from team2_chunking import load_chunks_from_json

# Importing from Team 3's internal files
from normalization import normalize_all
from embeddings import embed_chunks
from vector_store import init_vector_store, store_embeddings, retrieve_similar


def main():
    """
    Complete pipeline: Team 2 (JSON) → Team 3 (Vector Store) → Format for Team 4
    """
    import json
    
    print("=" * 80)
    print("TEAM 3: RAG PIPELINE INTEGRATION")
    print("=" * 80)
    
 
    # STEP 1: LOAD TEAM 2's DATA
 
    print("\n--- STEP 1: Load Data from Team 2 ---")
    
    # Replaced hardcoded transcript with Team 2's actual file loader
    # Make sure 'chunks.json' matches the file name Team 2 actually outputs
    chunks = load_chunks_from_json("chunks.json") 
    
    if not chunks:
        print("Could not load chunks. Ensure Team 2 has run their script and generated the JSON file.")
        return
        
    print(f"   Loaded {len(chunks)} chunks from Team 2")
    # STEP 2: TEAM 3 - NORMALIZE DATA

    print("\n--- STEP 2: Team 3 - Normalize Data ---")
    normalized_chunks = normalize_all(chunks)
    print(f"   Normalized {len(normalized_chunks)} chunks")
    
    # STEP 3: TEAM 3 - GENERATE EMBEDDINGS & STORE
  
    print("\n--- STEP 3: Team 3 - Embed and Store (SambaNova & ChromaDB) ---")
    try:
        embedded_chunks = embed_chunks(normalized_chunks)
        collection = init_vector_store("lecture_chunks")
        store_embeddings(collection, embedded_chunks)
        
        
        # STEP 4: RETRIEVE & FORMAT FOR TEAM 4
        
        print("\n--- STEP 4: Retrieve and Format for Team 4 ---")
        
        query = "How do we compute gradients?" # Test query
        print(f"\nQuery: '{query}'")
        
        # Retrieve the raw data from ChromaDB
        results = retrieve_similar(collection, query, n_results=3)
        
        # NEW: Format the raw ChromaDB results into the exact structure Team 4 needs
        team_4_input = []
        if results['documents'] and results['documents'][0]:
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                team_4_input.append({
                    "text": doc,
                    "metadata": {
                        "start_time_seconds": meta['start'] # Renamed key specifically for Team 4
                    }
                })
        
        print("\n Formatted Output ready to be passed to Team 4:")
        print(json.dumps(team_4_input, indent=2))
        
        print("\n" + "=" * 80)
        print("Done")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
