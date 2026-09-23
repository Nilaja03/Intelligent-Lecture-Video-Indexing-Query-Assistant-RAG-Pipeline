import os
import json

   
# IMPORTING ALL TEAMS' MODULES
   
# Team 1 Imports
try:
    from downloader import download_audio
    from transcriber import transcribe_audio
except ImportError as e:
    print(f" Missing Team 1 file: {e}. Make sure downloader.py is in the folder!")
    exit()

# Team 2 Imports
from team2_chunking import clean_transcript, create_chunks

# Team 3 Imports
from normalization import normalize_all
from embeddings import embed_chunks
from vector_store import init_vector_store, store_embeddings, retrieve_similar

# Team 4 Imports
from llm import generate_timestamped_answer, extract_video_id

def run_full_pipeline(video_link: str, user_query: str):
    print("=" * 60)
    print(" STARTING REAL E2E PIPELINE: TEAM 1 -> TEAM 4")
    print("=" * 60)

       
    # STEP 1: TEAM 1 (WHISPER TRANSCRIPTION)
       
    print(f"\n[Team 1] Processing Video Link: {video_link}")
    print("   -> Downloading audio (this might take a moment)...")
    audio_path = download_audio(video_link)
    
    print("   -> Transcribing audio with Whisper...")
    # This generates the exact list format Team 2 expects!
    raw_transcript = transcribe_audio(audio_path, model_size="base")
    print(f"   -> Extracted {len(raw_transcript)} raw segments.")

       
    # STEP 2: TEAM 2 (CLEANING & CHUNKING)
       
    print("\n[Team 2] Cleaning and Chunking Transcript...")
    cleaned_data = clean_transcript(raw_transcript)
    team_2_chunks = create_chunks(cleaned_data)
    print(f"   -> Created {len(team_2_chunks)} chunks with timestamps.")

       
    # STEP 3: TEAM 3 (NORMALIZATION, EMBEDDING & STORAGE)
       
    print("\n[Team 3] Normalizing Time Formats...")
    normalized_chunks = normalize_all(team_2_chunks)
    
    print("[Team 3] Generating Vectors...")
    embedded_chunks = embed_chunks(normalized_chunks)
    
    print("[Team 3] Storing Vectors into ChromaDB...")
    collection = init_vector_store("master_lecture_db")
    store_embeddings(collection, embedded_chunks)

       
    # STEP 4: TEAM 3 (RETRIEVAL BASED ON USER QUERY)
       
    print(f"\n[Team 3] Searching Database for: '{user_query}'...")
    results = retrieve_similar(collection, user_query, n_results=3)
    
    # Format ChromaDB output into the clean list Team 4 expects
    retrieved_chunks = []
    if results['documents'] and results['documents'][0]:
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            retrieved_chunks.append({
                "text": doc,
                "metadata": {"start_time_seconds": meta['start']}
            })
    print(f"   -> Found {len(retrieved_chunks)} relevant chunks to send to LLM.")

       
    # STEP 5: TEAM 4 (LLM GENERATION)
    
    print("\n[Team 4] Generating Final Answer & Clickable Timestamps...")
    final_output = generate_timestamped_answer(
        user_query=user_query, 
        retrieved_chunks=retrieved_chunks, 
        video_link=video_link
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE! FINAL OUTPUT TO TEAM 5 UI:")
    print("=" * 60)
    print(json.dumps(final_output, indent=2))
    return final_output


if __name__ == "__main__":

    # Interactive Inputs!
    test_link = input("\nEnter a real YouTube Link: ")
    test_query = input("Enter your question: ")
    
    run_full_pipeline(test_link, test_query)