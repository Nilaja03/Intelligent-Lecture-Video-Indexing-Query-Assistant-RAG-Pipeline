"""
Team 2: Chunking & Timestamp Metadata Module
Extracted from MLOPS_Project.ipynb and formatted as a reusable Python module

This module handles:
1. Cleaning raw transcript text
2. Creating chunks with timestamp metadata
3. Saving/loading chunks from JSON files

Input format (from Team 1 - Whisper transcription):
    [
        {
            "start": "00:00:01",
            "end": "00:00:05",
            "text": "Hello everyone welcome"
        }
    ]

Output format (to Team 3 - Vector Store):
    [
        {
            "chunk_id": 1,
            "text": "Hello everyone welcome",
            "start_time": "00:00:01",
            "end_time": "00:00:05"
        }
    ]
"""

import json


def clean_transcript(transcript):
    """
    Step 2 from Team 2: Clean the text
    
    Removes leading/trailing whitespace from transcript items.
    This ensures all text is properly formatted before chunking.
    
    Args:
        transcript (list): List of dicts with keys:
            - "start" (str): Timestamp in HH:MM:SS format
            - "end" (str): Timestamp in HH:MM:SS format
            - "text" (str): Transcript text (may have whitespace)
    
    Returns:
        list: Cleaned transcript with whitespace removed from text
        
    Example:
        >>> transcript = [
        ...     {"start": "00:00:01", "end": "00:00:05", "text": "  Hello  "}
        ... ]
        >>> clean_transcript(transcript)
        [{'start': '00:00:01', 'end': '00:00:05', 'text': 'Hello'}]
    """
    for item in transcript:
        if "text" in item:
            item["text"] = item["text"].strip()
    
    return transcript


def create_chunks(transcript):
    """
    Step 3 from Team 2: Create chunks with timestamp metadata
    
    Converts cleaned transcript into chunks with unique IDs and preserved timestamps.
    Each chunk represents a contiguous segment of the transcript with associated timing.
    
    Args:
        transcript (list): List of dicts with keys:
            - "start" (str): Timestamp in HH:MM:SS format
            - "end" (str): Timestamp in HH:MM:SS format
            - "text" (str): Transcript text (should be cleaned)
    
    Returns:
        list: List of chunk dicts with structure:
            {
                "chunk_id": int,           # Unique identifier (1-indexed)
                "text": str,               # Transcript segment
                "start_time": str,         # HH:MM:SS format
                "end_time": str            # HH:MM:SS format
            }
    
    Example:
        >>> transcript = [
        ...     {"start": "00:00:01", "end": "00:00:05", "text": "Hello everyone"},
        ...     {"start": "00:00:05", "end": "00:00:10", "text": "Welcome to class"}
        ... ]
        >>> chunks = create_chunks(transcript)
        >>> len(chunks)
        2
        >>> chunks[0]
        {
            'chunk_id': 1,
            'text': 'Hello everyone',
            'start_time': '00:00:01',
            'end_time': '00:00:05'
        }
    """
    chunks = []

    for i, item in enumerate(transcript):
        chunk = {
            "chunk_id": i + 1,                    # 1-indexed chunk ID
            "text": item["text"],                 # Transcript segment
            "start_time": item["start"],          # Keep as HH:MM:SS string
            "end_time": item["end"]               # Keep as HH:MM:SS string
        }
        chunks.append(chunk)

    return chunks


def save_chunks_to_json(chunks, output_file="chunks.json"):
    """
    Step 6 from Team 2: Save chunks as JSON file
    
    Serializes chunks to a JSON file for storage and later retrieval.
    Useful for debugging or checkpointing intermediate results.
    
    Args:
        chunks (list): List of chunk dicts (output from create_chunks())
        output_file (str, optional): Path to output JSON file. Defaults to "chunks.json"
    
    Returns:
        None
        
    Side Effects:
        Creates/overwrites a JSON file at output_file path
        
    Example:
        >>> chunks = create_chunks(transcript)
        >>> save_chunks_to_json(chunks, "my_chunks.json")
           File saved: my_chunks.json
    """
    try:
        with open(output_file, "w") as f:
            json.dump(chunks, f, indent=4)
        
        print(f"File saved: {output_file}")
    except IOError as e:
        print(f"   Error saving file: {e}")


def load_chunks_from_json(input_file="chunks.json"):
    """
    Load chunks from JSON file (for testing/debugging)
    
    Deserializes chunks from a JSON file back into Python objects.
    Useful for resuming work from a checkpoint.
    
    Args:
        input_file (str, optional): Path to input JSON file. Defaults to "chunks.json"
    
    Returns:
        list: List of chunk dicts in the same format as create_chunks() output
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
        
    Example:
        >>> chunks = load_chunks_from_json("my_chunks.json")
        >>> len(chunks)
        42
    """
    try:
        with open(input_file, "r") as f:
            chunks = json.load(f)
        
        print(f"   File loaded: {input_file} ({len(chunks)} chunks)")
        return chunks
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        return []
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {input_file}")
        return []


def get_chunk_statistics(chunks):
    """
    Get statistics about the chunks (useful for quality checking)
    
    Computes summary statistics about a set of chunks.
    
    Args:
        chunks (list): List of chunk dicts from create_chunks()
    
    Returns:
        dict: Dictionary with keys:
            - "total_chunks" (int): Number of chunks
            - "total_characters" (int): Sum of all text lengths
            - "avg_chunk_length" (float): Average characters per chunk
            - "min_chunk_length" (int): Shortest chunk (chars)
            - "max_chunk_length" (int): Longest chunk (chars)
    
    Example:
        >>> chunks = create_chunks(transcript)
        >>> stats = get_chunk_statistics(chunks)
        >>> print(f"Average chunk: {stats['avg_chunk_length']:.0f} chars")
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "total_characters": 0,
            "avg_chunk_length": 0,
            "min_chunk_length": 0,
            "max_chunk_length": 0
        }
    
    lengths = [len(chunk["text"]) for chunk in chunks]
    
    return {
        "total_chunks": len(chunks),
        "total_characters": sum(lengths),
        "avg_chunk_length": sum(lengths) / len(chunks),
        "min_chunk_length": min(lengths),
        "max_chunk_length": max(lengths)
    }


def filter_empty_chunks(chunks):
    """
    Remove chunks with empty or whitespace-only text
    
    Filters out chunks that don't contain meaningful content.
    
    Args:
        chunks (list): List of chunk dicts from create_chunks()
    
    Returns:
        list: Filtered list with empty chunks removed
        
    Example:
        >>> chunks = [
        ...     {"chunk_id": 1, "text": "Hello", "start_time": "00:00:01", "end_time": "00:00:05"},
        ...     {"chunk_id": 2, "text": "   ", "start_time": "00:00:05", "end_time": "00:00:10"},
        ...     {"chunk_id": 3, "text": "World", "start_time": "00:00:10", "end_time": "00:00:15"}
        ... ]
        >>> filtered = filter_empty_chunks(chunks)
        >>> len(filtered)
        2
    """
    return [chunk for chunk in chunks if chunk["text"].strip()]


# # ============================================================================
# # EXAMPLE USAGE & TESTING
# # ============================================================================

# if __name__ == "__main__":
#     # Example: Complete pipeline
#     print("=" * 70)
#     print("Team 2: Chunking & Timestamp Metadata - Example Usage")
#     print("=" * 70)
    
#     # Step 1: Raw transcript from Team 1 (Whisper)
#     print("\n--- Step 1: Raw Transcript Input ---")
#     transcript = [
#         {"start": "00:00:01", "end": "00:00:05", "text": "  Hello everyone welcome  "},
#         {"start": "00:00:05", "end": "00:00:10", "text": "Today we learn about AI"},
#         {"start": "00:00:10", "end": "00:00:15", "text": "Neural networks are powerful"},
#         {"start": "00:00:15", "end": "00:00:20", "text": ""},  # Empty chunk for testing
#         {"start": "00:00:20", "end": "00:00:25", "text": "They can learn from data"}
#     ]
#     print(f"Input: {len(transcript)} transcript items")
#     for i, item in enumerate(transcript[:2]):
#         print(f"  [{i}] {item}")
    
#     # Step 2: Clean transcript
#     print("\n--- Step 2: Clean Transcript ---")
#     cleaned = clean_transcript(transcript)
#     print(f"Cleaned {len(cleaned)} items (whitespace removed)")
#     print(f"  Example: '{cleaned[0]['text']}'")
    
#     # Step 3: Create chunks
#     print("\n--- Step 3: Create Chunks ---")
#     chunks = create_chunks(cleaned)
#     print(f"Generated {len(chunks)} chunks")
#     print("\nFirst chunk:")
#     print(f"  {chunks[0]}")
#     print("\nSecond chunk:")
#     print(f"  {chunks[1]}")
    
#     # Step 4: Filter empty chunks
#     print("\n--- Step 4: Filter Empty Chunks ---")
#     filtered_chunks = filter_empty_chunks(chunks)
#     print(f"Filtered: {len(chunks)} → {len(filtered_chunks)} chunks")
#     print(f"  Removed {len(chunks) - len(filtered_chunks)} empty chunk(s)")
    
#     # Step 5: Get statistics
#     print("\n--- Step 5: Chunk Statistics ---")
#     stats = get_chunk_statistics(filtered_chunks)
#     print(f"  Total chunks: {stats['total_chunks']}")
#     print(f"  Total characters: {stats['total_characters']}")
#     print(f"  Average chunk length: {stats['avg_chunk_length']:.1f} chars")
#     print(f"  Min/Max length: {stats['min_chunk_length']}/{stats['max_chunk_length']} chars")
    
#     # Step 6: Save to JSON
#     print("\n--- Step 6: Save to JSON ---")
#     save_chunks_to_json(filtered_chunks, "example_chunks.json")
    
#     # Step 7: Load from JSON
#     print("\n--- Step 7: Load from JSON ---")
#     loaded_chunks = load_chunks_from_json("example_chunks.json")
#     print(f"Verification: Loaded {len(loaded_chunks)} chunks, matches saved: {loaded_chunks == filtered_chunks}")
    
#     print("\n" + "=" * 70)
#     print("   Team 2 module ready for integration with Team 3!")
#     print("=" * 70)
