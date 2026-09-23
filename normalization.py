import re

def time_to_seconds(time_val):
    """Convert 'MM:SS' or 'HH:MM:SS' → seconds, OR just return if already float/int"""
    
    # NEW FIX: If the value is already a raw number (like from Groq), just return it!
    if isinstance(time_val, (int, float)):
        return float(time_val)
        
    time_str = str(time_val)
    
    # NEW FIX: Handle cases where a string is just a number like "45.5"
    if ":" not in time_str:
        return float(time_str)
        
    # Convert "HH:MM:SS" or "MM:SS"
    # Upgraded from 'int' to 'float' to handle decimal seconds gracefully
    parts = list(map(float, time_str.split(":")))
    
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds


def parse_time_range(time_range):
    """Convert 'MM:SS-MM:SS' → start, end in seconds"""
    start_str, end_str = time_range.split("-")
    return time_to_seconds(start_str), time_to_seconds(end_str)


def normalize_chunk(chunk):
    
    # FORMAT D: TEAM 2'S FORMAT - Check FIRST (has start_time/end_time)
    if "text" in chunk and "start_time" in chunk and "end_time" in chunk:
        start_seconds = time_to_seconds(chunk["start_time"])
        end_seconds = time_to_seconds(chunk["end_time"])
        return {
            "text": chunk["text"],
            "start": float(start_seconds),
            "end": float(end_seconds),
        }

    # Format A (has start/end directly)
    elif "text" in chunk and "start" in chunk:
        return {
            "text": chunk["text"],
            "start": float(chunk["start"]),
            "end": float(chunk["end"]),
        }

    # Format B
    elif "content" in chunk:
        return {
            "text": chunk["content"],
            "start": float(chunk["start_time"]),
            "end": float(chunk["end_time"]),
        }

    # Format C
    elif "chunk" in chunk:
        start, end = parse_time_range(chunk["time"])
        return {
            "text": chunk["chunk"],
            "start": float(start),
            "end": float(end),
        }

    else:
        raise ValueError(f"Unknown format: {chunk}")
    

def normalize_all(chunks):
    return [normalize_chunk(c) for c in chunks]