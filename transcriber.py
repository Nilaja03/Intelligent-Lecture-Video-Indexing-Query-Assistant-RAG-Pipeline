import os
from openai import OpenAI

def transcribe_audio(audio_path: str, model_size: str = "base"):
    """
    Transcribes audio using the Cloud Whisper API.
    Now uses 'verbose_json' to extract REAL timestamps!
    """
    print("\nSending audio to Cloud for blazing fast transcription...")
    
    client=OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY","") 
    )
    
    try:
        with open(audio_path, "rb") as audio_file:
            # NEW: 'verbose_json' forces the API to return the exact timestamps
            response = client.audio.transcriptions.create(
                model="Whisper-Large-v3",
                file=audio_file,
                response_format="verbose_json" 
            )
            
            transcribed_segments = []
            
            # Loop through the real segments provided by the AI
            if hasattr(response, 'segments') and response.segments:
                for segment in response.segments:
                    
                    # Safely extract the data whether it's an object or dictionary
                    if isinstance(segment, dict):
                        start_time = segment.get("start", 0.0)
                        end_time = segment.get("end", 0.0)
                        text = segment.get("text", "")
                    else:
                        start_time = getattr(segment, "start", 0.0)
                        end_time = getattr(segment, "end", 0.0)
                        text = getattr(segment, "text", "")
                    
                    transcribed_segments.append({
                        "text": text.strip(),
                        "start": float(start_time),
                        "end": float(end_time)
                    })
            
            return transcribed_segments

    except Exception as e:
        print(f"API Error: {e}")
        return []

# import os
# from openai import OpenAI

# def transcribe_audio(audio_path: str, model_size: str = "base"):
#     """
#     Transcribes audio using SambaNova's Cloud Whisper API.
#     NO LOCAL AI MODELS ARE DOWNLOADED!
#     """
#     print("\nSending audio to Grok for blazing fast transcription...")
    
#     # Connect to SambaNova's API using the OpenAI Python package
#     # client = OpenAI(
#     #     base_url="https://api.sambanova.ai/v1",
#     #     api_key=os.environ.get("SAMBANOVA_API_KEY", "") 
#     # )
    
#     client=OpenAI(
#         base_url="https://api.groq.com/openai/v1",
#         api_key=os.environ.get("GROQ_API_KEY","") 
#     )
#     try:
#         with open(audio_path, "rb") as audio_file:
#             # Call SambaNova's Whisper-Large-v3 model
#             response = client.audio.transcriptions.create(
#                 model="Whisper-Large-v3",
#                 file=audio_file,
#                 response_format="json"
#             )
            
#             # SambaNova returns the full transcript text.
#             full_text = response.text
            
#             # To keep Team 2's chunking script happy, we will quickly slice 
#             # the massive text block into smaller sentences and assign them estimated timestamps.
#             sentences = [s.strip() + "." for s in full_text.split(".") if s.strip()]
            
#             transcribed_segments = []
#             current_time = 0.0
            
#             for sentence in sentences:
#                 transcribed_segments.append({
#                     "text": sentence,
#                     "start": current_time,
#                     "end": current_time + 4.0 # Estimate 4 seconds per sentence
#                 })
#                 current_time += 4.0
                
#             return transcribed_segments

#     except Exception as e:
#         print(f"SambaNova API Error: {e}")
#         return []