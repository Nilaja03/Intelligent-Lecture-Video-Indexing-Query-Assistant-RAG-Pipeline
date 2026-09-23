from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
import os
import shutil
from downloader import download_audio
from transcriber import transcribe_audio
import json

app = FastAPI(
    title="Smart Video Assistant - Team 1 API",
    description="API for downloading and transcribing YouTube videos or Local Files."
)

class VideoRequest(BaseModel):
    url: str
    apply_llm_formatting: bool = False
    model_size: str = "base"

@app.post("/process_url")
def process_url(request: VideoRequest):
    """
    Downloads audio from a YouTube URL and transcribes it using faster-whisper.
    """
    try:
        print(f"Processing URL: {request.url}")
        audio_path = download_audio(request.url)
        
        
        
        segments = transcribe_audio(audio_path, model_size=request.model_size)
        
        # Save to the hard-coded file for Team 2 as per README
        output_filename = "transcription_output.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(segments, f, indent=4, ensure_ascii=False)
            
        return {
            "status": "success", 
            "message": f"Saved to {output_filename}",
            "segments": segments
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def process_upload(file: UploadFile = File(...), model_size: str = "base"):
    """
    Upload a local audio/video file directly for transcription.
    """
    if not os.path.exists("downloads"):
        os.makedirs("downloads")
    
    file_path = f"downloads/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        segments = transcribe_audio(file_path, model_size=model_size)
        
        # Save to the hard-coded file for Team 2 as per README
        output_filename = "transcription_output.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(segments, f, indent=4, ensure_ascii=False)
            
        return {
            "status": "success", 
            "message": f"Saved to {output_filename}",
            "segments": segments
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
