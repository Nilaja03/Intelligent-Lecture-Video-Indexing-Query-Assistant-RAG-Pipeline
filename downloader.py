import yt_dlp
import os

def download_audio(youtube_url: str, output_dir: str = "downloads") -> str:
    """
    Downloads the audio from a YouTube video and converts it to MP3 using yt-dlp.
    
    Args:
        youtube_url (str): The URL of the YouTube video.
        output_dir (str): The directory where the MP3 will be saved.
        
    Returns:
        str: The path to the downloaded MP3 file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '32',
        }],
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'quiet': False
    }
    
    print(f"Downloading audio from: {youtube_url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info['id']
        mp3_path = os.path.join(output_dir, f"{video_id}.mp3")
        return mp3_path
