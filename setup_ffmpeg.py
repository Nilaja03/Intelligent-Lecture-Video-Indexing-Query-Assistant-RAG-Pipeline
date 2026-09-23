import os
import urllib.request
import zipfile

url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
zip_path = "ffmpeg.zip"

print("Downloading FFmpeg... (This is ~130MB and might take a minute)")
urllib.request.urlretrieve(url, zip_path)

print("Extracting FFmpeg binaries...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for file_info in zip_ref.infolist():
        if file_info.filename.endswith("ffmpeg.exe"):
            with zip_ref.open(file_info) as source, open("ffmpeg.exe", "wb") as target:
                target.write(source.read())
        elif file_info.filename.endswith("ffprobe.exe"):
            with zip_ref.open(file_info) as source, open("ffprobe.exe", "wb") as target:
                target.write(source.read())

print("Cleaning up...")
os.remove(zip_path)

print("Done! ffmpeg.exe and ffprobe.exe have been securely placed in the current directory.")
