import yt_dlp
import os
import sys

def download_track(url, filename="track.mp3"):
    """
    Downloads high-quality audio from YouTube and saves it as an MP3.
    """
    print(f"[*] Downloading: {url}")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': filename.replace('.mp3', ''), # yt-dlp adds .mp3 automatically
        'quiet': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    print(f"[SUCCESS] Downloaded to {filename}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python dj_downloader.py <youtube_url> <track1|track2>")
    else:
        url = sys.argv[1]
        name = sys.argv[2] + ".mp3"
        download_track(url, name)
