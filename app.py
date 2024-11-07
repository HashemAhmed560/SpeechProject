import yt_dlp
import whisper
import os

# Create necessary directories if they don't exist
for directory in ["Audio", "Caption", "Whisper"]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_audio(video_url, title):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'Audio/{title}.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

def download_caption(video_url, title):
    ydl_opts = {
        'writesubtitles': True,
        'subtitleslangs': ['en'],
        'outtmpl': f'Caption/{title}.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

def generate_subtitles(audio_file, title):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    whisper.utils.write_srt(result["segments"], file=f'Whisper/{title}.srt')

# Your search query
query = "Your Query Here"

# Search for videos
ydl_opts = {}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(f"ytsearch{N}:{query}", download=False)
    video_urls = [video['webpage_url'] for video in info_dict['entries']]

# Download audio, captions, and generate subtitles
for video_url in video_urls:
    with yt_dlp.YoutubeDL() as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        title = info_dict.get('title', None)

        if title:
            download_audio(video_url, title)
            try:
                download_caption(video_url, title)
            except:
                generate_subtitles(f'Audio/{title}.wav', title)
