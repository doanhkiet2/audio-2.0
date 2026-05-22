import yt_dlp
from datetime import datetime
import os

youtube_link = "https://www.youtube.com/watch?v=IuoozJ_9QyQ"
output_folder = "./data/raw/audio"


def make_filename(info):
    upload_date = info.get("upload_date")
    if upload_date:
        dt = datetime.strptime(upload_date, "%Y%m%d")
    else:
        dt = datetime.now()

    mm_dd = dt.strftime("%m_%d")
    hh_mm = dt.strftime("%H_%M")

    video_id = info.get("id", "unknown")

    return f"{mm_dd}__{hh_mm}__{video_id}"


def download_youtube_wav(url, output_dir="."):

    os.makedirs(output_dir, exist_ok=True)

    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        filename = make_filename(info)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{output_dir}/{filename}.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        "postprocessor_args": ["-ac", "1", "-ar", "24000", "-sample_fmt", "s16"],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


download_youtube_wav(youtube_link, output_folder)
