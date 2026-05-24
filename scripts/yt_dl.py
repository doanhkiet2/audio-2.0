import yt_dlp
from datetime import datetime
import os

youtube_link = "https://www.youtube.com/watch?v=HfXCWVQIVF8"
output_folder = "./data/raw/audio/nuoiconchophu"

youtube_link_ar = [
    # "https://www.youtube.com/watch?v=-AgoLffPA98",
    # "https://www.youtube.com/watch?v=C9MpEZ-BWNE",
    # "https://www.youtube.com/watch?v=x6Kav8s5jSo",
    # "https://www.youtube.com/watch?v=NBVlwn9rccU",
    # "https://www.youtube.com/watch?v=hZYNwIcK64A",
    # "https://www.youtube.com/watch?v=MofmySg0oGU",
    # "https://www.youtube.com/watch?v=rDhMPGEgQu0",
    # "https://www.youtube.com/watch?v=VGp7hMzI164",
    # "https://www.youtube.com/watch?v=xw1L_RTxWqc",
    # "https://www.youtube.com/watch?v=JBaRKyqy8hM",
    # "https://www.youtube.com/watch?v=P_8vZprxMYg",
    # "https://www.youtube.com/watch?v=B4Cm9HKApz4",
    # "https://www.youtube.com/watch?v=3pT7rwFR0Fc",
    # "https://www.youtube.com/watch?v=bkW6L5L_c2Q",
    # "https://www.youtube.com/watch?v=U7U1kItj5aE",
    # "https://www.youtube.com/watch?v=GdPjwvlh0eU",
    # "https://www.youtube.com/watch?v=6lRUgVGAbPw",
    # "https://www.youtube.com/watch?v=qAud0zSa9u8",
    # "https://www.youtube.com/watch?v=WDGqYOlbdHc",
    # "https://www.youtube.com/watch?v=ymNJCO8RtWM",
    # "https://www.youtube.com/watch?v=_4LGwlwO9mc",
    # "https://www.youtube.com/watch?v=AFV-3nr2Ivs",
    # "https://www.youtube.com/watch?v=VKswn2BETWA",
    # "https://www.youtube.com/watch?v=oO0cFuWQjMU",
    # "https://www.youtube.com/watch?v=b1HQ3Z2YwPc",
    # "https://www.youtube.com/watch?v=51Csh3gvFEo",
    # "https://www.youtube.com/watch?v=Ca9sd_g668E",
    # "https://www.youtube.com/watch?v=5FoIfME92AM",
    # "https://www.youtube.com/watch?v=Y9uriqGNMfw",
    # "https://www.youtube.com/watch?v=I1dNk_-1y8U",
    # "https://www.youtube.com/watch?v=xQuz9ZfVtrg",
    # "https://www.youtube.com/watch?v=8jqUg6bTwsY",
    # "https://www.youtube.com/watch?v=Y7nU4l0VQvQ",
    # "https://www.youtube.com/watch?v=6pMmFGPDB8s",
    # "https://www.youtube.com/watch?v=dZl4rIAZVlo",
    "https://www.youtube.com/watch?v=4VWuiXuMwFs",
    "https://www.youtube.com/watch?v=59tJ85fYwVI",
    "https://www.youtube.com/watch?v=zrPNW_dRrB8",
    "https://www.youtube.com/watch?v=eIgJaDVrhs4",
    "https://www.youtube.com/watch?v=R1M3UIEViuc",
    "https://www.youtube.com/watch?v=wf6w0642tTk",
    "https://www.youtube.com/watch?v=0Il1iZwc6OM",
    "https://www.youtube.com/watch?v=sJPECoux9jE",
    "https://www.youtube.com/watch?v=EmS8rLj9mQ8",
    "https://www.youtube.com/watch?v=7CWlhYEQ2l4",
    "https://www.youtube.com/watch?v=j8IOJHzUPmc",
    "https://www.youtube.com/watch?v=oeSSytdRm4s",
    "https://www.youtube.com/watch?v=LNUelJ3q4xA",
]


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


for i in youtube_link_ar:
    try:
        download_youtube_wav(i, output_folder)
        print("DONE:", i)
    except Exception as e:
        print("ERROR:", i, e)
