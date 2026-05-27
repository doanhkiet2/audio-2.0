import json
from youtube_transcript_api import YouTubeTranscriptApi


def ms(seconds: float) -> int:
    return int(seconds * 1000)


def ms_to_timestamp(ms: int) -> str:
    hours = ms // 3600000
    minutes = (ms % 3600000) // 60000
    seconds = (ms % 60000) // 1000
    milliseconds = ms % 1000

    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


# Example
start_ms = 137840
duration_ms = 4038
end_ms = 141878

print(ms_to_timestamp(start_ms))  # 00:02:17.840
print(ms_to_timestamp(duration_ms))  # 00:00:04.038
print(ms_to_timestamp(end_ms))  # 00:02:21.878


def download_transcript(video_id: str, output_file="transcript.json"):
    api = YouTubeTranscriptApi()

    # Chỉ định tiếng Việt
    fetched_transcript = api.fetch(video_id, languages=["vi"])

    result = []

    for item in fetched_transcript:
        start_ms = ms(item.start)
        duration_ms = ms(item.duration)

        result.append(
            {
                "text": item.text,
                "start_time": ms_to_timestamp(start_ms),
                "duration_time": ms_to_timestamp(duration_ms),
                "end_time": ms_to_timestamp(end_ms),
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Saved:", output_file)


if __name__ == "__main__":
    video_id = "IuoozJ_9QyQ"

    download_transcript(video_id)
