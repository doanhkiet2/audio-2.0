from scripts.vad_segment_old import split_by_vad
from pydub import AudioSegment
import whisper


model = whisper.load_model("base")


def process_audio(file_path):
    audio = AudioSegment.from_file(file_path)

    segments = split_by_vad(file_path)

    results = []

    for start, end in segments:

        chunk = audio[int(start * 1000): int(end * 1000)]

        tmp_path = "temp.wav"
        chunk.export(tmp_path, format="wav")

        res = model.transcribe(tmp_path)

        text = res["text"].strip()

        if len(text) > 0:
            results.append({
                "start": start,
                "end": end,
                "text": text
            })

    return results