import json
from pathlib import Path

import torch
import torchaudio
import whisper
from pydub import AudioSegment
from silero_vad import load_silero_vad, get_speech_timestamps


# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

AUDIO_FILE = BASE_DIR / "data" / "raw" / "audio" / "test.wav"
OUTPUT_FILE = BASE_DIR / "output" / "vad_whisper.json"

BASE_DIR.joinpath("output").mkdir(exist_ok=True)


# =========================
# LOAD MODELS
# =========================

print("Loading models...")

vad_model = load_silero_vad()
whisper_model = whisper.load_model("base")


# =========================
# LOAD AUDIO (TORCH FOR VAD)
# =========================

def load_audio(path):
    wav, sr = torchaudio.load(path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        sr = 16000

    return wav.squeeze(0), sr


wav, sr = load_audio(AUDIO_FILE)


# =========================
# VAD SEGMENTATION
# =========================

print("Running VAD...")

speech_timestamps = get_speech_timestamps(
    wav,
    vad_model,
    sampling_rate=sr,
    threshold=0.5
)

segments = []

for t in speech_timestamps:

    start = t["start"] / sr
    end = t["end"] / sr

    duration = end - start

    # filter noise
    if 1.0 <= duration <= 15.0:
        segments.append((start, end))


print(f"VAD segments: {len(segments)}")


# =========================
# LOAD AUDIO (PYDUB FOR CUTTING)
# =========================

audio = AudioSegment.from_file(AUDIO_FILE)


# =========================
# WHISPER PROCESS
# =========================

results = []

print("Running Whisper...")

for i, (start, end) in enumerate(segments):

    # margin padding
    start_p = max(0, start - 0.2)
    end_p = end + 0.2

    chunk = audio[int(start_p * 1000): int(end_p * 1000)]

    tmp_path = BASE_DIR / "output" / "temp.wav"
    chunk.export(tmp_path, format="wav")

    res = whisper_model.transcribe(str(tmp_path))

    text = res["text"].strip()

    if len(text.split()) < 2:
        continue

    results.append({
        "start": round(start, 2),
        "end": round(end, 2),
        "text": text
    })

    if i % 10 == 0:
        print(f"Processed {i}/{len(segments)}")


# =========================
# SAVE OUTPUT
# =========================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)


print("\nDONE")
print(f"Saved: {OUTPUT_FILE}")
print(f"Segments: {len(results)}")