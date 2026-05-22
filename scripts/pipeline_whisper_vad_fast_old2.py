import json
from pathlib import Path

import whisper
import torch
import torchaudio
from silero_vad import load_silero_vad, get_speech_timestamps


# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

AUDIO_FILE = BASE_DIR / "data" / "raw" / "audio" / "test.wav"
OUTPUT_FILE = BASE_DIR / "output" / "dataset.json"

BASE_DIR.joinpath("output").mkdir(exist_ok=True)


# =========================
# LOAD MODELS
# =========================

print("Loading models...")

whisper_model = whisper.load_model("base")
vad_model = load_silero_vad()


# =========================
# LOAD AUDIO FOR VAD
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
# VAD SEGMENTS
# =========================

print("Running VAD...")

speech = get_speech_timestamps(
    wav,
    vad_model,
    sampling_rate=sr,
    threshold=0.5
)

segments = []

for t in speech:
    start = t["start"] / sr
    end = t["end"] / sr

    duration = end - start

    if 1.0 <= duration <= 15.0:
        segments.append((start, end))


print(f"VAD segments: {len(segments)}")


# =========================
# WHISPER (1 TIME ONLY)
# =========================

print("Running Whisper...")

result = whisper_model.transcribe(str(AUDIO_FILE))
whisper_segments = result["segments"]


# =========================
# CLEAN ALIGN (NO DUPLICATE)
# =========================

def overlap(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))


used_whisper = set()
results = []

print("Aligning...")

for v_start, v_end in segments:

    best_idx = -1
    best_score = 0

    for i, w in enumerate(whisper_segments):

        if i in used_whisper:
            continue

        w_start = w["start"]
        w_end = w["end"]

        score = overlap(v_start, v_end, w_start, w_end)

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx != -1:
        if best_score < 0.3:
            continue
        used_whisper.add(best_idx)
        text = whisper_segments[best_idx]["text"].strip()

        if len(text.split()) < 3:
            continue
        results.append({
            "start": round(v_start, 2),
            "end": round(v_end, 2),
            "text": whisper_segments[best_idx]["text"].strip()
        })


# =========================
# SAVE OUTPUT
# =========================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)


print("\nDONE")
print(f"Saved: {OUTPUT_FILE}")
print(f"Segments: {len(results)}")