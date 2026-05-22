import json
from pathlib import Path

import whisper
import torchaudio
from silero_vad import load_silero_vad, get_speech_timestamps

# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

AUDIO_FILE = BASE_DIR / "data" / "raw" / "audio" / "09_04__00_00__IuoozJ_9QyQ.wav"
OUTPUT_FILE = BASE_DIR / "output" / "09_04__00_00__IuoozJ_9QyQ.json"

BASE_DIR.joinpath("output").mkdir(exist_ok=True)


# =========================
# LOAD MODELS
# =========================

print("Loading models...")

vad_model = load_silero_vad()
whisper_model = whisper.load_model("base")


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

speech = get_speech_timestamps(wav, vad_model, sampling_rate=sr, threshold=0.5)

segments = []

for t in speech:
    start = t["start"] / sr
    end = t["end"] / sr

    duration = end - start

    if 1.0 <= duration <= 20.0:
        segments.append((start, end))


print(f"VAD segments: {len(segments)}")


# =========================
# WHISPER (ONLY ONCE)
# =========================

print("Running Whisper (full audio)...")

result = whisper_model.transcribe(str(AUDIO_FILE))
whisper_segments = result["segments"]


# =========================
# ALIGN (STABLE + NO DRIFT)
# =========================

results = []
used = set()
for v_start, v_end in segments:

    matched_indices = []

    for i, w in enumerate(whisper_segments):

        if i in used:
            continue

        w_start = w["start"]
        w_end = w["end"]

        # overlap check
        overlap = max(0, min(v_end, w_end) - max(v_start, w_start))

        if overlap > 0:
            matched_indices.append(i)

    if not matched_indices:
        continue

    # =========================
    # FIX CORE BUG HERE
    # =========================

    start_time = min(whisper_segments[i]["start"] for i in matched_indices)
    end_time = max(whisper_segments[i]["end"] for i in matched_indices)

    text = " ".join(whisper_segments[i]["text"].strip() for i in matched_indices)

    # mark used
    for i in matched_indices:
        used.add(i)

    # FILTER
    if len(text.split()) < 2:
        continue

    results.append(
        {"start": round(start_time, 2), "end": round(end_time, 2), "text": text}
    )


# =========================
# SAVE OUTPUT
# =========================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)


print("\nDONE")
print(f"Saved: {OUTPUT_FILE}")
print(f"Segments: {len(results)}")
