import os
import json
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm

# =========================
# CONFIG (SAFE PATH MODE)
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_FILE = BASE_DIR / "output" / "dataset.json"

AUDIO_SOURCE_DIR = BASE_DIR / "data" / "raw" / "audio"

OUTPUT_DIR = BASE_DIR / "dataset_final" / "audio"

META_FILE = BASE_DIR / "dataset_final" / "metadata.jsonl"

# create folders
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
META_FILE.parent.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD DATASET
# =========================

if not DATASET_FILE.exists():
    raise Exception(f"Dataset file not found: {DATASET_FILE}")

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# =========================
# FIND AUDIO FILES (mp3 + wav)
# =========================

audio_files = (
    list(AUDIO_SOURCE_DIR.glob("*.mp3")) +
    list(AUDIO_SOURCE_DIR.glob("*.wav"))
)

if not audio_files:
    raise Exception(f"No audio files found in: {AUDIO_SOURCE_DIR}")

# lấy file đầu tiên (bạn có thể nâng cấp multi-file sau)
audio_path = audio_files[0]

print("\n======================")
print("AUDIO LOADED")
print("======================")
print("File:", audio_path)

# =========================
# LOAD AUDIO (AUTO FORMAT)
# =========================

audio = AudioSegment.from_file(audio_path)

# =========================
# PROCESS CHUNKING
# =========================

metadata = []

print("\n======================")
print("CHUNKING AUDIO")
print("======================\n")

for i, item in enumerate(tqdm(dataset, desc="Cutting audio")):

    start = max(0, float(item["start"]) - 0.2)
    end = float(item["end"]) + 0.2
    text = item["text"]
    if len(text.split()) < 3:
        continue

    start_ms = int(start * 1000)
    end_ms = int(end * 1000)

    chunk = audio[start_ms:end_ms]

    # skip too small chunks
    if len(chunk) < 500:
        continue

    out_path = OUTPUT_DIR / f"{start:.2f}-{end:.2f}.wav"

    chunk.export(out_path, format="wav")

    metadata.append({
        "audio": str(out_path),
        "text": text,
        "start": start,
        "end": end,
        "duration": round(end - start, 2)
    })

# =========================
# SAVE METADATA
# =========================

with open(META_FILE, "w", encoding="utf-8") as f:
    for m in metadata:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

# =========================
# DONE
# =========================

print("\n======================")
print("DONE")
print("======================")
print(f"Audio file: {audio_path}")
print(f"Chunks created: {len(metadata)}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"Metadata file: {META_FILE}")