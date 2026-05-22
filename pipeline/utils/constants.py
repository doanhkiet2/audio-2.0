from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw" / "audio"
WAV16K_DIR = DATA_DIR / "processed" / "wav16k"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"

for p in [RAW_DIR, WAV16K_DIR, TRANSCRIPT_DIR]:
    p.mkdir(parents=True, exist_ok=True)