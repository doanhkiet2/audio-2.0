import json
import gc
import subprocess
from pathlib import Path
from tqdm import tqdm

import whisperx

# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_META = BASE_DIR / "output" / "test_build_train_audio" / "metadata_whisperx.jsonl"
AUDIO_BASE_DIR = INPUT_META.parent

OUTPUT_JSONL = (
    BASE_DIR
    / "output"
    / "test_build_train_audio"
    / "whisperx_forced_align_result.jsonl"
)

SEG_START_INDEX = 0
SEG_END_INDEX = 10

DEVICE = "cpu"
LANGUAGE = "vi"

# =========================
# LOAD
# =========================


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def resolve_audio_path(audio_value):
    p = Path(audio_value)

    if p.is_absolute() and p.exists():
        return p

    candidate = AUDIO_BASE_DIR / p

    if candidate.exists():
        return candidate

    return None


def get_audio_duration(path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    return float(result.stdout.strip())


# =========================
# MAIN
# =========================


def main():
    rows = load_jsonl(INPUT_META)

    start_idx = SEG_START_INDEX if SEG_START_INDEX is not None else 0
    end_idx = SEG_END_INDEX if SEG_END_INDEX is not None else len(rows)

    rows = rows[start_idx:end_idx]

    print(f"[INFO] Input meta : {INPUT_META}")
    print(f"[INFO] Output     : {OUTPUT_JSONL}")
    print(f"[INFO] Rows       : {len(rows)}")
    print(f"[INFO] Device     : {DEVICE}")
    print(f"[INFO] Mode       : forced align only")

    if OUTPUT_JSONL.exists():
        OUTPUT_JSONL.unlink()

    print("[INFO] Loading align model...")
    align_model, metadata = whisperx.load_align_model(
        language_code=LANGUAGE,
        device=DEVICE,
    )

    kept = 0
    skipped = 0
    errors = 0

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out:
        for i, row in enumerate(tqdm(rows, desc="WhisperX forced align")):
            audio_value = row.get("audio")
            audio_path = resolve_audio_path(audio_value)

            text_expected = (row.get("text") or "").strip()

            if audio_path is None:
                print(f"[SKIP] Missing audio: {audio_value}")
                skipped += 1
                continue

            if not text_expected:
                print(f"[SKIP] Missing text at row {start_idx + i}")
                skipped += 1
                continue

            try:
                audio = whisperx.load_audio(str(audio_path))
                audio_duration = get_audio_duration(audio_path)

                forced_segments = [
                    {
                        "start": 0.0,
                        "end": audio_duration,
                        "text": text_expected,
                    }
                ]

                aligned = whisperx.align(
                    forced_segments,
                    align_model,
                    metadata,
                    audio,
                    DEVICE,
                    return_char_alignments=False,
                )

                out_row = {
                    "index": start_idx + i,
                    "audio": audio_value,
                    "text_expected": text_expected,
                    "original_start": row.get("original_start"),
                    "original_end": row.get("original_end"),
                    "cut_start": row.get("cut_start"),
                    "cut_end": row.get("cut_end"),
                    "audio_duration": round(audio_duration, 3),
                    "whisperx_segments": aligned.get("segments", []),
                }

                out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                kept += 1

            except Exception as e:
                print(f"[ERROR] row={start_idx + i} audio={audio_value}: {e}")
                errors += 1

            gc.collect()

    print("\n[DONE] WhisperX forced align completed")
    print(f"[DONE] Kept   : {kept}")
    print(f"[DONE] Skipped: {skipped}")
    print(f"[DONE] Errors : {errors}")
    print(f"[DONE] Output : {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
