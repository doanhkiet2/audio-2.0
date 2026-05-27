import json
import subprocess
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]

# =========================
# CONFIG
# =========================

INPUT_META = BASE_DIR / "output" / "factory_output" / "factory_dataset_added.jsonl"

AUDIO_INPUT_DIR = BASE_DIR / "data" / "input_audio"

AUDIO_OUT_DIR = BASE_DIR / "output" / "vad_trim_test" / "input"
FINAL_META = BASE_DIR / "output" / "test_build_train_audio" / "metadata_whisperx.jsonl"

# None = chạy từ đầu / tới cuối
SEG_START_INDEX = 75000
SEG_END_INDEX = 75200
# Ví dụ test:
# SEG_START_INDEX = 0
# SEG_END_INDEX = 100

PADDING_SECONDS = 1.5

OUTPUT_SAMPLE_RATE = 16000

AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# HELPERS
# =========================


def resolve_audio_path(audio_path: str) -> Path | None:
    p = Path(audio_path)

    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(BASE_DIR / p)
        candidates.append(AUDIO_INPUT_DIR / p.name)

    for c in candidates:
        if c.exists():
            return c

    return None


def get_audio_duration(path: Path) -> float | None:
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

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        return float(result.stdout.strip())

    except Exception:
        return None


def run_ffmpeg_cut(
    input_audio: Path,
    output_audio: Path,
    start: float,
    end: float,
):
    duration = end - start

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start),
        "-i",
        str(input_audio),
        "-t",
        str(duration),
        "-ac",
        "1",
        "-ar",
        str(OUTPUT_SAMPLE_RATE),
        str(output_audio),
    ]

    subprocess.run(cmd, check=True)


def load_jsonl(path: Path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


# =========================
# MAIN
# =========================


def main():
    rows = load_jsonl(INPUT_META)

    start_idx = SEG_START_INDEX if SEG_START_INDEX is not None else 0
    end_idx = SEG_END_INDEX if SEG_END_INDEX is not None else len(rows)

    rows_to_process = rows[start_idx:end_idx]

    print(f"[INFO] Input metadata : {INPUT_META}")
    print(f"[INFO] Audio input dir: {AUDIO_INPUT_DIR}")
    print(f"[INFO] Total rows     : {len(rows)}")
    print(f"[INFO] Processing     : {start_idx} -> {end_idx}")
    print(f"[INFO] Rows selected  : {len(rows_to_process)}")
    print(f"[INFO] Audio output   : {AUDIO_OUT_DIR}")
    print(f"[INFO] Final metadata : {FINAL_META}")
    print(f"[INFO] Padding        : {PADDING_SECONDS}s")
    print(f"[INFO] Sample rate    : {OUTPUT_SAMPLE_RATE}")

    kept = 0
    skipped = 0
    errors = 0

    with open(FINAL_META, "w", encoding="utf-8") as out:
        for local_i, row in enumerate(tqdm(rows_to_process, desc="Cutting audio")):
            source_index = start_idx + local_i

            audio_path = row.get("audio_path")
            text = (row.get("text") or row.get("matched_text") or "").strip()
            start = row.get("start")
            end = row.get("end")

            if not audio_path or not text or start is None or end is None:
                skipped += 1
                continue

            input_audio = resolve_audio_path(audio_path)

            if input_audio is None:
                print(f"[SKIP] Missing audio: {audio_path}")
                skipped += 1
                continue

            start = float(start)
            end = float(end)

            if end <= start:
                print(f"[SKIP] Bad time: start={start}, end={end}")
                skipped += 1
                continue

            audio_duration = get_audio_duration(input_audio)

            cut_start = max(0.0, start - PADDING_SECONDS)
            cut_end = end + PADDING_SECONDS

            if audio_duration is not None:
                cut_end = min(audio_duration, cut_end)

            if cut_end <= cut_start:
                print(f"[SKIP] Bad padded time: {cut_start} - {cut_end}")
                skipped += 1
                continue

            output_name = f"{kept:06d}.wav"
            output_audio = AUDIO_OUT_DIR / output_name

            try:
                run_ffmpeg_cut(
                    input_audio=input_audio,
                    output_audio=output_audio,
                    start=cut_start,
                    end=cut_end,
                )

            except subprocess.CalledProcessError as e:
                print(f"[ERROR] ffmpeg failed at row {source_index}: {e}")
                errors += 1
                continue

            final_row = {
                "audio": f"audio_whisperx/{output_name}",
                "text": text,
                "duration": round(cut_end - cut_start, 3),
                "original_start": start,
                "original_end": end,
                "cut_start": round(cut_start, 3),
                "cut_end": round(cut_end, 3),
                "padding_seconds": PADDING_SECONDS,
                "score": row.get("score"),
                "chars_per_sec": row.get("chars_per_sec"),
                "source_audio": str(input_audio),
                "source_audio_path": audio_path,
                "source_index": source_index,
            }

            out.write(json.dumps(final_row, ensure_ascii=False) + "\n")
            kept += 1

    print("\n[DONE] Build WhisperX audio completed")
    print(f"[DONE] Kept   : {kept}")
    print(f"[DONE] Skipped: {skipped}")
    print(f"[DONE] Errors : {errors}")
    print(f"[DONE] Audio  : {AUDIO_OUT_DIR}")
    print(f"[DONE] Meta   : {FINAL_META}")


if __name__ == "__main__":
    main()
