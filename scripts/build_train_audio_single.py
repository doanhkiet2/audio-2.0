import json
import subprocess
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_META = BASE_DIR / "dataset_final" / "metadata.jsonl"

AUDIO_OUT_DIR = BASE_DIR / "dataset_final" / "audio"
FINAL_META = BASE_DIR / "dataset_final" / "metadata_final.jsonl"

AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_ffmpeg_cut(input_audio: Path, output_audio: Path, start: float, end: float):
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
        "22050",
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


def main():
    rows = load_jsonl(INPUT_META)

    print(f"[INFO] Input metadata: {INPUT_META}")
    print(f"[INFO] Total rows: {len(rows)}")
    print(f"[INFO] Audio output dir: {AUDIO_OUT_DIR}")
    print(f"[INFO] Final metadata: {FINAL_META}")

    kept = 0
    skipped = 0
    errors = 0

    with open(FINAL_META, "w", encoding="utf-8") as out:
        for i, row in enumerate(tqdm(rows, desc="Cutting audio")):
            audio_path = row.get("audio_path")
            text = row.get("text", "").strip()
            start = row.get("start")
            end = row.get("end")

            if not audio_path or not text or start is None or end is None:
                skipped += 1
                continue

            input_audio = BASE_DIR / audio_path

            if not input_audio.exists():
                print(f"[SKIP] Missing audio: {input_audio}")
                skipped += 1
                continue

            if end <= start:
                print(f"[SKIP] Bad time: start={start}, end={end}")
                skipped += 1
                continue

            output_name = f"{kept:06d}.wav"
            output_audio = AUDIO_OUT_DIR / output_name

            try:
                run_ffmpeg_cut(
                    input_audio=input_audio,
                    output_audio=output_audio,
                    start=float(start),
                    end=float(end),
                )
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] ffmpeg failed at row {i}: {e}")
                errors += 1
                continue

            final_row = {
                "audio": f"audio/{output_name}",
                "text": text,
                "duration": round(float(end) - float(start), 3),
                "score": row.get("score"),
                "chars_per_sec": row.get("chars_per_sec"),
                "source_audio": audio_path,
                "start": start,
                "end": end,
            }

            out.write(json.dumps(final_row, ensure_ascii=False) + "\n")
            kept += 1

    print("\n[DONE] Build train audio completed")
    print(f"[DONE] Kept: {kept}")
    print(f"[DONE] Skipped: {skipped}")
    print(f"[DONE] Errors: {errors}")
    print(f"[DONE] Audio dir: {AUDIO_OUT_DIR}")
    print(f"[DONE] Metadata: {FINAL_META}")


if __name__ == "__main__":
    main()
