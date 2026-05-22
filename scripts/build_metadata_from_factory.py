import json
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = BASE_DIR / "data" / "output_dataset"
OUTPUT_DIR = BASE_DIR / "dataset_final"
META_FILE = OUTPUT_DIR / "metadata.jsonl"

MIN_SCORE = 82

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    json_files = sorted(INPUT_DIR.glob("*.json"))

    print(f"[INFO] Input dir: {INPUT_DIR}")
    print(f"[INFO] Found json files: {len(json_files)}")
    print(f"[INFO] Output: {META_FILE}")

    total = 0
    kept = 0
    skipped = 0

    with open(META_FILE, "w", encoding="utf-8") as out:
        for json_file in tqdm(json_files, desc="Building metadata"):
            try:
                segments = load_json(json_file)
            except Exception as e:
                print(f"[ERROR] Cannot read {json_file.name}: {e}")
                continue

            if not isinstance(segments, list):
                print(f"[SKIP] {json_file.name} is not a list")
                continue

            for idx, seg in enumerate(segments):
                total += 1

                score = seg.get("score", 0)
                text = seg.get("matched_text", "").strip()
                audio_path = seg.get("audio_path", "")

                if score < MIN_SCORE:
                    skipped += 1
                    continue

                if not text:
                    skipped += 1
                    continue

                if not audio_path:
                    skipped += 1
                    continue

                row = {
                    "audio_path": audio_path,
                    "text": text,
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "score": score,
                    "chars_per_sec": seg.get("chars_per_sec"),
                    "source_json": json_file.name,
                }

                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1

    print("[DONE] Build metadata completed")
    print(f"[DONE] Total segments: {total}")
    print(f"[DONE] Kept: {kept}")
    print(f"[DONE] Skipped: {skipped}")
    print(f"[DONE] Saved: {META_FILE}")


if __name__ == "__main__":
    main()
