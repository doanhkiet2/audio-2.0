import json
import re
from pathlib import Path
from collections import Counter
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = BASE_DIR / "output" / "aligned"
REPORT_FILE = BASE_DIR / "output" / "reports" / "pre_factory_text_quality_report.txt"

NUMBER_RE = re.compile(r"\d")
WEIRD_RE = re.compile(r"[^a-zA-ZÀ-ỹà-ỹ0-9\s.,!?;:'\"“”‘’…\-–—()]")

TEXT_FIELDS = ["matched_text", "whisper_text"]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    return data


def main():
    json_files = sorted(INPUT_DIR.glob("*.json"))

    print(f"[INFO] Input dir: {INPUT_DIR}")
    print(f"[INFO] Found json files: {len(json_files)}")
    print(f"[INFO] Report: {REPORT_FILE}")

    total_segments = 0
    number_rows = []
    weird_rows = []
    char_counter = Counter()

    for json_file in tqdm(json_files, desc="Checking aligned json"):
        try:
            rows = load_json(json_file)
        except Exception as e:
            print(f"[ERROR] Cannot read {json_file.name}: {e}")
            continue

        if not isinstance(rows, list):
            print(f"[SKIP] Not a list: {json_file.name}")
            continue

        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue

            total_segments += 1

            for field in TEXT_FIELDS:
                text = row.get(field, "")

                if not text:
                    continue

                char_counter.update(text)

                if NUMBER_RE.search(text):
                    number_rows.append(
                        {
                            "file": json_file.name,
                            "index": idx,
                            "field": field,
                            "start": row.get("start"),
                            "end": row.get("end"),
                            "text": text,
                        }
                    )

                if WEIRD_RE.search(text):
                    weird_rows.append(
                        {
                            "file": json_file.name,
                            "index": idx,
                            "field": field,
                            "start": row.get("start"),
                            "end": row.get("end"),
                            "text": text,
                        }
                    )

    with open(REPORT_FILE, "w", encoding="utf-8") as out:
        out.write("========== PRE FACTORY TEXT QUALITY REPORT ==========\n")
        out.write(f"Input dir: {INPUT_DIR}\n")
        out.write(f"Json files: {len(json_files)}\n")
        out.write(f"Total segments: {total_segments}\n")
        out.write(f"Rows with numbers: {len(number_rows)}\n")
        out.write(f"Rows with weird chars: {len(weird_rows)}\n\n")

        out.write("========== ROWS WITH NUMBERS ==========\n")
        for item in number_rows:
            out.write(
                f"[{item['file']} | idx={item['index']} | {item['field']} "
                f"| {item['start']} - {item['end']}]\n"
            )
            out.write(item["text"] + "\n\n")

        out.write("========== ROWS WITH WEIRD CHARS ==========\n")
        for item in weird_rows:
            out.write(
                f"[{item['file']} | idx={item['index']} | {item['field']} "
                f"| {item['start']} - {item['end']}]\n"
            )
            out.write(item["text"] + "\n\n")

        out.write("========== TOP CHARACTERS ==========\n")
        for char, count in char_counter.most_common():
            out.write(f"{repr(char)}: {count}\n")

    print("[DONE] Pre-factory text quality check completed")
    print(f"[DONE] Total segments: {total_segments}")
    print(f"[DONE] Rows with numbers: {len(number_rows)}")
    print(f"[DONE] Rows with weird chars: {len(weird_rows)}")
    print(f"[DONE] Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
