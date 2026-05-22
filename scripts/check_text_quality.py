import json
import re
from pathlib import Path
from collections import Counter
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]

META_FILE = BASE_DIR / "dataset_final" / "metadata_final.jsonl"
REPORT_FILE = BASE_DIR / "dataset_final" / "text_quality_report.txt"

NUMBER_RE = re.compile(r"\d")
WEIRD_RE = re.compile(r"[^a-zA-ZÀ-ỹà-ỹ0-9\s.,!?;:'\"“”‘’…\-–—()]")


def main():
    rows = []

    with open(META_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    number_rows = []
    weird_rows = []
    char_counter = Counter()

    for idx, row in enumerate(tqdm(rows, desc="Checking text")):
        text = row.get("text", "")

        char_counter.update(text)

        if NUMBER_RE.search(text):
            number_rows.append((idx, row["audio"], text))

        if WEIRD_RE.search(text):
            weird_rows.append((idx, row["audio"], text))

    with open(REPORT_FILE, "w", encoding="utf-8") as out:
        out.write("========== TEXT QUALITY REPORT ==========\n")
        out.write(f"Total rows: {len(rows)}\n")
        out.write(f"Rows with numbers: {len(number_rows)}\n")
        out.write(f"Rows with weird chars: {len(weird_rows)}\n\n")

        out.write("========== ROWS WITH NUMBERS ==========\n")
        for idx, audio, text in number_rows:
            out.write(f"[{idx}] {audio}\n{text}\n\n")

        out.write("========== ROWS WITH WEIRD CHARS ==========\n")
        for idx, audio, text in weird_rows:
            out.write(f"[{idx}] {audio}\n{text}\n\n")

        out.write("========== TOP CHARACTERS ==========\n")
        for char, count in char_counter.most_common():
            out.write(f"{repr(char)}: {count}\n")

    print("[DONE] Text quality check completed")
    print(f"[DONE] Total rows: {len(rows)}")
    print(f"[DONE] Rows with numbers: {len(number_rows)}")
    print(f"[DONE] Rows with weird chars: {len(weird_rows)}")
    print(f"[DONE] Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
