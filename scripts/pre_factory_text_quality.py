import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = BASE_DIR / "output" / "align_output" / "aligned"
REPORT_DIR = BASE_DIR / "output" / "reports"
REPORT_FILE = REPORT_DIR / "pre_factory_text_quality_report.txt"

TEXT_FIELDS = ["matched_text"]
# DEBUG_FIELDS = ["whisper_text"]

# TEXT_FIELDS = TRAIN_FIELDS
NUMBER_RE = re.compile(r"\d")

WEIRD_RE = re.compile(r"[^a-zA-ZÀ-ỹà-ỹ0-9\s.,!?;:'\"“”‘’…\-–—()]")


# =========================
# LOAD JSON
# =========================


# =========================
# EXTRACT WORD CONTEXT
# =========================


def extract_context(text, match, window=3):

    tokens = []

    for m in re.finditer(r"\S+", text):
        tokens.append(
            (
                m.group(),
                m.start(),
                m.end(),
            )
        )

    hit = None

    for i, (_, s, e) in enumerate(tokens):

        if s <= match.start() < e or s < match.end() <= e:
            hit = i
            break

    if hit is None:
        return ""

    left = max(0, hit - window)
    right = min(
        len(tokens),
        hit + window + 1,
    )

    return " ".join(t[0] for t in tokens[left:right])


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    return data


# =========================
# MAIN
# =========================


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(INPUT_DIR.rglob("*.json"))

    print(f"[INFO] Input dir: {INPUT_DIR}")
    print(f"[INFO] Found json files: {len(json_files)}")
    print(f"[INFO] Report: {REPORT_FILE}")

    total_segments = 0

    read_errors = []
    format_errors = []
    row_errors = []

    number_rows = []
    weird_rows = []

    char_counter = Counter()

    file_stats = defaultdict(
        lambda: {
            "segments": 0,
            "numbers": 0,
            "weird_chars": 0,
            "row_errors": 0,
        }
    )

    for json_file in tqdm(json_files, desc="Checking aligned json"):
        rel_file = json_file.relative_to(INPUT_DIR)

        try:
            rows = load_json(json_file)

        except Exception as e:
            read_errors.append(
                {
                    "file": str(rel_file),
                    "error": str(e),
                }
            )
            continue

        if not isinstance(rows, list):
            format_errors.append(
                {
                    "file": str(rel_file),
                    "error": f"Expected list, got {type(rows).__name__}",
                }
            )
            continue

        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                row_errors.append(
                    {
                        "file": str(rel_file),
                        "index": idx,
                        "error": f"Expected dict row, got {type(row).__name__}",
                    }
                )
                file_stats[str(rel_file)]["row_errors"] += 1
                continue

            total_segments += 1
            file_stats[str(rel_file)]["segments"] += 1

            for field in TEXT_FIELDS:
                text = row.get(field, "")

                if text is None:
                    text = ""

                if not isinstance(text, str):
                    row_errors.append(
                        {
                            "file": str(rel_file),
                            "index": idx,
                            "field": field,
                            "error": f"Expected string, got {type(text).__name__}",
                        }
                    )
                    file_stats[str(rel_file)]["row_errors"] += 1
                    continue

                if not text.strip():
                    continue

                char_counter.update(text)

                number_match = NUMBER_RE.search(text)

                if number_match:

                    number_rows.append(
                        {
                            "file": str(rel_file),
                            "index": idx,
                            "field": field,
                            "start": row.get("start"),
                            "end": row.get("end"),
                            "found": number_match.group(),
                            "context": extract_context(
                                text,
                                number_match,
                            ),
                            "text": text,
                        }
                    )

                    file_stats[str(rel_file)]["numbers"] += 1

                weird_match = WEIRD_RE.search(text)

                if weird_match:

                    weird_rows.append(
                        {
                            "file": str(rel_file),
                            "index": idx,
                            "field": field,
                            "start": row.get("start"),
                            "end": row.get("end"),
                            "found": weird_match.group(),
                            "context": extract_context(
                                text,
                                weird_match,
                            ),
                            "text": text,
                        }
                    )

                    file_stats[str(rel_file)]["weird_chars"] += 1

    with open(REPORT_FILE, "w", encoding="utf-8") as out:
        out.write("========== PRE FACTORY TEXT QUALITY REPORT ==========\n")
        out.write(f"Input dir: {INPUT_DIR}\n")
        out.write(f"Json files found: {len(json_files)}\n")
        out.write(f"Total segments checked: {total_segments}\n")
        out.write(f"Read errors: {len(read_errors)}\n")
        out.write(f"Format errors: {len(format_errors)}\n")
        out.write(f"Row errors: {len(row_errors)}\n")
        out.write(f"Rows with numbers: {len(number_rows)}\n")
        out.write(f"Rows with weird chars: {len(weird_rows)}\n\n")

        out.write("========== FILE SUMMARY ==========\n")
        for file, stat in sorted(file_stats.items()):
            out.write(
                f"{file} | "
                f"segments={stat['segments']} | "
                f"numbers={stat['numbers']} | "
                f"weird_chars={stat['weird_chars']} | "
                f"row_errors={stat['row_errors']}\n"
            )
        out.write("\n")

        out.write("========== READ ERRORS ==========\n")
        for item in read_errors:
            out.write(f"[{item['file']}]\n")
            out.write(item["error"] + "\n\n")

        out.write("========== FORMAT ERRORS ==========\n")
        for item in format_errors:
            out.write(f"[{item['file']}]\n")
            out.write(item["error"] + "\n\n")

        out.write("========== ROW ERRORS ==========\n")
        for item in row_errors:
            out.write(
                f"[{item['file']} | idx={item['index']} "
                f"| field={item.get('field', '-')}] "
                f"{item['error']}\n"
            )
        out.write("\n")

        out.write("========== ROWS WITH NUMBERS ==========\n")
        for item in number_rows:
            out.write(
                f"[{item['file']} | idx={item['index']} | {item['field']} "
                f"| {item['start']} - {item['end']}]\n"
            )
            out.write(f"FOUND   : {repr(item['found'])}\n")

            out.write(f"CONTEXT : {repr(item['context'])}\n")

            out.write(item["text"] + "\n\n")

        out.write("========== ROWS WITH WEIRD CHARS ==========\n")
        for item in weird_rows:
            out.write(
                f"[{item['file']} | idx={item['index']} | {item['field']} "
                f"| {item['start']} - {item['end']}]\n"
            )
            out.write(f"FOUND   : {repr(item['found'])}\n")

            out.write(f"CONTEXT : {repr(item['context'])}\n")

            out.write(item["text"] + "\n\n")

        out.write("========== TOP CHARACTERS ==========\n")
        for char, count in char_counter.most_common():
            out.write(f"{repr(char)}: {count}\n")

    print("\n[DONE] Pre-factory text quality check completed")
    print(f"[DONE] Json files found: {len(json_files)}")
    print(f"[DONE] Total segments: {total_segments}")
    print(f"[DONE] Read errors: {len(read_errors)}")
    print(f"[DONE] Format errors: {len(format_errors)}")
    print(f"[DONE] Row errors: {len(row_errors)}")
    print(f"[DONE] Rows with numbers: {len(number_rows)}")
    print(f"[DONE] Rows with weird chars: {len(weird_rows)}")
    print(f"[DONE] Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
