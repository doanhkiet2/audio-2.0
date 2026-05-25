import json
import re
from pathlib import Path
from tqdm import tqdm

# =========================
# CONFIG
# =========================

ALIGN_JSON_PATH = Path("data/align_input/align_map.json")

INPUT_DIR = None
OUTPUT_DIR = None

# Nếu muốn chạy theo folder thì sửa thành:
# INPUT_DIR = Path("data/transcripts")
# OUTPUT_DIR = Path("data/transcripts_clean")

REJECT_FILE = Path("data/youtube_rejected.txt")
missing_files = []

# =========================
# FORMAT TIME
# =========================


def format_time(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    return f"{h:02}:{m:02}:{s:02}"


# =========================
# PARSE LINE
# =========================


def parse_line(line):
    line = line.strip()

    if not line:
        return None

    line = line.replace("\ufeff", "")
    line = re.sub(r"\s+", " ", line)

    match = re.match(r"^(\d{1,2}:\d{2}(?::\d{2})?)(.*)$", line)

    if not match:
        return None

    timestamp = match.group(1)
    text = match.group(2).strip()

    parts = timestamp.split(":")

    try:
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
            total_seconds = minutes * 60 + seconds

        elif len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            total_seconds = hours * 3600 + minutes * 60 + seconds

        else:
            return None

    except Exception:
        return None

        # =========================

    # REMOVE YOUTUBE GARBAGE
    # =========================

    # Vietnamese:
    # "3 giờ, 1 phút, 2 giây"
    # "1 giờ, 1 giây"
    # "3 giờ"
    text = re.sub(
        r"^\d+\s*giờ" r"(?:\s*,?\s*\d+\s*phút)?" r"(?:\s*,?\s*\d+\s*giây)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Vietnamese:
    # "1 phút, 6 giây"
    # "1 phút"
    text = re.sub(
        r"^\d+\s*phút" r"(?:\s*,?\s*\d+\s*giây)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Vietnamese:
    # "44 giây"
    text = re.sub(
        r"^\d+\s*giây",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # English:
    # "3 hours, 1 minute, 2 seconds"
    # "1 hour, 1 second"
    # "3 hours"
    text = re.sub(
        r"^\d+\s*hours?" r"(?:\s*,?\s*\d+\s*minutes?)?" r"(?:\s*,?\s*\d+\s*seconds?)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # English:
    # "1 minute, 3 seconds"
    # "1 minute"
    text = re.sub(
        r"^\d+\s*minutes?" r"(?:\s*,?\s*\d+\s*seconds?)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # English:
    # "3 seconds"
    text = re.sub(
        r"^\d+\s*seconds?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # cleanup spaces
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return None

    return {
        "time": format_time(total_seconds),
        "text": text,
    }


# =========================
# CLEAN ONE FILE
# =========================


def clean_one_file(input_file, output_file):
    input_file = Path(input_file)
    output_file = Path(output_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    REJECT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    cleaned = []
    rejected = []

    for line in raw_lines:
        parsed = parse_line(line)

        if parsed:
            cleaned.append(parsed)
        else:
            rejected.append(line.strip())

    final_lines = []
    seen = set()

    for item in cleaned:
        key = (item["time"], item["text"])

        if key not in seen:
            seen.add(key)
            final_lines.append(item)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in final_lines:
            f.write(f"{item['time']} {item['text']}\n")

    with open(REJECT_FILE, "a", encoding="utf-8") as f:
        for line in rejected:
            f.write(f"{input_file} | {line}\n")

    return len(final_lines), len(rejected)


# =========================
# BUILD JOBS
# =========================


def build_jobs_from_folder(input_dir, output_dir):
    jobs = []

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for input_file in sorted(input_dir.glob("*.txt")):
        output_file = output_dir / input_file.name

        jobs.append(
            {
                "input_file": input_file,
                "output_file": output_file,
            }
        )

    return jobs


def build_jobs_from_align_json(align_json_path):
    with open(align_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    jobs = []

    for item in data["items"]:
        input_file = Path(item["source_youtube_path"])

        output_file = input_file.with_name(
            input_file.stem + "_clean" + input_file.suffix
        )

        jobs.append(
            {
                "input_file": input_file,
                "output_file": output_file,
            }
        )

    return jobs


# =========================
# MAIN
# =========================

if REJECT_FILE.exists():
    REJECT_FILE.unlink()

if INPUT_DIR and OUTPUT_DIR:
    jobs = build_jobs_from_folder(INPUT_DIR, OUTPUT_DIR)
else:
    jobs = build_jobs_from_align_json(ALIGN_JSON_PATH)

print(f"\nTotal jobs: {len(jobs)}\n")

total_cleaned = 0
total_rejected = 0
missing = 0

for job in tqdm(jobs):
    input_file = job["input_file"]
    output_file = job["output_file"]

    if not input_file.exists():
        print(f"MISSING: {input_file}")
        missing += 1
        continue

    cleaned_count, rejected_count = clean_one_file(
        input_file=input_file,
        output_file=output_file,
    )

    total_cleaned += cleaned_count
    total_rejected += rejected_count

print("\n======================")
print(f"Jobs     : {len(jobs)}")
print(f"Missing  : {missing}")
print(f"Cleaned  : {total_cleaned}")
print(f"Rejected : {total_rejected}")
print("======================\n")

print("Saved rejected lines:")
print(REJECT_FILE)
