import re
from tqdm import tqdm

INPUT_FILE = "data/transcripts/gabieude8.txt"
OUTPUT_FILE = "data/transcripts/gabieude8_clean.txt"


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

    # remove invisible unicode chars
    line = line.replace("\ufeff", "")

    # normalize weird spaces
    line = re.sub(r"\s+", " ", line)

    # =========================
    # MATCH TIMESTAMP
    # supports:
    # 0:04abc
    # 1:06xyz
    # 01:02:03hello
    # =========================
    match = re.match(r"^(\d{1,2}:\d{2}(?::\d{2})?)(.*)$", line)

    if not match:
        return None

    timestamp = match.group(1)
    text = match.group(2).strip()

    # =========================
    # CONVERT TIME
    # =========================
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

    except:
        return None

    # =========================
    # REMOVE YOUTUBE GARBAGE
    # =========================

    # remove:
    # "44 giây"
    text = re.sub(r"^\d+\s*giây", "", text, flags=re.IGNORECASE)

    # remove:
    # "1 phút, 6 giây"
    text = re.sub(r"^\d+\s*phút\s*,?\s*\d*\s*giây?", "", text, flags=re.IGNORECASE)

    # remove duplicated timestamp garbage
    text = re.sub(r"^\d+\s*phút", "", text, flags=re.IGNORECASE)

    # cleanup spaces
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return None

    return {"time": format_time(total_seconds), "text": text}


# =========================
# LOAD
# =========================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_lines = f.readlines()

print(f"\nLoaded {len(raw_lines)} raw lines\n")

# =========================
# PROCESS
# =========================
cleaned = []
rejected = []

for line in tqdm(raw_lines):

    parsed = parse_line(line)

    if parsed:
        cleaned.append(parsed)
    else:
        rejected.append(line.strip())


# =========================
# DEDUPLICATE
# =========================
final_lines = []

seen = set()

for item in cleaned:

    key = (item["time"], item["text"])

    if key not in seen:

        seen.add(key)

        final_lines.append(item)


# =========================
# SAVE CLEAN
# =========================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for item in final_lines:

        f.write(f"{item['time']} {item['text']}\n")


# =========================
# SAVE REJECT
# =========================
with open("data/youtube_rejected.txt", "w", encoding="utf-8") as f:

    for line in rejected:
        f.write(line + "\n")


# =========================
# DONE
# =========================
print("\n======================")
print(f"Cleaned : {len(final_lines)}")
print(f"Rejected: {len(rejected)}")
print("======================\n")

print(f"Saved clean transcript:")
print(OUTPUT_FILE)

print(f"\nSaved rejected lines:")
print("data/youtube_rejected.txt\n")
