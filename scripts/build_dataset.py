import os
import re
import json

from pathlib import Path


# =========================
# CONFIG
# =========================

ALIGNED_FILE = "output/aligned.txt"
OUTPUT_JSON = "output/dataset.json"

MIN_DURATION = 1.0
MAX_DURATION = 20.0

MIN_SCORE = 85  # lọc noise nhẹ


os.makedirs("output", exist_ok=True)


# =========================
# PARSE LINE
# =========================

def parse_line(line):
    """
    Format:
    [start - end] (score=xx) text
    """

    match = re.match(
        r"\[(.*?)\s*-\s*(.*?)\]\s*\(score=(.*?)\)\s*(.*)",
        line.strip()
    )

    if not match:
        return None

    return {
        "start": float(match.group(1)),
        "end": float(match.group(2)),
        "score": float(match.group(3)),
        "text": match.group(4).strip()
    }


# =========================
# CLEAN TEXT
# =========================

def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# =========================
# QUALITY LABEL
# =========================

def get_quality(score):

    if score >= 95:
        return "good"
    elif score >= 85:
        return "medium"
    else:
        return "bad"


# =========================
# MAIN
# =========================

def build_dataset():

    dataset = []

    with open(ALIGNED_FILE, "r", encoding="utf-8") as f:

        for line in f:

            parsed = parse_line(line)
            if not parsed:
                continue

            start = parsed["start"]
            end = parsed["end"]
            score = parsed["score"]
            text = clean_text(parsed["text"])

            duration = end - start

            # =========================
            # FILTER NOISE
            # =========================

            if duration < MIN_DURATION:
                continue

            if duration > MAX_DURATION:
                continue

            if score < MIN_SCORE:
                continue

            # =========================
            # BUILD ITEM
            # =========================

            item = {
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(duration, 2),
                "text": text,
                "score": round(score, 2),
                "quality": get_quality(score)
            }

            dataset.append(item)

    # =========================
    # SAVE JSON
    # =========================

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print("\n======================")
    print("DATASET BUILT DONE")
    print("======================")
    print(f"Total samples: {len(dataset)}")
    print(f"Saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    build_dataset()