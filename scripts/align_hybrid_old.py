import os
import json
import re
from rapidfuzz import fuzz
from tqdm import tqdm

# ==================================================
# CONFIG
# ==================================================

WHISPER_FILE = "data/transcripts/test.json"
YOUTUBE_FILE = "data/youtube_clean.txt"

OUTPUT_FILE = "output/aligned_v1.json"
REJECT_FILE = "output/rejected_v1.json"

SIMILARITY_THRESHOLD = 75

MIN_DURATION = 1.0
MAX_DURATION = 20.0

MIN_CHARS_PER_SEC = 3
MAX_CHARS_PER_SEC = 35

PREVIOUS_SHIFT = 1.0

os.makedirs("output", exist_ok=True)


# ==================================================
# NORMALIZE
# ==================================================


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_count(text):
    return len(text.split())


# ==================================================
# LOAD WHISPER JSON (ONLY CHANGE)
# ==================================================

with open(WHISPER_FILE, "r", encoding="utf-8") as f:
    whisper_segments = json.load(f)


# ==================================================
# LOAD YOUTUBE TXT
# ==================================================


def hhmmss_to_seconds(time_str):
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s


youtube_segments = []

with open(YOUTUBE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        match = re.match(r"^(\d{2}:\d{2}:\d{2})\s+(.*)$", line)
        if not match:
            continue

        youtube_segments.append(
            {"start": hhmmss_to_seconds(match.group(1)), "text": match.group(2).strip()}
        )


# ==================================================
# EXTRACT BEST SUBSTRING (UNCHANGED V1 CORE)
# ==================================================


def extract_best_substring(whisper_text, context, min_words=4):

    words = context.split()

    best_score = 0
    best_text = context

    whisper_norm = normalize_text(whisper_text)
    whisper_words = word_count(whisper_norm)

    for start in range(len(words)):
        for end in range(start + min_words, len(words) + 1):

            candidate = " ".join(words[start:end])
            candidate_norm = normalize_text(candidate)

            base_score = fuzz.ratio(whisper_norm, candidate_norm)

            char_ratio = min(len(candidate_norm), len(whisper_norm)) / max(
                len(candidate_norm), len(whisper_norm)
            )
            word_ratio = min(whisper_words, word_count(candidate_norm)) / max(
                whisper_words, word_count(candidate_norm)
            )

            score = base_score * 0.7 + char_ratio * 100 * 0.15 + word_ratio * 100 * 0.15

            if candidate.strip().endswith((".", "!", "?", '"')):
                score += 5

            if score > best_score:
                best_score = score
                best_text = candidate

    return best_text, best_score


# ==================================================
# ALIGN
# ==================================================

accepted = []
rejected = []

print("\nALIGN V1 RUNNING...\n")

for w in tqdm(whisper_segments):

    w_start = w["start"]
    w_end = w["end"]
    whisper_text = w["text"]

    # =========================
    # GET CANDIDATES (V1 LOGIC)
    # =========================

    candidates = []
    candidate_set = set()

    # previous segment (UNCHANGED V1)
    previous_segment = None

    for y in youtube_segments:
        if y["start"] < w_start - PREVIOUS_SHIFT:
            previous_segment = y
        else:
            break

    if previous_segment:
        text = previous_segment["text"]
        candidates.append(text)
        candidate_set.add(text)

    # window inside VAD
    for y in youtube_segments:
        if w_start <= y["start"] <= w_end:
            if y["text"] not in candidate_set:
                candidates.append(y["text"])
                candidate_set.add(y["text"])

    if not candidates:
        rejected.append({"reason": "NO_CANDIDATE", **w})
        continue

    context = " ".join(candidates)

    # =========================
    # MATCH
    # =========================

    best_text, score = extract_best_substring(whisper_text, context)

    duration = w_end - w_start

    chars_per_sec = len(best_text) / duration if duration > 0 else 999

    accepted_flag = (
        score >= SIMILARITY_THRESHOLD
        and MIN_DURATION <= duration <= MAX_DURATION
        and MIN_CHARS_PER_SEC <= chars_per_sec <= MAX_CHARS_PER_SEC
    )

    result = {
        "start": w_start,
        "end": w_end,
        "score": round(score, 1),
        "chars_per_sec": round(chars_per_sec, 2),
        "text": best_text,
    }

    if accepted_flag:
        accepted.append(result)
    else:
        rejected.append(result)


# ==================================================
# SAVE
# ==================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(accepted, f, ensure_ascii=False, indent=2)

with open(REJECT_FILE, "w", encoding="utf-8") as f:
    json.dump(rejected, f, ensure_ascii=False, indent=2)


# ==================================================
# STATS
# ==================================================

print("\n========================")
print(f"Accepted : {len(accepted)}")
print(f"Rejected : {len(rejected)}")
print("========================\n")

print("Saved:")
print(OUTPUT_FILE)
print(REJECT_FILE)
