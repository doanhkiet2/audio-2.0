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

OUTPUT_FILE = "output/aligned_v2.json"
REJECT_FILE = "output/rejected_v2.json"

# ==================================================
# MAIN FILTER (PHRASE 1)
# ==================================================

SIMILARITY_THRESHOLD = 82

MIN_DURATION = 1.0
MAX_DURATION = 20.0

MIN_CHARS_PER_SEC = 3
MAX_CHARS_PER_SEC = 35

PREVIOUS_SHIFT = 1.0

# ==================================================
# STRUCTURE FILTER (PHRASE 2)
# ==================================================

MAX_PUNCT_MISS = 1

os.makedirs("output", exist_ok=True)


# ==================================================
# NORMALIZE
# ==================================================


def normalize_text(text):

    text = text.lower()

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==================================================
# WORD COUNT
# ==================================================


def word_count(text):

    return len(text.split())


# ==================================================
# TIME
# ==================================================


def hhmmss_to_seconds(time_str):

    h, m, s = map(int, time_str.split(":"))

    return h * 3600 + m * 60 + s


# ==================================================
# SPLIT BY PUNCTUATION
# ==================================================


def split_by_punctuation(text):

    # bỏ quote khỏi logic structure
    text = re.sub(r"[\"“”']", "", text)

    # mọi dấu câu xem như tương đương
    chunks = re.split(r"[.,?!:]+", text)

    results = []

    for c in chunks:

        c = c.strip()

        if not c:
            continue

        results.append(len(c.split()))

    return results


# ==================================================
# PHRASE 2
# STRICT STRUCTURE VALIDATOR
# ==================================================


def punctuation_structure_pass(
    whisper_text,
    candidate_text,
):

    w_chunks = split_by_punctuation(whisper_text)

    c_chunks = split_by_punctuation(candidate_text)

    if not w_chunks or not c_chunks:
        return False

    i = 0
    j = 0

    miss_used = 0

    while i < len(w_chunks) and j < len(c_chunks):

        K = w_chunks[i]
        P = c_chunks[j]

        # ======================================
        # EXACT MATCH
        # ======================================

        if K == P:

            i += 1
            j += 1

            continue

        # ======================================
        # WHISPER SHORTER
        # candidate thiếu dấu câu
        # merge nhiều whisper chunks
        # ======================================

        if K < P:

            if miss_used >= MAX_PUNCT_MISS:
                return False

            merged = K

            next_i = i

            success = False

            while next_i + 1 < len(w_chunks):

                next_i += 1

                merged += w_chunks[next_i]

                if merged == P:

                    miss_used += 1

                    i = next_i + 1
                    j += 1

                    success = True

                    break

            if not success:
                return False

            continue

        # ======================================
        # WHISPER LONGER
        # reject ngay
        # ======================================

        if K > P:
            return False

    # ======================================
    # MUST END TOGETHER
    # ======================================

    if i != len(w_chunks):
        return False

    if j != len(c_chunks):
        return False

    return True


# ==================================================
# LOAD WHISPER JSON
# ==================================================

with open(WHISPER_FILE, "r", encoding="utf-8") as f:

    whisper_segments = json.load(f)


# ==================================================
# LOAD YOUTUBE
# ==================================================

youtube_segments = []

with open(YOUTUBE_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        match = re.match(
            r"^(\d{2}:\d{2}:\d{2})\s+(.*)$",
            line,
        )

        if not match:
            continue

        youtube_segments.append(
            {
                "start": hhmmss_to_seconds(match.group(1)),
                "text": match.group(2).strip(),
            }
        )


# ==================================================
# PHRASE 1
# EXTRACT BEST SUBSTRING
# ==================================================


def extract_best_substring(
    whisper_text,
    context,
    min_words=4,
):

    words = context.split()

    best_score = -999999
    best_text = ""

    whisper_norm = normalize_text(whisper_text)

    whisper_words = word_count(whisper_norm)

    for start in range(len(words)):

        for end in range(
            start + min_words,
            len(words) + 1,
        ):

            candidate = " ".join(words[start:end])

            candidate_norm = normalize_text(candidate)

            # ======================================
            # FUZZY SCORE
            # ======================================

            base_score = fuzz.ratio(
                whisper_norm,
                candidate_norm,
            )

            # ======================================
            # LENGTH SCORE
            # ======================================

            char_ratio = min(
                len(candidate_norm),
                len(whisper_norm),
            ) / max(
                len(candidate_norm),
                len(whisper_norm),
            )

            candidate_words = word_count(candidate_norm)

            word_ratio = min(
                whisper_words,
                candidate_words,
            ) / max(
                whisper_words,
                candidate_words,
            )

            # ======================================
            # FINAL SCORE
            # ======================================

            score = base_score * 0.7 + char_ratio * 100 * 0.15 + word_ratio * 100 * 0.15

            # clean ending bonus

            if candidate.strip().endswith((".", "!", "?", '"')):
                score += 5

            # ======================================
            # SAVE BEST
            # ======================================

            if score > best_score:

                best_score = score
                best_text = candidate

    return best_text, best_score


# ==================================================
# STATS
# ==================================================

print("\n========================")
print(f"Whisper segments : {len(whisper_segments)}")
print(f"YouTube segments : {len(youtube_segments)}")
print("========================\n")


# ==================================================
# ALIGN
# ==================================================

accepted = []
rejected = []

for w in tqdm(whisper_segments):

    w_start = w["start"]
    w_end = w["end"]

    whisper_text = w["text"]

    # ==============================================
    # GET CANDIDATES
    # ==============================================

    candidates = []

    candidate_set = set()

    # ----------------------------------------------
    # PREVIOUS SEGMENT
    # ----------------------------------------------

    previous_segment = None

    for y in youtube_segments:

        if y["start"] < w_start - PREVIOUS_SHIFT:
            previous_segment = y
        else:
            break

    if previous_segment:

        text = previous_segment["text"]

        if text not in candidate_set:

            candidates.append(text)

            candidate_set.add(text)

    # ----------------------------------------------
    # INSIDE WINDOW
    # ----------------------------------------------

    for y in youtube_segments:

        if y["start"] >= w_start and y["start"] <= w_end:

            text = y["text"]

            if text not in candidate_set:

                candidates.append(text)

                candidate_set.add(text)

    # ==============================================
    # NO CANDIDATE
    # ==============================================

    if not candidates:

        rejected.append(
            {
                "reason": "NO_CANDIDATE",
                **w,
            }
        )

        continue

    # ==============================================
    # CONTEXT
    # ==============================================

    context = " ".join(candidates)

    # ==============================================
    # PHRASE 1
    # ==============================================

    best_text, score = extract_best_substring(
        whisper_text,
        context,
    )

    # ==============================================
    # PHRASE 2
    # ==============================================

    structure_ok = punctuation_structure_pass(
        whisper_text,
        best_text,
    )

    if not structure_ok:

        rejected.append(
            {
                "reason": "STRUCTURE_FAIL",
                "start": w_start,
                "end": w_end,
                "score": round(score, 1),
                "whisper_text": whisper_text,
                "matched_text": best_text,
            }
        )

        continue

    # ==============================================
    # FILTER
    # ==============================================

    duration = w_end - w_start

    chars_per_sec = len(best_text) / duration if duration > 0 else 999

    accepted_flag = (
        score >= SIMILARITY_THRESHOLD
        and duration >= MIN_DURATION
        and duration <= MAX_DURATION
        and chars_per_sec >= MIN_CHARS_PER_SEC
        and chars_per_sec <= MAX_CHARS_PER_SEC
    )

    # ==============================================
    # RESULT
    # ==============================================

    result = {
        "start": w_start,
        "end": w_end,
        "score": round(score, 1),
        "chars_per_sec": round(
            chars_per_sec,
            2,
        ),
        "whisper_text": whisper_text,
        "matched_text": best_text,
    }

    if accepted_flag:
        accepted.append(result)

    else:
        rejected.append(
            {
                "reason": "LOW_SCORE",
                **result,
            }
        )


# ==================================================
# SAVE
# ==================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        accepted,
        f,
        ensure_ascii=False,
        indent=2,
    )

with open(
    REJECT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        rejected,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ==================================================
# FINAL STATS
# ==================================================

print("\n========================")
print(f"Accepted : {len(accepted)}")
print(f"Rejected : {len(rejected)}")
print("========================\n")

print("Saved files:")
print(OUTPUT_FILE)
print(REJECT_FILE)
