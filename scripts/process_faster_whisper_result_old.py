import json
import re
from pathlib import Path
from tqdm import tqdm
from rapidfuzz import fuzz

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]

ENABLE_WORD_COUNT_REJECT = False
MAX_WORD_COUNT_DIFF = 0


FASTER_WHISPER_RESULT = (
    BASE_DIR
    / "output"
    / "test_build_train_audio"
    / "faster_whisper_word_timestamp_result.jsonl"
)

OUTPUT_JSONL = (
    BASE_DIR
    / "output"
    / "test_build_train_audio"
    / "faster_whisper_adjusted_compare.jsonl"
)

MIN_FIRST_WORD_SCORE = 30
MIN_LAST_WORD_SCORE = 30
EDGE_PENALTY_THRESHOLD = 75
EDGE_PENALTY = 15
MIN_SCORE = 70
MIN_TEXT_SCORE = 75
MIN_AVG_WORD_PROB = 0.55

MAX_WORD_GAP = 0.8
MAX_DELTA_START = 1.2
MAX_DELTA_END = 1.2

TIME_MARGIN = 1.0


def first_last_word_score(expected_text, span_words):
    expected_words = normalize_text(expected_text).split()
    matched_words = normalize_text(" ".join(w["word"] for w in span_words)).split()

    if not expected_words or not matched_words:
        return 0, 0

    first_score = fuzz.ratio(expected_words[0], matched_words[0])
    last_score = fuzz.ratio(expected_words[-1], matched_words[-1])

    return first_score, last_score


# =========================
# TEXT UTILS
# =========================


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\sÀ-ỹà-ỹ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def flatten_words(segments):
    words = []

    for seg in segments:
        for w in seg.get("words", []):
            word = (w.get("word") or "").strip()

            if not word:
                continue

            if w.get("start") is None or w.get("end") is None:
                continue

            prob = w.get("probability")
            if prob is None:
                prob = w.get("score")

            words.append(
                {
                    "word": word,
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "probability": None if prob is None else float(prob),
                }
            )

    return words


def get_avg_prob(span_words):
    probs = [w["probability"] for w in span_words if w.get("probability") is not None]

    if not probs:
        return None

    return sum(probs) / len(probs)


def get_max_word_gap(span_words):
    if len(span_words) < 2:
        return 0.0

    gaps = []

    for i in range(len(span_words) - 1):
        gap = span_words[i + 1]["start"] - span_words[i]["end"]
        gaps.append(gap)

    return max(gaps) if gaps else 0.0


def find_best_word_span(
    expected_text,
    words,
    expected_rel_start,
    expected_rel_end,
    time_margin=1.0,
):
    expected_norm = normalize_text(expected_text)
    expected_word_count = len(expected_norm.split())

    if not expected_norm:
        return None, "EMPTY_EXPECTED_TEXT"

    if not words:
        return None, "NO_WORDS"

    expected_duration = expected_rel_end - expected_rel_start

    if expected_duration <= 0:
        return None, "BAD_EXPECTED_DURATION"

    min_len = max(1, int(expected_word_count * 0.75))
    max_len = max(1, int(expected_word_count * 1.25))

    min_duration = expected_duration * 0.70
    max_duration = expected_duration * 1.30

    search_start = max(0.0, expected_rel_start - time_margin)
    search_end = expected_rel_end + time_margin

    best = None
    word_count_reject_count = 0
    for i in range(len(words)):
        for j in range(i + min_len, min(len(words), i + max_len) + 1):
            span_words = words[i:j]
            span_word_count = len(span_words)

            if ENABLE_WORD_COUNT_REJECT:
                if abs(span_word_count - expected_word_count) > MAX_WORD_COUNT_DIFF:
                    word_count_reject_count += 1
                    continue

            span_start = span_words[0]["start"]
            span_end = span_words[-1]["end"]
            span_duration = span_end - span_start

            if span_start < search_start or span_end > search_end:
                continue

            if span_duration < min_duration or span_duration > max_duration:
                continue

            span_text = " ".join(w["word"] for w in span_words)
            span_norm = normalize_text(span_text)

            text_score = fuzz.ratio(expected_norm, span_norm)
            first_score, last_score = first_last_word_score(expected_text, span_words)
            if first_score < MIN_FIRST_WORD_SCORE:
                continue

            if last_score < MIN_LAST_WORD_SCORE:
                continue
            edge_penalty = 0

            if first_score < EDGE_PENALTY_THRESHOLD:
                edge_penalty += EDGE_PENALTY

            if last_score < EDGE_PENALTY_THRESHOLD:
                edge_penalty += EDGE_PENALTY
            start_penalty = abs(span_start - expected_rel_start) * 10
            end_penalty = abs(span_end - expected_rel_end) * 10
            duration_penalty = abs(span_duration - expected_duration) * 8

            score = (
                text_score
                - start_penalty
                - end_penalty
                - duration_penalty
                - edge_penalty
            )

            avg_prob = get_avg_prob(span_words)
            max_word_gap = get_max_word_gap(span_words)

            if best is None or score > best["score"]:
                best = {
                    "score": round(score, 2),
                    "text_score": round(text_score, 2),
                    # DEBUG
                    "first_word_score": round(first_score, 2),
                    "last_word_score": round(last_score, 2),
                    "edge_penalty": round(edge_penalty, 2),
                    "avg_word_probability": (
                        None if avg_prob is None else round(avg_prob, 4)
                    ),
                    "max_word_gap": round(max_word_gap, 3),
                    "word_start_index": i,
                    "word_end_index": j - 1,
                    "matched_text": span_text,
                    "relative_start": span_start,
                    "relative_end": span_end,
                    "expected_rel_start": round(expected_rel_start, 3),
                    "expected_rel_end": round(expected_rel_end, 3),
                    "span_duration": round(span_duration, 3),
                    "expected_duration": round(expected_duration, 3),
                    "word_count": len(span_words),
                    "expected_word_count": expected_word_count,
                }

    if best is None:
        return None, "NO_GOOD_SPAN"

    return best, None


def judge_result(best, new_start, new_end, original_start, original_end):
    reasons = []

    if best["score"] < MIN_SCORE:
        reasons.append("LOW_MATCH_SCORE")

    if best["text_score"] < MIN_TEXT_SCORE:
        reasons.append("LOW_TEXT_SCORE")

    avg_prob = best.get("avg_word_probability")
    if avg_prob is not None and avg_prob < MIN_AVG_WORD_PROB:
        reasons.append("LOW_WORD_PROBABILITY")

    if best.get("max_word_gap", 0) > MAX_WORD_GAP:
        reasons.append("LARGE_WORD_GAP")

    if original_start is not None:
        delta_start = abs(new_start - float(original_start))
        if delta_start > MAX_DELTA_START:
            reasons.append("DELTA_START_TOO_LARGE")

    if original_end is not None:
        delta_end = abs(new_end - float(original_end))
        if delta_end > MAX_DELTA_END:
            reasons.append("DELTA_END_TOO_LARGE")

    if not reasons:
        return "OK", "OK"

    return "REJECT", "|".join(reasons)


# =========================
# MAIN
# =========================


def main():
    total = 0
    ok = 0
    rejected = 0
    no_words = 0

    if OUTPUT_JSONL.exists():
        OUTPUT_JSONL.unlink()

    with open(FASTER_WHISPER_RESULT, "r", encoding="utf-8") as fin, open(
        OUTPUT_JSONL, "a", encoding="utf-8"
    ) as out:

        for line in tqdm(fin, desc="Processing faster-whisper result"):
            total += 1
            row = json.loads(line)

            expected_text = row.get("text_expected", "")
            cut_start = float(row.get("cut_start", 0))

            original_start = row.get("original_start")
            original_end = row.get("original_end")

            segments = row.get("segments", [])

            if segments:
                words = flatten_words(segments)
            else:
                words = row.get("words", [])
            words = flatten_words(segments)

            if not words:
                no_words += 1
                rejected += 1

                out_row = {
                    "index": row.get("index"),
                    "audio": row.get("audio"),
                    "status": "REJECT",
                    "reject_reason": "NO_WORDS",
                    "old_start": original_start,
                    "old_end": original_end,
                    "new_start": None,
                    "new_end": None,
                    "text_expected": expected_text,
                    "faster_text_pred": row.get("text_pred"),
                }

                out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                continue

            expected_rel_start = float(original_start) - cut_start
            expected_rel_end = float(original_end) - cut_start

            best, reason = find_best_word_span(
                expected_text=expected_text,
                words=words,
                expected_rel_start=expected_rel_start,
                expected_rel_end=expected_rel_end,
                time_margin=TIME_MARGIN,
            )

            if best is None:
                rejected += 1

                out_row = {
                    "index": row.get("index"),
                    "audio": row.get("audio"),
                    "status": "REJECT",
                    "reject_reason": reason,
                    "old_start": original_start,
                    "old_end": original_end,
                    "new_start": original_start,
                    "new_end": original_end,
                    "delta_start": 0,
                    "delta_end": 0,
                    "cut_start": row.get("cut_start"),
                    "cut_end": row.get("cut_end"),
                    "expected_rel_start": round(expected_rel_start, 3),
                    "expected_rel_end": round(expected_rel_end, 3),
                    "text_expected": expected_text,
                    "faster_text_pred": row.get("text_pred"),
                }

                out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                continue

            new_start = cut_start + best["relative_start"]
            new_end = cut_start + best["relative_end"]

            status, reject_reason = judge_result(
                best=best,
                new_start=new_start,
                new_end=new_end,
                original_start=original_start,
                original_end=original_end,
            )

            if status == "OK":
                ok += 1
            else:
                rejected += 1

            out_row = {
                "index": row.get("index"),
                "audio": row.get("audio"),
                "status": status,
                "reject_reason": reject_reason,
                "old_start": original_start,
                "old_end": original_end,
                "new_start": round(new_start, 3),
                "new_end": round(new_end, 3),
                "delta_start": (
                    round(new_start - float(original_start), 3)
                    if original_start is not None
                    else None
                ),
                "delta_end": (
                    round(new_end - float(original_end), 3)
                    if original_end is not None
                    else None
                ),
                "cut_start": row.get("cut_start"),
                "cut_end": row.get("cut_end"),
                "relative_start": round(best["relative_start"], 3),
                "relative_end": round(best["relative_end"], 3),
                "expected_rel_start": best.get("expected_rel_start"),
                "expected_rel_end": best.get("expected_rel_end"),
                "text_score": best.get("text_score"),
                "match_score": best["score"],
                "avg_word_probability": best.get("avg_word_probability"),
                "max_word_gap": best.get("max_word_gap"),
                "span_duration": best.get("span_duration"),
                "expected_duration": best.get("expected_duration"),
                "word_count": best.get("word_count"),
                "expected_word_count": best.get("expected_word_count"),
                "text_expected": expected_text,
                "faster_text_pred": row.get("text_pred"),
                "faster_matched_text": best["matched_text"],
                "word_start_index": best["word_start_index"],
                "word_end_index": best["word_end_index"],
            }

            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print("\nDONE")
    print(f"TOTAL    : {total}")
    print(f"OK       : {ok}")
    print(f"REJECTED : {rejected}")
    print(f"NO_WORDS : {no_words}")
    print(f"OUTPUT   : {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
