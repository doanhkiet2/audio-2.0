import json
import re
from pathlib import Path
from tqdm import tqdm
from rapidfuzz import fuzz

# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

WHISPERX_RESULT = (
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

MIN_SCORE = 70


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
            word = w.get("word", "").strip()

            if not word:
                continue

            if w.get("start") is None or w.get("end") is None:
                continue

            words.append(
                {
                    "word": word,
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "score": w.get("score"),
                }
            )

    return words


def find_best_word_span(
    expected_text,
    words,
    expected_rel_start,
    expected_rel_end,
    time_margin=1.0,
):
    expected_norm = normalize_text(expected_text)
    expected_word_count = len(expected_norm.split())

    if not expected_norm or not words:
        return None

    expected_duration = expected_rel_end - expected_rel_start

    min_len = max(1, int(expected_word_count * 0.75))
    max_len = max(1, int(expected_word_count * 1.25))

    min_duration = expected_duration * 0.70
    max_duration = expected_duration * 1.30

    search_start = max(0.0, expected_rel_start - time_margin)
    search_end = expected_rel_end + time_margin

    best = None

    for i in range(len(words)):
        for j in range(i + min_len, min(len(words), i + max_len) + 1):
            span_words = words[i:j]

            span_start = span_words[0]["start"]
            span_end = span_words[-1]["end"]
            span_duration = span_end - span_start

            # bắt buộc nằm gần vùng thời gian cũ
            if span_start < search_start or span_end > search_end:
                continue

            # bắt buộc duration không được ngắn/dài quá lệch
            if span_duration < min_duration or span_duration > max_duration:
                continue

            span_text = " ".join(w["word"] for w in span_words)
            span_norm = normalize_text(span_text)

            text_score = fuzz.ratio(expected_norm, span_norm)

            start_penalty = abs(span_start - expected_rel_start) * 10
            end_penalty = abs(span_end - expected_rel_end) * 10
            duration_penalty = abs(span_duration - expected_duration) * 8

            score = text_score - start_penalty - end_penalty - duration_penalty

            if best is None or score > best["score"]:
                best = {
                    "score": round(score, 2),
                    "text_score": round(text_score, 2),
                    "word_start_index": i,
                    "word_end_index": j - 1,
                    "matched_text": span_text,
                    "relative_start": span_start,
                    "relative_end": span_end,
                    "expected_rel_start": round(expected_rel_start, 3),
                    "expected_rel_end": round(expected_rel_end, 3),
                    "span_duration": round(span_duration, 3),
                    "expected_duration": round(expected_duration, 3),
                }

    return best


# =========================
# MAIN
# =========================


def main():
    total = 0
    kept = 0
    low_score = 0
    no_words = 0

    if OUTPUT_JSONL.exists():
        OUTPUT_JSONL.unlink()

    with open(WHISPERX_RESULT, "r", encoding="utf-8") as fin, open(
        OUTPUT_JSONL, "a", encoding="utf-8"
    ) as out:

        for line in tqdm(fin, desc="Processing WhisperX result"):
            total += 1

            row = json.loads(line)

            expected_text = row.get("text_expected", "")
            cut_start = float(row.get("cut_start", 0))

            original_start = row.get("original_start")
            original_end = row.get("original_end")

            segments = row.get("whisperx_segments", [])
            words = flatten_words(segments)

            if not words:
                no_words += 1

                out_row = {
                    "index": row.get("index"),
                    "audio": row.get("audio"),
                    "status": "NO_WORDS",
                    "old_start": original_start,
                    "old_end": original_end,
                    "new_start": None,
                    "new_end": None,
                    "text_expected": expected_text,
                }

                out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                continue

            expected_rel_start = float(original_start) - cut_start
            expected_rel_end = float(original_end) - cut_start

            best = find_best_word_span(
                expected_text=expected_text,
                words=words,
                expected_rel_start=expected_rel_start,
                expected_rel_end=expected_rel_end,
                time_margin=1.0,
            )

            if best is None:
                no_words += 1

                out_row = {
                    "index": row.get("index"),
                    "audio": row.get("audio"),
                    "status": "NO_GOOD_SPAN",
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
                }

                out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                continue

            new_start = cut_start + best["relative_start"]
            new_end = cut_start + best["relative_end"]

            status = "OK"

            if best["score"] < MIN_SCORE:
                status = "LOW_SCORE"
                low_score += 1
            else:
                kept += 1

            out_row = {
                "index": row.get("index"),
                "audio": row.get("audio"),
                "status": status,
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
                "text_expected": expected_text,
                "whisperx_matched_text": best["matched_text"],
                "word_start_index": best["word_start_index"],
                "word_end_index": best["word_end_index"],
            }

            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print("\nDONE")
    print(f"TOTAL     : {total}")
    print(f"OK        : {kept}")
    print(f"LOW_SCORE : {low_score}")
    print(f"NO_WORDS  : {no_words}")
    print(f"OUTPUT    : {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
