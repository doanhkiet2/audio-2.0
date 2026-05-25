import json
import re
from pathlib import Path

from rapidfuzz import fuzz
from tqdm import tqdm

# ==================================================
# CONFIG
# ==================================================
SPLIT_DURATION_SECONDS = 20 * 60 + 0.015
ALIGN_MAP_PATH = Path("data/align_input/align_map_clean.json")

WHISPER_FOLDER = None
YOUTUBE_FOLDER = None

# Nếu muốn chạy folder trực tiếp thì bật:
# WHISPER_FOLDER = Path("data/align_input/whisper_sub")
# YOUTUBE_FOLDER = Path("data/align_input/youtube_sub")

OUTPUT_FOLDER = Path("data/align_output/aligned")
REJECT_FOLDER = Path("data/align_output/rejected")

# ==================================================
# FILTER
# ==================================================

SIMILARITY_THRESHOLD = 82

MIN_DURATION = 1.0
MAX_DURATION = 20.0

MIN_CHARS_PER_SEC = 3
MAX_CHARS_PER_SEC = 35

PREVIOUS_SHIFT = 1.0

MAX_PUNCT_MISS = 1

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
REJECT_FOLDER.mkdir(parents=True, exist_ok=True)


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


def hhmmss_to_seconds(time_str):
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s


# ==================================================
# STRUCTURE CHECK
# ==================================================


def split_by_punctuation(text):
    text = re.sub(r"[\"“”']", "", text)

    chunks = re.split(r"[.,?!:]+", text)

    results = []

    for c in chunks:
        c = c.strip()

        if not c:
            continue

        results.append(len(c.split()))

    return results


def punctuation_structure_pass(whisper_text, candidate_text):
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

        if K == P:
            i += 1
            j += 1
            continue

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

        if K > P:
            return False

    if i != len(w_chunks):
        return False

    if j != len(c_chunks):
        return False

    return True


# ==================================================
# LOAD
# ==================================================


def get_part_index(file_path):
    """
    Ví dụ:
    _Y7nU4l0VQvQ_8.json -> 8
    _abc_12.json -> 12
    """
    match = re.search(r"_(\d+)\.json$", file_path.name)

    if not match:
        return 1

    return int(match.group(1))


def sort_whisper_files(files):
    """
    Tránh lỗi sort kiểu:
    _1, _10, _11, _2
    """
    return sorted(files, key=get_part_index)


def load_whisper_segments(whisper_files):
    all_segments = []

    whisper_files = sort_whisper_files(whisper_files)

    for file in whisper_files:
        part_index = get_part_index(file)

        time_offset = SPLIT_DURATION_SECONDS * (part_index - 1)

        with open(file, "r", encoding="utf-8") as f:
            segments = json.load(f)

        if not segments:
            continue

        # Bỏ segment đầu và cuối CỦA TỪNG FILE 20 PHÚT
        if len(segments) <= 2:
            continue

        segments = segments[1:-1]

        for seg in segments:
            new_seg = dict(seg)

            # giữ nguyên local time trong file 20 phút
            new_seg["start"] = seg["start"]
            new_seg["end"] = seg["end"]

            # thêm global time chỉ để so với YouTube
            new_seg["global_start"] = seg["start"] + time_offset
            new_seg["global_end"] = seg["end"] + time_offset

            new_seg["source_file"] = file.name
            new_seg["part_index"] = part_index
            new_seg["time_offset"] = round(time_offset, 3)

            all_segments.append(new_seg)

    return all_segments


def load_youtube_segments(youtube_file):
    youtube_segments = []

    with open(youtube_file, "r", encoding="utf-8") as f:
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

    return youtube_segments


# ==================================================
# MATCH
# ==================================================


def extract_best_substring(whisper_text, context, min_words=4):
    words = context.split()

    best_score = -999999
    best_text = ""

    whisper_norm = normalize_text(whisper_text)
    whisper_words = word_count(whisper_norm)

    for start in range(len(words)):
        for end in range(start + min_words, len(words) + 1):
            candidate = " ".join(words[start:end])
            candidate_norm = normalize_text(candidate)

            base_score = fuzz.ratio(
                whisper_norm,
                candidate_norm,
            )

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

            score = base_score * 0.7 + char_ratio * 100 * 0.15 + word_ratio * 100 * 0.15

            if candidate.strip().endswith((".", "!", "?", '"')):
                score += 5

            if score > best_score:
                best_score = score
                best_text = candidate

    return best_text, best_score


# ==================================================
# ALIGN ONE JOB
# ==================================================


def align_one_job(job):
    youtube_file = Path(job["youtube_file"])
    whisper_files = [Path(x) for x in job["whisper_files"]]

    output_file = OUTPUT_FOLDER / f"{job['name']}.json"
    reject_file = REJECT_FOLDER / f"{job['name']}_rejected.json"

    whisper_segments = load_whisper_segments(whisper_files)
    youtube_segments = load_youtube_segments(youtube_file)

    accepted = []
    rejected = []

    print("\n========================")
    print(f"JOB              : {job['name']}")
    print(f"YouTube file     : {youtube_file}")
    print(f"Whisper files    : {len(whisper_files)}")
    print(f"Whisper segments : {len(whisper_segments)}")
    print(f"YouTube segments : {len(youtube_segments)}")
    print("========================\n")

    for w in tqdm(whisper_segments):
        w_start = w["global_start"]
        w_end = w["global_end"]
        whisper_text = w["text"]

        candidates = []
        candidate_set = set()

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

        for y in youtube_segments:
            if y["start"] >= w_start and y["start"] <= w_end:
                text = y["text"]

                if text not in candidate_set:
                    candidates.append(text)
                    candidate_set.add(text)

        if not candidates:
            rejected.append(
                {
                    "reason": "NO_CANDIDATE",
                    **w,
                }
            )
            continue

        context = " ".join(candidates)

        best_text, score = extract_best_substring(
            whisper_text,
            context,
        )

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

        duration = w_end - w_start
        chars_per_sec = len(best_text) / duration if duration > 0 else 999

        accepted_flag = (
            score >= SIMILARITY_THRESHOLD
            and duration >= MIN_DURATION
            and duration <= MAX_DURATION
            and chars_per_sec >= MIN_CHARS_PER_SEC
            and chars_per_sec <= MAX_CHARS_PER_SEC
        )

        result = {
            "start": w["start"],
            "end": w["end"],
            # có thể giữ để debug, hoặc bỏ nếu không muốn
            "global_start": round(w["global_start"], 3),
            "global_end": round(w["global_end"], 3),
            "score": round(score, 1),
            "chars_per_sec": round(chars_per_sec, 2),
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

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            accepted,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(reject_file, "w", encoding="utf-8") as f:
        json.dump(
            rejected,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n========================")
    print(f"JOB      : {job['name']}")
    print(f"Accepted : {len(accepted)}")
    print(f"Rejected : {len(rejected)}")
    print("========================")
    print(output_file)
    print(reject_file)

    return len(accepted), len(rejected)


# ==================================================
# BUILD JOBS
# ==================================================


def build_jobs_from_align_map():
    with open(ALIGN_MAP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    jobs = []

    for item in data["items"]:
        youtube_file = Path(item["source_youtube_path"])
        whisper_folder = Path(item["whisper_folder_path"])
        whisper_prefix = item["whisper_prefix"]

        whisper_files = sort_whisper_files(
            whisper_folder.glob(f"{whisper_prefix}_*.json")
        )

        name = youtube_file.stem

        jobs.append(
            {
                "name": name,
                "youtube_file": youtube_file,
                "whisper_files": whisper_files,
            }
        )

    return jobs


def build_jobs_from_folders():
    whisper_folder = Path(WHISPER_FOLDER)
    youtube_folder = Path(YOUTUBE_FOLDER)

    jobs = []

    youtube_files = sorted(youtube_folder.rglob("*.txt"))

    for youtube_file in youtube_files:
        whisper_prefix = "_" + youtube_file.stem

        whisper_files = sorted(whisper_folder.glob(f"{whisper_prefix}_*.json"))

        jobs.append(
            {
                "name": youtube_file.stem,
                "youtube_file": youtube_file,
                "whisper_files": whisper_files,
            }
        )

    return jobs


# ==================================================
# MAIN
# ==================================================

if WHISPER_FOLDER and YOUTUBE_FOLDER:
    print("MODE: folder")
    jobs = build_jobs_from_folders()
else:
    print("MODE: align_map")
    jobs = build_jobs_from_align_map()

print(f"\nTotal jobs: {len(jobs)}\n")

total_accepted = 0
total_rejected = 0
missing = 0

for job in jobs:
    youtube_file = Path(job["youtube_file"])
    whisper_files = job["whisper_files"]

    if not youtube_file.exists():
        print(f"MISSING YOUTUBE: {youtube_file}")
        missing += 1
        continue

    if not whisper_files:
        print(f"MISSING WHISPER: {job['name']}")
        missing += 1
        continue

    accepted_count, rejected_count = align_one_job(job)

    total_accepted += accepted_count
    total_rejected += rejected_count

print("\n========================")
print(f"Jobs     : {len(jobs)}")
print(f"Missing  : {missing}")
print(f"Accepted : {total_accepted}")
print(f"Rejected : {total_rejected}")
print("========================\n")
