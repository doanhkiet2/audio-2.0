import json
import wave
from pathlib import Path
from statistics import mean
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = BASE_DIR / "dataset_final"
META_FILE = DATASET_DIR / "metadata_final.jsonl"
AUDIO_DIR = DATASET_DIR / "audio"

MIN_DURATION = 1.0
MAX_DURATION = 15.0


def get_wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        return frames / float(rate)


def load_jsonl(path: Path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Bad JSON at line {line_no}: {e}")
                continue

            rows.append((line_no, row))

    return rows


def main():
    print(f"[INFO] Dataset dir: {DATASET_DIR}")
    print(f"[INFO] Metadata: {META_FILE}")
    print(f"[INFO] Audio dir: {AUDIO_DIR}")

    if not META_FILE.exists():
        print(f"[ERROR] Missing metadata file: {META_FILE}")
        return

    if not AUDIO_DIR.exists():
        print(f"[ERROR] Missing audio dir: {AUDIO_DIR}")
        return

    rows = load_jsonl(META_FILE)

    missing_audio = []
    empty_text = []
    bad_duration = []
    duration_mismatch = []

    durations = []

    for line_no, row in tqdm(rows, desc="Validating dataset"):
        audio_rel = row.get("audio")
        text = row.get("text", "").strip()
        meta_duration = row.get("duration")

        if not text:
            empty_text.append(line_no)

        if not audio_rel:
            missing_audio.append((line_no, "missing audio field"))
            continue

        audio_path = DATASET_DIR / audio_rel

        if not audio_path.exists():
            missing_audio.append((line_no, audio_rel))
            continue

        try:
            real_duration = get_wav_duration(audio_path)
        except Exception as e:
            missing_audio.append((line_no, f"{audio_rel} | cannot read wav: {e}"))
            continue

        durations.append(real_duration)

        if real_duration < MIN_DURATION or real_duration > MAX_DURATION:
            bad_duration.append((line_no, audio_rel, real_duration))

        if meta_duration is not None:
            diff = abs(float(meta_duration) - real_duration)
            if diff > 0.2:
                duration_mismatch.append(
                    (line_no, audio_rel, float(meta_duration), real_duration, diff)
                )

    print("\n========== DATASET FINAL VALIDATION ==========")
    print(f"Total metadata rows: {len(rows)}")
    print(f"Missing / unreadable audio: {len(missing_audio)}")
    print(f"Empty text rows: {len(empty_text)}")
    print(f"Bad duration rows: {len(bad_duration)}")
    print(f"Duration mismatch rows: {len(duration_mismatch)}")

    if durations:
        total_seconds = sum(durations)
        print("\n[DURATION]")
        print(f"Total audio: {total_seconds / 60:.2f} minutes")
        print(f"Min: {min(durations):.2f}s")
        print(f"Max: {max(durations):.2f}s")
        print(f"Avg: {mean(durations):.2f}s")

    if missing_audio:
        print("\n[MISSING / UNREADABLE AUDIO]")
        for item in missing_audio[:20]:
            print(item)

    if empty_text:
        print("\n[EMPTY TEXT]")
        print(empty_text[:20])

    if bad_duration:
        print("\n[BAD DURATION]")
        for item in bad_duration[:20]:
            print(item)

    if duration_mismatch:
        print("\n[DURATION MISMATCH]")
        for item in duration_mismatch[:20]:
            print(item)

    if (
        not missing_audio
        and not empty_text
        and not bad_duration
        and not duration_mismatch
    ):
        print("\n[DONE] Dataset final looks clean ✅")
    else:
        print("\n[WARN] Dataset has issues. Check logs above.")


if __name__ == "__main__":
    main()
