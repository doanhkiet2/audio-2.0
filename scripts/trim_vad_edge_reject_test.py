import json
import gc
from pathlib import Path

import soundfile as sf
from tqdm import tqdm
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

# =========================
# CONFIG
# =========================

INPUT_DIR = Path("output/vad_trim_test/input")
OUTPUT_DIR = Path("output/vad_trim_test/output")

ACCEPT_LOG = Path("output/vad_trim_test/accepted.jsonl")
REJECT_LOG = Path("output/vad_trim_test/rejected.jsonl")

LIMIT_FILES = 200

SAMPLE_RATE = 16000

EDGE_REJECT_MS = 200  # nếu có speech trong 200ms đầu/cuối => reject
KEEP_SILENCE_MS = 100  # file pass thì giữ lại 100ms silence đầu/cuối

MIN_FINAL_DURATION = 0.8
MAX_FINAL_DURATION = 20.0

VAD_THRESHOLD = 0.5
MIN_SPEECH_DURATION_MS = 120
MIN_SILENCE_DURATION_MS = 80

AUDIO_EXTS = [".wav", ".mp3", ".flac", ".m4a"]

# =========================
# UTILS
# =========================


def write_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ms_to_sample(ms):
    return int(SAMPLE_RATE * ms / 1000)


def sample_to_ms(sample):
    return sample * 1000 / SAMPLE_RATE


def clear_logs():
    for p in [ACCEPT_LOG, REJECT_LOG]:
        if p.exists():
            p.unlink()


def reject(reason, audio_path, extra=None):
    row = {
        "file": str(audio_path),
        "status": "reject",
        "reason": reason,
    }

    if extra:
        row.update(extra)

    write_jsonl(REJECT_LOG, row)


# =========================
# MAIN
# =========================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clear_logs()

    print("[INFO] Loading Silero VAD...")
    model = load_silero_vad()

    audio_files = []
    for ext in AUDIO_EXTS:
        audio_files.extend(INPUT_DIR.rglob(f"*{ext}"))

    audio_files = sorted(audio_files)[:LIMIT_FILES]

    print(f"[INFO] Input files: {len(audio_files)}")
    print(f"[INFO] Edge reject: {EDGE_REJECT_MS}ms")
    print(f"[INFO] Keep silence: {KEEP_SILENCE_MS}ms")

    for audio_path in tqdm(audio_files):
        try:
            wav = read_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
            total_samples = len(wav)
            duration_sec = total_samples / SAMPLE_RATE

            speech_ts = get_speech_timestamps(
                wav,
                model,
                sampling_rate=SAMPLE_RATE,
                threshold=VAD_THRESHOLD,
                min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
                min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
                speech_pad_ms=0,
            )

            if not speech_ts:
                reject(
                    "no_speech",
                    audio_path,
                    {
                        "duration_sec": round(duration_sec, 3),
                    },
                )
                continue

            first_speech_start = speech_ts[0]["start"]
            last_speech_end = speech_ts[-1]["end"]

            first_speech_ms = sample_to_ms(first_speech_start)
            tail_silence_ms = sample_to_ms(total_samples - last_speech_end)

            # =========================
            # HARD REJECT EDGE 200MS
            # =========================

            if first_speech_ms < EDGE_REJECT_MS:
                reject(
                    "speech_in_first_200ms",
                    audio_path,
                    {
                        "duration_sec": round(duration_sec, 3),
                        "first_speech_ms": round(first_speech_ms, 1),
                    },
                )
                continue

            if tail_silence_ms < EDGE_REJECT_MS:
                reject(
                    "speech_in_last_200ms",
                    audio_path,
                    {
                        "duration_sec": round(duration_sec, 3),
                        "tail_silence_ms": round(tail_silence_ms, 1),
                    },
                )
                continue

            # =========================
            # TRIM SILENCE
            # =========================

            keep = ms_to_sample(KEEP_SILENCE_MS)

            trim_start = max(0, first_speech_start - keep)
            trim_end = min(total_samples, last_speech_end + keep)

            trimmed = wav[trim_start:trim_end]
            final_duration = len(trimmed) / SAMPLE_RATE

            if final_duration < MIN_FINAL_DURATION:
                reject(
                    "too_short_after_trim",
                    audio_path,
                    {
                        "final_duration": round(final_duration, 3),
                    },
                )
                continue

            if final_duration > MAX_FINAL_DURATION:
                reject(
                    "too_long_after_trim",
                    audio_path,
                    {
                        "final_duration": round(final_duration, 3),
                    },
                )
                continue

            out_path = OUTPUT_DIR / audio_path.name
            sf.write(str(out_path), trimmed.numpy(), SAMPLE_RATE, subtype="PCM_16")

            write_jsonl(
                ACCEPT_LOG,
                {
                    "file": str(audio_path),
                    "output_file": str(out_path),
                    "status": "accept",
                    "original_duration": round(duration_sec, 3),
                    "final_duration": round(final_duration, 3),
                    "first_speech_ms": round(first_speech_ms, 1),
                    "tail_silence_ms": round(tail_silence_ms, 1),
                    "trim_start_ms": round(sample_to_ms(trim_start), 1),
                    "trim_end_ms": round(sample_to_ms(trim_end), 1),
                    "speech_chunks": len(speech_ts),
                },
            )

        except Exception as e:
            reject(
                "error",
                audio_path,
                {
                    "error": str(e),
                },
            )

        gc.collect()

    print("[DONE]")
    print(f"Accepted log: {ACCEPT_LOG}")
    print(f"Rejected log: {REJECT_LOG}")
    print(f"Output audio: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
