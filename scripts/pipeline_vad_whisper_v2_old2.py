import gc
import json
import time
from pathlib import Path

import torch
from pydub import AudioSegment
import numpy as np
import whisper
from silero_vad import get_speech_timestamps, load_silero_vad
from tqdm import tqdm

# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = BASE_DIR / "data" / "input_audio" / "splitedtest"
OUTPUT_DIR = BASE_DIR / "data" / "input_audio" / "splitedtest"

WHISPER_MODEL_NAME = "base"

SUPPORTED_EXTS = [".wav", ".mp3", ".m4a", ".flac"]

VAD_THRESHOLD = 0.5
MIN_SEGMENT_DURATION = 1.0
MAX_SEGMENT_DURATION = 20.0

SLEEP_AFTER_EACH_FILE = 3

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# HELPERS
# =========================


def cleanup_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_audio_for_vad(audio_path, device):
    audio = AudioSegment.from_file(str(audio_path))

    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples = samples / 32768.0

    wav = torch.from_numpy(samples).to(device)

    return wav, 16000


def get_audio_files(input_dir: Path):
    files = []

    for ext in SUPPORTED_EXTS:
        files.extend(input_dir.glob(f"*{ext}"))

    return sorted(files)


def process_one_file(audio_file: Path, vad_model, whisper_model):
    output_file = OUTPUT_DIR / f"{audio_file.stem}.json"

    if output_file.exists():
        print(f"[SKIP] Already exists: {output_file.name}")
        return

    print(f"\n[INFO] Processing: {audio_file.name}")

    # =========================
    # LOAD AUDIO FOR VAD
    # =========================

    wav, sr = load_audio_for_vad(audio_path, device)

    # =========================
    # VAD
    # =========================

    print("[INFO] Running VAD...")
    speech = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=sr,
        threshold=VAD_THRESHOLD,
    )

    vad_segments = []

    for t in speech:
        start = t["start"] / sr
        end = t["end"] / sr
        duration = end - start

        if MIN_SEGMENT_DURATION <= duration <= MAX_SEGMENT_DURATION:
            vad_segments.append((start, end))

    print(f"[INFO] VAD segments: {len(vad_segments)}")

    # Giải phóng wav sau VAD
    del wav
    cleanup_memory()

    # =========================
    # WHISPER FULL AUDIO
    # =========================

    print("[INFO] Running Whisper...")
    result = whisper_model.transcribe(str(audio_file))
    whisper_segments = result.get("segments", [])

    # =========================
    # ALIGN VAD + WHISPER
    # =========================

    results = []
    used = set()

    for v_start, v_end in vad_segments:
        matched_indices = []

        for i, w in enumerate(whisper_segments):
            if i in used:
                continue

            w_start = w["start"]
            w_end = w["end"]

            overlap = max(0, min(v_end, w_end) - max(v_start, w_start))

            if overlap > 0:
                matched_indices.append(i)

        if not matched_indices:
            continue

        start_time = min(whisper_segments[i]["start"] for i in matched_indices)
        end_time = max(whisper_segments[i]["end"] for i in matched_indices)
        text = " ".join(
            whisper_segments[i]["text"].strip() for i in matched_indices
        ).strip()

        for i in matched_indices:
            used.add(i)

        if len(text.split()) < 2:
            continue

        results.append(
            {
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "text": text,
                "audio_path": str(audio_file.relative_to(BASE_DIR)),
            }
        )

    # =========================
    # SAVE
    # =========================

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved: {output_file}")
    print(f"[DONE] Segments: {len(results)}")

    del result
    del whisper_segments
    del vad_segments
    del results
    cleanup_memory()

    if SLEEP_AFTER_EACH_FILE > 0:
        print(f"[INFO] Cooling down {SLEEP_AFTER_EACH_FILE}s...")
        time.sleep(SLEEP_AFTER_EACH_FILE)


def main():
    print("[INFO] Loading models...")
    vad_model = load_silero_vad()
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME)

    audio_files = get_audio_files(INPUT_DIR)

    print(f"[INFO] Input dir: {INPUT_DIR}")
    print(f"[INFO] Output dir: {OUTPUT_DIR}")
    print(f"[INFO] Found audio files: {len(audio_files)}")

    if not audio_files:
        print("[ERROR] No audio files found.")
        return

    for audio_file in tqdm(audio_files, desc="Whisper VAD files"):
        try:
            process_one_file(audio_file, vad_model, whisper_model)
        except Exception as e:
            print(f"[ERROR] Failed: {audio_file.name}")
            print(f"[ERROR] {e}")
        finally:
            cleanup_memory()

    print("\n[DONE] All files processed")


if __name__ == "__main__":
    main()
