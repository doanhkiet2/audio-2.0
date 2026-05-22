import gc
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torchaudio
import whisper
from silero_vad import load_silero_vad, get_speech_timestamps
from tqdm import tqdm

# =========================
# COLAB CONFIG
# =========================

INPUT_DIR = Path("/content/drive/MyDrive/audio_input")
OUTPUT_DIR = Path("/content/drive/MyDrive/output_json")
LOG_FILE = Path("/content/drive/MyDrive/logs/whisper_vad_done.jsonl")

WHISPER_MODEL = "medium"  # base / small / medium / large-v3

VAD_THRESHOLD = 0.5
MIN_SEGMENT_DURATION = 1.0
MAX_SEGMENT_DURATION = 20.0

SLEEP_AFTER_EACH_FILE = 2

SUPPORTED_EXTS = [".wav", ".mp3", ".m4a", ".flac"]


# =========================
# UTILS
# =========================


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_done_files():
    done = set()

    if not LOG_FILE.exists():
        return done

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                done.add(item["input"])
            except Exception:
                pass

    return done


def write_done_log(input_path, output_path):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    item = {
        "input": str(input_path),
        "output": str(output_path),
        "done_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_audio_files():
    files = []

    for ext in SUPPORTED_EXTS:
        files.extend(INPUT_DIR.glob(f"*{ext}"))

    return sorted(files)


def load_audio_for_vad(audio_path, device):
    wav, sr = torchaudio.load(str(audio_path))

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        sr = 16000

    wav = wav.squeeze(0).to(device)

    return wav, sr


# =========================
# MAIN PROCESS
# =========================


def process_one_file(audio_file, vad_model, whisper_model, device):
    output_file = OUTPUT_DIR / f"{audio_file.stem}.json"

    if output_file.exists():
        print(f"[SKIP] Output exists: {output_file.name}")
        write_done_log(audio_file, output_file)
        return

    print("\n" + "=" * 80)
    print(f"[RUN] {audio_file.name}")

    # -------- VAD --------
    print("[INFO] Running VAD...")

    wav, sr = load_audio_for_vad(audio_file, device)

    with torch.no_grad():
        speech = get_speech_timestamps(
            wav,
            vad_model,
            sampling_rate=sr,
            threshold=VAD_THRESHOLD,
        )

    vad_segments = []

    for item in speech:
        start = item["start"] / sr
        end = item["end"] / sr
        duration = end - start

        if MIN_SEGMENT_DURATION <= duration <= MAX_SEGMENT_DURATION:
            vad_segments.append((start, end))

    print(f"[INFO] VAD segments: {len(vad_segments)}")

    del wav
    cleanup_memory()

    # -------- WHISPER --------
    print("[INFO] Running Whisper GPU...")

    result = whisper_model.transcribe(
        str(audio_file),
        fp16=torch.cuda.is_available(),
        verbose=False,
    )

    whisper_segments = result.get("segments", [])

    final_segments = []
    used = set()

    for v_start, v_end in vad_segments:
        matched = []

        for i, w in enumerate(whisper_segments):
            if i in used:
                continue

            w_start = float(w["start"])
            w_end = float(w["end"])

            overlap = max(0, min(v_end, w_end) - max(v_start, w_start))

            if overlap > 0:
                matched.append(i)

        if not matched:
            continue

        start_time = min(whisper_segments[i]["start"] for i in matched)
        end_time = max(whisper_segments[i]["end"] for i in matched)

        text = " ".join(whisper_segments[i]["text"].strip() for i in matched).strip()

        for i in matched:
            used.add(i)

        if len(text.split()) < 2:
            continue

        final_segments.append(
            {
                "start": round(float(start_time), 2),
                "end": round(float(end_time), 2),
                "text": text,
                "audio_path": str(audio_file),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_segments, f, ensure_ascii=False, indent=2)

    write_done_log(audio_file, output_file)

    print(f"[DONE] Saved: {output_file}")
    print(f"[DONE] Segments: {len(final_segments)}")

    del result
    del whisper_segments
    del final_segments
    del vad_segments

    cleanup_memory()

    if SLEEP_AFTER_EACH_FILE > 0:
        time.sleep(SLEEP_AFTER_EACH_FILE)


def main():
    print("[INFO] Colab Whisper VAD Pipeline")

    device = get_device()
    print(f"[INFO] Device: {device}")

    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    print(f"[INFO] Input dir: {INPUT_DIR}")
    print(f"[INFO] Output dir: {OUTPUT_DIR}")
    print(f"[INFO] Log file: {LOG_FILE}")
    print(f"[INFO] Whisper model: {WHISPER_MODEL}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading VAD model...")
    vad_model = load_silero_vad().to(device)
    vad_model.eval()

    print("[INFO] Loading Whisper model...")
    whisper_model = whisper.load_model(
        WHISPER_MODEL,
        device=device,
    )

    audio_files = get_audio_files()
    done_files = load_done_files()

    audio_files = [p for p in audio_files if str(p) not in done_files]

    print(f"[INFO] Remaining files: {len(audio_files)}")

    if not audio_files:
        print("[DONE] Nothing to process.")
        return

    for audio_file in tqdm(audio_files, desc="Colab Whisper VAD"):
        try:
            process_one_file(
                audio_file=audio_file,
                vad_model=vad_model,
                whisper_model=whisper_model,
                device=device,
            )
        except Exception as e:
            print(f"[ERROR] Failed: {audio_file.name}")
            print(e)
            break
        finally:
            cleanup_memory()

    print("\n[DONE] Finished")


if __name__ == "__main__":
    main()
