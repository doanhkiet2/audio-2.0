import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
import gc
import json
import time
from pathlib import Path

import torch
import torchaudio
import whisper
from silero_vad import get_speech_timestamps, load_silero_vad
from tqdm import tqdm


from src.config_loader import load_config

SUPPORTED_EXTS = [".wav", ".mp3", ".m4a", ".flac"]


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_done_files(log_file: Path):
    done = set()

    if not log_file.exists():
        return done

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                done.add(item["input"])
            except Exception:
                pass

    return done


def write_done_log(log_file: Path, input_path: Path, output_path: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)

    item = {
        "input": str(input_path),
        "output": str(output_path),
        "done_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_audio_for_vad(path: Path):
    wav, sr = torchaudio.load(str(path))

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
        sr = 16000

    return wav.squeeze(0), sr


def get_audio_files(input_dir: Path):
    files = []
    for ext in SUPPORTED_EXTS:
        files.extend(input_dir.glob(f"*{ext}"))
    return sorted(files)


def process_one_file(audio_file: Path, vad_model, whisper_model, cfg):
    output_dir = cfg["output_dir"]
    log_file = cfg["log_file"]

    output_file = output_dir / f"{audio_file.stem}.json"

    if output_file.exists():
        print(f"[SKIP] Output exists: {output_file.name}")
        write_done_log(log_file, audio_file, output_file)
        return

    print(f"\n[INFO] Processing: {audio_file.name}")

    wav, sr = load_audio_for_vad(audio_file)

    print("[INFO] Running VAD...")
    speech = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=sr,
        threshold=cfg["vad_threshold"],
    )

    vad_segments = []
    for t in speech:
        start = t["start"] / sr
        end = t["end"] / sr
        duration = end - start

        if cfg["min_segment_duration"] <= duration <= cfg["max_segment_duration"]:
            vad_segments.append((start, end))

    print(f"[INFO] VAD segments: {len(vad_segments)}")

    del wav
    cleanup_memory()

    print("[INFO] Running Whisper...")
    result = whisper_model.transcribe(str(audio_file))
    whisper_segments = result.get("segments", [])

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
                "audio_path": str(audio_file),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    write_done_log(log_file, audio_file, output_file)

    print(f"[DONE] Saved: {output_file}")
    print(f"[DONE] Segments: {len(results)}")

    del result
    del whisper_segments
    del vad_segments
    del results

    cleanup_memory()

    if cfg["sleep_after_each_file"] > 0:
        print(f"[INFO] Cooling down {cfg['sleep_after_each_file']}s...")
        time.sleep(cfg["sleep_after_each_file"])


def main():
    cfg = load_config()

    print(f"[INFO] APP_ENV: {__import__('os').getenv('APP_ENV', 'local')}")
    print(f"[INFO] Input dir: {cfg['input_dir']}")
    print(f"[INFO] Output dir: {cfg['output_dir']}")
    print(f"[INFO] Log file: {cfg['log_file']}")

    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    cfg["log_file"].parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading models...")
    vad_model = load_silero_vad()
    whisper_model = whisper.load_model(cfg["whisper_model"])

    audio_files = get_audio_files(cfg["input_dir"])
    done_files = load_done_files(cfg["log_file"])

    audio_files = [p for p in audio_files if str(p) not in done_files]

    print(f"[INFO] Remaining audio files: {len(audio_files)}")

    if not audio_files:
        print("[DONE] Nothing to process.")
        return

    for audio_file in tqdm(audio_files, desc="Whisper VAD files"):
        try:
            process_one_file(audio_file, vad_model, whisper_model, cfg)
        except Exception as e:
            print(f"[ERROR] Failed: {audio_file.name}")
            print(f"[ERROR] {e}")
            break
        finally:
            cleanup_memory()

    print("\n[DONE] All possible files processed")


if __name__ == "__main__":
    main()
