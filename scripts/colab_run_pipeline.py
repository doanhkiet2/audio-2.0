# colab_run_pipeline.py

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

# =========================
# CONFIG
# =========================

INPUT_AUDIOS = [
    "/content/drive/MyDrive/voice-project-2.0/input_audio/audio_1.wav",
    "/content/drive/MyDrive/voice-project-2.0/input_audio/audio_2.wav",
    "/content/drive/MyDrive/voice-project-2.0/input_audio/audio_3.wav",
]

PIPELINE_SCRIPT = (
    "/content/drive/MyDrive/voice-project-2.0/scripts/pipeline_vad_whisper_v2.py"
)

OUTPUT_DIR = Path("/content/drive/MyDrive/voice-project-2.0/output_json")
LOG_FILE = Path("/content/drive/MyDrive/voice-project-2.0/logs/done_files.jsonl")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


# =========================
# LOAD DONE LOG
# =========================


def load_done_files():
    done = set()

    if not LOG_FILE.exists():
        return done

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                done.add(item["input"])
            except:
                pass

    return done


def write_done_log(input_path, output_path):
    item = {
        "input": input_path,
        "output": str(output_path),
        "done_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


# =========================
# MAIN LOOP
# =========================

done_files = load_done_files()

for input_audio in INPUT_AUDIOS:
    input_audio = str(input_audio)

    if input_audio in done_files:
        print(f"[SKIP] Already done: {input_audio}")
        continue

    audio_name = Path(input_audio).stem
    output_json = OUTPUT_DIR / f"{audio_name}.json"

    if output_json.exists():
        print(f"[SKIP] Output exists: {output_json}")
        write_done_log(input_audio, output_json)
        continue

    print("=" * 80)
    print(f"[RUN] {input_audio}")
    print(f"[OUT] {output_json}")

    cmd = [
        "python",
        PIPELINE_SCRIPT,
        "--input",
        input_audio,
        "--output",
        str(output_json),
    ]

    try:
        subprocess.run(cmd, check=True)

        if output_json.exists():
            write_done_log(input_audio, output_json)
            print(f"[DONE] {input_audio}")
        else:
            print(f"[WARN] Script ran but output not found: {output_json}")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed: {input_audio}")
        print(e)
        break
