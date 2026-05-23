from pathlib import Path
import subprocess
import json
import gc
import time

# =========================
# AUTO INSTALL
# =========================
subprocess.run(
    "pip -q install faster-whisper tqdm",
    shell=True,
)

# =========================
# IMPORT
# =========================
from faster_whisper import WhisperModel

# =========================
# CONFIG
# =========================
INPUT_DIR = Path("/content/drive/MyDrive/audio_input")
OUTPUT_DIR = Path("/content/drive/MyDrive/output_json")
TMP_DIR = Path("/content/tmp_wav16k")


LOG_DIR = Path("/content/drive/MyDrive/logs")

SUCCESS_LOG = LOG_DIR / "success.log"
ERROR_LOG = LOG_DIR / "error.log"

LOG_DIR.mkdir(parents=True, exist_ok=True)


def write_log(path, msg):
    with open(path, "a", encoding="utf8") as f:
        f.write(msg + "\n")


MODEL_SIZE = "medium"

DEVICE = "cuda"
COMPUTE_TYPE = "float16"

BEAM_SIZE = 1
SKIP_DONE = True

for p in [INPUT_DIR, OUTPUT_DIR, TMP_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
print("\nLoading model...")
model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
)

print("Model ready")

# =========================
# AUDIO TYPES
# =========================
AUDIO_EXTS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".opus",
    ".mp4",
    ".mkv",
}


# =========================
# CONVERT
# =========================
def convert_audio(src, dst):

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# =========================
# TRANSCRIBE
# =========================
def process_file(audio):

    output = OUTPUT_DIR / f"{audio.stem}.json"

    if SKIP_DONE and output.exists():
        print(f"SKIP: {audio.name}")
        return

    tmp = TMP_DIR / f"{audio.stem}.wav"

    print(f"\nSTART {audio.name}")

    start = time.time()

    convert_audio(audio, tmp)

    segments, info = model.transcribe(
        str(tmp),
        language="vi",
        vad_filter=True,
        beam_size=BEAM_SIZE,
        condition_on_previous_text=False,
        temperature=0.0,
    )

    result = []

    for s in segments:
        result.append(
            {
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text.strip(),
            }
        )

    tmp_json = output.with_suffix(".tmp")

    with open(tmp_json, "w", encoding="utf8") as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    tmp_json.replace(output)

    tmp.unlink(missing_ok=True)

    del result

    gc.collect()
    print(f"DONE {audio.name}")
    elapsed = (time.time() - start) / 60

    write_log(SUCCESS_LOG, f"{audio.name}|{elapsed:.2f}min")

    print(f"TIME: {(time.time()-start)/60:.2f} min")


# =========================
# RUN
# =========================
# files = []

# for p in INPUT_DIR.rglob("*"):

#     if p.is_file() and p.suffix.lower() in AUDIO_EXTS:

#         files.append(p)

# print(f"\nFound {len(files)} files")

# for file in files:

#     try:

#         process_file(file)

#     except Exception as e:

#         print(f"ERROR {file.name}")

#         print(e)

# print("\nALL DONE")

# =========================
# RUN - SNAPSHOT SAFE MODE
# =========================
SNAPSHOT_FILE = OUTPUT_DIR / "_snapshot_files.txt"

files = []

for p in sorted(INPUT_DIR.rglob("*")):

    if not p.is_file():
        continue

    if p.suffix.lower() not in AUDIO_EXTS:
        continue

    # tránh ăn nhầm file vừa upload xong / đang sync
    age_sec = time.time() - p.stat().st_mtime
    if age_sec < 120:
        print(f"WAIT/SKIP NEW FILE: {p.name}")
        continue

    files.append(p)

with open(SNAPSHOT_FILE, "w", encoding="utf8") as f:
    for p in files:
        f.write(str(p) + "\n")

print(f"\nFound {len(files)} files")
print(f"Snapshot saved to: {SNAPSHOT_FILE}")

for file in files:

    try:

        process_file(file)

    except Exception as e:
        print(f"ERROR {file.name}")

        write_log(ERROR_LOG, f"{file.name}|{str(e)}")

print("\nALL DONE")
