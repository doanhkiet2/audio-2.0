import subprocess
from pathlib import Path
from tqdm import tqdm

# =========================
# CONFIG
# =========================

INPUT_DIR = "data/raw/audio/chinhchiendebaovenang"
OUTPUT_DIR = "data/input_audio/splited"

SKIP_FILES = {}

CHUNK_MINUTES = 20

SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_CODEC = "pcm_s16le"

AUDIO_EXTS = [".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"]


def run_cmd(cmd):
    subprocess.run(cmd, check=True)


def get_duration_seconds(input_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(input_path),
    ]

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    return float(result.stdout.strip())


def split_one_file(input_path: Path, output_dir: Path):
    duration = get_duration_seconds(input_path)
    chunk_seconds = CHUNK_MINUTES * 60

    total_chunks = int(duration // chunk_seconds)
    if duration % chunk_seconds > 0:
        total_chunks += 1

    input_stem = input_path.stem

    print(f"\n[INFO] Input: {input_path}")
    print(f"[INFO] Duration: {duration / 60:.2f} minutes")
    print(f"[INFO] Total chunks: {total_chunks}")

    for i in tqdm(range(total_chunks), desc=f"Splitting {input_stem}"):
        start = i * chunk_seconds
        output_file = output_dir / f"{input_stem}_{i + 1}.wav"

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(start),
            "-i",
            str(input_path),
            "-t",
            str(chunk_seconds),
            "-ac",
            str(CHANNELS),
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            AUDIO_CODEC,
            str(output_file),
        ]

        try:
            run_cmd(cmd)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed chunk {i + 1} of {input_path.name}: {e}")
            continue


def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[ERROR] Input dir not found: {input_dir}")
        return

    audio_files = [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]

    if not audio_files:
        print(f"[ERROR] No audio files found in: {input_dir}")
        return

    print(f"[INFO] Input dir: {input_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Found audio files: {len(audio_files)}")
    print(f"[INFO] Chunk size: {CHUNK_MINUTES} minutes")

    for input_path in audio_files:
        if input_path.name in SKIP_FILES:
            print(f"[SKIP] {input_path.name}")
            continue

        try:
            split_one_file(input_path, output_dir)
        except Exception as e:
            print(f"[ERROR] Failed file {input_path.name}: {e}")
            continue

    print("\n[DONE] All audio prepared successfully")


if __name__ == "__main__":
    main()
