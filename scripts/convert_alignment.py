import os
import json

INPUT_FILE = "data/04_06__00_00__Mq_c_AfzhT4.json"
AUDIO_DIR = "data/input_audio"
OUTPUT_DIR = "data/alignment_output"


def run():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 👉 lấy tên file từ INPUT_FILE
    base_name = os.path.basename(INPUT_FILE).replace(".json", "")

    audio_name = base_name + ".wav"

    audio_path = os.path.join(AUDIO_DIR, audio_name)

    out_path = os.path.join(OUTPUT_DIR, audio_name + ".json")

    # inject audio_path
    for seg in data:
        seg["audio_path"] = audio_path

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"WROTE: {out_path}")


if __name__ == "__main__":
    run()
