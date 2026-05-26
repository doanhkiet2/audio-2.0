import json
import re
from pathlib import Path
from tqdm import tqdm

# =========================
# CONFIG
# =========================

JSONL_INPUT_PATH = Path("output/factory_output/factory_dataset.jsonl")

ALIGN_MAP_PATH = Path("data/align_input/align_map_clean.json")


AUDIO_FOLDER = Path("data/input_audio")

JSONL_OUTPUT_PATH = Path("output/factory_output/factory_dataset_added.jsonl")


# =========================
# LOAD ALIGN MAP
# =========================


def build_map(align_map_path):

    with open(
        align_map_path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    result = {}

    for item in data["items"]:

        youtube = Path(item["source_youtube_path"]).stem

        result[youtube] = item["whisper_prefix"]

    return result


# =========================
# GET PART SUFFIX
# =========================


def extract_suffix(filename):

    filename = Path(filename).stem

    filename = re.sub(
        r"_clean$",
        "",
        filename,
    )

    m = re.search(
        r" - Copy \((\d+)\)$",
        filename,
    )

    if not m:
        return "_1"

    return f"_{m.group(1)}"


# =========================
# MAIN
# =========================


def run():

    align_map = build_map(ALIGN_MAP_PATH)

    # debug
    print("\n=== ALIGN MAP SAMPLE ===")
    print("count:", len(align_map))

    for i, k in enumerate(align_map.keys()):
        print(repr(k))
        if i >= 10:
            break

    #

    total = 0
    missing = 0
    updated = 0

    if JSONL_OUTPUT_PATH.exists():
        JSONL_OUTPUT_PATH.unlink()

    with open(
        JSONL_INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as fin:

        for line in tqdm(fin):

            total += 1

            row = json.loads(line)

            source_json = row.get("source_json", "")

            filename = Path(source_json).stem

            filename = re.sub(
                r"_clean$",
                "",
                filename,
            )
            # debug
            if total <= 10:
                print("\n=== JSONL SOURCE SAMPLE ===")
                print("source_json:", repr(source_json))
                print("filename   :", repr(filename))
                print("in map?    :", filename in align_map)
            #

            whisper_prefix = align_map.get(filename)

            if whisper_prefix is None:
                missing += 1

                row["audio_path"] = ""

            else:

                suffix = extract_suffix(source_json)

                audio_name = whisper_prefix + suffix + ".wav"

                row["audio_path"] = str(AUDIO_FOLDER / audio_name)

                updated += 1

            with open(
                JSONL_OUTPUT_PATH,
                "a",
                encoding="utf-8",
            ) as fout:

                fout.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print()
    print("DONE")
    print(f"TOTAL   : {total}")
    print(f"UPDATED : {updated}")
    print(f"MISSING : {missing}")
    print(JSONL_OUTPUT_PATH)


if __name__ == "__main__":
    run()
