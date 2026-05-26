import json
from pathlib import Path
from tqdm import tqdm

from config_loader import load_config
from validator import validate_segment
from cleaner import clean_text

OUTPUT_JSONL = "factory_dataset.jsonl"
REJECT_JSONL = "factory_rejected.jsonl"


def write_jsonl(path: Path, item: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_reject(reject_file: Path, reason: str, item: dict):
    row = dict(item)
    row["reject_reason"] = reason
    write_jsonl(reject_file, row)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        return None

    return data


def run():
    # =========================
    # 1. LOAD CONFIG
    # =========================
    cfg = load_config()

    input_dir = Path(cfg["paths"]["input_dir"])
    output_dir = Path(cfg["paths"]["output_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_file = output_dir / OUTPUT_JSONL
    reject_file = output_dir / REJECT_JSONL

    if dataset_file.exists():
        dataset_file.unlink()

    if reject_file.exists():
        reject_file.unlink()

    # =========================
    # 2. GET INPUT JSON FILES
    # =========================
    json_files = sorted(input_dir.rglob("*.json"))

    print(f"[INFO] Input dir : {input_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Json files: {len(json_files)}")
    print("[INFO] Checkpoint: DISABLED")
    print(f"[INFO] Dataset file: {dataset_file}")
    print(f"[INFO] Reject file : {reject_file}")

    stats = {
        "FILES": 0,
        "KEEP": 0,
        "REJECT": 0,
        "READ_ERROR": 0,
        "BAD_FORMAT": 0,
    }

    # =========================
    # 3. MAIN LOOP
    # =========================
    for json_file in tqdm(json_files, desc="Phase 3A Factory"):
        stats["FILES"] += 1

        try:
            segments = load_json(json_file)

        except Exception as e:
            stats["READ_ERROR"] += 1

            write_jsonl(
                reject_file,
                {
                    "reject_reason": "READ_ERROR",
                    "source_json": str(json_file),
                    "error": str(e),
                },
            )

            tqdm.write(f"[READ_ERROR] {json_file}: {e}")
            continue

        if segments is None:
            stats["BAD_FORMAT"] += 1

            write_jsonl(
                reject_file,
                {
                    "reject_reason": "BAD_FORMAT",
                    "source_json": str(json_file),
                    "error": "JSON root is not list or dict",
                },
            )

            tqdm.write(f"[BAD_FORMAT] {json_file}")
            continue

        if not segments:
            stats["BAD_FORMAT"] += 1

            write_jsonl(
                reject_file,
                {
                    "reject_reason": "EMPTY_FILE",
                    "source_json": str(json_file),
                },
            )

            tqdm.write(f"[EMPTY_FILE] {json_file}")
            continue

        file_keep = 0
        file_reject = 0

        for idx, seg in enumerate(segments):
            base_info = {
                "source_json": str(json_file),
                "source_index": idx,
            }

            if not isinstance(seg, dict):
                write_jsonl(
                    reject_file,
                    {
                        **base_info,
                        "reject_reason": "BAD_ROW_TYPE",
                        "row_type": type(seg).__name__,
                        "row": repr(seg),
                    },
                )

                stats["REJECT"] += 1
                file_reject += 1
                continue

            seg = dict(seg)
            seg.update(base_info)

            # Validate trước
            if not validate_segment(seg):
                write_reject(
                    reject_file,
                    "VALIDATE_FAIL",
                    seg,
                )

                stats["REJECT"] += 1
                file_reject += 1
                continue

            # Clean text
            original_text = seg.get("matched_text", "")
            cleaned_text = clean_text(original_text)

            # Cleaner trả "" nghĩa là reject segment lỗi
            if not cleaned_text:
                seg["original_matched_text"] = original_text

                write_reject(
                    reject_file,
                    "CLEAN_TEXT_EMPTY",
                    seg,
                )

                stats["REJECT"] += 1
                file_reject += 1
                continue

            seg["matched_text"] = cleaned_text

            # Nếu seg chưa có audio_path thì tạm dùng source_file nếu có
            if "audio_path" not in seg:
                seg["audio_path"] = seg.get("source_file", "")

            write_jsonl(dataset_file, seg)

            stats["KEEP"] += 1
            file_keep += 1

        tqdm.write(f"{json_file.name} done | " f"keep={file_keep} reject={file_reject}")

    # =========================
    # 4. FINAL STATS
    # =========================
    print("\nFINAL STATS:")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print(f"\nDataset file: {dataset_file}")
    print(f"Reject file : {reject_file}")


if __name__ == "__main__":
    run()
