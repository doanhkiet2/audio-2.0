import os
from tqdm import tqdm

from config_loader import load_config
from loader import load_alignment_segments
from validator import validate_segment
from cleaner import clean_text
from writer import write_dataset_item
from checkpoint import load_checkpoint, save_checkpoint


def run():
    # =========================
    # 1. LOAD CONFIG
    # =========================
    cfg = load_config()

    input_dir = cfg["paths"]["input_dir"]
    align_dir = cfg["paths"]["align_dir"]
    output_dir = cfg["paths"]["output_dir"]

    skip_done = cfg["runtime"].get("skip_done_files", True)

    # =========================
    # 2. LOAD CHECKPOINT
    # =========================
    ckpt = load_checkpoint()
    done_files = set(ckpt.get("done_files", []))

    # =========================
    # 3. GET INPUT FILES
    # =========================
    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".wav") or f.endswith(".mp3")
    ]

    stats = {"KEEP": 0, "REJECT": 0, "SKIP": 0}
    for f in files:

        print("\n[DEBUG FILE]:", f)
        print("[DEBUG ALIGN DIR]:", align_dir)

        segments = load_alignment_segments(f, align_dir)

        print("[DEBUG SEGMENTS]:", segments)

        if not segments:
            stats["REJECT"] += 1
            continue
    # =========================
    # 4. MAIN LOOP
    # =========================
    for f in tqdm(files, desc="Phase 3A Factory"):

        # skip if already processed
        if skip_done and f in done_files:
            stats["SKIP"] += 1
            continue

        # load Phase 1.5 alignment output (segment list)
        segments = load_alignment_segments(f, align_dir)

        # if missing alignment → mark done & skip
        if not segments:
            done_files.add(f)
            save_checkpoint(list(done_files))
            stats["REJECT"] += 1
            continue

        # =========================
        # 5. PROCESS SEGMENTS
        # =========================
        DEBUG = True
        for seg in segments:
            if DEBUG:
                print(seg)
                print("VALID:", validate_segment(seg))

            # basic validation
            if not validate_segment(seg):
                stats["REJECT"] += 1
                continue

            # clean text
            seg["matched_text"] = clean_text(seg["matched_text"])

            # attach metadata
            seg["audio_path"] = f

            # write dataset
            write_dataset_item(seg, output_dir)

            stats["KEEP"] += 1

        # =========================
        # 6. CHECKPOINT UPDATE
        # =========================
        done_files.add(f)
        save_checkpoint(list(done_files))

        tqdm.write(f"{os.path.basename(f)} done | {stats}")

    # =========================
    # 7. FINAL STATS
    # =========================
    print("\nFINAL STATS:")
    print(stats)


if __name__ == "__main__":
    run()
