import os
import json
import glob
from statistics import mean

INPUT_DIR = "data/output_dataset"


def analyze_file(path):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # =========================
    # NORMALIZE DATA FORMAT
    # =========================

    if isinstance(data, dict) and "segments" in data:
        data = data["segments"]
    if isinstance(data, dict):
        data = [data]
    if isinstance(data, str):
        data = json.loads(data)

    if not isinstance(data, list):
        print(f"[SKIP] bad format: {path}")
        return None

    durations = []
    scores = []
    cps = []

    for seg in data:

        if not isinstance(seg, dict):
            continue

        if "start" not in seg or "end" not in seg:
            continue

        duration = seg["end"] - seg["start"]
        durations.append(duration)

        scores.append(seg.get("score", 0))
        cps.append(seg.get("chars_per_sec", 0))

    if not durations:
        return None

    return {
        "file": os.path.basename(path),
        "segments": len(durations),
        "avg_duration": sum(durations) / len(durations),
        "avg_score": sum(scores) / len(scores),
        "avg_chars_per_sec": sum(cps) / len(cps),
    }


def run():

    files = glob.glob(os.path.join(INPUT_DIR, "*.json"))

    report = []

    for f in files:
        result = analyze_file(f)
        if result is not None:
            report.append(result)

    print("\n=== DATASET QC REPORT ===\n")

    print("TOTAL FILES:", len(report))

    if not report:
        print("EMPTY DATASET")
        return

    print("\nAVG SCORE:", mean([r["avg_score"] for r in report]))
    print("AVG DURATION:", mean([r["avg_duration"] for r in report]))
    print("AVG CPS:", mean([r["avg_chars_per_sec"] for r in report]))

    print("\nSAMPLE:")
    for r in report[:5]:
        print(r)


if __name__ == "__main__":
    run()
