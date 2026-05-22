import json
from pathlib import Path
from statistics import mean
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]

META_FILE = BASE_DIR / "dataset_final" / "metadata.jsonl"


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(values, p):
    values = sorted(values)
    idx = int(len(values) * p / 100)
    idx = min(idx, len(values) - 1)
    return values[idx]


def main():
    print(f"[INFO] Reading: {META_FILE}")

    rows = load_jsonl(META_FILE)

    durations = []
    scores = []
    cps_values = []

    for row in tqdm(rows, desc="Analyzing metadata"):
        start = row.get("start")
        end = row.get("end")

        if start is not None and end is not None:
            durations.append(end - start)

        if row.get("score") is not None:
            scores.append(row["score"])

        if row.get("chars_per_sec") is not None:
            cps_values.append(row["chars_per_sec"])

    print("\n========== SUMMARY ==========")
    print(f"Total rows: {len(rows)}")

    if durations:
        print("\n[DURATION]")
        print(f"Min: {min(durations):.2f}s")
        print(f"Max: {max(durations):.2f}s")
        print(f"Avg: {mean(durations):.2f}s")
        print(f"P50: {pct(durations, 50):.2f}s")
        print(f"P90: {pct(durations, 90):.2f}s")
        print(f"P95: {pct(durations, 95):.2f}s")

    if scores:
        print("\n[SCORE]")
        print(f"Min: {min(scores):.2f}")
        print(f"Max: {max(scores):.2f}")
        print(f"Avg: {mean(scores):.2f}")
        print(f"P50: {pct(scores, 50):.2f}")
        print(f"P90: {pct(scores, 90):.2f}")
        print(f"P95: {pct(scores, 95):.2f}")

    if cps_values:
        print("\n[CHARS PER SEC]")
        print(f"Min: {min(cps_values):.2f}")
        print(f"Max: {max(cps_values):.2f}")
        print(f"Avg: {mean(cps_values):.2f}")
        print(f"P50: {pct(cps_values, 50):.2f}")
        print(f"P90: {pct(cps_values, 90):.2f}")
        print(f"P95: {pct(cps_values, 95):.2f}")


if __name__ == "__main__":
    main()
