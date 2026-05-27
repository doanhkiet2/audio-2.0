from pathlib import Path
import shutil
import re

# =========================
# CONFIG
# =========================

INPUT_DIR = Path(r"data/input_audio")
OUTPUT_DIR = Path(r"data/input_audio")

COPY_MODE = False
# True  = copy
# False = rename/move

PATTERN = re.compile(r"^\d{2}_\d{2}__\d{2}_\d{2}__")

# =========================
# RUN
# =========================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

count = 0
skip = 0

for file in INPUT_DIR.iterdir():

    if not file.is_file():
        continue

    # đổi prefix thành "_"
    new_name = PATTERN.sub("_", file.name)

    if new_name == file.name:
        print(f"SKIP: {file.name}")
        skip += 1
        continue

    out_file = OUTPUT_DIR / new_name

    if out_file.exists():
        print(f"EXISTS: {out_file.name}")
        skip += 1
        continue

    if COPY_MODE:
        shutil.copy2(file, out_file)
    else:
        file.rename(out_file)

    print(f"{file.name} -> {new_name}")
    count += 1

print("\nDONE")
print(f"PROCESSED: {count}")
print(f"SKIPPED  : {skip}")
