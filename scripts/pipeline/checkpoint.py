import os
import json

CKPT_PATH = "data/logs/checkpoint.json"


def load_checkpoint():
    if not os.path.exists(CKPT_PATH):
        return {"done_files": []}

    with open(CKPT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(done_files):
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

    with open(CKPT_PATH, "w", encoding="utf-8") as f:
        json.dump({"done_files": done_files}, f, indent=2)
