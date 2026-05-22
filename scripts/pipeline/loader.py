import os
import json


def load_alignment_segments(file_path, align_dir):

    file_name = os.path.basename(file_path)
    json_path = os.path.join(align_dir, file_name + ".json")

    if not os.path.exists(json_path):
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
