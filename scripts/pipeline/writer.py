import os, json
import uuid


def write_dataset_item(seg, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    file_name = f"{uuid.uuid4().hex}.json"
    out_path = os.path.join(output_dir, file_name)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([seg], f, ensure_ascii=False, indent=2)
