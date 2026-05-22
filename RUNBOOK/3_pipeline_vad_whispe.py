import subprocess
import sys
from pathlib import Path

# =========================
# 🎯 CHỈ SỬA DÒNG NÀY
TARGET_PATH = "/home/admin1/voice-project-2.0/pipeline/pipeline.py"
# =========================

def run_target(path_str: str):
    path = Path(path_str).resolve()

    if not path.exists():
        print(f"❌ File không tồn tại: {path}")
        return

    if not path.suffix == ".py":
        print("❌ Chỉ hỗ trợ file .py")
        return

    print(f"🚀 Running: {path}")

    result = subprocess.run(
        [sys.executable, str(path)],
        check=False
    )

    if result.returncode == 0:
        print("✅ Done")
    else:
        print("❌ Script failed")

if __name__ == "__main__":
    run_target(TARGET_PATH)