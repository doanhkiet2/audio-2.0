# PROJECT CONTEXT

## 1. MỤC TIÊU

Build Vietnamese TTS Voice Clone dataset pipeline.

Output cuối:

RAW AUDIO
→ CLEAN
→ WHISPER
→ ALIGN
→ FILTER
→ DATASET
→ TRAIN READY

---

## 2. KIẾN TRÚC

voice-project-2.0

data/
├── raw/audio
├── processed
├── wav16k
├── normalized
├── segments

output/

dataset_final/

scripts/

docs/

---

## 3. QUY ƯỚC CODE

### Path

- dùng pathlib
- không hardcode path

Ví dụ:

BASE_DIR = Path(__file__).resolve().parents[1]

---

### Progress UI

mọi script chạy >30 giây:

bắt buộc có:

- tqdm
- print tiến trình
- ETA

---

### Logging

ưu tiên:

[INFO]
[SKIP]
[ERROR]
[DONE]

---

## 4. QUY TẮC DATASET

Giữ:

- transcript khớp audio
- segment tự nhiên
- duration hợp lý

Loại:

- lệch nội dung
- số đọc sai
- text bẩn
- mất dấu

---

## 5. FILE QUAN TRỌNG

scripts/

align_hybrid.py

build_dataset.py

dataset_clean_pipeline.py

---

## 6. PHASE HIỆN TẠI

Phase 3
Dataset Cleaning

---

## 7. KPI

Target:

score >95

reject <35%

dataset ≥30h sạch

---

## 8. CÁCH LÀM VIỆC

Khi mở chat mới:

1.
Đọc docs/PROJECT_CONTEXT.md

2.
Đọc docs/NEXT_STEP.md

3.
đọc docs/RULES.md

3.
Chỉ sau đó mới đề xuất code
