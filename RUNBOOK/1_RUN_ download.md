```md
# Step 4 — Run Hybrid Alignment (Whisper + YouTube)

## Mục tiêu:

Kết hợp:

- Whisper transcript (có timestamp)
- YouTube transcript (text sạch)

👉 để tạo transcript chất lượng cao hơn:


output/aligned.txt


---

## File chạy:

Script:


scripts/align_hybrid.py


Run:

```bash
python scripts/align_hybrid.py
Input:
1. Whisper transcript
data/whisper.txt

Format:

[0.00 - 2.50] xin chao moi nguoi
[2.50 - 5.20] hom nay chung ta se...
2. YouTube transcript (clean text + timestamp)
data/youtube_clean.txt

Format:

00:00:00 Xin chào mọi người
00:00:05 Hôm nay chúng ta sẽ...
Output:
1. File chính (accepted alignment)
output/aligned.txt

Format:

[0.00 - 2.50] Xin chào mọi người.
[2.50 - 5.20] Hôm nay chúng ta sẽ...
2. File bị loại (debug)
output/rejected.txt

Chứa các đoạn:

không match tốt
score thấp
sai timing
noise
Logic hoạt động:
1. Parse dữ liệu
Whisper → parse timestamp range
YouTube → parse timestamp + text
2. Matching

Với mỗi whisper segment:

tìm candidate YouTube text trong khoảng thời gian
thêm previous segment (offset PREVIOUS_SHIFT)
tạo context
3. Scoring

Dùng:

fuzzy matching (rapidfuzz)
word ratio
char ratio
punctuation bonus
4. Filter

Một segment chỉ được ACCEPT nếu:

similarity ≥ 75
duration hợp lệ (1–20s)
chars/sec hợp lý (3–35)
Output format:
[start - end] (score=X.X) text