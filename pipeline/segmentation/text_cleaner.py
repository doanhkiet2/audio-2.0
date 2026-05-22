import re
import unicodedata


# -------------------------
# 1. normalize unicode
# -------------------------
def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


# -------------------------
# 2. fix spacing issues
# -------------------------
def fix_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text.strip()


# -------------------------
# 3. common vietnamese whisper fixes
# -------------------------
VIET_FIX_MAP = {
    "tôn hùng": "tôm hùm",
    "tôn hùng úc": "tôm hùm úc",
    "thư kế": "thừa kế",
    "vũ khống": "vu khống",
    "dân dân": "Dương Dương",
    "tạm minh viễn": "Tạ Minh Viễn",
    "hứa thân nhi": "Hứa Thanh Nhi",
    "hứa thang nhi": "Hứa Thanh Nhi",
    "sản": "sàn",
    "giác": "rác",
}


def fix_common_errors(text: str) -> str:
    lower = text.lower()

    for wrong, correct in VIET_FIX_MAP.items():
        lower = lower.replace(wrong, correct.lower())

    return lower


# -------------------------
# 4. restore capitalization (light)
# -------------------------
def restore_capitalization(text: str) -> str:
    sentences = re.split(r"([.!?])", text)
    out = []

    for i in range(0, len(sentences), 2):
        part = sentences[i].strip()
        if not part:
            continue

        part = part[0].upper() + part[1:]
        out.append(part)

        if i + 1 < len(sentences):
            out.append(sentences[i + 1])

    return "".join(out)


# -------------------------
# MAIN CLEAN FUNCTION
# -------------------------
def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = fix_common_errors(text)
    text = fix_spacing(text)
    text = restore_capitalization(text)

    return text