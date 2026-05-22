import re

# ==================================================
# ASR FIX MAP
# Chỉ dùng cho lỗi nghe sai / nhận diện sai.
# Không dùng cho số -> chữ.
# ==================================================

VIET_FIX_MAP = {
    "tôn hùng": "tôm hùm",
    "thư kế": "thừa kế",
    "vũ khống": "vu khống",
}


# ==================================================
# NUMBER NORMALIZATION
# ==================================================

DIGIT_WORDS = {
    "0": "không",
    "1": "một",
    "2": "hai",
    "3": "ba",
    "4": "bốn",
    "5": "năm",
    "6": "sáu",
    "7": "bảy",
    "8": "tám",
    "9": "chín",
}

UNITS = [
    "",
    "một",
    "hai",
    "ba",
    "bốn",
    "năm",
    "sáu",
    "bảy",
    "tám",
    "chín",
]


def read_two_digits(n: int) -> str:
    if n < 10:
        return UNITS[n]

    tens = n // 10
    unit = n % 10

    if tens == 1:
        if unit == 0:
            return "mười"
        if unit == 5:
            return "mười lăm"
        return "mười " + UNITS[unit]

    result = UNITS[tens] + " mươi"

    if unit == 0:
        return result
    if unit == 1:
        return result + " mốt"
    if unit == 4:
        return result + " tư"
    if unit == 5:
        return result + " lăm"

    return result + " " + UNITS[unit]


def read_three_digits(n: int) -> str:
    if n < 100:
        return read_two_digits(n)

    hundred = n // 100
    rest = n % 100

    result = UNITS[hundred] + " trăm"

    if rest == 0:
        return result

    if rest < 10:
        return result + " lẻ " + UNITS[rest]

    return result + " " + read_two_digits(rest)


def number_to_vietnamese(n: int) -> str:
    if n < 0:
        return "âm " + number_to_vietnamese(abs(n))

    if n < 1000:
        return read_three_digits(n)

    if n < 1_000_000:
        thousands = n // 1000
        rest = n % 1000

        result = number_to_vietnamese(thousands) + " nghìn"

        if rest == 0:
            return result

        if rest < 100:
            return result + " không trăm " + read_two_digits(rest)

        return result + " " + read_three_digits(rest)

    return " ".join(DIGIT_WORDS[d] for d in str(n))


def read_digits_one_by_one(text: str) -> str:
    return " ".join(DIGIT_WORDS[d] for d in text)


def normalize_context_numbers(text: str) -> str:
    # Case truyện/audio đặc biệt
    text = re.sub(r"\b2\s+ca\b", "song ca", text)
    text = re.sub(r"\b3\s+ca\b", "tam ca", text)

    # Hotline / mã số hay đọc từng chữ số
    text = re.sub(
        r"\b120\b",
        lambda m: read_digits_one_by_one(m.group(0)),
        text,
    )

    return text


def normalize_numbers(text: str) -> str:
    text = normalize_context_numbers(text)

    def replace_number(match):
        raw = match.group(0)

        # Bỏ dấu phẩy/chấm trong số lớn kiểu 1,000 hoặc 1.000
        clean = raw.replace(",", "").replace(".", "")

        try:
            n = int(clean)
        except ValueError:
            return raw

        return number_to_vietnamese(n)

    text = re.sub(r"\b\d[\d,.]*\b", replace_number, text)

    return text


# ==================================================
# TEXT CLEANER
# ==================================================


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower().strip()

    # Fix lỗi ASR trước
    for wrong, right in VIET_FIX_MAP.items():
        text = text.replace(wrong, right)

    # Normalize số
    text = normalize_numbers(text)

    # Dọn khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()

    return text
