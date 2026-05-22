VIET_FIX_MAP = {"tôn hùng": "tôm hùm", "thư kế": "thừa kế", "vũ khống": "vu khống"}


def clean_text(text: str) -> str:
    text = text.lower().strip()

    for k, v in VIET_FIX_MAP.items():
        text = text.replace(k, v)

    return text
