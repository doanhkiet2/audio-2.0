def validate_segment(seg):

    if not seg:
        return False

    if "matched_text" not in seg:
        return False

    if not seg["matched_text"] or len(seg["matched_text"].strip()) == 0:
        return False

    # optional safety checks only (KHÔNG reject mạnh)
    if seg.get("start", None) is None:
        return False

    if seg.get("end", None) is None:
        return False

    return True
