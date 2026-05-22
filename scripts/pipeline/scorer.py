def score_sample(sample):
    """
    Phase 3 v1 scoring
    """

    confidence = sample.get("confidence", 0)
    alignment = sample.get("alignment_score", 0)

    score = (confidence + alignment) / 2

    if score >= 0.85:
        return score, "KEEP"
    elif score >= 0.65:
        return score, "REVIEW"
    else:
        return score, "REJECT"
