import string


def normalize_text(text: str, mode: str = "asr_fair") -> str:
    """
    Normalize text for metric calculation.

    Modes:
    - "standard": lowercase + whitespace collapse (preserves punctuation)
    - "asr_fair": lowercase + punctuation removal + whitespace collapse
    """
    text = text.lower()

    if mode == "asr_fair":
        text = text.translate(str.maketrans("", "", string.punctuation))

    return " ".join(text.split())
