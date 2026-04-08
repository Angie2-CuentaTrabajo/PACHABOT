from __future__ import annotations

import re
import unicodedata


def fix_mojibake(text: str) -> str:
    """Repair common UTF-8/Windows-1252 mojibake when detected."""

    suspicious_tokens = ("Ã", "Â", "â€œ", "â€", "â€“", "â€”")
    if not any(token in text for token in suspicious_tokens):
        return text

    for encoding_name in ("cp1252", "latin-1"):
        try:
            repaired = text.encode(encoding_name).decode("utf-8")
            if repaired.count("Ã") < text.count("Ã"):
                return repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text


def strip_accents(text: str) -> str:
    """Remove accents for rule-based matching while preserving base characters."""

    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def clean_text(text: str) -> str:
    """Normalize whitespace and remove noisy empty lines."""

    text = fix_mojibake(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_search(text: str) -> str:
    """Prepare text for keyword matching."""

    cleaned = clean_text(text).lower()
    return strip_accents(cleaned)
