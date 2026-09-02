"""Utility functions for text cleaning, monetary parsing, date normalization,
and bounding box geometry.
"""

import re
import math
import datetime as dt
import numpy as np
from .config import DATE_RE, MONEY_TOKEN_RE


def clean_text(value) -> str:
    """Normalize whitespace and strip leading/trailing spaces."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def norm_upper(text) -> str:
    """Clean text and convert to uppercase for matching."""
    return clean_text(text).upper()


def ocr_numeric_normalize(value) -> str:
    """Normalize OCR character confusions in numbers (O->0, I/l/|->1)."""
    s = str(value).strip()
    s = s.replace("₹", "")
    s = re.sub(r"\bRs\.?\b", "", s, flags=re.I)
    s = s.replace("O", "0").replace("o", "0")
    s = s.replace("I", "1").replace("l", "1")
    s = s.replace("|", "1")
    return s.replace(" ", "")


def normalize_money(value) -> str | None:
    """Extract and format a monetary decimal string (e.g. '1234.56')."""
    if value is None:
        return None

    s = ocr_numeric_normalize(value)
    s = re.sub(r"[^0-9.\-]", "", s)

    if not s or s in {"-", "."}:
        return None

    if s.count("-") > 1 or ("-" in s and not s.startswith("-")):
        return None

    try:
        n = float(s)
    except Exception:
        return None

    if not math.isfinite(n):
        return None

    return f"{n:.2f}"


def money_float(value) -> float | None:
    """Convert monetary value to float or None."""
    v = normalize_money(value)
    return float(v) if v is not None else None


def looks_like_reference_number(value) -> bool:
    """Detect if a numeric string is likely a reference/PO/document ID rather than money.
    Pure digit strings of 7+ digits are almost always identifiers.
    """
    v = normalize_money(value)
    if v is None:
        return False

    digits = re.sub(r"\D", "", v.split(".")[0])
    return len(digits) >= 7


def is_percent_token(text) -> bool:
    """Check if token represents a percentage (e.g. '18%' or numeric <= 50)."""
    t = str(text).strip()
    if "%" in t:
        return True

    v = money_float(t)
    return v is not None and 0 <= v <= 50


def normalize_date(value) -> str | None:
    """Parse various invoice date formats into ISO 8601 (YYYY-MM-DD)."""
    value = clean_text(value)
    if not value:
        return None

    found = DATE_RE.search(value)
    if found:
        value = found.group(0)
    value = value.replace(" ", "")

    formats = [
        "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y",
        "%d.%m.%y", "%d/%m/%y", "%d-%m-%y",
        "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
    ]

    for fmt in formats:
        try:
            parsed = dt.datetime.strptime(value, fmt)
            if parsed.year < 2000:
                parsed = parsed.replace(year=parsed.year + 2000)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return None


def y_center(box) -> float:
    """Calculate vertical center of a bounding box [x1, y1, x2, y2]."""
    return (box[1] + box[3]) / 2.0


def x_center(box) -> float:
    """Calculate horizontal center of a bounding box [x1, y1, x2, y2]."""
    return (box[0] + box[2]) / 2.0


def row_height(box) -> float:
    """Calculate height of a bounding box."""
    return max(1.0, float(box[3] - box[1]))


def clamp_box(box, width: int, height: int) -> list[float]:
    """Clamp box coordinates to image dimensions [0, width, 0, height]."""
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(float(width), float(x1))),
        max(0.0, min(float(height), float(y1))),
        max(0.0, min(float(width), float(x2))),
        max(0.0, min(float(height), float(y2))),
    ]
