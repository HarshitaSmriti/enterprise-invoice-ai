"""Document type routing and cross-model inference dispatcher."""

import re
from .config import GSTIN_RE, GST_STATE_CODES


def detect_document_type(words: list[str]) -> str:
    """Classify document as Indian GST Invoice or International Invoice."""
    text_corpus = " ".join(words).upper()

    # 1. GSTIN Check
    for word in words:
        cand = re.sub(r"[^A-Za-z0-9]", "", word).upper()
        if GSTIN_RE.fullmatch(cand) and cand[:2] in GST_STATE_CODES:
            return "INDIAN_GST"

    # 2. Keywords
    gst_keywords = ["CGST", "SGST", "IGST", "GSTIN", "TAXABLE VALUE", "BILL QTY", "GR QTY"]
    hits = sum(1 for kw in gst_keywords if kw in text_corpus)
    if hits >= 2 or "₹" in text_corpus or "INR" in text_corpus:
        return "INDIAN_GST"

    return "INTERNATIONAL"
