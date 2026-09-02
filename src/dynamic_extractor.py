"""Generic, document-agnostic key-value, table, and document type extractor.

Extracts whatever structured information is actually present in a document
without enforcing a rigid predefined invoice schema.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ----------------------------------------------------------------------
# 1. Document Type Detection
# ----------------------------------------------------------------------
def classify_document_type(words: List[str]) -> str:
    """Classify document type based on vocabulary and keyword distribution."""
    text = " ".join(words).lower()

    if any(k in text for k in ["grocery", "pantry", "supermarket", "shopping list", "produce", "cartons", "bakery", "dairy"]):
        return "grocery_list"
    if any(k in text for k in ["purchase order", "po number", "p.o. number", "po-", "buyer:", "order date"]):
        return "purchase_order"
    if any(k in text for k in ["incident report", "equipment report", "inspection report", "badge id"]):
        return "incident_report"
    if any(k in text for k in ["tax invoice", "gstin", "bill to", "invoice no", "inv no", "bill no", "invoice date"]):
        return "invoice"
    if any(k in text for k in ["cash receipt", "sales receipt", "register receipt", "cashier", "change due", "receipt id", "receipt"]):
        return "receipt"
    if any(k in text for k in ["agreement", "contract", "parties", "witnesseth", "terms & conditions"]):
        return "legal_agreement"
    if any(k in text for k in ["form", "registration", "application", "applicant"]):
        return "form"

    return "general_document"


# ----------------------------------------------------------------------
# 2. Dynamic Key-Value Pair Extraction
# ----------------------------------------------------------------------
def extract_dynamic_key_values(
    words: List[str],
    boxes: List[List[float]],
    scores: Optional[List[float]] = None,
    page: int = 1,
) -> List[Dict[str, Any]]:
    """Extract all visible label-value pairs from text lines and spatial layout."""
    if not words or len(words) != len(boxes):
        return []

    if scores is None or len(scores) != len(words):
        scores = [0.95] * len(words)

    # 1. Cluster words into lines based on vertical overlap
    lines = _cluster_words_into_lines(words, boxes, scores)

    extracted_fields = []
    seen_keys = set()

    for line in lines:
        line_text = line["text"]
        line_box = line["box"]
        line_score = line["confidence"]
        items = line["items"]

        # Strategy A: Check each token in the line for colon delimiters
        has_colon_field = False
        for it_idx, it in enumerate(items):
            w = it["word"]
            b = it["box"]
            s = it["score"]

            if ":" in w and not w.startswith("http") and not re.match(r'^\d{1,2}:\d{2}', w):
                parts = w.split(":", 1)
                k_candidate = parts[0].strip()
                v_candidate = parts[1].strip()

                # Case 1: Both label and value are inside this single OCR token (e.g. "PO Number: PO-2026-8842")
                if len(k_candidate) >= 2 and len(v_candidate) >= 1:
                    k_clean = _clean_field_name(k_candidate)
                    if k_clean and k_clean not in seen_keys and not k_clean.replace("_", "").isdigit():
                        seen_keys.add(k_clean)
                        extracted_fields.append({
                            "name": k_clean,
                            "label": k_candidate,
                            "value": _clean_value(v_candidate),
                            "confidence": round(float(s), 3),
                            "box": [round(float(v), 1) for v in b],
                            "page": page,
                            "source": "inline_colon",
                        })
                        has_colon_field = True
                        continue

                # Case 2: Colon is at end of token (e.g. "Store:", "Date:"), value follows in subsequent tokens
                if len(k_candidate) >= 2 and len(v_candidate) == 0:
                    # Look backwards to include any words in multi-word label (e.g. "Payment", "Terms:")
                    lbl_start = it_idx
                    while lbl_start > 0 and (items[lbl_start]["box"][0] - items[lbl_start - 1]["box"][2]) <= 25 and (it_idx - lbl_start) < 3:
                        lbl_start -= 1
                    k_full = " ".join([items[j]["word"] for j in range(lbl_start, it_idx + 1)]).rstrip(":").strip()
                    k_clean = _clean_field_name(k_full)

                    # Look forward for value tokens up to next colon token
                    val_items = []
                    for fwd_idx in range(it_idx + 1, len(items)):
                        if ":" in items[fwd_idx]["word"] and not items[fwd_idx]["word"].startswith("http"):
                            break
                        val_items.append(items[fwd_idx])

                    v_full = " ".join([vi["word"] for vi in val_items]).strip()

                    if k_clean and v_full and k_clean not in seen_keys and not k_clean.replace("_", "").isdigit():
                        seen_keys.add(k_clean)
                        p_box = list(b)
                        if val_items:
                            p_box[0] = items[lbl_start]["box"][0]
                            p_box[2] = val_items[-1]["box"][2]
                        extracted_fields.append({
                            "name": k_clean,
                            "label": k_full,
                            "value": _clean_value(v_full),
                            "confidence": round(float(s), 3),
                            "box": [round(float(v), 1) for v in p_box],
                            "page": page,
                            "source": "token_colon",
                        })
                        has_colon_field = True

        if has_colon_field:
            continue

        # Strategy B: Common label prefixes without colon
        prefix_pattern = r'^(Total|Subtotal|Estimated Total|Order Date|Required Date|Due Date|Budget|Date|Shopper|Vendor|Customer|Buyer|Supplier|Store|Payment Terms|Action Required|Assigned Technician|Supervisor Sign-off)\s+([$€£₹A-Za-z0-9].+)$'
        p_match = re.match(prefix_pattern, line_text, re.IGNORECASE)
        if p_match:
            k_raw = p_match.group(1).strip()
            v_raw = p_match.group(2).strip()
            k_clean = _clean_field_name(k_raw)
            if k_clean and v_raw and k_clean not in seen_keys:
                seen_keys.add(k_clean)
                extracted_fields.append({
                    "name": k_clean,
                    "label": k_raw,
                    "value": _clean_value(v_raw),
                    "confidence": round(float(line_score), 3),
                    "box": [round(float(v), 1) for v in line_box],
                    "page": page,
                    "source": "prefix_pattern",
                })

    return extracted_fields


# ----------------------------------------------------------------------
# 3. Dynamic Generic Table Extraction
# ----------------------------------------------------------------------
def extract_dynamic_tables(
    words: List[str],
    boxes: List[List[float]],
    scores: Optional[List[float]] = None,
    page: int = 1,
) -> List[Dict[str, Any]]:
    """Detect and reconstruct arbitrary tabular data without hardcoded column schemas."""
    if not words or len(words) != len(boxes):
        return []

    if scores is None:
        scores = [0.95] * len(words)

    lines = _cluster_words_into_lines(words, boxes, scores)
    if len(lines) < 3:
        return []

    # Identify candidate header line
    header_idx = -1
    columns: List[str] = []
    col_bounds: List[Tuple[float, float]] = []

    for idx, line in enumerate(lines[:-1]):
        line_words = line["words"]
        line_boxes = line["boxes"]
        text_upper = line["text"].upper()

        if len(line_words) >= 3 or any(k in text_upper for k in ["DESCRIPTION", "ITEM", "QTY", "PRICE", "RATE", "AMOUNT", "UNITS", "SKU", "CATEGORY", "CODE", "COST"]):
            segments = _segment_header_columns(line_words, line_boxes)
            if len(segments) >= 2:
                header_idx = idx
                columns = [s["name"] for s in segments]
                col_bounds = [(s["x1"], s["x2"]) for s in segments]
                break

    if header_idx == -1 or not columns:
        return []

    # Extract row entries under the header
    table_rows = []
    y_header_bottom = lines[header_idx]["box"][3]

    for line in lines[header_idx + 1:]:
        y_top = line["box"][1]
        line_text = line["text"]

        # Stop at summary/notes footer or key-value section
        if re.search(r'\b(TOTAL|SUBTOTAL|TAX|SALES TAX|NOTES|ESTIMATED TOTAL|PAYMENT|AUTHORIZED SIGNATURE|SUPERVISOR|AUTH CODE|THANK YOU)\b', line_text, re.IGNORECASE):
            break

        if y_top < y_header_bottom:
            continue

        row_dict = {col: "" for col in columns}
        line_words = line["words"]
        line_boxes = line["boxes"]

        for w, b in zip(line_words, line_boxes):
            x_mid = (b[0] + b[2]) / 2.0
            best_col = None
            min_dist = float("inf")
            for col_name, (cx1, cx2) in zip(columns, col_bounds):
                if cx1 - 40 <= x_mid <= cx2 + 40:
                    best_col = col_name
                    break
                dist = min(abs(x_mid - cx1), abs(x_mid - cx2))
                if dist < min_dist:
                    min_dist = dist
                    best_col = col_name

            if best_col:
                if row_dict[best_col]:
                    row_dict[best_col] += " " + w
                else:
                    row_dict[best_col] = w

        non_empty = [v for v in row_dict.values() if v.strip()]
        # If the line contains a colon and only 1 column has text, it is an attribute, not a data row
        if ":" in line_text and len(non_empty) <= 1:
            break

        if len(non_empty) >= 2 or (len(non_empty) >= 1 and len(row_dict[columns[0]]) > 2):
            table_rows.append(row_dict)

    if not table_rows:
        return []

    table_title = "Document Table"
    if header_idx > 0 and len(lines[header_idx - 1]["text"]) < 50:
        table_title = lines[header_idx - 1]["text"]

    return [
        {
            "title": table_title,
            "columns": columns,
            "rows": table_rows,
            "confidence": 0.92,
            "page": page,
        }
    ]


# ----------------------------------------------------------------------
# Internal Helpers
# ----------------------------------------------------------------------
def _cluster_words_into_lines(
    words: List[str],
    boxes: List[List[float]],
    scores: List[float],
    y_threshold: float = 8.0,
) -> List[Dict[str, Any]]:
    """Group words into horizontal lines sorted top-to-bottom, left-to-right."""
    items = []
    for w, b, s in zip(words, boxes, scores):
        items.append({"word": w, "box": b, "score": s})

    items.sort(key=lambda it: (it["box"][1], it["box"][0]))

    lines: List[Dict[str, Any]] = []
    for it in items:
        w_box = it["box"]
        y_mid = (w_box[1] + w_box[3]) / 2.0

        placed = False
        for line in lines:
            line_box = line["box"]
            line_ymid = (line_box[1] + line_box[3]) / 2.0
            if abs(y_mid - line_ymid) <= y_threshold:
                line["items"].append(it)
                line["box"][0] = min(line["box"][0], w_box[0])
                line["box"][1] = min(line["box"][1], w_box[1])
                line["box"][2] = max(line["box"][2], w_box[2])
                line["box"][3] = max(line["box"][3], w_box[3])
                placed = True
                break

        if not placed:
            lines.append({
                "items": [it],
                "box": list(w_box),
            })

    processed_lines = []
    for line in lines:
        line["items"].sort(key=lambda it: it["box"][0])
        line_words = [it["word"] for it in line["items"]]
        line_boxes = [it["box"] for it in line["items"]]
        line_scores = [it["score"] for it in line["items"]]

        processed_lines.append({
            "text": " ".join(line_words),
            "words": line_words,
            "boxes": line_boxes,
            "items": line["items"],
            "box": line["box"],
            "confidence": float(np.mean(line_scores)) if line_scores else 0.95,
        })

    processed_lines.sort(key=lambda l: l["box"][1])
    return processed_lines


def _segment_header_columns(words: List[str], boxes: List[List[float]]) -> List[Dict[str, Any]]:
    """Group header words into column headers with horizontal intervals."""
    if not words:
        return []

    columns = []
    curr_words = [words[0]]
    curr_x1 = boxes[0][0]
    curr_x2 = boxes[0][2]

    for w, b in zip(words[1:], boxes[1:]):
        x1, x2 = b[0], b[2]
        gap = x1 - curr_x2

        if gap <= 25:
            curr_words.append(w)
            curr_x2 = x2
        else:
            columns.append({
                "name": " ".join(curr_words),
                "x1": curr_x1,
                "x2": curr_x2,
            })
            curr_words = [w]
            curr_x1 = x1
            curr_x2 = x2

    if curr_words:
        columns.append({
            "name": " ".join(curr_words),
            "x1": curr_x1,
            "x2": curr_x2,
        })

    return columns


def _clean_field_name(raw: str) -> str:
    """Normalize raw label string into clean snake_case identifier."""
    cleaned = re.sub(r'[:#*_\-–]+', ' ', raw).strip()
    cleaned = re.sub(r'\\s+', '_', cleaned).lower()
    return cleaned


def _clean_value(val: str) -> str:
    """Clean extracted value string."""
    return re.sub(r'[\\r\\n]+', ' ', val).strip()
