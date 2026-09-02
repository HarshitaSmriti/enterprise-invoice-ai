"""Generic, document-agnostic key-value, table, and document type extractor.

Derives all structure, labels, values, and table grids dynamically from OCR text
and spatial bounding boxes without relying on hardcoded field lists or table schemas.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ----------------------------------------------------------------------
# 1. Document Type Detection (Metadata Hint Only)
# ----------------------------------------------------------------------
def classify_document_type(words: List[str]) -> str:
    """Classify document type based on vocabulary distribution for metadata display."""
    text = " ".join(words).lower()

    if any(k in text for k in ["tax invoice", "gstin", "bill to", "invoice no", "inv no", "bill no", "invoice date"]):
        return "invoice"
    if any(k in text for k in ["cash receipt", "sales receipt", "register receipt", "cashier", "change due", "receipt id", "receipt"]):
        return "receipt"
    if any(k in text for k in ["purchase order", "po number", "p.o. number", "po-", "buyer:", "order date"]):
        return "purchase_order"
    if any(k in text for k in ["grocery", "pantry", "supermarket", "shopping list", "produce", "cartons", "bakery", "dairy"]):
        return "grocery_list"
    if any(k in text for k in ["incident report", "equipment report", "inspection report", "badge id"]):
        return "incident_report"
    if any(k in text for k in ["agreement", "contract", "parties", "witnesseth", "terms & conditions"]):
        return "legal_agreement"
    if any(k in text for k in ["form", "registration", "application", "applicant", "onboarding"]):
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
    exclude_boxes: Optional[List[List[float]]] = None,
) -> List[Dict[str, Any]]:
    """Extract all visible label-value pairs from text lines and spatial layout.
    
    Discovers:
    - Inline colon pairs: 'Key: Value'
    - Token colon pairs: 'Key:' followed by value tokens
    - Spatial gap pairs: Left label, Right value separated by horizontal distance
    """
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

        # If line falls inside a table bounding box, skip it so table data stays in tables
        if exclude_boxes:
            line_ymid = (line_box[1] + line_box[3]) / 2.0
            if any(tb[1] - 4 <= line_ymid <= tb[3] + 4 for tb in exclude_boxes):
                continue

        # Strategy A: Colon-delimited key-value pairs (inline or token-separated)
        has_colon_field = False
        for it_idx, it in enumerate(items):
            w = it["word"]
            b = it["box"]
            s = it["score"]

            if ":" in w and not w.startswith("http") and not re.match(r'^\d{1,2}:\d{2}', w):
                parts = w.split(":", 1)
                k_candidate = parts[0].strip()
                v_candidate = parts[1].strip()

                # Case A1: Single OCR box contains both Key and Value (e.g. "PO Number: PO-2026-8842")
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

                # Case A2: Colon is attached to label token (e.g. "Store:", "Date:"), value follows in subsequent tokens
                if len(k_candidate) >= 2 and len(v_candidate) == 0:
                    lbl_start = it_idx
                    while lbl_start > 0 and (items[lbl_start]["box"][0] - items[lbl_start - 1]["box"][2]) <= 25 and (it_idx - lbl_start) < 4:
                        lbl_start -= 1
                    k_full = " ".join([items[j]["word"] for j in range(lbl_start, it_idx + 1)]).rstrip(":").strip()
                    k_clean = _clean_field_name(k_full)

                    # Gather value tokens until next colon token or line end
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

        # Strategy B: Purely geometric spatial label-value pairs without colon
        # Discovers arbitrary left-aligned labels paired with right-aligned values
        if len(items) >= 2:
            for split_idx in range(1, min(len(items), 5)):
                left_items = items[:split_idx]
                right_items = items[split_idx:]

                k_text = " ".join([it["word"] for it in left_items]).strip()
                v_text = " ".join([it["word"] for it in right_items]).strip()

                gap = right_items[0]["box"][0] - left_items[-1]["box"][2]

                if gap >= 20 and len(k_text) <= 32 and len(v_text) >= 1:
                    # Valid label begins with an uppercase letter or alphanumeric code
                    if k_text[0].isupper() and not any(k_text.lower().startswith(w) for w in ["the ", "this ", "please ", "during ", "note "]):
                        k_clean = _clean_field_name(k_text)
                        if k_clean and k_clean not in seen_keys and not k_clean.replace("_", "").isdigit():
                            seen_keys.add(k_clean)
                            extracted_fields.append({
                                "name": k_clean,
                                "label": k_text,
                                "value": _clean_value(v_text),
                                "confidence": round(float(line_score), 3),
                                "box": [round(float(v), 1) for v in line_box],
                                "page": page,
                                "source": "spatial_gap_pair",
                            })
                            break

    return extracted_fields


# ----------------------------------------------------------------------
# 3. Dynamic Generic Table Extraction (Purely Geometric)
# ----------------------------------------------------------------------
def extract_dynamic_tables(
    words: List[str],
    boxes: List[List[float]],
    scores: Optional[List[float]] = None,
    page: int = 1,
) -> List[Dict[str, Any]]:
    """Detect and reconstruct arbitrary tabular data through geometric column analysis.
    
    Does NOT depend on predefined column names. Detects candidate headers by verifying
    that subsequent lines share the same horizontal column intervals.
    """
    if not words or len(words) != len(boxes):
        return []

    if scores is None:
        scores = [0.95] * len(words)

    lines = _cluster_words_into_lines(words, boxes, scores)
    if len(lines) < 3:
        return []

    tables_found = []
    used_line_indices = set()

    # Search for candidate header lines geometrically
    for idx, line in enumerate(lines[:-2]):
        if idx in used_line_indices:
            continue

        line_words = line["words"]
        line_boxes = line["boxes"]
        line_text = line["text"]

        # Table headers NEVER contain colons (colons indicate key-value pairs)
        if ":" in line_text:
            continue

        # A candidate header must have at least 2 distinct horizontal segments
        segments = _segment_header_columns(line_words, line_boxes)
        if len(segments) < 2:
            continue

        # Validate header geometrically: subsequent lines must align with these columns
        subsequent_lines = lines[idx + 1:min(idx + 15, len(lines))]
        if not _validate_table_geometry(segments, subsequent_lines):
            continue

        # Header validated geometrically!
        columns = [s["name"] for s in segments]
        col_bounds = [(s["x1"], s["x2"]) for s in segments]
        y_header_bottom = line["box"][3]
        used_line_indices.add(idx)

        # Extract table rows
        table_rows = []
        last_y = y_header_bottom

        for row_idx, r_line in enumerate(subsequent_lines, start=idx + 1):
            y_top = r_line["box"][1]
            r_text = r_line["text"]

            # Stop if vertical gap is unusually large (> 3.5x row height)
            if y_top - last_y > 80:
                break

            # Stop if line has a colon and only 1 text block (e.g. key-value footer)
            if ":" in r_text and len(r_line["words"]) <= 4:
                break

            # Map tokens in this row to discovered columns
            row_dict = {col: "" for col in columns}
            for w, b in zip(r_line["words"], r_line["boxes"]):
                x_mid = (b[0] + b[2]) / 2.0
                best_col = None
                min_dist = float("inf")
                for col_name, (cx1, cx2) in zip(columns, col_bounds):
                    if cx1 - 45 <= x_mid <= cx2 + 45:
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
            if len(non_empty) >= 2 or (len(non_empty) >= 1 and len(row_dict[columns[0]]) > 2):
                table_rows.append(row_dict)
                used_line_indices.add(row_idx)
                last_y = r_line["box"][3]
            else:
                # If subsequent line does not match table layout, table has ended
                if table_rows:
                    break

        if len(table_rows) >= 2:
            table_title = "Document Table"
            if idx > 0 and len(lines[idx - 1]["text"]) < 60 and not lines[idx - 1]["text"].endswith(":"):
                table_title = lines[idx - 1]["text"]

            table_box = [
                float(min(col_bounds[0][0], line["box"][0])),
                float(line["box"][1]),
                float(max(col_bounds[-1][1], line["box"][2])),
                float(last_y),
            ]

            tables_found.append({
                "title": table_title,
                "columns": columns,
                "rows": table_rows,
                "box": table_box,
                "confidence": 0.92,
                "page": page,
            })

    return tables_found


# ----------------------------------------------------------------------
# Internal Helpers
# ----------------------------------------------------------------------
def _validate_table_geometry(segments: List[Dict[str, Any]], subsequent_lines: List[Dict[str, Any]]) -> bool:
    """Validate whether candidate columns repeat across at least 2 subsequent lines."""
    if len(segments) < 2 or not subsequent_lines:
        return False

    matching_lines = 0
    for line in subsequent_lines[:6]:
        line_text = line["text"]
        if ":" in line_text and len(line["words"]) <= 3:
            continue

        hits = 0
        for seg in segments:
            cx1, cx2 = seg["x1"], seg["x2"]
            if any(cx1 - 45 <= (b[0] + b[2]) / 2.0 <= cx2 + 45 for b in line["boxes"]):
                hits += 1

        if hits >= 2:
            matching_lines += 1

    return matching_lines >= 2


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
    cleaned = re.sub(r'\s+', '_', cleaned).lower()
    return cleaned


def _clean_value(val: str) -> str:
    """Clean extracted value string."""
    return re.sub(r'[\r\n]+', ' ', val).strip()
