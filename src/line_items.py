"""Geometry-aware invoice table and line-item reconstruction (v10/v11)."""

import re
from .config import (
    BILL_HEADER_MARKERS,
    CONSUMPTION_MARKERS,
    SUMMARY_WORDS,
    HEADER_WORDS,
    DATE_RE,
)
from .preprocessing import candidate_rows
from .utils import (
    clean_text,
    norm_upper,
    normalize_money,
    money_float,
    looks_like_reference_number,
    x_center,
)


def is_summary_row(text: str) -> bool:
    t = norm_upper(text)
    return any(k in t for k in SUMMARY_WORDS)


def is_consumption_row(text: str) -> bool:
    t = norm_upper(text)
    return any(k in t for k in CONSUMPTION_MARKERS)


def find_bill_table_bounds(rows: list[dict]) -> tuple[int | None, int | None]:
    """Identify the header and termination boundaries of the actual BILL item table."""
    header_idx = None

    for i, row in enumerate(rows):
        t = norm_upper(row["text"])
        score = sum(marker in t for marker in BILL_HEADER_MARKERS)
        if score >= 3 and ("PO" in t or "MATERIAL" in t) and "QTY" in t:
            header_idx = i
            break

    if header_idx is None:
        return None, None

    end_idx = len(rows)
    for i in range(header_idx + 1, len(rows)):
        t = norm_upper(rows[i]["text"])
        if "CONSUMPTION DETAILS" in t:
            end_idx = i
            break
        if re.search(r"\bTOTAL\b", t) and not any(
            x in t for x in ("GST", "VALUE", "AMOUNT IN WORDS")
        ):
            end_idx = i
            break

    return header_idx, end_idx


def row_numbers(row: dict, words: list[str], boxes: list[list[float]]) -> list[dict]:
    result = []
    for idx in row["indices"]:
        token = str(words[idx]).strip()
        if not token or "%" in token or DATE_RE.fullmatch(token):
            continue

        value = normalize_money(token)
        if value is None or looks_like_reference_number(value):
            continue

        result.append({
            "index": idx,
            "value": value,
            "x": x_center(boxes[idx]),
            "raw": token,
        })

    return sorted(result, key=lambda n: n["x"])


def bill_row_numeric_columns(row: dict, words: list[str], boxes: list[list[float]]) -> dict | None:
    """Select the four deterministic numeric columns: GR QTY, BILL QTY, RATE, AMT."""
    nums = row_numbers(row, words, boxes)
    if len(nums) < 4:
        return None

    tail = nums[-4:]
    gr_qty, bill_qty, rate, amount = tail

    q1 = money_float(gr_qty["value"])
    q2 = money_float(bill_qty["value"])
    r = money_float(rate["value"])
    a = money_float(amount["value"])

    if any(v is None or v <= 0 for v in (q1, q2, r, a)):
        return None

    arithmetic_error = abs(q2 * r - a) / max(a, 1.0)
    if arithmetic_error > 0.05:  # slight tolerance for rounding
        return None

    return {
        "quantity": bill_qty["value"],
        "unit_price": rate["value"],
        "amount": amount["value"],
        "confidence": max(0.85, 1.0 - arithmetic_error),
    }


def bill_row_description(row: dict, words: list[str], boxes: list[list[float]]) -> str:
    """Extract item description by stripping PO and the 4 numeric columns."""
    nums = row_numbers(row, words, boxes)
    if len(nums) < 4:
        return ""

    numeric_indices = {n["index"] for n in nums[-4:]}
    desc = []

    for idx in row["indices"]:
        token = clean_text(words[idx])
        if not token or idx in numeric_indices:
            continue

        compact = re.sub(r"[^A-Za-z0-9]", "", token)
        if compact.isdigit() and len(compact) >= 6:
            continue

        desc.append(token)

    return clean_text(" ".join(desc))


def reconstruct_line_items(words: list[str], boxes: list[list[float]]) -> list[dict]:
    """Reconstruct verified line items from document rows."""
    rows = candidate_rows(words, boxes)
    header_idx, end_idx = find_bill_table_bounds(rows)

    if header_idx is None:
        return []

    items = []
    for row in rows[header_idx + 1:end_idx]:
        text = clean_text(row["text"])
        upper = norm_upper(text)

        if not text or is_consumption_row(text) or is_summary_row(text):
            continue

        if re.search(r"\bTOTAL\b", upper):
            continue

        columns = bill_row_numeric_columns(row, words, boxes)
        if columns is None:
            continue

        description = bill_row_description(row, words, boxes)
        if len(description) < 2 or re.fullmatch(r"[\d\s.,/_-]+", description):
            continue

        items.append({
            "description": description,
            "quantity": columns["quantity"],
            "unit_price": columns["unit_price"],
            "amount": columns["amount"],
            "confidence": round(float(columns["confidence"]), 4),
        })

    # Deduplicate exact rows
    deduped = []
    seen = set()
    for item in items:
        sig = (
            re.sub(r"\W+", "", item["description"].upper()),
            item["quantity"],
            item["unit_price"],
            item["amount"],
        )
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(item)

    return deduped
