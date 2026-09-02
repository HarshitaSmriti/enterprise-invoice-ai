"""GSTIN and Indian tax/financial extraction utilities."""

import re
from collections import defaultdict
from .config import GSTIN_RE, GST_STATE_CODES, MONEY_ROW_RULES
from .preprocessing import candidate_rows
from .utils import (
    clean_text,
    normalize_money,
    money_float,
    is_percent_token,
    looks_like_reference_number,
)


def extract_gstins(words: list[str], boxes: list[list[float]]) -> list[dict]:
    """Scan words and candidate rows for valid Indian GSTIN numbers."""
    rows = candidate_rows(words, boxes)
    found = []

    # Check multi-token row combinations (handles spaces in GSTIN OCR)
    for row in rows:
        idxs = row["indices"]
        for start in range(len(idxs)):
            for width in range(1, min(4, len(idxs) - start) + 1):
                token_indices = idxs[start:start + width]
                candidate = "".join(
                    re.sub(r"[^A-Za-z0-9]", "", words[i])
                    for i in token_indices
                ).upper()

                for match in GSTIN_RE.finditer(candidate):
                    value = match.group(0).upper()
                    if value[:2] in GST_STATE_CODES:
                        found.append({
                            "value": value,
                            "raw": " ".join(words[i] for i in token_indices),
                            "confidence": 1.0,
                        })

    # Also check individual word tokens
    for word in words:
        candidate = re.sub(r"[^A-Za-z0-9]", "", str(word)).upper()
        for match in GSTIN_RE.finditer(candidate):
            value = match.group(0).upper()
            if value[:2] in GST_STATE_CODES:
                found.append({
                    "value": value,
                    "raw": word,
                    "confidence": 1.0,
                })

    # Deduplicate by GSTIN value
    unique = {}
    for item in found:
        unique[item["value"]] = item

    return list(unique.values())


def extract_money_fields(words: list[str], boxes: list[list[float]]) -> tuple[dict, dict]:
    """Extract financial summary fields using keyword rules and position heuristics."""
    rows = candidate_rows(words, boxes)
    output = {}
    diagnostics = defaultdict(list)

    for field, rules in MONEY_ROW_RULES.items():
        scored = []

        for row in rows:
            upper = row["text"].upper()

            for rule_group in rules:
                matched = next(
                    (label for label in rule_group if label in upper),
                    None
                )
                if not matched:
                    continue

                for idx in row["indices"]:
                    token = words[idx]
                    normalized = normalize_money(token)
                    if normalized is None or is_percent_token(token) or looks_like_reference_number(normalized):
                        continue

                    scored.append({
                        "value": normalized,
                        "raw": token,
                        "keyword": matched,
                        "row_text": row["text"],
                        "x": boxes[idx][0],
                        "y": row["y_center"],
                    })

        if not scored:
            continue

        # In summary rows, the right-most monetary amount is the actual amount
        scored.sort(key=lambda c: (c["y"], c["x"]))
        selected = scored[-1]
        output[field] = selected
        diagnostics[field] = scored

    # Reconcile TAX = CGST + SGST if both are found and tax is missing
    cgst = money_float(output.get("CENTRAL_GST", {}).get("value"))
    sgst = money_float(output.get("STATE_GST", {}).get("value"))
    if cgst is not None and sgst is not None:
        output["TAX"] = {
            "value": f"{cgst + sgst:.2f}",
            "raw": f"{cgst:.2f} + {sgst:.2f}",
            "keyword": "CGST+SGST",
            "row_text": "derived",
        }

    return output, diagnostics
