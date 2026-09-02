"""Validation engine checking structural integrity, arithmetic reconciliation, and schemas."""

import re
import datetime as dt
from .config import CANONICAL_FIELDS, MONEY_FIELDS, GSTIN_RE, GST_STATE_CODES
from .utils import looks_like_reference_number


def final_consistency_check(result: dict) -> tuple[bool, list[str]]:
    """Validate canonical extracted JSON against business rules and data schemas."""
    errors = []

    # 1. Required top-level fields
    for field in CANONICAL_FIELDS + ["GSTIN", "line_items", "_diagnostics"]:
        if field not in result:
            errors.append(f"Missing top-level field: {field}")

    # 2. Date formats
    for field in ["INVOICE_DATE", "DUE_DATE"]:
        val = result.get(field)
        if val is not None and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(val)):
            errors.append(f"{field} is not formatted as YYYY-MM-DD: '{val}'")

    # 3. Monetary decimal formatting
    for field in MONEY_FIELDS:
        val = result.get(field)
        if val is not None and not re.fullmatch(r"-?\d+\.\d{2}", str(val)):
            errors.append(f"{field} is not standard decimal money: '{val}'")

    # 4. GSTIN checksum / pattern
    gstin = result.get("GSTIN")
    if gstin:
        normalized = re.sub(r"[^A-Za-z0-9]", "", str(gstin)).upper()
        if not (GSTIN_RE.fullmatch(normalized) and normalized[:2] in GST_STATE_CODES):
            errors.append(f"Invalid GSTIN structure: '{gstin}'")

    # 5. Line items integrity
    seen = set()
    for i, item in enumerate(result.get("line_items", []), start=1):
        for key in ["description", "quantity", "unit_price", "amount", "confidence"]:
            if key not in item:
                errors.append(f"Line item {i} missing key: {key}")

        for key in ["quantity", "unit_price", "amount"]:
            val = item.get(key)
            if val and looks_like_reference_number(val):
                errors.append(f"Reference number leaked into line-item {key}: '{val}'")

        sig = (
            item.get("description", "").upper().strip(),
            item.get("quantity", ""),
            item.get("unit_price", ""),
            item.get("amount", ""),
        )
        if sig in seen:
            errors.append(f"Duplicate line item {i}")
        seen.add(sig)

    # 6. Global line sum reconciliation
    diagnostics = result.get("_diagnostics", {})
    if diagnostics.get("global_line_sum_matches_subtotal") is False:
        errors.append("Global invoice line-item amounts do not reconcile with SUBTOTAL.")

    # 7. Date sequence
    inv = result.get("INVOICE_DATE")
    due = result.get("DUE_DATE")
    if inv and due:
        try:
            if dt.datetime.strptime(due, "%Y-%m-%d").date() < dt.datetime.strptime(inv, "%Y-%m-%d").date():
                errors.append("Due date cannot be earlier than invoice date.")
        except Exception:
            pass

    return len(errors) == 0, errors
