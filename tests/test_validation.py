"""Unit tests for the validation engine and business rule consistency checks."""

from src.validation import final_consistency_check
from src.config import CANONICAL_FIELDS


def test_validation_pass_scenario():
    """Verify valid canonical document produces PASS."""
    canonical = {field: None for field in CANONICAL_FIELDS}
    canonical.update({
        "VENDOR_NAME": "ACME TEXTILES LTD",
        "CUSTOMER_NAME": "GLOBAL RETAIL INC",
        "INVOICE_NUMBER": "INV-2026-001",
        "INVOICE_DATE": "2026-08-15",
        "DUE_DATE": "2026-09-15",
        "GSTIN": "33ABCDE1234F1Z5",
        "SUBTOTAL": "1000.00",
        "CENTRAL_GST": "90.00",
        "STATE_GST": "90.00",
        "TAX": "180.00",
        "TOTAL_AMOUNT": "1180.00",
        "line_items": [
            {
                "description": "Cotton Yarn 40s",
                "quantity": "10.00",
                "unit_price": "100.00",
                "amount": "1000.00",
                "confidence": 0.95,
            }
        ],
        "_diagnostics": {
            "global_line_sum_matches_subtotal": True
        }
    })

    passed, errors = final_consistency_check(canonical)
    assert passed is True
    assert len(errors) == 0


def test_validation_due_date_earlier_than_invoice():
    """Verify validation flags due date before invoice date."""
    canonical = {field: None for field in CANONICAL_FIELDS}
    canonical.update({
        "INVOICE_DATE": "2026-08-15",
        "DUE_DATE": "2026-08-10",  # Earlier!
        "GSTIN": None,
        "line_items": [],
        "_diagnostics": {}
    })

    passed, errors = final_consistency_check(canonical)
    assert passed is False
    assert any("Due date cannot be earlier" in e for e in errors)


def test_validation_bad_gstin():
    """Verify validation flags malformed GSTIN."""
    canonical = {field: None for field in CANONICAL_FIELDS}
    canonical.update({
        "GSTIN": "INVALID999GSTIN",
        "line_items": [],
        "_diagnostics": {}
    })

    passed, errors = final_consistency_check(canonical)
    assert passed is False
    assert any("Invalid GSTIN" in e for e in errors)
