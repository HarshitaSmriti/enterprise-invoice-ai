"""Integration tests for the complete InvoiceProcessingPipeline."""

from pathlib import Path
from src.pipeline import InvoiceProcessingPipeline
from src.pdf_utils import load_pages
from src.config import SAMPLE_DATA_DIR, CANONICAL_FIELDS


def test_multipage_pdf_loading():
    """Verify multi-page PDF renders all pages into images."""
    multipage_pdf = SAMPLE_DATA_DIR / "8176000040.pdf"
    if multipage_pdf.exists():
        pages = load_pages(multipage_pdf)
        assert len(pages) == 5
        assert all(p.size[0] > 0 and p.size[1] > 0 for p in pages)


def test_end_to_end_single_page_pdf_pipeline():
    """Test full extraction pipeline on real single-page invoice 8176011266.pdf."""
    single_pdf = SAMPLE_DATA_DIR / "8176011266.pdf"
    if not single_pdf.exists():
        return

    pipeline = InvoiceProcessingPipeline()
    result = pipeline.process(single_pdf)

    # 1. Structure check
    for field in CANONICAL_FIELDS + ["GSTIN", "line_items", "_diagnostics"]:
        assert field in result

    # 2. Key fields extracted correctly from the invoice
    assert result["INVOICE_NUMBER"] == "8176011266"
    assert result["INVOICE_DATE"] == "2026-07-23"
    assert "SRI VENKATESWARA" in (result["CUSTOMER_NAME"] or "")

    # 3. Line items
    assert len(result["line_items"]) >= 1
    assert "Bermuda" in result["line_items"][0]["description"] or len(result["line_items"][0]["description"]) > 5

    # 4. Diagnostics
    assert result["_diagnostics"]["document_type"] == "INDIAN_GST"
    assert result["_diagnostics"]["consistency_check"]["status"] in ["PASS", "WARNINGS"]
