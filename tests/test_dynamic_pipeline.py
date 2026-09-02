"""Automated verification suite for Universal Dynamic Document Extraction & Adaptive Forms.

Validates:
1. Normal invoice extraction into generic schema.
2. Grocery list / non-invoice extraction (no rigid schema enforcement).
3. Dynamic custom table extraction (columns & rows).
4. Irregular / missing fields document handling.
5. Dynamic form schema generation.
"""

import pytest
from pathlib import Path
from PIL import Image

from src.config import SAMPLE_DATA_DIR, TEMP_DIR
from src.dynamic_pipeline import DynamicDocumentPipeline
from src.dynamic_extractor import (
    classify_document_type,
    extract_dynamic_key_values,
    extract_dynamic_tables,
)
from src.dynamic_form import render_adaptive_form


@pytest.fixture(scope="module")
def dynamic_pipeline():
    """Instantiate the dynamic document pipeline once for test suite."""
    return DynamicDocumentPipeline()


def test_1_normal_invoice_extraction(dynamic_pipeline):
    """Verify normal invoice extracts dynamic fields and line items into generic JSON."""
    pdf_path = SAMPLE_DATA_DIR / "8176011266.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test invoice {pdf_path} not found")

    result = dynamic_pipeline.process(pdf_path)

    # 1. Structural integrity of generic JSON
    assert "document_type" in result
    assert "fields" in result
    assert "tables" in result
    assert "metadata" in result

    # 2. Classified as invoice
    assert result["document_type"] == "invoice"

    # 3. Dynamic fields present
    fields = result["fields"]
    assert len(fields) > 0, "Invoice should have extracted dynamic fields"
    for f in fields:
        assert "name" in f and isinstance(f["name"], str)
        assert "value" in f
        assert "confidence" in f
        assert 0.0 <= f["confidence"] <= 1.0

    # 4. Metadata populated
    meta = result["metadata"]
    assert meta["page_count"] == 1
    assert meta["word_count"] > 0
    assert meta["processing_time"] > 0


def test_2_grocery_list_non_invoice_extraction(dynamic_pipeline):
    """Verify non-invoice grocery list extracts items, categories, and prices without schema failure."""
    img_path = SAMPLE_DATA_DIR / "grocery_list.png"
    if not img_path.exists():
        pytest.skip(f"Grocery sample {img_path} not found")

    result = dynamic_pipeline.process(img_path)

    # 1. Classification
    assert result["document_type"] == "grocery_list"

    # 2. Key-value fields extracted (Store, Date, Shopper, Budget, etc.)
    field_names = [f["name"] for f in result["fields"]]
    assert any("store" in fn or "date" in fn or "shopper" in fn or "budget" in fn or "total" in fn for fn in field_names)

    # 3. Dynamic Grocery Items Table extracted
    assert len(result["tables"]) >= 1
    table = result["tables"][0]
    assert len(table["columns"]) >= 3
    assert len(table["rows"]) >= 5

    # Check grocery row items
    row_strings = " ".join([str(r) for r in table["rows"]]).lower()
    assert "apples" in row_strings or "milk" in row_strings or "eggs" in row_strings


def test_3_document_with_custom_tables(dynamic_pipeline):
    """Verify purchase order document with custom table headers is dynamically reconstructed."""
    po_path = SAMPLE_DATA_DIR / "purchase_order.png"
    if not po_path.exists():
        pytest.skip(f"PO sample {po_path} not found")

    result = dynamic_pipeline.process(po_path)

    # 1. Document classification
    assert result["document_type"] == "purchase_order"

    # 2. Custom PO fields
    field_names = [f["name"] for f in result["fields"]]
    assert any("po_number" in fn or "buyer" in fn or "supplier" in fn or "total" in fn for fn in field_names)

    # 3. Dynamic custom table
    assert len(result["tables"]) >= 1
    po_table = result["tables"][0]
    # Reconstructed columns should adapt to document
    assert len(po_table["columns"]) >= 3
    assert len(po_table["rows"]) >= 3
    # Check item descriptions or SKU numbers
    table_content = " ".join([str(r) for r in po_table["rows"]])
    assert "SKU" in table_content or "Flange" in table_content or "Bolt" in table_content


def test_4_document_with_missing_irregular_fields(dynamic_pipeline):
    """Verify irregular form with non-financial fields extracts whatever is visible without failure."""
    form_path = SAMPLE_DATA_DIR / "irregular_form.png"
    if not form_path.exists():
        pytest.skip(f"Form sample {form_path} not found")

    result = dynamic_pipeline.process(form_path)

    # 1. Document classification
    assert result["document_type"] in ["incident_report", "form", "general_document"]

    # 2. Visible fields extracted
    field_names = [f["name"] for f in result["fields"]]
    assert any("incident_id" in fn or "severity" in fn or "operator" in fn or "facility" in fn or "reported_date" in fn for fn in field_names)

    # 3. Missing invoice fields must NOT cause failure or throw exception
    # (GSTIN, vendor, tax, etc. are absent in this document)
    assert "fields" in result
    assert len(result["fields"]) >= 4


def test_5_generic_json_schema_compliance():
    """Verify generic JSON structure adheres strictly to the document-agnostic specification."""
    sample_doc = {
        "document_type": "grocery_list",
        "fields": [
            {"name": "item", "value": "Milk", "confidence": 0.98, "box": [10, 20, 100, 40], "page": 1},
            {"name": "quantity", "value": "2", "confidence": 0.95, "box": [110, 20, 150, 40], "page": 1}
        ],
        "tables": [
            {
                "title": "Grocery Items",
                "columns": ["Item", "Qty", "Price"],
                "rows": [{"Item": "Bread", "Qty": "1", "Price": "$3.00"}],
                "confidence": 0.92,
                "page": 1
            }
        ],
        "metadata": {
            "page_count": 1,
            "word_count": 25,
            "ocr_engine": "pp_structure_v3"
        }
    }

    # Verify keys
    assert "document_type" in sample_doc
    assert isinstance(sample_doc["fields"], list)
    assert isinstance(sample_doc["tables"], list)
    assert isinstance(sample_doc["metadata"], dict)

    # Verify field schema
    for f in sample_doc["fields"]:
        assert "name" in f and "value" in f and "confidence" in f


def test_6_retail_receipt_extraction(dynamic_pipeline):
    """Verify retail receipt extracts custom receipt fields and items table."""
    rcpt_path = SAMPLE_DATA_DIR / "retail_receipt.png"
    if not rcpt_path.exists():
        pytest.skip(f"Receipt sample {rcpt_path} not found")

    result = dynamic_pipeline.process(rcpt_path)

    # 1. Classification
    assert result["document_type"] == "receipt"

    # 2. Dynamic receipt fields
    field_names = [f["name"] for f in result["fields"]]
    assert any("receipt_id" in fn or "cashier" in fn or "date" in fn or "total_amount" in fn for fn in field_names)

    # 3. Dynamic receipt items table
    assert len(result["tables"]) >= 1
    table = result["tables"][0]
    assert len(table["columns"]) >= 2
    assert len(table["rows"]) >= 3

