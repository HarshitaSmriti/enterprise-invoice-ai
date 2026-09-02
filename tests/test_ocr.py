"""Unit tests for OCR bounding box handling and engine adapters."""

from pathlib import Path
from src.ocr import bbox4, dedupe_ocr_records, get_pp_structure, get_direct_ocr, ocr_image
from src.config import SAMPLE_DATA_DIR


def test_bbox4_conversions():
    """Test standard 4-point polygon to [x1, y1, x2, y2]."""
    poly = [[10, 20], [100, 20], [100, 50], [10, 50]]
    box = bbox4(poly)
    assert box == [10.0, 20.0, 100.0, 50.0]

    flat = [5, 15, 80, 60]
    assert bbox4(flat) == [5.0, 15.0, 80.0, 60.0]


def test_dedupe_ocr_records():
    """Test deduplication and coordinate sorting of OCR records."""
    records = [
        {"word": "TOTAL", "box": [50.0, 100.0, 100.0, 120.0], "confidence": 0.99},
        {"word": "TOTAL", "box": [50.0, 100.0, 100.0, 120.0], "confidence": 0.99},
        {"word": "INVOICE", "box": [50.0, 20.0, 150.0, 40.0], "confidence": 0.95},
    ]
    deduped = dedupe_ocr_records(records)
    assert len(deduped) == 2
    # Check top-to-bottom sorting: INVOICE at y=20-40 comes before TOTAL at y=100-120
    assert deduped[0]["word"] == "INVOICE"
    assert deduped[1]["word"] == "TOTAL"


def test_ocr_engine_execution():
    """Verify OCR runs on a real image without failure."""
    test_img = SAMPLE_DATA_DIR / "test_document.png"
    if test_img.exists():
        words, boxes, approx, engine = ocr_image(test_img)
        assert len(words) > 0
        assert len(words) == len(boxes)
        assert len(approx) == len(words)
        assert engine in ["pp_structure_v3", "paddleocr_direct"]
