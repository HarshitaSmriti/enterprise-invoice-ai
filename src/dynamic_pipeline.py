"""Universal, document-agnostic document extraction pipeline.

Integrates PP-StructureV3/PaddleOCR, dynamic table detection, key-value geometry,
and LayoutLMv3 models (Model A & Model B) into a unified generic JSON schema.
"""

from __future__ import annotations

import time
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image

# DLL guard
import torch

from src.config import DEVICE, PADDLE_DEVICE, TEMP_DIR
from src.pdf_utils import load_pages, validate_file
from src.ocr import ocr_image
from src.model_loader import get_models
from src.field_extractor import run_token_classifier, build_spans
from src.dynamic_extractor import (
    classify_document_type,
    extract_dynamic_key_values,
    extract_dynamic_tables,
)
from src.line_items import reconstruct_line_items
from src.reconciliation import merge_page_fields


class DynamicDocumentPipeline:
    """End-to-end dynamic document extraction engine."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or DEVICE
        self._models = None

    def _ensure_models(self):
        if self._models is None:
            self._models = get_models(device=self.device)
        return self._models

    def process(self, input_path: Union[str, Path]) -> Dict[str, Any]:
        """Execute full dynamic extraction on arbitrary document."""
        start_time = time.perf_counter()
        valid_path = validate_file(input_path)

        # 1. Load document pages
        pages = load_pages(valid_path)
        if not pages:
            raise ValueError(f"No viewable pages rendered for {valid_path}")

        all_page_fields = []
        all_tables = []
        total_words = 0
        engines_used = set()
        document_text_corpus = []

        # 2. Process each page
        for page_idx, page_img in enumerate(pages, start=1):
            temp_page_file = TEMP_DIR / f"_dyn_p{page_idx}_{Path(input_path).stem}.png"
            page_img.save(temp_page_file, format="PNG")

            try:
                # OCR & Layout
                words, boxes, approx, ocr_engine = ocr_image(temp_page_file)
                engines_used.add(ocr_engine)
                total_words += len(words)
                document_text_corpus.extend(words)

                # A. Extract dynamic key-value pairs
                kv_fields = extract_dynamic_key_values(words, boxes, page=page_idx)

                # B. Extract dynamic tables
                tables = extract_dynamic_tables(words, boxes, page=page_idx)

                # C. Check if specialized LayoutLMv3 invoice models add value
                # Used only when document domain is applicable (Invoice / GST)
                doc_type_hint = classify_document_type(words)
                ml_fields = []

                if doc_type_hint == "invoice":
                    try:
                        models = self._ensure_models()
                        preds_a = run_token_classifier(
                            page_img, words, boxes,
                            models["model_a"], models["processor_a"], models["id2label_a"],
                            device=self.device
                        )
                        preds_b = run_token_classifier(
                            page_img, words, boxes,
                            models["model_b"], models["processor_b"], models["id2label_b"],
                            device=self.device
                        )
                        spans = build_spans(preds_a + preds_b)
                        # Add confident spans as dynamic fields
                        for s in spans:
                            label = s["label"].replace("B-", "").replace("I-", "").lower()
                            if s.get("confidence", 0) >= 0.70 and len(s.get("text", "").strip()) > 1:
                                ml_fields.append({
                                    "name": label,
                                    "label": label.replace("_", " ").title(),
                                    "value": s["text"].strip(),
                                    "confidence": round(float(s["confidence"]), 3),
                                    "box": s.get("box"),
                                    "page": page_idx,
                                    "source": "layoutlmv3",
                                })

                        # If generic table didn't catch line items but billing table exists
                        if not tables and doc_type_hint == "invoice":
                            inv_items = reconstruct_line_items(words, boxes)
                            if inv_items:
                                tables.append({
                                    "title": "Line Items",
                                    "columns": ["description", "quantity", "unit_price", "amount"],
                                    "rows": inv_items,
                                    "confidence": 0.90,
                                    "page": page_idx,
                                })
                    except Exception:
                        pass

                # Merge and deduplicate fields for this page
                merged_fields = _merge_dynamic_fields(kv_fields, ml_fields)
                all_page_fields.extend(merged_fields)
                all_tables.extend(tables)

            finally:
                if temp_page_file.exists():
                    temp_page_file.unlink()

        # 3. Detect overall document type
        overall_doc_type = classify_document_type(document_text_corpus)

        # 4. Assemble canonical generic JSON
        elapsed = round(time.perf_counter() - start_time, 2)
        generic_output = {
            "document_type": overall_doc_type,
            "fields": all_page_fields,
            "tables": all_tables,
            "metadata": {
                "source_file": Path(input_path).name,
                "page_count": len(pages),
                "word_count": total_words,
                "ocr_engine": ", ".join(sorted(engines_used)),
                "device": self.device,
                "paddle_device": PADDLE_DEVICE,
                "processing_time": elapsed,
                "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            }
        }

        return generic_output


def _merge_dynamic_fields(geo_fields: List[Dict[str, Any]], ml_fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge geometric key-value fields and ML token spans with deduplication."""
    field_map = {}

    # Geometric fields provide reliable label-value boundaries
    for f in geo_fields:
        field_map[f["name"]] = f

    # ML fields provide named entity recognition
    for f in ml_fields:
        k = f["name"]
        if k not in field_map:
            field_map[k] = f
        else:
            # If ML field has higher confidence, update value
            if f.get("confidence", 0) > field_map[k].get("confidence", 0):
                field_map[k] = f

    # Sort fields by page, then box y-coordinate
    result = list(field_map.values())
    result.sort(key=lambda item: (item.get("page", 1), item.get("box", [0, 0, 0, 0])[1]))
    return result
