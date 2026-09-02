"""Unified Enterprise Invoice Processing Pipeline."""

import time
from pathlib import Path
import json

from .config import CANONICAL_FIELDS, TEMP_DIR, OUTPUTS_DIR, DEVICE, PADDLE_DEVICE
from .pdf_utils import load_pages, validate_file
from .ocr import ocr_image
from .model_loader import get_models
from .field_extractor import run_token_classifier, build_spans
from .line_items import reconstruct_line_items
from .reconciliation import merge_page_fields, merge_page_results
from .router import detect_document_type
from .validation import final_consistency_check


class InvoiceProcessingPipeline:
    """Production invoice extraction pipeline coordinator."""

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.models = get_models(device=device)

    def process(self, input_path: str | Path) -> dict:
        """Run complete extraction pipeline on uploaded document."""
        start_time = time.perf_counter()
        doc_path = validate_file(input_path)

        page_images = load_pages(doc_path)
        page_results = []

        all_words = []

        for page_no, page_image in enumerate(page_images, start=1):
            temp_page_file = TEMP_DIR / f"_temp_p_{page_no}_{int(time.time()*1000)}.png"

            try:
                page_image.save(temp_page_file, format="PNG")
                words, boxes, approx, ocr_engine = ocr_image(temp_page_file)
                all_words.extend(words)

                # 1. Model A inference
                preds_a = run_token_classifier(
                    page_image, words, boxes,
                    self.models["model_a"],
                    self.models["processor_a"],
                    self.models["id2label_a"],
                    device=self.device,
                )

                # 2. Model B inference
                preds_b = run_token_classifier(
                    page_image, words, boxes,
                    self.models["model_b"],
                    self.models["processor_b"],
                    self.models["id2label_b"],
                    device=self.device,
                )

                spans_a = build_spans(preds_a)
                spans_b = build_spans(preds_b)

                # 3. Merge fields & OCR structural evidence
                fields, metadata, money_diag, gstins = merge_page_fields(
                    spans_a + spans_b, words, boxes
                )

                # 4. Table & line-item reconstruction
                line_items = reconstruct_line_items(words, boxes)
                for item in line_items:
                    item["page"] = page_no

                page_results.append({
                    "fields": fields,
                    "metadata": metadata,
                    "line_items": line_items,
                    "gstins": gstins,
                    "diagnostics": {
                        "page": page_no,
                        "ocr_word_count": len(words),
                        "ocr_engine": ocr_engine,
                        "model_a_spans": len(spans_a),
                        "model_b_spans": len(spans_b),
                        "line_items_count": len(line_items),
                    },
                })

            finally:
                if temp_page_file.exists():
                    try:
                        temp_page_file.unlink()
                    except Exception:
                        pass
                try:
                    page_image.close()
                except Exception:
                    pass

        # 5. Merge multi-page results
        merged = merge_page_results(page_results)

        # 6. Canonical JSON assembly
        fields = merged.get("fields", {})
        canonical = {field: fields.get(field) for field in CANONICAL_FIELDS}
        canonical["GSTIN"] = fields.get("GSTIN")
        canonical["line_items"] = merged.get("line_items", [])
        canonical["_field_metadata"] = merged.get("field_metadata", {})
        canonical["_diagnostics"] = merged.get("diagnostics", {})

        # 7. Document type detection
        doc_type = detect_document_type(all_words)
        canonical["_diagnostics"]["document_type"] = doc_type

        # 8. Consistency validation
        passed, errors = final_consistency_check(canonical)
        canonical["_diagnostics"]["consistency_check"] = {
            "status": "PASS" if passed else ("WARNINGS" if canonical["line_items"] else "FAIL"),
            "errors": errors,
        }

        canonical["_diagnostics"]["source_file"] = Path(input_path).name
        canonical["_diagnostics"]["device"] = self.device
        canonical["_diagnostics"]["paddle_device"] = PADDLE_DEVICE
        canonical["_diagnostics"]["total_seconds"] = round(time.perf_counter() - start_time, 3)

        return canonical
