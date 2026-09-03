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

        from concurrent.futures import ThreadPoolExecutor
        page_images = load_pages(doc_path)
        page_results = []

        all_words = []

        for page_no, page_image in enumerate(page_images, start=1):
            temp_page_file = TEMP_DIR / f"_temp_p_{page_no}_{int(time.time()*1000)}.png"

            try:
                words, boxes = [], []
                ocr_engine = "digital_pdf_fast"

                # Fast-path for digital PDFs: extract text & exact 1000-scaled bounding boxes in ~15ms
                if doc_path.suffix.lower() == ".pdf":
                    try:
                        import warnings
                        warnings.filterwarnings("ignore", category=DeprecationWarning)
                        warnings.filterwarnings("ignore", message=".*fitz.*")
                        import pymupdf
                        try:
                            pymupdf.TOOLS.mupdf_display_errors(False)
                        except Exception:
                            pass
                        with pymupdf.open(str(doc_path)) as pdf_doc:
                            if page_no - 1 < len(pdf_doc):
                                pdf_page = pdf_doc[page_no - 1]
                                pw, ph = pdf_page.rect.width, pdf_page.rect.height
                                raw_words = pdf_page.get_text("words")
                                if len(raw_words) >= 15:
                                    for x0, y0, x1, y1, text, b, l, widx in raw_words:
                                        t = text.strip()
                                        if t:
                                            words.append(t)
                                            boxes.append([
                                                max(0, min(1000, int(round(x0 / pw * 1000)))),
                                                max(0, min(1000, int(round(y0 / ph * 1000)))),
                                                max(0, min(1000, int(round(x1 / pw * 1000)))),
                                                max(0, min(1000, int(round(y1 / ph * 1000)))),
                                            ])
                    except Exception:
                        words, boxes = [], []

                # Fallback to OCR for scanned PDFs or raster image uploads
                if not words:
                    page_image.save(temp_page_file, format="PNG")
                    words, boxes, approx, ocr_engine = ocr_image(temp_page_file)

                all_words.extend(words)

                # Concurrent forward passes for Model A (universal) and Model B (GST)
                def _run_a():
                    return run_token_classifier(
                        page_image, words, boxes,
                        self.models["model_a"],
                        self.models["processor_a"],
                        self.models["id2label_a"],
                        device=self.device,
                    )

                def _run_b():
                    return run_token_classifier(
                        page_image, words, boxes,
                        self.models["model_b"],
                        self.models["processor_b"],
                        self.models["id2label_b"],
                        device=self.device,
                    )

                with ThreadPoolExecutor(max_workers=2) as executor:
                    fa = executor.submit(_run_a)
                    fb = executor.submit(_run_b)
                    preds_a = fa.result()
                    preds_b = fb.result()

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
