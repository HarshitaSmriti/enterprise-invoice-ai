"""OCR adapter module integrating PP-StructureV3 with PaddleOCR direct fallback.

Handles OCR record extraction from native PaddleOCR 3.x parallel arrays,
materialized JSON, and legacy walkers, returning normalized bounding boxes and words.
"""

import os
# Set PaddleX flags before importing paddleocr to avoid network connectivity checks and timeouts
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_PDX_EAGER_INIT"] = "False"
os.environ["PADDLE_PDX_MODEL_SOURCE"] = "huggingface"

import json
from pathlib import Path
import numpy as np

# Ensure torch is loaded before paddle to avoid Windows DLL issues
import torch  # noqa: F401
import paddle
import paddleocr
from paddleocr import PPStructureV3, PaddleOCR

from .config import PADDLE_DEVICE
from .utils import y_center

# Global singleton engine instances
_PP_STRUCTURE_ENGINE = None
_DIRECT_OCR_ENGINE = None


def bbox4(box) -> list[float]:
    """Convert arbitrary polygon/box-like array to [x1, y1, x2, y2]."""
    if box is None:
        raise ValueError("Empty OCR box")

    arr = np.asarray(box, dtype=float)

    if arr.ndim == 1 and arr.size >= 4:
        return [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])]

    if arr.ndim == 2 and arr.shape[1] >= 2:
        xs = arr[:, 0]
        ys = arr[:, 1]
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

    flat = arr.reshape(-1)
    if flat.size >= 4:
        return [float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])]

    raise ValueError(f"Unsupported OCR box format: {box}")


def _materialize_json(obj):
    """Retrieve JSON payload from OCR result if available."""
    if obj is None:
        return None

    try:
        val = getattr(obj, "json")
    except Exception:
        return obj

    try:
        val = val() if callable(val) else val
    except Exception:
        return obj

    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return obj

    return val


def _append_parallel_ocr_arrays(payload, records: list) -> int:
    """Extract native PaddleOCR 3.x parallel arrays (texts, scores, boxes)."""
    if not isinstance(payload, dict):
        return 0

    added = 0
    candidates = [payload]
    nested = payload.get("res")
    if isinstance(nested, dict):
        candidates.insert(0, nested)

    for obj in candidates:
        texts = obj.get("rec_texts") or obj.get("texts")
        scores = obj.get("rec_scores") or obj.get("scores")
        boxes = (
            obj.get("rec_boxes")
            or obj.get("rec_polys")
            or obj.get("dt_polys")
            or obj.get("boxes")
            or obj.get("polys")
        )

        if texts is None or boxes is None:
            continue

        texts = list(texts)
        boxes = list(boxes)
        scores = list(scores) if scores is not None else [1.0] * len(texts)

        n = min(len(texts), len(boxes))
        for i in range(n):
            text = str(texts[i]).strip()
            if not text:
                continue

            try:
                box = bbox4(boxes[i])
            except Exception:
                continue

            try:
                score = float(scores[i]) if i < len(scores) else 1.0
            except Exception:
                score = 1.0

            records.append({
                "word": text,
                "box": box,
                "confidence": score,
            })
            added += 1

        if added:
            break

    return added


def _walk_paddle(obj, records: list, depth: int = 0):
    """Recursive walker for older or nested PaddleOCR response dicts."""
    if obj is None or depth > 20:
        return

    materialized = _materialize_json(obj)
    if materialized is not obj:
        _walk_paddle(materialized, records, depth + 1)
        return

    if isinstance(obj, dict):
        _append_parallel_ocr_arrays(obj, records)

        text = obj.get("text") or obj.get("transcription")
        box = obj.get("bbox") or obj.get("box") or obj.get("points") or obj.get("coordinate")
        score = obj.get("confidence") or obj.get("score") or obj.get("rec_score")

        if text is not None and box is not None:
            try:
                records.append({
                    "word": str(text).strip(),
                    "box": bbox4(box),
                    "confidence": float(score) if score is not None else 1.0,
                })
            except Exception:
                pass

        for val in obj.values():
            _walk_paddle(val, records, depth + 1)

    elif isinstance(obj, (list, tuple)):
        # Standard tuple: [polygon, (text, score)]
        if (
            len(obj) == 2
            and isinstance(obj[1], (list, tuple))
            and len(obj[1]) >= 1
            and isinstance(obj[1][0], str)
        ):
            try:
                records.append({
                    "word": str(obj[1][0]).strip(),
                    "box": bbox4(obj[0]),
                    "confidence": float(obj[1][1]) if len(obj[1]) > 1 else 1.0,
                })
                return
            except Exception:
                pass

        for val in obj:
            _walk_paddle(val, records, depth + 1)


def dedupe_ocr_records(records: list[dict]) -> list[dict]:
    """Deduplicate records with identical words and box positions, sort reading order."""
    output = []
    seen = set()

    for record in records:
        word = str(record["word"]).strip()
        if not word:
            continue

        try:
            box = bbox4(record["box"])
        except Exception:
            continue

        key = (word, tuple(round(float(v), 1) for v in box))
        if key in seen:
            continue

        seen.add(key)
        output.append({
            "word": word,
            "box": box,
            "confidence": float(record.get("confidence", 1.0)),
        })

    # Sort primarily top-to-bottom, secondarily left-to-right
    output.sort(key=lambda r: ((r["box"][1] + r["box"][3]) / 2.0, r["box"][0]))
    return output


def get_pp_structure():
    """Lazy initialize and return singleton PPStructureV3 engine."""
    global _PP_STRUCTURE_ENGINE
    if _PP_STRUCTURE_ENGINE is not None:
        return _PP_STRUCTURE_ENGINE

    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    kwargs = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "use_seal_recognition": False,
        "use_table_recognition": False,
        "use_formula_recognition": False,
        "use_chart_recognition": False,
        "use_region_detection": False,
        "text_recognition_batch_size": 16,
        "device": PADDLE_DEVICE,
    }

    try:
        _PP_STRUCTURE_ENGINE = PPStructureV3(**kwargs)
    except TypeError:
        minimal = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "use_table_recognition": False,
            "use_formula_recognition": False,
            "device": PADDLE_DEVICE,
        }
        _PP_STRUCTURE_ENGINE = PPStructureV3(**minimal)

    return _PP_STRUCTURE_ENGINE


def get_direct_ocr():
    """Lazy initialize and return singleton PaddleOCR direct engine (lightweight mobile model)."""
    global _DIRECT_OCR_ENGINE
    if _DIRECT_OCR_ENGINE is not None:
        return _DIRECT_OCR_ENGINE

    kwargs = {
        "lang": "en",
        "ocr_version": "PP-OCRv4",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "device": PADDLE_DEVICE,
    }

    try:
        _DIRECT_OCR_ENGINE = PaddleOCR(**kwargs)
    except TypeError:
        minimal = {
            "lang": "en",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "device": PADDLE_DEVICE,
        }
        _DIRECT_OCR_ENGINE = PaddleOCR(**minimal)

    return _DIRECT_OCR_ENGINE


def _run_engine(engine, image_path: str | Path) -> list[dict]:
    """Execute prediction on image and parse records."""
    results = engine.predict(str(image_path))
    records = []

    for result in results:
        payload = _materialize_json(result)
        if isinstance(payload, dict):
            _append_parallel_ocr_arrays(payload, records)
        _walk_paddle(payload, records)

        if not records:
            try:
                texts = getattr(result, "rec_texts", None)
                boxes = getattr(result, "rec_boxes", None) or getattr(result, "dt_polys", None)
                scores = getattr(result, "rec_scores", None)

                if texts is not None and boxes is not None:
                    texts = list(texts)
                    boxes = list(boxes)
                    scores = list(scores) if scores is not None else [1.0] * len(texts)

                    for i in range(min(len(texts), len(boxes))):
                        text = str(texts[i]).strip()
                        if not text:
                            continue
                        records.append({
                            "word": text,
                            "box": bbox4(boxes[i]),
                            "confidence": float(scores[i]) if i < len(scores) else 1.0,
                        })
            except Exception:
                pass

    return dedupe_ocr_records(records)


def ocr_image(image_path: str | Path) -> tuple[list[str], list[list[float]], list[bool], str]:
    """Run OCR on image, selecting lightweight high-speed engine by default on CPU.
    
    Returns:
        (words, boxes, approx, engine_name)
    """
    mode = os.getenv("OCR_ENGINE_MODE", "auto").lower()

    if mode == "server":
        primary_fn = get_pp_structure
        primary_name = "pp_structure_v3"
        fallback_fn = get_direct_ocr
        fallback_name = "paddleocr_mobile"
    else:
        primary_fn = get_direct_ocr
        primary_name = "paddleocr_mobile"
        fallback_fn = get_pp_structure
        fallback_name = "pp_structure_v3"

    try:
        engine = primary_fn()
        records = _run_engine(engine, image_path)
        if records:
            words = [r["word"] for r in records]
            boxes = [r["box"] for r in records]
            approx = [r["confidence"] < 0.5 for r in records]
            return words, boxes, approx, primary_name
    except Exception:
        pass

    # Fallback
    engine = fallback_fn()
    records = _run_engine(engine, image_path)
    if not records:
        raise RuntimeError(f"OCR returned zero records for {image_path}")

    words = [r["word"] for r in records]
    boxes = [r["box"] for r in records]
    approx = [r["confidence"] < 0.5 for r in records]
    return words, boxes, approx, fallback_name
