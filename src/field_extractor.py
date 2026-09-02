"""LayoutLMv3 token inference and entity span assembly."""

import re
import numpy as np
import torch

from .config import FIELD_ALIASES, DEVICE
from .preprocessing import normalize_box_1000, make_chunks
from .utils import clean_text


@torch.inference_mode()
def run_token_classifier(
    image,
    words: list[str],
    boxes: list[list[float]],
    model,
    processor,
    id2label: dict[int, str],
    device: str = DEVICE,
) -> list[dict]:
    """Execute token classification on an image and word boxes using LayoutLMv3."""
    width, height = image.size
    predictions = []

    for start, end in make_chunks(len(words)):
        chunk_words = words[start:end]
        chunk_boxes = boxes[start:end]

        normalized_boxes = [
            normalize_box_1000(b, width, height)
            for b in chunk_boxes
        ]

        encoded = processor(
            image,
            chunk_words,
            boxes=normalized_boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        encoded = {
            k: v.to(device)
            for k, v in encoded.items()
            if hasattr(v, "to")
        }

        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1)
        conf, pred_ids = probs.max(dim=-1)

        try:
            mapping = processor.tokenizer(
                chunk_words,
                boxes=normalized_boxes,
                truncation=True,
                padding="max_length",
                max_length=512,
            ).word_ids()
        except Exception:
            mapping = [None] * pred_ids.shape[1]

        seen_word = set()
        for token_pos, word_id in enumerate(mapping):
            if word_id is None or word_id in seen_word or word_id >= len(chunk_words):
                continue

            seen_word.add(word_id)
            global_idx = start + word_id
            label = id2label.get(int(pred_ids[0, token_pos].item()), "O")

            predictions.append({
                "word_index": global_idx,
                "word": words[global_idx],
                "box": boxes[global_idx],
                "label": label,
                "confidence": float(conf[0, token_pos].item()),
            })

    return predictions


def canonical_field(label: str) -> str | None:
    """Map raw predicted label to canonical field name."""
    if not label or label == "O":
        return None

    value = str(label).upper()
    value = re.sub(r"^[BIOES]-", "", value)
    value = value.replace(" ", "_").replace("-", "_")
    return FIELD_ALIASES.get(value)


def build_spans(predictions: list[dict]) -> list[dict]:
    """Group contiguous token predictions into labeled entity spans."""
    spans = []
    current = None

    for pred in predictions:
        label = canonical_field(pred["label"])
        if label is None:
            if current:
                spans.append(current)
                current = None
            continue

        if current is None or current["label"] != label:
            if current:
                spans.append(current)

            current = {
                "label": label,
                "word_indices": [pred["word_index"]],
                "words": [pred["word"]],
                "boxes": [pred["box"]],
                "confidences": [pred["confidence"]],
            }
        else:
            current["word_indices"].append(pred["word_index"])
            current["words"].append(pred["word"])
            current["boxes"].append(pred["box"])
            current["confidences"].append(pred["confidence"])

    if current:
        spans.append(current)

    for span in spans:
        span["value"] = clean_text(" ".join(span["words"]))
        span["confidence"] = float(np.mean(span["confidences"]))
        span["box"] = [
            min(b[0] for b in span["boxes"]),
            min(b[1] for b in span["boxes"]),
            max(b[2] for b in span["boxes"]),
            max(b[3] for b in span["boxes"]),
        ]

    return spans
