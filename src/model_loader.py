"""Model loading and caching module for LayoutLMv3 models."""

from pathlib import Path
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

from .config import FIELD_MODEL_DIR, GST_MODEL_DIR, DEVICE

_LOADED_MODELS = {}


REQUIRED_MODEL_FILES = [
    "config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def load_layoutlm(model_dir: str | Path, model_name: str, device: str = DEVICE):
    """Load a LayoutLMv3 token classifier and processor from a local checkpoint."""
    model_dir = Path(model_dir).resolve()
    missing_files = [f for f in REQUIRED_MODEL_FILES if not (model_dir / f).exists()]

    if missing_files:
        raise FileNotFoundError(
            f"Required files for {model_name} missing from '{model_dir}': {missing_files}. "
            "Ensure the trained weights are mounted or set FIELD_MODEL_DIR / GST_MODEL_DIR."
        )

    processor = LayoutLMv3Processor.from_pretrained(str(model_dir), apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    return model, processor, id2label


def get_models(device: str = DEVICE, field_dir: str | Path = FIELD_MODEL_DIR, gst_dir: str | Path = GST_MODEL_DIR):
    """Load and cache both Model A (universal) and Model B (GST) in memory."""
    global _LOADED_MODELS
    cache_key = f"{device}_{field_dir}_{gst_dir}"
    if cache_key in _LOADED_MODELS:
        return _LOADED_MODELS[cache_key]

    model_a, proc_a, id2label_a = load_layoutlm(
        field_dir, "Model A (Universal)", device=device
    )
    model_b, proc_b, id2label_b = load_layoutlm(
        gst_dir, "Model B (GST-specific)", device=device
    )

    _LOADED_MODELS[cache_key] = {
        "model_a": model_a,
        "processor_a": proc_a,
        "id2label_a": id2label_a,
        "model_b": model_b,
        "processor_b": proc_b,
        "id2label_b": id2label_b,
    }
    return _LOADED_MODELS[cache_key]
