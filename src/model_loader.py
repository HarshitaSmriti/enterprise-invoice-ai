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


def _ensure_preprocessor_config(model_dir: Path):
    """Ensure preprocessor_config.json exists for LayoutLMv3Processor."""
    preproc_file = model_dir / "preprocessor_config.json"
    if not preproc_file.exists():
        import json
        proc_file = model_dir / "processor_config.json"
        img_proc_data = {
            "apply_ocr": False,
            "do_normalize": True,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_processor_type": "LayoutLMv3ImageProcessor",
            "image_std": [0.5, 0.5, 0.5],
            "resample": 2,
            "rescale_factor": 0.00392156862745098,
            "size": {"height": 224, "width": 224},
            "tesseract_config": "",
        }
        if proc_file.exists():
            try:
                with open(proc_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "image_processor" in data:
                    img_proc_data.update(data["image_processor"])
            except Exception:
                pass
        with open(preproc_file, "w", encoding="utf-8") as f:
            json.dump(img_proc_data, f, indent=2)


def load_layoutlm(model_dir: str | Path, model_name: str, device: str = DEVICE):
    """Load a LayoutLMv3 token classifier and processor from a local checkpoint."""
    model_dir = Path(model_dir).resolve()
    _ensure_preprocessor_config(model_dir)

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
