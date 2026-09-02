"""Unit tests for LayoutLMv3 model loading and token inference."""

import torch
from src.config import FIELD_MODEL_DIR, GST_MODEL_DIR, DEVICE
from src.model_loader import load_layoutlm, get_models


def test_field_level_model_loading():
    """Verify Model A (Universal) loads from local safetensors with correct config."""
    model, processor, id2label = load_layoutlm(FIELD_MODEL_DIR, "Model A", device=DEVICE)
    assert model is not None
    assert processor is not None
    assert len(id2label) == 26
    assert "B-VENDOR_NAME" in id2label.values()
    assert "B-TOTAL_AMOUNT" in id2label.values()


def test_gst_level_model_loading():
    """Verify Model B (GST) loads from local safetensors with correct config."""
    model, processor, id2label = load_layoutlm(GST_MODEL_DIR, "Model B", device=DEVICE)
    assert model is not None
    assert processor is not None
    assert len(id2label) == 26
    assert "B-CENTRAL_GST" in id2label.values()
    assert "B-STATE_GST" in id2label.values()


def test_get_models_cached_singleton():
    """Verify get_models caches instances in memory."""
    models1 = get_models()
    models2 = get_models()
    assert models1 is models2
    assert "model_a" in models1
    assert "model_b" in models1
