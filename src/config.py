"""Central configuration module for Enterprise Document AI.

Defines model directories, canonical schemas, regexes, and device parameters.
"""

import os
# Set PaddleX flags to avoid foreign server network checks and timeouts in cloud environments
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_PDX_EAGER_INIT"] = "False"
import os
import re
import tempfile
from pathlib import Path
import torch

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Configurable model weight directories (supports environment variable overrides for deployment)
MODEL_BASE_DIR = Path(os.getenv("MODEL_DIR", str(PROJECT_ROOT)))
FIELD_MODEL_DIR = Path(os.getenv("FIELD_MODEL_DIR", str(MODEL_BASE_DIR / "field_level")))
GST_MODEL_DIR = Path(os.getenv("GST_MODEL_DIR", str(MODEL_BASE_DIR / "gst_level")))

# Writable temporary directories in /tmp
_tmp_base = Path(tempfile.gettempdir())
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", str(_tmp_base / "invoice_ai_outputs")))
TEMP_DIR = Path(os.getenv("TEMP_DIR", str(OUTPUTS_DIR / "_temp")))
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"

try:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# File formats supported
SUPPORTED_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"
}

# Runtime execution devices (auto-detects CUDA/GPU with clean CPU fallback and env override)
DEVICE = os.getenv("TORCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
PADDLE_DEVICE = os.getenv("PADDLE_DEVICE", "gpu:0" if torch.cuda.is_available() else "cpu")

# Environmental flags for Paddle / oneDNN
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_DISABLE_DNNL"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Canonical target field list
CANONICAL_FIELDS = [
    "VENDOR_NAME",
    "CUSTOMER_NAME",
    "ADDRESS",
    "INVOICE_NUMBER",
    "INVOICE_DATE",
    "DUE_DATE",
    "REF_NO",
    "CENTRAL_GST",
    "STATE_GST",
    "IGST",
    "SUBTOTAL",
    "TAX",
    "TOTAL_AMOUNT",
]

MONEY_FIELDS = {
    "CENTRAL_GST",
    "STATE_GST",
    "IGST",
    "SUBTOTAL",
    "TAX",
    "TOTAL_AMOUNT",
}

# Indian GST State Codes (01 to 37)
GST_STATE_CODES = {f"{i:02d}" for i in range(1, 38)}

# Regular expressions
GSTIN_RE = re.compile(
    r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b",
    re.I,
)

DATE_RE = re.compile(
    r"\b(?:"
    r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
    r"|"
    r"\d{4}[./-]\d{1,2}[./-]\d{1,2}"
    r")\b"
)

MONEY_TOKEN_RE = re.compile(r"^[\u20b9Rs\.\s]*[\d,]+\.?\d*$")

# Keyword dictionaries for financial rows
MONEY_ROW_RULES = {
    "CENTRAL_GST": [
        ("CENTRAL GST",), ("CGST",),
    ],
    "STATE_GST": [
        ("STATE GST",), ("SGST",),
    ],
    "IGST": [
        ("INTEGRATED GST",), ("IGST",),
    ],
    "SUBTOTAL": [
        ("BASIC TOTAL",), ("SUB TOTAL",), ("SUBTOTAL",),
        ("TAXABLE VALUE",), ("TAXABLE AMOUNT",),
    ],
    "TOTAL_AMOUNT": [
        ("BILL AMOUNT",), ("GRAND TOTAL",),
        ("INVOICE TOTAL",), ("TOTAL AMOUNT",),
        ("TOTAL VALUE",), ("NET PAYABLE",), ("NET AMOUNT",),
    ],
}

# Table boundary detection markers (v10/v11 structural table model)
BILL_HEADER_MARKERS = (
    "PO NO",
    "MATERIAL DESCRIPTION",
    "GR QTY",
    "BILL QTY",
    "RATE",
    "AMT INR",
)

CONSUMPTION_MARKERS = (
    "CONSUMPTION DETAILS",
    "MATERIAL CODE",
    "MATERIAL DESC",
    "DO NO",
    "DO DATE",
    "ISSUE QTY",
    "RECEIVE QTY",
    "DEBIT/CREDIT",
)

SUMMARY_WORDS = {
    "SUBTOTAL", "BASIC TOTAL", "TAXABLE VALUE", "CGST", "CENTRAL GST",
    "SGST", "STATE GST", "IGST", "TAX", "GRAND TOTAL", "BILL AMOUNT",
    "TOTAL AMOUNT", "TOTAL VALUE", "NET PAYABLE", "ROUND OFF",
    "AMOUNT IN WORDS", "TERMS", "CONDITION", "REMARKS", "NOTE",
}

HEADER_WORDS = {
    "DESCRIPTION", "ITEM", "PRODUCT", "QTY", "QUANTITY", "RATE",
    "UNIT PRICE", "PRICE", "AMOUNT", "TOTAL", "HSN", "SAC",
}

# Canonical field normalization aliases
FIELD_ALIASES = {
    "VENDOR": "VENDOR_NAME",
    "VENDOR_NAME": "VENDOR_NAME",
    "SUPPLIER": "VENDOR_NAME",
    "SUPPLIER_NAME": "VENDOR_NAME",

    "CUSTOMER": "CUSTOMER_NAME",
    "CUSTOMER_NAME": "CUSTOMER_NAME",
    "BUYER": "CUSTOMER_NAME",

    "ADDRESS": "ADDRESS",
    "BILLING_ADDRESS": "ADDRESS",

    "INVOICE_NO": "INVOICE_NUMBER",
    "INVOICE_NUMBER": "INVOICE_NUMBER",
    "BILL_NO": "INVOICE_NUMBER",

    "INVOICE_DATE": "INVOICE_DATE",
    "DATE": "INVOICE_DATE",
    "DUE_DATE": "DUE_DATE",

    "REF_NO": "REF_NO",
    "REFERENCE_NO": "REF_NO",

    "CGST": "CENTRAL_GST",
    "CENTRAL_GST": "CENTRAL_GST",
    "SGST": "STATE_GST",
    "STATE_GST": "STATE_GST",
    "IGST": "IGST",
    "INTEGRATED_GST": "IGST",

    "SUBTOTAL": "SUBTOTAL",
    "BASIC_TOTAL": "SUBTOTAL",
    "TAXABLE_VALUE": "SUBTOTAL",
    "TAX": "TAX",
    "TOTAL": "TOTAL_AMOUNT",
    "TOTAL_AMOUNT": "TOTAL_AMOUNT",
    "GRAND_TOTAL": "TOTAL_AMOUNT",
}
