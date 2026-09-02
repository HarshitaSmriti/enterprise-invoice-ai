---
title: Enterprise Invoice AI
emoji: 🧾
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.42.0
app_file: app.py
pinned: false
---

# Enterprise Document AI — Intelligent Document Extraction System

A production-ready Enterprise Document AI platform featuring dual LayoutLMv3 models (Universal Invoice & Indian GST), PP-StructureV3 OCR, geometric line-item table reconstruction, and schema-agnostic dynamic document understanding.

---

## 1. Project Overview

This platform provides end-to-end automated processing for enterprise invoices and complex business documents. It solves key real-world challenges in document parsing:
- **Dual-Model Inference**: Combines a Universal Invoice model (Model A) and an Indian GST-specific model (Model B) with intelligent confidence routing and reconciliation.
- **Structural Table Extraction**: Distinguishes invoice billing tables from irrelevant consumption tables or running totals, extracting numeric columns deterministically (GR QTY, BILL QTY, RATE, AMT INR) with arithmetic consistency verification.
- **Indian GST Intelligence**: Extracts and validates 15-character Indian GSTINs with state code verification (01–37), cross-validating Central GST (CGST), State GST (SGST), Integrated GST (IGST), and taxable values.
- **Multi-Page Handling**: Processes multi-page PDF documents sequentially with memory management and aggregates cross-page line items and summary financials.
- **Validation Engine**: Performs automated arithmetic checks (Subtotal + Taxes == Total Amount, Line Item Sum == Subtotal) and enforces business rules (date sequence validation, ISO-8601 formatting, monetary decimal precision).
- **Dynamic Document AI**: Includes a schema-agnostic multimodal extraction component for non-standard documents (contracts, receipts, bills of lading) with an interactive form editor.

---

## 2. System Architecture

`
[ Uploaded Document (PDF / Image) ]
                 │
                 ▼
     [ Document Validation & Rendering ]
     (PyMuPDF / Pillow at 170 DPI)
                 │
                 ▼
   [ PP-StructureV3 / PaddleOCR Engine ]
 (Word Tokens, Bounding Boxes, Confidence)
                 │
                 ▼
    [ LayoutLMv3 Preprocessing ]
   (1000-Scale Normalization & Chunking)
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
[ Model A (Universal) ]   [ Model B (GST) ]
  (field_level/)            (gst_level/)
      └──────────┬──────────┘
                 │
                 ▼
  [ Router & Reconciliation Engine ]
  - Header customer extraction (rejects table contamination)
  - Labeled invoice & PO numbers
  - Financial rows & GSTIN regex verification
  - Geometric billing table reconstruction
                 │
                 ▼
     [ Validation Engine ]
  - Mathematical integrity (Subtotal + Tax == Total)
  - Date ordering (Due Date >= Invoice Date)
  - GSTIN structure & state codes
                 │
                 ▼
  [ Canonical JSON Schema + Streamlit UI ]
`

---

## 3. Folder Structure

`
Enterprise_AI_static/
│
├── app.py                      # Production Streamlit application (Tabbed: Static + Dynamic)
├── requirements.txt            # Production dependencies
├── README.md                   # Complete architectural and operations manual
├── .gitignore                  # Git ignore rules for training caches and venvs
├── .gitattributes              # Git LFS configuration for *.safetensors
│
├── field_level/                # Universal LayoutLMv3 Model A
│   ├── config.json
│   ├── model.safetensors
│   ├── processor_config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── gst_level/                  # Indian GST LayoutLMv3 Model B
│   ├── config.json
│   ├── model.safetensors
│   ├── processor_config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── src/                        # Modular production pipeline
│   ├── __init__.py             # Windows DLL load guard (torch before paddle)
│   ├── config.py               # Paths, constants, canonical fields, regexes
│   ├── pdf_utils.py            # PDF rendering, multi-page parsing, image decoding
│   ├── ocr.py                  # PP-StructureV3 & PaddleOCR adapter with deduplication
│   ├── preprocessing.py        # 1000-scale box normalization, chunking, row clustering
│   ├── model_loader.py         # Cached model loading for Model A & Model B
│   ├── field_extractor.py      # Universal Model A token classification & span builder
│   ├── gst_extractor.py        # GST Model B token classification, GSTIN & tax extractor
│   ├── line_items.py           # Geometric billing table reconstruction & arithmetic checks
│   ├── reconciliation.py       # Cross-model reconciliation & structural evidence merging
│   ├── router.py               # Document type routing (Indian GST vs International)
│   ├── validation.py           # Financial consistency check, GSTIN check, date logic
│   ├── pipeline.py             # High-level pipeline orchestrator
│   └── utils.py                # String cleaning, date parsing, decimal formatting
│
├── dynamic/                    # Dynamic schema-agnostic extraction module
│   ├── __init__.py
│   ├── service.py              # Connector for multimodal backend API
│   └── form_renderer.py        # Recursive editable dynamic form component
│
├── notebooks/                  # Production inference notebooks for reference
│   └── GST_Invoice_Extractor_PRODUCTION_v11_ONE_RUN_RECONCILIATION_FIXED.ipynb
│
├── sample_data/                # Real sample invoices for testing and demo
│   ├── 8176011266.pdf          # 1-Page GST processing invoice
│   ├── 8176000040.pdf          # 5-Page multi-page invoice
│   └── test_document.png       # Image document
│
├── tests/                      # Automated test suite (9 unit + 2 integration tests)
│   ├── __init__.py
│   ├── test_model_loading.py   # Verifies Model A and Model B loading & configs
│   ├── test_ocr.py             # Verifies bounding box conversion & OCR execution
│   ├── test_validation.py      # Verifies schema & arithmetic consistency checks
│   └── test_pipeline.py        # End-to-end integration tests on real invoices
│
└── outputs/                    # Output directory for exported JSON and reports
`

---

## 4. Trained Models

The application uses two fine-tuned LayoutLMv3 models directly from local safetensors:

1. **ield_level/ (Universal Model A)**
   - Specialized for general invoice field understanding.
   - Extracts standard entities: VENDOR_NAME, CUSTOMER_NAME, ADDRESS, INVOICE_NUMBER, INVOICE_DATE, DUE_DATE, REF_NO, SUBTOTAL, TAX, TOTAL_AMOUNT, and line items.
2. **gst_level/ (GST Model B)**
   - Specialized for Indian GST invoices.
   - Trained with BIO tagging across 26 labels targeting tax breakdown components: CENTRAL_GST, STATE_GST, SUBTOTAL, TOTAL_AMOUNT, BILL_NO, etc.

Both models are loaded in evaluation mode on CUDA if available, falling back to CPU.

---

## 5. Supported File Formats

The application natively supports:
- **PDF** (.pdf) — Multi-page documents rendered at 170 DPI via PyMuPDF.
- **PNG** (.png)
- **JPG / JPEG** (.jpg, .jpeg)
- **WEBP** (.webp)
- **TIFF** (.tif, .tiff)

---

## 6. Installation & Setup

### Prerequisites
- Python 3.11 (64-bit)
- Git & Git LFS

### Step 1: Clone and Set Up Git LFS
`ash
git clone <repo-url>
cd Enterprise_AI_static
git lfs install
git lfs pull
`

### Step 2: Install Dependencies
`ash
pip install -r requirements.txt
`

> **Note on Windows DLL Load Order**: PyTorch must be imported prior to Paddle to avoid shm.dll loading conflicts. This is automatically handled by src/__init__.py.

---

## 7. Running the Streamlit Application

Launch the application with:

`ash
streamlit run app.py
`

The web interface will be accessible at http://localhost:8501.

### User Interface Features
- **Invoice Extraction Tab**:
  - Drag & drop file uploader.
  - Sidebar quick-load selector with real test invoices.
  - High-resolution document preview.
  - One-click extraction with live progress indicator.
  - KPI summary metrics cards (Total Amount, Subtotal, Tax, Invoice No, Date, GSTIN).
  - Tabbed breakdown: Invoice Information, Financials & Taxes, Line Items Table, Validation & Audit, and Canonical JSON.
  - Download buttons for Canonical JSON and Summary Text Report.
- **Dynamic Document Extraction Tab**:
  - Schema-agnostic extraction for unstructured or non-invoice documents.
  - Guidance prompt input for target entity discovery.
  - Interactive editable form renderer for dynamic key-values.

---

## 8. Running the Automated Test Suite

Run all unit and integration tests:

`ash
pytest tests/ -v
`

Test suite coverage:
- 	est_model_loading.py: Checks Model A and Model B loading, config validity, and singleton caching.
- 	est_ocr.py: Verifies polygon bounding box conversion, OCR record deduplication, and OCR execution.
- 	est_validation.py: Tests pass scenario, date sequencing logic, and invalid GSTIN detection.
- 	est_pipeline.py: Tests multi-page PDF rendering and full end-to-end extraction on real invoice 8176011266.pdf.

---

## 9. Git LFS Configuration

Large model weights (*.safetensors) are tracked using Git LFS via .gitattributes:

`gitattributes
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
`

Training caches, optimizer states, virtual environments, and temporary files are excluded in .gitignore.

---

## 10. Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| NotImplementedError: ConvertPirAttribute2RuntimeAttribute | PaddlePaddle 3.3.x CPU oneDNN bug on Windows | Ensure paddlepaddle==3.2.2 and paddleocr==3.4.1 are installed as pinned in equirements.txt. |
| OSError: [WinError 127] shm.dll | DLL conflict when importing Paddle before PyTorch | src/__init__.py automatically imports PyTorch first. Ensure imports go through src. |
| OutOfMemoryError on GPU | Large multi-page PDF exceeding VRAM | The pipeline processes pages sequentially and clears cache (	orch.cuda.empty_cache()) per page. |
| Missing line items | Non-standard invoice table formatting | Line items fallback extracts predicted token spans if standard table header markers are not found. |
