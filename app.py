"""Enterprise Document AI - Production Invoice Extraction Application.

End-to-End Invoice Processing with:
1. Document Upload (PDF, PNG, JPG, JPEG, TIFF)
2. OCR & Layout via PP-StructureV3
3. Field & Token Classification via LayoutLMv3 Universal (field_level) & Indian GST (gst_level)
4. Multi-Page Tax Reconciliation & Arithmetic Consistency Validation
5. Line-Item Table Reconstruction
6. Interactive Streamlit Results Display & JSON Export
"""

import io
import json
import time
from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st

# Windows DLL order guard: import torch before paddle
import torch  # noqa: F401

from src.config import (
    SUPPORTED_EXTENSIONS,
    CANONICAL_FIELDS,
    MONEY_FIELDS,
    SAMPLE_DATA_DIR,
    DEVICE,
    PADDLE_DEVICE,
    TEMP_DIR,
    FIELD_MODEL_DIR,
    GST_MODEL_DIR,
)
from src.pipeline import InvoiceProcessingPipeline
from src.pdf_utils import load_pages, validate_file

# Streamlit Page Setup
st.set_page_config(
    page_title="Enterprise Invoice AI",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enterprise UI styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .badge-pass {
        background-color: #DCFCE7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }
    .badge-warn {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }
    .badge-info {
        background-color: #E0F2FE;
        color: #0369A1;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }
    .field-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .field-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .field-value {
        font-size: 1.05rem;
        color: #0F172A;
        font-weight: 600;
        word-break: break-word;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Invoice Extraction Pipeline & Models...")
def get_invoice_pipeline() -> InvoiceProcessingPipeline:
    """Initialize and cache the production InvoiceProcessingPipeline."""
    return InvoiceProcessingPipeline(device=DEVICE)


# -----------------------------------------------------------------------------
# Sidebar: System Status & Sample Invoices
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System Status")
    st.success(f"🖥️ **PyTorch Device:** `{DEVICE.upper()}`")
    st.info(f"🔍 **Paddle Device:** `{PADDLE_DEVICE.upper()}`")
    
    with st.expander("📁 Verified Model Checkpoints", expanded=False):
        st.caption(f"**Universal Model:** `{FIELD_MODEL_DIR.name}`")
        st.caption(f"**GST Model:** `{GST_MODEL_DIR.name}`")
        st.caption("Active OCR: PP-StructureV3 / PP-OCRv5")

    st.divider()
    st.header("📂 Sample Invoices")
    sample_choice = st.selectbox(
        "Load Pre-packaged Invoice",
        options=[
            "-- Select an invoice --",
            "8176011266.pdf (Real 1-Page Indian GST Invoice)",
            "8176000040.pdf (Real 5-Page Multi-Page Invoice)",
            "test_document.png (Cloud Services Tax Invoice)",
        ],
    )

    selected_sample_path = None
    if sample_choice.startswith("8176011266"):
        selected_sample_path = SAMPLE_DATA_DIR / "8176011266.pdf"
    elif sample_choice.startswith("8176000040"):
        selected_sample_path = SAMPLE_DATA_DIR / "8176000040.pdf"
    elif sample_choice.startswith("test_document"):
        selected_sample_path = SAMPLE_DATA_DIR / "test_document.png"

    st.divider()
    st.caption("Supported formats: PDF, PNG, JPG, JPEG, TIFF")


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">Enterprise Invoice AI & GST Extraction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Automated end-to-end extraction with PP-StructureV3, trained LayoutLMv3 models, tax reconciliation, and arithmetic validation.</div>',
    unsafe_allow_html=True,
)

# Step 1: File Upload & Input Selection
col_upload, col_preview = st.columns([1.2, 1], gap="medium")

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Invoice Document (PDF or Image)",
        type=["pdf", "png", "jpg", "jpeg", "webp", "tif", "tiff"],
        key="main_invoice_uploader",
    )

    target_path = None
    target_display_name = ""

    if uploaded_file is not None:
        target_path = TEMP_DIR / f"upload_{uploaded_file.name}"
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        target_display_name = uploaded_file.name
        st.success(f"Uploaded: **{uploaded_file.name}** ({len(uploaded_file.getvalue()) / 1024:.1f} KB)")
    elif selected_sample_path is not None and selected_sample_path.exists():
        target_path = selected_sample_path
        target_display_name = selected_sample_path.name
        st.info(f"Loaded sample: **{target_display_name}**")

    btn_extract = st.button(
        "🚀 Extract Invoice",
        type="primary",
        use_container_width=True,
        disabled=(target_path is None),
    )

with col_preview:
    st.subheader("Document Preview")
    if target_path is not None and target_path.exists():
        try:
            pages = load_pages(target_path)
            st.image(
                pages[0],
                caption=f"Page 1 of {len(pages)}: {target_display_name}",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Preview unavailable: {e}")
    else:
        st.info("Upload an invoice or select a sample to preview here.")

st.divider()

# Step 2: End-to-End Extraction Execution
if btn_extract and target_path is not None:
    pipeline = get_invoice_pipeline()
    with st.spinner("Processing invoice through PP-StructureV3 and LayoutLMv3 models..."):
        try:
            t0 = time.perf_counter()
            invoice_result = pipeline.process(target_path)
            elapsed = round(time.perf_counter() - t0, 2)
            invoice_result["_diagnostics"]["processing_time"] = elapsed
            st.session_state["invoice_result"] = invoice_result
            st.session_state["invoice_name"] = target_display_name
            st.success(f"✅ Extraction completed in **{elapsed}s**!")
        except Exception as err:
            st.error(f"Extraction failed: {err}")
            st.exception(err)

# Step 3: Display Extracted Results
if "invoice_result" in st.session_state:
    res = st.session_state["invoice_result"]
    diag = res.get("_diagnostics", {})
    check = diag.get("consistency_check", {})
    status = check.get("status", "PASS")
    doc_type = diag.get("document_type", "INVOICE")

    # Status & Compliance Header
    status_cols = st.columns([1.5, 1, 1, 1])
    with status_cols[0]:
        if status == "PASS":
            st.markdown('<span class="badge-pass">✓ Validation Passed (Arithmetic & GST Verified)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-warn">⚠ Review Required (Validation Warnings)</span>', unsafe_allow_html=True)
        st.caption(f"Document: **{st.session_state.get('invoice_name')}** • Type: `{doc_type}`")
    with status_cols[1]:
        st.metric("Total Amount", f"₹ {res.get('TOTAL_AMOUNT') or '-'}")
    with status_cols[2]:
        st.metric("Subtotal", f"₹ {res.get('SUBTOTAL') or '-'}")
    with status_cols[3]:
        st.metric("Tax Amount", f"₹ {res.get('TAX') or '-'}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed Results Tabs
    tab_overview, tab_items, tab_audit, tab_json = st.tabs([
        "📋 Reconciled Invoice Fields",
        "📦 Line Items Table",
        "🔍 Tax & Audit Validation",
        "💻 Canonical JSON & Export",
    ])

    # --- TAB 1: RECONCILED FIELDS ---
    with tab_overview:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### 📄 Invoice Identification")
            st.markdown(f"""
            <div class="field-card">
                <div class="field-label">Invoice Number</div>
                <div class="field-value">{res.get('INVOICE_NUMBER') or '-'}</div>
            </div>
            <div class="field-card">
                <div class="field-label">Invoice Date</div>
                <div class="field-value">{res.get('INVOICE_DATE') or '-'}</div>
            </div>
            <div class="field-card">
                <div class="field-label">Due Date</div>
                <div class="field-value">{res.get('DUE_DATE') or '-'}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("##### 🏢 Vendor (Supplier)")
            st.markdown(f"""
            <div class="field-card">
                <div class="field-label">Vendor Name</div>
                <div class="field-value">{res.get('VENDOR_NAME') or '-'}</div>
            </div>
            <div class="field-card">
                <div class="field-label">Vendor GSTIN</div>
                <div class="field-value">{res.get('GSTIN') or '-'}</div>
            </div>
            <div class="field-card">
                <div class="field-label">Vendor Address</div>
                <div class="field-value">{res.get('VENDOR_ADDRESS') or '-'}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown("##### 👤 Customer (Billed To)")
            st.markdown(f"""
            <div class="field-card">
                <div class="field-label">Customer Name</div>
                <div class="field-value">{res.get('CUSTOMER_NAME') or '-'}</div>
            </div>
            <div class="field-card">
                <div class="field-label">Customer GSTIN</div>
                <div class="field-value">{res.get('CUSTOMER_GSTIN') or '-'}</div>
            </div>
            <div class="field-card">
                <div class="field-label">Customer Address</div>
                <div class="field-value">{res.get('CUSTOMER_ADDRESS') or '-'}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("##### 💰 Financial & Tax Breakdown")
        f_cols = st.columns(6)
        f_cols[0].metric("Subtotal", f"₹ {res.get('SUBTOTAL') or '-'}")
        f_cols[1].metric("CGST", f"₹ {res.get('CENTRAL_GST') or '-'}")
        f_cols[2].metric("SGST", f"₹ {res.get('STATE_GST') or '-'}")
        f_cols[3].metric("IGST", f"₹ {res.get('INTEGRATED_GST') or '-'}")
        f_cols[4].metric("Total Tax", f"₹ {res.get('TAX') or '-'}")
        f_cols[5].metric("Grand Total", f"₹ {res.get('TOTAL_AMOUNT') or '-'}")

    # --- TAB 2: LINE ITEMS TABLE ---
    with tab_items:
        st.subheader("Reconstructed Line Items")
        line_items = res.get("line_items", [])
        if line_items:
            df_items = pd.DataFrame(line_items)
            display_cols = [c for c in ["description", "quantity", "unit_price", "amount", "page"] if c in df_items.columns]
            st.dataframe(df_items[display_cols], use_container_width=True)
            st.caption(f"Extracted **{len(line_items)}** item lines across document.")
        else:
            st.info("No structured line item table rows detected on this invoice.")

    # --- TAB 3: TAX & AUDIT VALIDATION ---
    with tab_audit:
        st.subheader("Arithmetic & Compliance Audit")
        errors = check.get("errors", [])
        if not errors:
            st.success("✅ **All consistency checks passed!** Subtotal + Tax equals Grand Total, and GSTIN formatting is valid.")
        else:
            for err in errors:
                st.warning(f"⚠️ {err}")

        st.divider()
        st.markdown("##### 🛠️ Extraction Diagnostics")
        d_cols = st.columns(4)
        d_cols[0].metric("OCR Engine", diag.get("ocr_engine", "PP-StructureV3"))
        d_cols[1].metric("Total Words Read", diag.get("total_words", "-"))
        d_cols[2].metric("Pages Processed", diag.get("total_pages", 1))
        d_cols[3].metric("Runtime", f"{diag.get('processing_time', '-')}s")

    # --- TAB 4: CANONICAL JSON & EXPORT ---
    with tab_json:
        st.subheader("Canonical Invoice Output (JSON)")
        st.json(res)

        json_bytes = json.dumps(res, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            "📥 Download Invoice JSON",
            data=json_bytes,
            file_name=f"{st.session_state.get('invoice_name', 'invoice')}_extracted.json",
            mime="application/json",
            use_container_width=True,
        )
