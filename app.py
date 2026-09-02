"""Enterprise Document AI - Production Streamlit Application.

Modular, production-ready interface for:
1. Dynamic Document AI & Adaptive Forms (Fully document-agnostic extraction for arbitrary documents)
2. Specialized Invoice Extraction & Audit (LayoutLMv3 Universal + GST models, geometry table reconstruction, tax reconciliation)
3. Hosted Multimodal Service (Vision-Language connector)
"""

import io
import json
import time
import datetime as dt
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
)
from src.dynamic_pipeline import DynamicDocumentPipeline
from src.dynamic_form import render_adaptive_form
from src.pipeline import InvoiceProcessingPipeline
from src.pdf_utils import load_pages, validate_file
from dynamic.form_renderer import render_dynamic_form
from dynamic.service import DynamicExtractorClient

# Streamlit Page Setup
st.set_page_config(
    page_title="Enterprise Document AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom enterprise styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-dynamic {
        background-color: #EDE9FE;
        color: #5B21B6;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-pass {
        background-color: #DCFCE7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-warn {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Dynamic Document AI Pipeline...")
def get_dynamic_pipeline():
    """Cache the universal document-agnostic extraction pipeline."""
    return DynamicDocumentPipeline()


@st.cache_resource(show_spinner="Loading Specialized Invoice Pipeline...")
def get_invoice_pipeline():
    """Cache the specialized invoice extraction pipeline."""
    return InvoiceProcessingPipeline()


# Sidebar Controls
with st.sidebar:
    st.header("⚙️ System Status")
    st.success(f"🖥️ **PyTorch Device:** `{DEVICE.upper()}`")
    st.info(f"🔍 **Paddle Device:** `{PADDLE_DEVICE.upper()}`")
    st.caption("Active Engines: PP-StructureV3 + LayoutLMv3 Models")

    st.divider()
    st.header("📂 Quick Load Samples")
    sample_choice = st.selectbox(
        "Select Test Document",
        options=[
            "-- Choose a document --",
            "grocery_list.png (Grocery List / Non-Invoice)",
            "purchase_order.png (Purchase Order with Custom Tables)",
            "irregular_form.png (Incident Form with Irregular Fields)",
            "8176011266.pdf (Real 1-Page GST Invoice)",
            "8176000040.pdf (Real 5-Page Multi-Page Document)",
            "test_document.png (Image Document)",
        ]
    )

    selected_sample_path = None
    if sample_choice.startswith("grocery_list"):
        selected_sample_path = SAMPLE_DATA_DIR / "grocery_list.png"
    elif sample_choice.startswith("purchase_order"):
        selected_sample_path = SAMPLE_DATA_DIR / "purchase_order.png"
    elif sample_choice.startswith("irregular_form"):
        selected_sample_path = SAMPLE_DATA_DIR / "irregular_form.png"
    elif sample_choice.startswith("8176011266"):
        selected_sample_path = SAMPLE_DATA_DIR / "8176011266.pdf"
    elif sample_choice.startswith("8176000040"):
        selected_sample_path = SAMPLE_DATA_DIR / "8176000040.pdf"
    elif sample_choice.startswith("test_document"):
        selected_sample_path = SAMPLE_DATA_DIR / "test_document.png"

    st.divider()
    st.caption("Supported formats: PDF, PNG, JPG, JPEG, WEBP, TIFF")


# Top-level Navigation Tabs
tab_dynamic_engine, tab_invoice_engine, tab_hosted_service = st.tabs([
    "🌐 Dynamic Document AI & Adaptive Forms",
    "🧾 Specialized Invoice Extraction & Audit",
    "☁️ Hosted Multimodal Service",
])

# ==============================================================================
# TAB 1: DYNAMIC DOCUMENT AI & ADAPTIVE FORMS (PRIMARY UNIVERSAL ENGINE)
# ==============================================================================
with tab_dynamic_engine:
    st.markdown('<div class="main-header">Universal Document AI & Adaptive Forms</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Document-agnostic extraction for arbitrary documents (Invoices, Receipts, Grocery Lists, POs, Forms, Agreements)</div>',
        unsafe_allow_html=True
    )

    col_dyn_input, col_dyn_preview = st.columns([1.2, 1], gap="medium")

    with col_dyn_input:
        st.subheader("Upload Any Document")
        uploaded_dyn = st.file_uploader(
            "Drag & Drop any document here",
            type=["pdf", "png", "jpg", "jpeg", "webp", "tif", "tiff"],
            key="dyn_uploader_input",
            help="Extracts whatever meaningful information is present without enforcing a fixed invoice schema."
        )

        dyn_target_path = None
        dyn_target_name = None

        if uploaded_dyn is not None:
            temp_dyn_upload = TEMP_DIR / f"dyn_{uploaded_dyn.name}"
            with open(temp_dyn_upload, "wb") as f:
                f.write(uploaded_dyn.getvalue())
            dyn_target_path = temp_dyn_upload
            dyn_target_name = uploaded_dyn.name
        elif selected_sample_path is not None and selected_sample_path.exists():
            dyn_target_path = selected_sample_path
            dyn_target_name = selected_sample_path.name
            st.info(f"📑 Using sample document: **{dyn_target_name}**")

        extract_dyn_btn = st.button("🚀 Extract Document Dynamically", type="primary", use_container_width=True, key="run_dyn_main")

    with col_dyn_preview:
        st.subheader("Document Preview")
        if dyn_target_path is not None and dyn_target_path.exists():
            try:
                preview_pages = load_pages(dyn_target_path)
                st.image(preview_pages[0], caption=f"Page 1 of {len(preview_pages)}: {dyn_target_name}", use_container_width=True)
                if len(preview_pages) > 1:
                    st.caption(f"ℹ️ Multi-page document detected: **{len(preview_pages)} pages**.")
            except Exception as e:
                st.warning(f"Preview unavailable: {e}")
        else:
            st.info("Upload a document or select a sample from the sidebar to view preview.")

    # Execution
    if extract_dyn_btn and dyn_target_path is not None:
        try:
            pipeline = get_dynamic_pipeline()
            with st.spinner("Extracting document structure, layout, tables, and attributes..."):
                t_start = time.perf_counter()
                res = pipeline.process(dyn_target_path)
                t_elapsed = round(time.perf_counter() - t_start, 2)

            st.session_state["dyn_result"] = res
            st.session_state["dyn_elapsed"] = t_elapsed
            st.session_state["dyn_target_name"] = dyn_target_name

        except Exception as exc:
            st.error(f"❌ Dynamic Extraction Failed: {type(exc).__name__}: {str(exc)}")
            st.exception(exc)

    # Dynamic Results & Adaptive Forms Display
    if "dyn_result" in st.session_state:
        dyn_data = st.session_state["dyn_result"]
        doc_type = dyn_data.get("document_type", "general_document").upper()
        fields_list = dyn_data.get("fields", [])
        tables_list = dyn_data.get("tables", [])
        meta = dyn_data.get("metadata", {})

        st.divider()

        # Status Bar
        status_cols = st.columns([1.5, 1, 1, 1])
        with status_cols[0]:
            st.markdown(f'<span class="badge-dynamic">📑 Classified Type: {doc_type}</span>', unsafe_allow_html=True)
            st.caption(f"File: **{st.session_state.get('dyn_target_name')}** • Engine: `{meta.get('ocr_engine', 'OCR')}`")
        with status_cols[1]:
            st.metric("Processing Time", f"{meta.get('processing_time', 0):.2f}s")
        with status_cols[2]:
            st.metric("Dynamic Fields", f"{len(fields_list)}")
        with status_cols[3]:
            st.metric("Tables Found", f"{len(tables_list)}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabbed View: Adaptive Form vs Raw JSON
        form_tab, json_tab = st.tabs([
            "📝 Adaptive Dynamic Form",
            "💻 Canonical Generic JSON & Export",
        ])

        with form_tab:
            # Renders adaptive form inputs and interactive table editors
            edited_result = render_adaptive_form(dyn_data, form_key="main_dynamic_form")

        with json_tab:
            st.subheader("Canonical Generic JSON Schema")
            st.json(dyn_data)

            st.divider()
            dl_col1, dl_col2 = st.columns(2)
            json_str = json.dumps(dyn_data, indent=2, ensure_ascii=False)
            with dl_col1:
                st.download_button(
                    "📥 Download Generic JSON",
                    data=json_str,
                    file_name=f"{st.session_state.get('dyn_target_name', 'doc')}_extracted.json",
                    mime="application/json",
                    use_container_width=True
                )

            # Generate plain text report
            report_lines = [
                "ENTERPRISE DOCUMENT AI — DYNAMIC EXTRACTION REPORT",
                "=" * 60,
                f"Source Document : {st.session_state.get('dyn_target_name')}",
                f"Document Type   : {doc_type}",
                f"Page Count      : {meta.get('page_count', 1)}",
                f"Total Words     : {meta.get('word_count', 0)}",
                f"Processing Time : {meta.get('processing_time', 0)}s",
                "-" * 60,
                f"DYNAMIC FIELDS ({len(fields_list)}):",
            ]
            for f in fields_list:
                report_lines.append(f"  {f.get('name'):25s}: {f.get('value')} (conf: {f.get('confidence')})")

            report_lines.extend(["-" * 60, f"TABLES ({len(tables_list)}):"])
            for t_idx, tb in enumerate(tables_list, 1):
                report_lines.append(f"  Table {t_idx} [{tb.get('title')}]: Columns: {tb.get('columns')}")
                for r in tb.get('rows', []):
                    report_lines.append(f"    {r}")

            report_txt = "\n".join(report_lines)
            with dl_col2:
                st.download_button(
                    "📄 Download Form Summary (TXT)",
                    data=report_txt,
                    file_name=f"{st.session_state.get('dyn_target_name', 'doc')}_summary.txt",
                    mime="text/plain",
                    use_container_width=True
                )

# ==============================================================================
# TAB 2: SPECIALIZED INVOICE EXTRACTION & AUDIT
# ==============================================================================
with tab_invoice_engine:
    st.markdown('<div class="main-header">Specialized Invoice Extraction & Reconciliation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Specialized Indian GST & Universal Invoice Extraction with LayoutLMv3 and Arithmetic Verification</div>',
        unsafe_allow_html=True
    )

    inv_col1, inv_col2 = st.columns([1.2, 1], gap="medium")
    with inv_col1:
        inv_file = st.file_uploader(
            "Upload Invoice for Specialized Tax Reconciliation",
            type=["pdf", "png", "jpg", "jpeg", "webp", "tif", "tiff"],
            key="specialized_inv_uploader"
        )
        inv_target = None
        if inv_file is not None:
            temp_inv = TEMP_DIR / f"spec_inv_{inv_file.name}"
            with open(temp_inv, "wb") as f:
                f.write(inv_file.getvalue())
            inv_target = temp_inv
        elif selected_sample_path is not None and "81760" in selected_sample_path.name:
            inv_target = selected_sample_path
            st.info(f"📑 Using sample invoice: **{inv_target.name}**")

        run_inv_btn = st.button("🚀 Run Specialized Invoice Extraction", type="primary", key="run_spec_inv_btn")

    with inv_col2:
        st.subheader("Invoice Preview")
        if inv_target is not None and inv_target.exists():
            try:
                pages_inv = load_pages(inv_target)
                st.image(pages_inv[0], caption=f"Page 1 of {len(pages_inv)}: {inv_target.name}", use_container_width=True)
            except Exception:
                pass

    if run_inv_btn and inv_target is not None:
        try:
            inv_pipeline = get_invoice_pipeline()
            with st.spinner("Processing through LayoutLMv3 Universal & GST models with tax reconciliation..."):
                inv_res = inv_pipeline.process(inv_target)
            st.session_state["spec_inv_result"] = inv_res
        except Exception as e:
            st.error(f"Invoice extraction error: {e}")

    if "spec_inv_result" in st.session_state:
        res = st.session_state["spec_inv_result"]
        diag = res.get("_diagnostics", {})
        check = diag.get("consistency_check", {})
        status = check.get("status", "PASS")

        if status == "PASS":
            st.markdown('<span class="badge-pass">✓ Validated Document</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-warn">⚠ Extraction with Warnings</span>', unsafe_allow_html=True)

        kpis = st.columns(4)
        kpis[0].metric("Total Amount", f"₹ {res.get('TOTAL_AMOUNT') or '-'}")
        kpis[1].metric("Subtotal", f"₹ {res.get('SUBTOTAL') or '-'}")
        kpis[2].metric("Tax Total", f"₹ {res.get('TAX') or '-'}")
        kpis[3].metric("Invoice No", f"{res.get('INVOICE_NUMBER') or '-'}")

        with st.expander("💻 Raw Invoice Canonical Output", expanded=False):
            st.json(res)

# ==============================================================================
# TAB 3: HOSTED MULTIMODAL SERVICE
# ==============================================================================
with tab_hosted_service:
    st.markdown('<div class="main-header">Hosted Multimodal Document AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Remote Vision-Language Model Service Connector (Qwen2.5-VL Backend)</div>',
        unsafe_allow_html=True
    )

    client = DynamicExtractorClient()
    health_status = client.check_health()
    if health_status.get("online"):
        st.success(f"🟢 Remote Service Online (Status: {health_status.get('status_code')})")
    else:
        st.warning(f"🟡 Remote Service Connecting: {health_status.get('error', 'Offline')}")

    host_file = st.file_uploader("Upload Document for Multimodal Extraction", type=["pdf", "png", "jpg", "jpeg", "webp"], key="host_uploader")
    custom_prompt = st.text_area("Optional Guidance Prompt", placeholder="e.g. Focus on liability clauses, total penalty, and key signatories.")

    if st.button("🚀 Call Remote Multimodal API", key="call_remote_btn") and host_file is not None:
        with st.spinner("Communicating with remote API..."):
            try:
                host_file.seek(0)
                remote_resp = client.extract(
                    filename=host_file.name,
                    file_bytes=host_file.read(),
                    mime_type=host_file.type or "application/octet-stream",
                    custom_instructions=custom_prompt,
                )
                st.success("✅ Remote extraction complete!")
                st.json(remote_resp)
            except Exception as err:
                st.error(f"Remote service call failed: {err}")
