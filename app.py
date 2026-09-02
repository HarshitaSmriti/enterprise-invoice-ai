"""Enterprise Document AI - Production Streamlit Application.

Modular, production-ready interface for:
1. Static Invoice Extraction (LayoutLMv3 Universal + GST models, geometry table reconstruction, tax reconciliation)
2. Dynamic Document Extraction (Schema-agnostic multimodal extraction)
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
    .badge-fail {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading LayoutLMv3 models and OCR engines into memory...")
def get_cached_pipeline():
    """Cache the extraction pipeline to ensure models are loaded only once."""
    return InvoiceProcessingPipeline()


# Top-level Tabs
tab_invoice, tab_dynamic = st.tabs([
    "📄 Enterprise Invoice Extraction",
    "🌐 Dynamic Document Extraction"
])

# ==============================================================================
# TAB 1: INVOICE EXTRACTION (STATIC PIPELINE)
# ==============================================================================
with tab_invoice:
    st.markdown('<div class="main-header">Enterprise Document AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Automated Invoice Understanding with LayoutLMv3 (Universal + Indian GST) & PP-StructureV3</div>',
        unsafe_allow_html=True
    )

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ System Status")
        st.success(f"🖥️ **PyTorch Device:** `{DEVICE.upper()}`")
        st.info(f"🔍 **Paddle Device:** `{PADDLE_DEVICE.upper()}`")
        st.caption("Models: `field_level` (Universal) & `gst_level` (GST)")

        st.divider()
        st.header("📂 Quick Load Sample")
        sample_choice = st.selectbox(
            "Select Real Test Invoice",
            options=[
                "-- Choose a sample document --",
                "8176011266.pdf (1-Page GST Processing Invoice)",
                "8176000040.pdf (5-Page Multi-Page Invoice)",
                "test_document.png (Image Invoice)",
            ]
        )

        selected_sample_path = None
        if sample_choice.startswith("8176011266"):
            selected_sample_path = SAMPLE_DATA_DIR / "8176011266.pdf"
        elif sample_choice.startswith("8176000040"):
            selected_sample_path = SAMPLE_DATA_DIR / "8176000040.pdf"
        elif sample_choice.startswith("test_document"):
            selected_sample_path = SAMPLE_DATA_DIR / "test_document.png"

        st.divider()
        st.caption("Supported formats: PDF, PNG, JPG, JPEG, WEBP, TIFF")

    # Upload Section
    col_input, col_preview = st.columns([1.2, 1], gap="medium")

    with col_input:
        st.subheader("Upload Invoice")
        uploaded_file = st.file_uploader(
            "Drag & Drop invoice here",
            type=["pdf", "png", "jpg", "jpeg", "webp", "tif", "tiff"],
            help="Supported: PDF, PNG, JPG, JPEG, WEBP, TIFF"
        )

        # Handle file resolution (upload vs sample selection)
        target_path = None
        target_name = None

        if uploaded_file is not None:
            temp_upload = TEMP_DIR / f"upload_{uploaded_file.name}"
            with open(temp_upload, "wb") as f:
                f.write(uploaded_file.getvalue())
            target_path = temp_upload
            target_name = uploaded_file.name
        elif selected_sample_path is not None and selected_sample_path.exists():
            target_path = selected_sample_path
            target_name = selected_sample_path.name
            st.info(f"📑 Using sample invoice: **{target_name}**")

        extract_clicked = st.button("🚀 Extract Invoice", type="primary", use_container_width=True)

    with col_preview:
        st.subheader("Document Preview")
        if target_path is not None and target_path.exists():
            try:
                pages = load_pages(target_path)
                st.image(pages[0], caption=f"Page 1 of {len(pages)}: {target_name}", use_container_width=True)
                if len(pages) > 1:
                    st.caption(f"ℹ️ Document has **{len(pages)} pages**. All pages will be processed.")
            except Exception as e:
                st.warning(f"Preview unavailable: {e}")
        else:
            st.info("Upload an invoice or select a sample from the sidebar to see a preview.")

    # Processing and Results Display
    if extract_clicked and target_path is not None:
        try:
            pipeline = get_cached_pipeline()

            with st.spinner("Processing document through PP-StructureV3 and LayoutLMv3 ensemble..."):
                t_start = time.perf_counter()
                result = pipeline.process(target_path)
                elapsed = round(time.perf_counter() - t_start, 2)

            st.session_state["invoice_result"] = result
            st.session_state["invoice_elapsed"] = elapsed
            st.session_state["invoice_target_name"] = target_name

        except Exception as exc:
            st.error(f"❌ Extraction Failed: {type(exc).__name__}: {str(exc)}")
            st.exception(exc)

    # Render Extraction Results
    if "invoice_result" in st.session_state:
        res = st.session_state["invoice_result"]
        diag = res.get("_diagnostics", {})
        check = diag.get("consistency_check", {})
        status = check.get("status", "PASS")
        errors = check.get("errors", [])
        warnings = diag.get("warnings", [])

        st.divider()

        # Header Status Bar
        status_cols = st.columns([1.5, 1, 1, 1])
        with status_cols[0]:
            if status == "PASS":
                st.markdown('<span class="badge-pass">✓ Validated Document</span>', unsafe_allow_html=True)
            elif status == "WARNINGS":
                st.markdown('<span class="badge-warn">⚠ Extraction with Warnings</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-fail">✗ Inconsistency Detected</span>', unsafe_allow_html=True)
            st.caption(f"File: **{st.session_state.get('invoice_target_name')}** • Type: `{diag.get('document_type', 'INVOICE')}`")

        with status_cols[1]:
            st.metric("Inference Time", f"{diag.get('total_seconds', 0):.2f}s")
        with status_cols[2]:
            st.metric("Total Pages", f"{diag.get('pages', 1)}")
        with status_cols[3]:
            st.metric("Line Items", f"{len(res.get('line_items', []))}")

        # KPI Summary Cards
        st.markdown("<br>", unsafe_allow_html=True)
        kpi_cols = st.columns(6)
        currency_sym = "₹ " if diag.get("document_type") == "INDIAN_GST" else ""

        with kpi_cols[0]:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{currency_sym}{res.get("TOTAL_AMOUNT") or "-"}</div><div class="metric-lbl">Total Amount</div></div>', unsafe_allow_html=True)
        with kpi_cols[1]:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{currency_sym}{res.get("SUBTOTAL") or "-"}</div><div class="metric-lbl">Subtotal</div></div>', unsafe_allow_html=True)
        with kpi_cols[2]:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{currency_sym}{res.get("TAX") or "-"}</div><div class="metric-lbl">Tax Total</div></div>', unsafe_allow_html=True)
        with kpi_cols[3]:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{res.get("INVOICE_NUMBER") or "-"}</div><div class="metric-lbl">Invoice No</div></div>', unsafe_allow_html=True)
        with kpi_cols[4]:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{res.get("INVOICE_DATE") or "-"}</div><div class="metric-lbl">Invoice Date</div></div>', unsafe_allow_html=True)
        with kpi_cols[5]:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{res.get("GSTIN") or "-"}</div><div class="metric-lbl">GSTIN</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabbed Detail View
        tab_fields, tab_financial, tab_items, tab_audit, tab_json = st.tabs([
            "📋 Invoice Information",
            "💰 Financial & GST Breakdown",
            "📦 Line Items Table",
            "🛡️ Validation & Audit",
            "💻 Canonical JSON & Export"
        ])

        # 1. Invoice Information
        with tab_fields:
            col_v, col_c = st.columns(2)
            with col_v:
                st.subheader("Vendor / Supplier")
                st.write(f"**Vendor Name:** {res.get('VENDOR_NAME') or 'Not Detected'}")
                st.write(f"**Address:** {res.get('ADDRESS') or 'Not Detected'}")
                st.write(f"**Supplier GSTIN:** {res.get('GSTIN') or 'Not Detected'}")

            with col_c:
                st.subheader("Customer & Document Info")
                st.write(f"**Customer Name:** {res.get('CUSTOMER_NAME') or 'Not Detected'}")
                st.write(f"**Invoice Number:** `{res.get('INVOICE_NUMBER') or 'Not Detected'}`")
                st.write(f"**Invoice Date:** `{res.get('INVOICE_DATE') or 'Not Detected'}`")
                st.write(f"**Due Date:** `{res.get('DUE_DATE') or 'Not Specified'}`")
                st.write(f"**Reference / PO No:** `{res.get('REF_NO') or 'Not Detected'}`")

        # 2. Financial & GST Breakdown
        with tab_financial:
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                st.subheader("Summary Values")
                f_data = [
                    {"Field": "Subtotal / Basic Total", "Amount": res.get("SUBTOTAL")},
                    {"Field": "Central GST (CGST)", "Amount": res.get("CENTRAL_GST")},
                    {"Field": "State GST (SGST)", "Amount": res.get("STATE_GST")},
                    {"Field": "Integrated GST (IGST)", "Amount": res.get("IGST")},
                    {"Field": "Total Tax (Derived/Declared)", "Amount": res.get("TAX")},
                    {"Field": "Grand Total Amount", "Amount": res.get("TOTAL_AMOUNT")},
                ]
                st.dataframe(pd.DataFrame(f_data), use_container_width=True, hide_index=True)

            with f_col2:
                st.subheader("Tax Reconciliation Status")
                sub = float(res.get("SUBTOTAL") or 0)
                tax = float(res.get("TAX") or 0)
                tot = float(res.get("TOTAL_AMOUNT") or 0)
                if sub and tot:
                    diff = abs((sub + tax) - tot)
                    if diff <= 1.0:
                        st.success(f"✓ Arithmetic check passed: Subtotal ({sub:.2f}) + Tax ({tax:.2f}) = {tot:.2f}")
                    else:
                        st.warning(f"⚠️ Reconcile discrepancy: Subtotal ({sub:.2f}) + Tax ({tax:.2f}) != {tot:.2f} (diff={diff:.2f})")
                else:
                    st.info("Subtotal or Total was not detected for arithmetic comparison.")

        # 3. Line Items Table
        with tab_items:
            items = res.get("line_items", [])
            if items:
                st.subheader(f"Extracted Table Rows ({len(items)} items)")
                df_items = pd.DataFrame(items)
                rename_map = {
                    "description": "Item / Material Description",
                    "quantity": "Quantity",
                    "unit_price": "Unit Price (Rate)",
                    "amount": "Total Amount",
                    "confidence": "Confidence",
                    "page": "Page #",
                }
                cols_to_show = [c for c in rename_map.keys() if c in df_items.columns]
                df_display = df_items[cols_to_show].rename(columns=rename_map)
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            else:
                st.info("No confident tabular line items were reconstructed.")

        # 4. Validation & Audit
        with tab_audit:
            st.subheader("Automated Integrity Verification")

            if not errors and not warnings:
                st.success("✓ All schema, format, and business rule validations passed with zero issues.")

            if errors:
                st.error("Validation Errors Encountered:")
                for err in errors:
                    st.markdown(f"- ❌ {err}")

            if warnings:
                st.warning("Validation Warnings & Advisories:")
                for w in warnings:
                    st.markdown(f"- ⚠️ {w}")

            st.divider()
            st.subheader("Field Provenance & Extraction Confidence")
            meta = res.get("_field_metadata", {})
            if meta:
                meta_rows = []
                for k, v in meta.items():
                    meta_rows.append({
                        "Field": k,
                        "Value": v.get("value") or res.get(k),
                        "Source Method": v.get("source"),
                        "Confidence": f"{float(v.get('confidence', 0))*100:.1f}%",
                    })
                st.dataframe(pd.DataFrame(meta_rows), use_container_width=True, hide_index=True)

        # 5. Canonical JSON & Export
        with tab_json:
            st.subheader("Canonical Extracted JSON")
            st.json(res)

            st.divider()
            dl_col1, dl_col2 = st.columns(2)

            json_str = json.dumps(res, indent=2, ensure_ascii=False)
            with dl_col1:
                st.download_button(
                    "📥 Download Canonical JSON",
                    data=json_str,
                    file_name=f"{st.session_state.get('invoice_target_name', 'invoice')}_extracted.json",
                    mime="application/json",
                    use_container_width=True
                )

            # Generate plain-text report
            report_lines = [
                "ENTERPRISE DOCUMENT AI — EXTRACTION REPORT",
                "=" * 60,
                f"Source Document : {st.session_state.get('invoice_target_name')}",
                f"Generated Date  : {dt.datetime.now().isoformat(timespec='seconds')}",
                f"Validation Status: {status}",
                f"Total Pages     : {diag.get('pages', 1)}",
                f"Execution Time  : {diag.get('total_seconds', 0):.2f}s",
                "-" * 60,
                "CANONICAL FIELDS:",
            ]
            for fld in CANONICAL_FIELDS + ["GSTIN"]:
                report_lines.append(f"  {fld:20s}: {res.get(fld)}")

            report_lines.extend(["-" * 60, f"LINE ITEMS ({len(items)}):"])
            for idx, it in enumerate(items, 1):
                report_lines.append(f"  {idx}. {it.get('description','')} | Qty: {it.get('quantity','')} | Rate: {it.get('unit_price','')} | Amt: {it.get('amount','')}")

            report_text = "\n".join(report_lines)

            with dl_col2:
                st.download_button(
                    "📄 Download Summary Report (TXT)",
                    data=report_text,
                    file_name=f"{st.session_state.get('invoice_target_name', 'invoice')}_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

# ==============================================================================
# TAB 2: DYNAMIC DOCUMENT EXTRACTION (MULTIMODAL)
# ==============================================================================
with tab_dynamic:
    st.markdown('<div class="main-header">Dynamic Multimodal Document AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Schema-Agnostic Extraction for Unstructured Documents (Agreements, Receipts, Shipping Bills)</div>',
        unsafe_allow_html=True
    )

    client = DynamicExtractorClient()

    d_col1, d_col2 = st.columns([1, 1.3], gap="large")

    with d_col1:
        st.subheader("1. Ingestion & Dynamic Guidance")
        dyn_file = st.file_uploader(
            "Upload Document for Dynamic Extraction",
            type=["pdf", "png", "jpg", "jpeg", "webp"],
            key="dyn_uploader"
        )

        custom_guide = st.text_area(
            "Optional Guidance Prompt",
            placeholder="e.g. Focus on extracting buyer obligations, dates, and total commercial penalty clauses.",
            help="Directs the multimodal model on specific entities of interest."
        )

        run_dyn_btn = st.button("🚀 Run Dynamic Extraction", type="primary", key="dyn_run_btn")

        if dyn_file is not None and not dyn_file.name.lower().endswith(".pdf"):
            try:
                st.image(Image.open(dyn_file), caption="Document Preview", use_container_width=True)
            except Exception:
                pass

    with d_col2:
        st.subheader("2. Dynamic Entity Extraction Form")

        if run_dyn_btn and dyn_file is not None:
            with st.spinner("Calling Multimodal Dynamic Extraction Engine..."):
                try:
                    dyn_file.seek(0)
                    resp = client.extract(
                        filename=dyn_file.name,
                        file_bytes=dyn_file.read(),
                        mime_type=dyn_file.type or "application/octet-stream",
                        custom_instructions=custom_guide,
                    )
                    st.session_state["dyn_response"] = resp
                    st.success("✅ Dynamic extraction completed successfully!")
                except Exception as ex:
                    st.error(f"❌ Dynamic Extraction API unavailable or failed: {ex}")
                    st.info("💡 Note: Dynamic extraction calls the hosted Vision-Language backend. Verify your backend endpoint or connection.")

        if "dyn_response" in st.session_state:
            dyn_data = st.session_state["dyn_response"].get("data", {})
            st.markdown(f"**Document Type Identified:** `{dyn_data.get('document_type', 'Document')}`")

            # Render editable form
            doc_fields = dyn_data.get("document", {})
            if doc_fields:
                st.subheader("Editable Key-Values")
                edited = render_dynamic_form(doc_fields, key_prefix="dyn_ui")

            with st.expander("💻 Raw Dynamic Output", expanded=False):
                st.json(st.session_state["dyn_response"])
        else:
            st.info("Upload any arbitrary document to dynamically extract structured entities without a predefined schema.")
