"""Adaptive Dynamic Form Component for Streamlit.

Consumes generic document JSON and dynamically renders appropriate editable
inputs (text, number, date) and interactive table editors (st.data_editor).
"""

from typing import Any, Dict, List
import pandas as pd
import streamlit as st


def render_adaptive_form(doc_json: Dict[str, Any], form_key: str = "adaptive_doc_form") -> Dict[str, Any]:
    """Render adaptive form controls driven completely by extracted document JSON."""
    if not doc_json:
        st.info("No document data available to generate form.")
        return doc_json

    edited_json = {
        "document_type": doc_json.get("document_type", "general_document"),
        "fields": [],
        "tables": [],
        "metadata": doc_json.get("metadata", {})
    }

    doc_type = doc_json.get("document_type", "document").replace("_", " ").title()
    fields = doc_json.get("fields", [])
    tables = doc_json.get("tables", [])

    st.markdown(f"### 📝 Adaptive Form: **{doc_type}**")
    st.caption("Fields and tables below were generated dynamically from visible document content.")

    # -------------------------------------------------------------
    # 1. Dynamic Key-Value Fields Section
    # -------------------------------------------------------------
    if fields:
        st.subheader("Extracted Document Attributes")
        
        # Split fields into 2 balanced columns
        cols = st.columns(2)
        
        for idx, field in enumerate(fields):
            col_target = cols[idx % 2]
            name = field.get("name", f"field_{idx}")
            label = field.get("label") or name.replace("_", " ").title()
            val = str(field.get("value", "") or "")
            conf = field.get("confidence", 1.0)
            f_key = f"{form_key}_fld_{idx}_{name}"

            with col_target:
                # Determine best input type
                is_numeric = False
                clean_num = val.replace("$", "").replace("€", "").replace("£", "").replace("₹", "").replace(",", "").strip()
                try:
                    num_val = float(clean_num)
                    is_numeric = True
                except ValueError:
                    is_numeric = False

                if is_numeric and len(clean_num) <= 10 and not name.endswith(("_id", "_no", "_number", "_code", "_phone")):
                    if "." in clean_num:
                        new_val = st.number_input(
                            f"{label} (Confidence: {conf*100:.0f}%)",
                            value=float(num_val),
                            format="%.2f",
                            key=f_key,
                        )
                    else:
                        new_val = st.number_input(
                            f"{label} (Confidence: {conf*100:.0f}%)",
                            value=int(num_val),
                            step=1,
                            key=f_key,
                        )
                    new_val_str = str(new_val)
                elif len(val) > 80:
                    new_val_str = st.text_area(
                        f"{label} (Confidence: {conf*100:.0f}%)",
                        value=val,
                        key=f_key,
                    )
                else:
                    new_val_str = st.text_input(
                        f"{label} (Confidence: {conf*100:.0f}%)",
                        value=val,
                        key=f_key,
                    )

                edited_field = dict(field)
                edited_field["value"] = new_val_str
                edited_json["fields"].append(edited_field)

    # -------------------------------------------------------------
    # 2. Dynamic Tables Section
    # -------------------------------------------------------------
    if tables:
        st.divider()
        st.subheader(f"Extracted Tables ({len(tables)})")

        for t_idx, table in enumerate(tables):
            title = table.get("title", f"Table {t_idx + 1}")
            rows = table.get("rows", [])
            cols_list = table.get("columns", [])

            st.markdown(f"#### 📊 {title}")
            if rows:
                df = pd.DataFrame(rows)
                # Ensure columns order
                if cols_list:
                    valid_cols = [c for c in cols_list if c in df.columns]
                    other_cols = [c for c in df.columns if c not in valid_cols]
                    df = df[valid_cols + other_cols]

                edited_df = st.data_editor(
                    df,
                    use_container_width=True,
                    num_rows="dynamic",
                    key=f"{form_key}_table_{t_idx}",
                )
                
                edited_table = dict(table)
                edited_table["rows"] = edited_df.to_dict(orient="records")
                edited_json["tables"].append(edited_table)
            else:
                st.info("Table detected with no readable rows.")

    return edited_json
