"""Dynamic Form Component for Streamlit.

Renders editable form fields dynamically based on extracted JSON data.
"""

from typing import Any, Dict
import streamlit as st


def render_dynamic_form(data: Dict[str, Any], key_prefix: str = "dyn_field") -> Dict[str, Any]:
    """Recursively render editable Streamlit input components for dynamic JSON key-values."""
    edited_data = {}

    if not data or not isinstance(data, dict):
        st.info("No structured data to display in form.")
        return edited_data

    for key, value in data.items():
        if key.startswith("_"):
            continue

        unique_key = f"{key_prefix}_{key}"
        label = key.replace("_", " ").title()

        if isinstance(value, dict):
            with st.expander(f"📁 {label}", expanded=True):
                edited_data[key] = render_dynamic_form(value, key_prefix=unique_key)

        elif isinstance(value, list):
            with st.expander(f"📋 {label} ({len(value)} items)", expanded=True):
                edited_list = []
                for idx, item in enumerate(value):
                    if isinstance(item, dict):
                        st.markdown(f"**Item {idx + 1}**")
                        edited_item = render_dynamic_form(item, key_prefix=f"{unique_key}_{idx}")
                        edited_list.append(edited_item)
                        st.divider()
                    else:
                        item_val = st.text_input(
                            f"Item {idx + 1}",
                            value=str(item) if item is not None else "",
                            key=f"{unique_key}_{idx}",
                        )
                        edited_list.append(item_val)
                edited_data[key] = edited_list

        elif isinstance(value, bool):
            edited_data[key] = st.checkbox(label, value=value, key=unique_key)

        elif isinstance(value, (int, float)):
            if isinstance(value, int):
                edited_data[key] = st.number_input(label, value=int(value), step=1, key=unique_key)
            else:
                edited_data[key] = st.number_input(
                    label, value=float(value), step=0.01, format="%.2f", key=unique_key
                )

        else:
            str_val = str(value) if value is not None else ""
            if len(str_val) > 80:
                edited_data[key] = st.text_area(label, value=str_val, key=unique_key)
            else:
                edited_data[key] = st.text_input(label, value=str_val, key=unique_key)

    return edited_data
