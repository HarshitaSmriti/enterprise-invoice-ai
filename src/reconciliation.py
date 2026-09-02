"""Reconciliation engine combining Model A, Model B, and structural OCR evidence."""

import re
import datetime as dt
from .config import CANONICAL_FIELDS, MONEY_FIELDS, DATE_RE
from .preprocessing import candidate_rows
from .field_extractor import canonical_field
from .gst_extractor import extract_gstins, extract_money_fields
from .line_items import reconstruct_line_items
from .utils import clean_text, norm_upper, normalize_date, normalize_money, money_float


def extract_customer_from_invoice_header(rows: list[dict]) -> str | None:
    """Extract customer/buyer name from invoice header, avoiding material-table leaks."""
    for row in rows[:20]:
        t = clean_text(row["text"])
        u = norm_upper(t)

        if "BILL NO" not in u:
            continue

        left = re.split(r"\bBILL\s*NO\b", t, flags=re.I)[0]
        left = re.sub(r"\bFORM\s+[A-Z0-9_-]+\b", "", left, flags=re.I)
        left = re.sub(r"\bREF\s*NO\b.*$", "", left, flags=re.I)
        left = clean_text(left)

        if left and len(left) >= 3:
            return left

    customer_labels = (
        "CUSTOMER NAME", "CUSTOMER", "BUYER", "BILL TO", "SOLD TO",
        "CONSIGNEE", "SHIP TO",
    )
    for row in rows[:30]:
        t = clean_text(row["text"])
        u = norm_upper(t)
        for label in customer_labels:
            pos = u.find(label)
            if pos >= 0:
                value = clean_text(t[pos + len(label):].lstrip(" :#-"))
                if value and len(value) > 2:
                    return value

    return None


def extract_invoice_number(rows: list[dict]) -> dict | None:
    """Extract invoice number from labeled patterns."""
    labels = [
        "BILL NO", "BILL NUMBER", "INVOICE NO",
        "INVOICE NUMBER", "INVOICE #", "INV NO",
    ]
    bad = ["REF NO", "REFERENCE NO", "FORM NO"]

    for row in rows:
        t = row["text"]
        u = t.upper()
        if any(b in u for b in bad):
            continue

        for label in labels:
            pos = u.find(label)
            if pos < 0:
                continue

            tail = t[pos + len(label):]
            tail = re.sub(r"^[\s:#\-./]+", " ", tail).strip()

            m = re.search(r"\b[A-Za-z0-9][A-Za-z0-9./_-]{2,30}\b", tail)
            if m:
                value = m.group(0).strip(" .,:;")
                if not DATE_RE.fullmatch(value):
                    return {
                        "value": value,
                        "raw": tail,
                        "box": [row["x1"], row["y_center"], row["x2"], row["y_center"]],
                    }

    return None


def extract_ref_no(rows: list[dict]) -> str | None:
    """Extract reference / PO number from header lines."""
    for pos, row in enumerate(rows[:30]):
        u = norm_upper(row["text"])
        if "REF NO" not in u and "REFERENCE NO" not in u:
            continue

        tail = re.sub(r".*?REF(?:ERENCE)?\s*NO\.?", "", row["text"], flags=re.I)
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9./_-]{0,15}", tail)
        for token in tokens:
            if DATE_RE.fullmatch(token):
                continue
            digits = re.sub(r"\D", "", token)
            if (digits and len(digits) <= 6) or (not digits and len(token) <= 6):
                return token

        for nxt in rows[pos + 1:pos + 4]:
            candidate = clean_text(nxt["text"]).lstrip(" :#-/.")
            tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9./_-]{0,15}", candidate)
            for token in tokens:
                if DATE_RE.fullmatch(token):
                    continue
                digits = re.sub(r"\D", "", token)
                if (digits and len(digits) <= 6) or (not digits and len(token) <= 6):
                    return token

    return None


def merge_page_fields(model_spans: list[dict], words: list[str], boxes: list[list[float]]) -> tuple[dict, dict, dict, list]:
    """Merge LayoutLMv3 predictions with structural OCR evidence for one page."""
    fields = {}
    metadata = {}

    for span in model_spans:
        field = canonical_field(span.get("label"))
        if field not in CANONICAL_FIELDS:
            continue

        value = clean_text(span.get("value"))
        if not value:
            continue

        candidate = {
            "value": value,
            "source": "layoutlm",
            "confidence": float(span.get("confidence", 0)),
            "raw": value,
            "bbox": span.get("box"),
        }

        old = metadata.get(field)
        if old is None or candidate["confidence"] > old["confidence"]:
            fields[field] = value
            metadata[field] = candidate

    rows = candidate_rows(words, boxes)

    # 1. CUSTOMER NAME
    customer = extract_customer_from_invoice_header(rows)
    if customer:
        fields["CUSTOMER_NAME"] = customer
        metadata["CUSTOMER_NAME"] = {
            "source": "ocr_invoice_header",
            "confidence": 1.0,
            "raw": customer,
        }

    # 2. INVOICE NUMBER
    inv = extract_invoice_number(rows)
    if inv:
        fields["INVOICE_NUMBER"] = inv["value"]
        metadata["INVOICE_NUMBER"] = {
            "source": "ocr_labeled_invoice_number",
            "confidence": 1.0,
            "raw": inv["raw"],
            "bbox": inv["box"],
        }

    # 3. REF NO
    ref = extract_ref_no(rows)
    if ref:
        fields["REF_NO"] = ref
        metadata["REF_NO"] = {
            "source": "ocr_labeled_ref_no",
            "confidence": 1.0,
            "raw": ref,
        }

    # 4. DATES
    date_values = []
    for row in rows[:30]:
        for d in DATE_RE.findall(row["text"]):
            normalized = normalize_date(d)
            if normalized:
                date_values.append((normalized, row["text"]))

    if date_values:
        fields["INVOICE_DATE"] = date_values[0][0]
        metadata["INVOICE_DATE"] = {
            "source": "ocr_first_header_date",
            "confidence": 0.98,
            "raw": date_values[0][1],
        }

    # Due date sanity
    if fields.get("DUE_DATE"):
        normalized_due = normalize_date(fields["DUE_DATE"])
        if normalized_due:
            fields["DUE_DATE"] = normalized_due
        else:
            fields.pop("DUE_DATE", None)
            metadata.pop("DUE_DATE", None)

    if fields.get("INVOICE_DATE") and fields.get("DUE_DATE"):
        try:
            inv_d = dt.datetime.strptime(fields["INVOICE_DATE"], "%Y-%m-%d").date()
            due_d = dt.datetime.strptime(fields["DUE_DATE"], "%Y-%m-%d").date()
            if due_d < inv_d:
                fields.pop("DUE_DATE", None)
                metadata.pop("DUE_DATE", None)
        except Exception:
            pass

    # 5. FINANCIAL FIELDS
    money, money_diag = extract_money_fields(words, boxes)
    for field, candidate in money.items():
        if candidate.get("value") is not None:
            fields[field] = candidate["value"]
            metadata[field] = {
                "source": "ocr_labeled_financial_row",
                "confidence": 1.0,
                "raw": candidate.get("raw"),
                "keyword": candidate.get("keyword"),
                "row_text": candidate.get("row_text"),
            }

    # 6. GSTIN
    gstins = extract_gstins(words, boxes)
    if gstins:
        fields["GSTIN"] = gstins[0]["value"]
        metadata["GSTIN"] = {
            "source": "ocr_gstin_regex",
            "confidence": 1.0,
            "raw": gstins[0]["raw"],
        }

    for field in MONEY_FIELDS:
        if fields.get(field) is not None:
            fields[field] = normalize_money(fields[field])

    return fields, metadata, money_diag, gstins


def merge_page_results(page_results: list[dict]) -> dict:
    """Aggregate multi-page extraction results into a unified canonical document structure."""
    final_fields = {}
    final_metadata = {}
    all_items = []
    page_diagnostics = []

    for page_res in page_results:
        page_diagnostics.append(page_res["diagnostics"])
        all_items.extend(page_res["line_items"])

        for field, value in page_res["fields"].items():
            if value is None or value == "":
                continue

            candidate = page_res["metadata"].get(field, {
                "source": "page",
                "confidence": 0.5,
                "raw": value,
            })
            old = final_metadata.get(field)

            explicit = str(candidate.get("source", "")).startswith("ocr_")
            old_explicit = old is not None and str(old.get("source", "")).startswith("ocr_")

            if (
                old is None
                or (explicit and not old_explicit)
                or candidate.get("confidence", 0) > old.get("confidence", 0)
            ):
                final_fields[field] = value
                final_metadata[field] = candidate

    # Cross-page line-item deduplication
    deduped = []
    for item in all_items:
        sig = (
            re.sub(r"\W+", "", item.get("description", "").upper()),
            item.get("quantity", ""),
            item.get("unit_price", ""),
            item.get("amount", ""),
        )
        if not any(
            sig[0] == re.sub(r"\W+", "", old.get("description", "").upper())
            and sig[1:] == (old.get("quantity", ""), old.get("unit_price", ""), old.get("amount", ""))
            for old in deduped
        ):
            deduped.append(item)

    # Reconcile taxes
    cgst = money_float(final_fields.get("CENTRAL_GST"))
    sgst = money_float(final_fields.get("STATE_GST"))
    if cgst is not None and sgst is not None:
        final_fields["TAX"] = f"{cgst + sgst:.2f}"
        final_metadata["TAX"] = {
            "source": "derived_cgst_plus_sgst",
            "confidence": 1.0,
            "raw": f"{cgst:.2f} + {sgst:.2f}",
        }

    warnings = []
    subtotal = money_float(final_fields.get("SUBTOTAL"))
    tax = money_float(final_fields.get("TAX"))
    total = money_float(final_fields.get("TOTAL_AMOUNT"))

    if subtotal is not None and tax is not None and total is not None:
        diff = abs((subtotal + tax) - total)
        if diff > 1.0:
            warnings.append(
                f"Subtotal ({subtotal:.2f}) + Tax ({tax:.2f}) does not match Total ({total:.2f}) (diff={diff:.2f})."
            )

    if not deduped:
        warnings.append("No line items reconstructed from table.")

    global_line_sum = round(sum(money_float(i.get("amount")) or 0.0 for i in deduped), 2)
    line_sum_matches_subtotal = None
    if subtotal is not None and deduped:
        line_sum_matches_subtotal = abs(global_line_sum - subtotal) <= 0.05
        if not line_sum_matches_subtotal:
            warnings.append(
                f"Line item amount sum ({global_line_sum:.2f}) does not match SUBTOTAL ({subtotal:.2f})."
            )

    return {
        "fields": final_fields,
        "field_metadata": final_metadata,
        "line_items": deduped,
        "diagnostics": {
            "pages": len(page_results),
            "page_diagnostics": page_diagnostics,
            "warnings": warnings,
            "global_line_item_amount_sum": f"{global_line_sum:.2f}",
            "global_line_sum_matches_subtotal": line_sum_matches_subtotal,
        },
    }
