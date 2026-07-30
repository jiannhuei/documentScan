"""
=============================================================================
STAGE 1 (Expanded): Azure AI Document Intelligence - Full Context & Tables
=============================================================================
"""

import os
import time
import json
import re
import statistics
from difflib import SequenceMatcher
import pandas as pd
import cv2
import numpy as np
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
ENDPOINT = ""
KEY = ""
INPUT_DOCUMENT = "sample/Test5.pdf"
FIELD_KEYWORD_CONFIG = "DocIntResult/Test5.pdf_field_keywords.json"
OUTPUT_MARKDOWN = f"PaddleLocalResult/{os.path.basename(INPUT_DOCUMENT)}_full_document_context.md"
OUTPUT_EXCEL = f"PaddleLocalResult/{os.path.basename(INPUT_DOCUMENT)}_extracted_tables.xlsx"
OUTPUT_KEYWORD_JSON = f"PaddleLocalResult/{os.path.basename(INPUT_DOCUMENT)}_result.json"
CHECKBOX_TEMPLATE_CONFIG = os.environ.get("OCR_CHECKBOX_TEMPLATE_CONFIG", "DocIntResult/checkbox_template.json")
STRICT_CONFIDENCE_THRESHOLD = float(os.environ.get("OCR_STRICT_CONFIDENCE", "0.8"))
DEBUG_TEMPLATE_OVERLAY = os.environ.get("OCR_DEBUG_TEMPLATE_OVERLAY", "1") == "1"
DEBUG_TEMPLATE_OVERLAY_DIR = os.environ.get("OCR_DEBUG_TEMPLATE_OVERLAY_DIR", "DocIntResult/template_debug")

# Optional fixed checkbox template for this form (normalized coordinates).
# Coordinates are approximate and can be tuned if the source form layout shifts.
_CHECKBOX_TEMPLATE_CACHE = None

# Skip these fields from template-based extraction; rely on image detection and selection marks instead
TEMPLATE_SKIP_FIELDS = {"IsPoliticallyExposedPerson", "FatcaDocumentType", "PbCardAccountNumber", "OwnsZeroOrOneCreditCard", "OwnsTwoOrMoreCreditCards"}


def _resolve_field_config_path(config_path: str):
    candidates = []
    if config_path:
        candidates.append(config_path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in list(candidates):
        candidates.append(os.path.join(script_dir, candidate))

    seen = set()
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.exists(normalized):
            return normalized

    return os.path.normpath(config_path)


def _resolve_optional_config_path(config_path: str, fallback_name: str):
    candidates = []
    if config_path:
        candidates.append(config_path)
        candidates.append(os.path.basename(config_path))
    if fallback_name:
        candidates.append(fallback_name)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in list(candidates):
        candidates.append(os.path.join(script_dir, candidate))

    seen = set()
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.exists(normalized):
            return normalized
    return None


def _get_checkbox_templates():
    global _CHECKBOX_TEMPLATE_CACHE
    if _CHECKBOX_TEMPLATE_CACHE is not None:
        return _CHECKBOX_TEMPLATE_CACHE

    path = _resolve_optional_config_path(CHECKBOX_TEMPLATE_CONFIG, "checkbox_template.json")
    if not path:
        return _CHECKBOX_TEMPLATE_CACHE

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        templates = payload.get("templates") if isinstance(payload, dict) and "templates" in payload else payload
        if isinstance(templates, dict):
            _CHECKBOX_TEMPLATE_CACHE = templates
            return _CHECKBOX_TEMPLATE_CACHE
    except Exception:
        pass

    return _CHECKBOX_TEMPLATE_CACHE


def _normalize_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _clean_value(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _label_candidate_variants(line: str):
    base = _clean_value(line)
    if not base:
        return []

    variants = [base]
    stripped = re.sub(r"^[-*•\s]+", "", base)
    stripped = re.sub(r"\[[^\]]*\]", "", stripped)
    stripped = _clean_value(stripped)
    if stripped and stripped not in variants:
        variants.append(stripped)

    before_colon = _clean_value(base.split(":", 1)[0])
    if before_colon and before_colon not in variants:
        variants.append(before_colon)

    return variants


def _first_selected_checkbox_option(line: str):
    upper = str(line or "").upper()
    selected = r"[X/✓✔☑☒]"
    yes_selected = bool(
        re.search(rf"\[{selected}\]\s*YES", upper) or re.search(rf"YES\s*\[{selected}\]", upper)
    )
    no_selected = bool(
        re.search(rf"\[{selected}\]\s*NO", upper) or re.search(rf"NO\s*\[{selected}\]", upper)
    )

    if yes_selected and not no_selected:
        return "YES"
    if no_selected and not yes_selected:
        return "NO"
    if yes_selected and no_selected:
        return "BOTH"
    return ""


def _is_checked_statement(line: str):
    return bool(re.search(r"^\s*[-*•]?\s*\[[Xx/✓✔☑☒]\]", str(line or "")))


def _extract_percent_value(line: str):
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", str(line or ""))
    if not match:
        return ""
    return f"{match.group(1)}%"


def _match_keyword_from_line(line: str, field_cfg, context_text: str = ""):
    for candidate in _label_candidate_variants(line):
        keyword = _match_keyword(candidate, field_cfg, context_text)
        if keyword:
            return keyword
    return None


def _next_value_line(lines, start_idx, field_cfg):
    for j in range(start_idx, min(len(lines), start_idx + 4)):
        candidate = _clean_value(lines[j])
        if not candidate:
            continue
        if re.match(r"^\d+[.)]\s", candidate):
            continue
        if _match_keyword_from_line(candidate, field_cfg, candidate):
            continue
        if ":" in candidate and len(candidate.split(":", 1)[0]) < len(candidate):
            continue
        if not _is_informative_value(candidate):
            continue
        return candidate
    return ""


def _is_informative_value(value: str):
    cleaned = _clean_value(value)
    if not cleaned:
        return False

    if cleaned.startswith("#"):
        return False

    # Ignore placeholder or selection glyph-only values.
    normalized = _normalize_label(cleaned)
    if normalized in {"", "yesno"}:
        return False

    if re.fullmatch(r"[\[\](){}\-_/\\|.\s☐☑☒]+", cleaned):
        return False

    return True


def _extract_date_value(text: str):
    candidate = _clean_value(text)
    if not candidate:
        return ""

    numeric = re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", candidate)
    if numeric:
        return numeric.group(0)

    month_name = re.search(
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
        r"January|February|March|April|June|July|August|September|October|November|December)\s+\d{2,4}\b",
        candidate,
        flags=re.IGNORECASE,
    )
    if month_name:
        return month_name.group(0)

    return ""


def _polygon_to_bbox(polygon):
    if not polygon:
        return None
    xs = []
    ys = []
    for point in polygon:
        try:
            xs.append(float(point.x))
            ys.append(float(point.y))
        except Exception:
            try:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
            except Exception:
                continue
    if not xs or not ys:
        return None
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return {
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "cx": (x0 + x1) / 2.0,
        "cy": (y0 + y1) / 2.0,
        "h": max(0.001, y1 - y0),
    }


def _render_pdf_pages_as_images(pdf_path: str, scale: float = 2.0):
    try:
        import pypdfium2 as pdfium
    except Exception:
        return []

    if not os.path.exists(pdf_path) or not pdf_path.lower().endswith(".pdf"):
        return []

    images = []
    pdf = None
    try:
        pdf = pdfium.PdfDocument(pdf_path)
        for idx in range(len(pdf)):
            page = pdf[idx]
            try:
                bitmap = page.render(scale=scale)
                pil_image = bitmap.to_pil()
                images.append(pil_image)
            finally:
                close_page = getattr(page, "close", None)
                if callable(close_page):
                    close_page()
    except Exception:
        return []
    finally:
        if pdf is not None:
            close_pdf = getattr(pdf, "close", None)
            if callable(close_pdf):
                close_pdf()

    return images


def _detect_checked_box_centers_norm(pil_image):
    if pil_image is None:
        return []

    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = gray.shape[:2]
    checked = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 8 or h < 8 or w > 70 or h > 70:
            continue

        ratio = w / float(max(h, 1))
        if ratio < 0.75 or ratio > 1.35:
            continue

        roi = thresh[y:y + h, x:x + w]
        if roi.size == 0:
            continue

        fill_ratio = cv2.countNonZero(roi) / float(w * h)
        if fill_ratio < 0.18:
            continue

        # Keep likely checked boxes, avoid large dark blobs.
        if fill_ratio > 0.78:
            continue

        checked.append(((x + w / 2.0) / w_img, (y + h / 2.0) / h_img))

    # De-duplicate close centers.
    deduped = []
    for cx, cy in checked:
        if any(abs(cx - ox) < 0.01 and abs(cy - oy) < 0.01 for ox, oy in deduped):
            continue
        deduped.append((cx, cy))

    return deduped


def _is_template_box_checked(pil_image, coord_x: float, coord_y: float, box_size_norm: float = 0.026):
    if pil_image is None:
        return False

    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    h_img, w_img = gray.shape[:2]
    box_px = int(max(10, min(h_img, w_img) * box_size_norm))

    cx = int(coord_x * w_img)
    cy = int(coord_y * h_img)
    x0 = max(0, cx - box_px // 2)
    x1 = min(w_img, cx + box_px // 2)
    y0 = max(0, cy - box_px // 2)
    y1 = min(h_img, cy + box_px // 2)
    if x1 <= x0 or y1 <= y0:
        return False

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return False

    # Inner area excludes border lines of empty boxes.
    margin = max(1, int(min(roi.shape) * 0.22))
    inner = roi[margin: roi.shape[0] - margin, margin: roi.shape[1] - margin]
    if inner.size == 0:
        inner = roi

    bin_inv = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ink_ratio = cv2.countNonZero(bin_inv) / float(bin_inv.size)
    return ink_ratio >= 0.08


def _extract_checkbox_values_from_fixed_template(result: AnalyzeResult, doc_path: str):
    values = {}
    templates = _get_checkbox_templates()
    page_images = _render_pdf_pages_as_images(doc_path)
    if not page_images:
        return values

    if DEBUG_TEMPLATE_OVERLAY:
        _export_checkbox_template_debug_images(page_images, templates, DEBUG_TEMPLATE_OVERLAY_DIR)

    pages = getattr(result, "pages", []) or []
    if not pages:
        return values

    for page_idx, _page in enumerate(pages):
        page_key = f"page{page_idx}"
        template = templates.get(page_key)
        if not template or page_idx >= len(page_images):
            continue

        page_img = page_images[page_idx]
        for keyword, spec in template.items():
            if keyword in TEMPLATE_SKIP_FIELDS:
                continue
            
            box_size = float(spec.get("box_size", 0.026))
            kind = spec.get("type")

            if kind == "single":
                coord = spec.get("coord")
                if coord and _is_template_box_checked(page_img, coord[0], coord[1], box_size):
                    values[keyword] = spec.get("value", "YES")

            elif kind == "yesno":
                yes_coord = spec.get("yes")
                no_coord = spec.get("no")
                yes_checked = bool(yes_coord and _is_template_box_checked(page_img, yes_coord[0], yes_coord[1], box_size))
                no_checked = bool(no_coord and _is_template_box_checked(page_img, no_coord[0], no_coord[1], box_size))
                if yes_checked and not no_checked:
                    values[keyword] = "YES"
                elif no_checked and not yes_checked:
                    values[keyword] = "NO"
                elif yes_checked and no_checked:
                    values[keyword] = "BOTH"

            elif kind == "option":
                for option in spec.get("options", []):
                    coord = option.get("coord")
                    if coord and _is_template_box_checked(page_img, coord[0], coord[1], box_size):
                        values[keyword] = option.get("label", "")
                        break

    return values


def _export_checkbox_template_debug_images(page_images, templates, output_dir: str):
    if not page_images or not templates:
        return

    os.makedirs(output_dir, exist_ok=True)

    for page_idx, page_img in enumerate(page_images):
        page_key = f"page{page_idx}"
        template = templates.get(page_key)
        if not template:
            continue

        canvas = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2BGR)
        h_img, w_img = canvas.shape[:2]

        for keyword, spec in template.items():
            if keyword in TEMPLATE_SKIP_FIELDS:
                continue
            
            box_size = float(spec.get("box_size", 0.026))
            kind = spec.get("type")

            entries = []
            if kind == "single":
                coord = spec.get("coord")
                if coord:
                    entries.append((coord[0], coord[1], spec.get("value", "YES")))
            elif kind == "yesno":
                yes_coord = spec.get("yes")
                no_coord = spec.get("no")
                if yes_coord:
                    entries.append((yes_coord[0], yes_coord[1], "YES"))
                if no_coord:
                    entries.append((no_coord[0], no_coord[1], "NO"))
            elif kind == "option":
                for option in spec.get("options", []):
                    coord = option.get("coord")
                    if coord:
                        entries.append((coord[0], coord[1], option.get("label", "")))

            for x_norm, y_norm, label in entries:
                checked = _is_template_box_checked(page_img, x_norm, y_norm, box_size)
                box_px = int(max(10, min(h_img, w_img) * box_size))
                cx = int(x_norm * w_img)
                cy = int(y_norm * h_img)
                x0 = max(0, cx - box_px // 2)
                y0 = max(0, cy - box_px // 2)
                x1 = min(w_img - 1, cx + box_px // 2)
                y1 = min(h_img - 1, cy + box_px // 2)

                color = (0, 180, 0) if checked else (0, 0, 220)
                status = "checked" if checked else "unchecked"
                text = f"{keyword}:{label}:{status}" if label else f"{keyword}:{status}"

                cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
                cv2.putText(
                    canvas,
                    text,
                    (max(0, x0 - 2), max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        out_path = os.path.join(output_dir, f"page_{page_idx + 1}_template_overlay.png")
        cv2.imwrite(out_path, canvas)


def _build_normalized_page_lines_from_words(page):
    page_w = float(getattr(page, "width", 1.0) or 1.0)
    page_h = float(getattr(page, "height", 1.0) or 1.0)

    words = []
    for word in getattr(page, "words", []) or []:
        bbox = _polygon_to_bbox(getattr(word, "polygon", None))
        if not bbox:
            continue
        content = _clean_value(getattr(word, "content", ""))
        if not content:
            continue
        words.append(
            {
                "text": content,
                "x": bbox["x0"] / page_w,
                "cy": bbox["cy"] / page_h,
                "h": bbox["h"] / page_h,
            }
        )

    if not words:
        return []

    words.sort(key=lambda w: (w["cy"], w["x"]))
    median_h = statistics.median([w["h"] for w in words]) if words else 0.01
    y_tol = max(0.005, median_h * 0.7)

    lines = []
    for word in words:
        if not lines or abs(word["cy"] - lines[-1]["cy"]) > y_tol:
            lines.append({"cy": word["cy"], "words": [word]})
        else:
            lines[-1]["words"].append(word)
            row = lines[-1]
            row["cy"] = (row["cy"] * (len(row["words"]) - 1) + word["cy"]) / len(row["words"])

    for line in lines:
        line["words"].sort(key=lambda w: w["x"])
        line["text"] = " ".join(w["text"] for w in line["words"])

    return lines


def _extract_checkbox_values_from_image(result: AnalyzeResult, doc_path: str):
    values = {}
    page_images = _render_pdf_pages_as_images(doc_path)
    if not page_images:
        return values

    pages = getattr(result, "pages", []) or []
    for page_idx, page in enumerate(pages):
        if page_idx >= len(page_images):
            break

        checked_centers = _detect_checked_box_centers_norm(page_images[page_idx])
        if not checked_centers:
            continue

        lines = _build_normalized_page_lines_from_words(page)
        if not lines:
            continue

        for cx, cy in checked_centers:
            nearest_idx = min(range(len(lines)), key=lambda i: abs(lines[i]["cy"] - cy))
            if abs(lines[nearest_idx]["cy"] - cy) > 0.05:
                continue

            line = lines[nearest_idx]
            right_words = [w["text"] for w in line["words"] if w["x"] >= cx - 0.02]
            left_words = [w["text"] for w in line["words"] if w["x"] < cx - 0.02]
            local_text = " ".join(left_words[-6:] + right_words[:8])
            local_upper = local_text.upper()

            context_lines = lines[max(0, nearest_idx - 2): min(len(lines), nearest_idx + 3)]
            context_text = " ".join(x.get("text", "") for x in context_lines)
            context_upper = context_text.upper()

            yes_no = ""
            if "YES" in local_upper:
                yes_no = "YES"
            elif "NO" in local_upper:
                yes_no = "NO"

            if "PREFERENCE TO RECEIVE E-INVOICE" in context_upper and yes_no:
                values["EInvoicePreference"] = yes_no

            if "RESIDENT FOR PURPOSES IN ANY COUNTRY OTHER THAN MALAYSIA" in context_upper and yes_no:
                values["TaxResidentOutsideMalaysia"] = yes_no

            if "POLITICALLY EXPOSED PERSON" in context_upper:
                values["IsPoliticallyExposedPerson"] = "YES"

            if "PREFERENCE TO RECEIVE" in context_upper and "MARKETING" in context_upper and yes_no:
                values["MarketingPreference"] = yes_no

            if "OPT-IN" in context_upper and "CNP" in context_upper:
                values["OptInCnpOverseas"] = "YES"

            if "TWO (2) OR MORE CREDIT CARD ISSUERS" in context_upper:
                values["OwnsTwoOrMoreCreditCards"] = "YES"

            if "ONLY ONE (1) CREDIT CARD" in context_upper:
                values["OwnsZeroOrOneCreditCard"] = "YES"

            if "FORM W-9" in context_upper:
                values["FatcaDocumentType"] = "FORM W-9"

            if "FORM W-8BEN" in context_upper:
                values["FatcaDocumentType"] = "FORM W-8BEN"

    return values


def _build_page_lines_from_words(page):
    words = []
    for word in getattr(page, "words", []) or []:
        bbox = _polygon_to_bbox(getattr(word, "polygon", None))
        if not bbox:
            continue
        content = _clean_value(getattr(word, "content", ""))
        if not content:
            continue
        words.append({"text": content, "x": bbox["x0"], "cy": bbox["cy"], "h": bbox["h"]})

    if not words:
        return []

    words.sort(key=lambda w: (w["cy"], w["x"]))
    median_h = statistics.median([w["h"] for w in words]) if words else 0.01
    y_tol = max(0.01, median_h * 0.6)

    lines = []
    for word in words:
        if not lines or abs(word["cy"] - lines[-1]["cy"]) > y_tol:
            lines.append({"cy": word["cy"], "words": [word]})
        else:
            lines[-1]["words"].append(word)
            row = lines[-1]
            row["cy"] = (row["cy"] * (len(row["words"]) - 1) + word["cy"]) / len(row["words"])

    for line in lines:
        line["words"].sort(key=lambda w: w["x"])
        line["text"] = " ".join(w["text"] for w in line["words"])

    return lines


def _extract_selection_mark_values(result: AnalyzeResult):
    values = {}

    for page in getattr(result, "pages", []) or []:
        lines = _build_page_lines_from_words(page)
        if not lines:
            continue

        for mark in getattr(page, "selection_marks", []) or []:
            state = str(getattr(mark, "state", "")).lower()
            if state != "selected":
                continue

            mark_box = _polygon_to_bbox(getattr(mark, "polygon", None))
            if not mark_box:
                continue

            nearest_idx = min(range(len(lines)), key=lambda i: abs(lines[i]["cy"] - mark_box["cy"]))
            line = lines[nearest_idx]
            line_text = line.get("text", "")

            right_words = [w["text"] for w in line["words"] if w["x"] >= mark_box["cx"] - 0.02]
            left_words = [w["text"] for w in line["words"] if w["x"] < mark_box["cx"] - 0.02]
            local_text = " ".join(left_words[-6:] + right_words[:8])
            local_upper = local_text.upper()

            context_lines = lines[max(0, nearest_idx - 2): min(len(lines), nearest_idx + 3)]
            context_text = " ".join(x.get("text", "") for x in context_lines)
            context_upper = context_text.upper()

            mark_value = ""
            if "YES" in local_upper:
                mark_value = "YES"
            elif "NO" in local_upper:
                mark_value = "NO"

            if "PREFERENCE TO RECEIVE E-INVOICE" in context_upper and mark_value:
                values["EInvoicePreference"] = mark_value

            if "RESIDENT FOR PURPOSES IN ANY COUNTRY OTHER THAN MALAYSIA" in context_upper and mark_value:
                values["TaxResidentOutsideMalaysia"] = mark_value

            if "POLITICALLY EXPOSED PERSON" in context_upper and "YES" in context_upper:
                values["IsPoliticallyExposedPerson"] = "YES"

            if "PREFERENCE TO RECEIVE" in context_upper and "MARKETING" in context_upper and mark_value:
                values["MarketingPreference"] = mark_value

            if "OPT-IN" in context_upper and "CNP" in context_upper:
                values["OptInCnpOverseas"] = "YES"

            if "TWO (2) OR MORE CREDIT CARD ISSUERS" in context_upper:
                values["OwnsTwoOrMoreCreditCards"] = "YES"

            if "ONLY ONE (1) CREDIT CARD" in context_upper:
                values["OwnsZeroOrOneCreditCard"] = "YES"

            if "FORM W-9" in context_upper:
                values["FatcaDocumentType"] = "FORM W-9"

            if "FORM W-8BEN" in context_upper:
                values["FatcaDocumentType"] = "FORM W-8BEN"

    return values


def _selected_marker_pattern():
    return r"[Xx/✓✔☑☒]"


def _detect_yes_no_from_text(text: str):
    upper = str(text or "").upper()
    selected = _selected_marker_pattern()
    yes_selected = bool(
        re.search(rf"\[{selected}\]\s*YES", upper) or re.search(rf"YES\s*\[{selected}\]", upper)
    )
    no_selected = bool(
        re.search(rf"\[{selected}\]\s*NO", upper) or re.search(rf"NO\s*\[{selected}\]", upper)
    )
    if yes_selected and not no_selected:
        return "YES"
    if no_selected and not yes_selected:
        return "NO"
    if yes_selected and no_selected:
        return "BOTH"
    return ""


def _find_section_ranges(lines):
    section_starts = []
    for idx, line in enumerate(lines):
        match = re.match(r"^\s*#?\s*(\d{1,2})[.)\s-]", str(line or ""))
        if match:
            section_starts.append((int(match.group(1)), idx))

    ranges = {}
    for i, (section_no, start_idx) in enumerate(section_starts):
        end_idx = section_starts[i + 1][1] if i + 1 < len(section_starts) else len(lines)
        ranges[section_no] = (start_idx, end_idx)
    return ranges


def _section_text(lines, ranges, section_no):
    if section_no not in ranges:
        return ""
    start, end = ranges[section_no]
    return "\n".join(lines[start:end])


def _extract_section_specific_values(lines):
    values = {}
    ranges = _find_section_ranges(lines)
    
    # Special handling for credit card checkboxes that may not be in a numbered section
    # Search the entire content for these patterns
    all_text = "\n".join(lines)
    section8_lines = all_text.splitlines()
    for idx, line in enumerate(section8_lines):
        # Find ☒ symbol and check next few lines for credit card text
        if "☒" in line:
            # Look ahead in next few lines for the field text
            upcoming = " ".join(section8_lines[idx:min(idx+3, len(section8_lines))]).upper()
            if "ONLY ONE" in upcoming and "CREDIT CARD" in upcoming:
                values["OwnsZeroOrOneCreditCard"] = "YES"
            elif "TWO" in upcoming and "2" in upcoming and "MORE" in upcoming and "CREDIT" in upcoming:
                values["OwnsTwoOrMoreCreditCards"] = "YES"

    section6 = _section_text(lines, ranges, 6)
    if section6:
        e_invoice = _detect_yes_no_from_text(section6)
        if e_invoice:
            values["EInvoicePreference"] = e_invoice

    section7 = _section_text(lines, ranges, 7)
    if section7:
        tax_resident = _detect_yes_no_from_text(section7)
        if tax_resident:
            values["TaxResidentOutsideMalaysia"] = tax_resident

        if "FORM W-9" in section7.upper() and re.search(
            rf"\[{_selected_marker_pattern()}\]\s*FORM\s*W-?9|FORM\s*W-?9\s*\[{_selected_marker_pattern()}\]",
            section7,
            flags=re.IGNORECASE,
        ):
            values["FatcaDocumentType"] = "FORM W-9"

        if "FORM W-8BEN" in section7.upper() and re.search(
            rf"\[{_selected_marker_pattern()}\]\s*FORM\s*W-?8BEN|FORM\s*W-?8BEN\s*\[{_selected_marker_pattern()}\]",
            section7,
            flags=re.IGNORECASE,
        ):
            values["FatcaDocumentType"] = "FORM W-8BEN"

        date_match = re.search(
            r"DATE\s+FULL\s+DOCUMENTS\s+FURNISHED[^\n]*?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            section7,
            flags=re.IGNORECASE,
        )
        if date_match:
            values["FatcaDocumentsFurnishedDate"] = _clean_value(date_match.group(1))

    section8 = _section_text(lines, ranges, 8)
    if section8:
        if "POLITICALLY EXPOSED PERSON" in section8.upper():
            pep = _detect_yes_no_from_text(section8)
            if pep:
                values["IsPoliticallyExposedPerson"] = pep

        if "PREFERENCE TO RECEIVE" in section8.upper() and "YES" in section8.upper() and "NO" in section8.upper():
            marketing = _detect_yes_no_from_text(section8)
            if marketing:
                values["MarketingPreference"] = marketing

        for line in section8.splitlines():
            if "TWO (2) OR MORE CREDIT CARD ISSUERS" in line.upper() and _is_checked_statement(line):
                values["OwnsTwoOrMoreCreditCards"] = "YES"
            if "ONLY ONE (1) CREDIT CARD" in line.upper() and _is_checked_statement(line):
                values["OwnsZeroOrOneCreditCard"] = "YES"
            
            # If "two or more" line has NO checkbox, mark as empty to avoid false positive
            if "TWO (2) OR MORE CREDIT CARD ISSUERS" in line.upper() and not _is_checked_statement(line):
                values["OwnsTwoOrMoreCreditCards"] = ""
            
            # If "only one" line has NO checkbox, mark as empty  
            if "ONLY ONE (1) CREDIT CARD" in line.upper() and not _is_checked_statement(line):
                values["OwnsZeroOrOneCreditCard"] = ""

        # Handle credit card ownership checkboxes with symbol-based detection
        # Look for ☒ symbol and match to the next non-empty line with credit card text
        section8_lines = section8.splitlines()
        for idx, line in enumerate(section8_lines):
            # Find ☒ symbol and check next few lines for credit card text
            if "☒" in line:
                # Look ahead in next few lines for the field text
                upcoming = " ".join(section8_lines[idx:min(idx+3, len(section8_lines))]).upper()
                if "ONLY ONE" in upcoming and "CREDIT CARD" in upcoming:
                    values["OwnsZeroOrOneCreditCard"] = "YES"
                elif "TWO" in upcoming and "2" in upcoming and "MORE" in upcoming and "CREDIT" in upcoming:
                    values["OwnsTwoOrMoreCreditCards"] = "YES"

        principal_match = re.search(
            r"PRINCIPAL\s+CARD\s+APPLICANT[^\n]*DATE[^\n]*?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{2,4})",
            section8,
            flags=re.IGNORECASE,
        )
        if principal_match:
            values["PrincipalApplicantSignatureDate"] = _clean_value(principal_match.group(1))

        supplementary_match = re.search(
            r"SUPPLEMENTARY\s+CAR[DO]\s+APPLICANT[^\n]*DATE[^\n]*?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{2,4})",
            section8,
            flags=re.IGNORECASE,
        )
        if supplementary_match:
            values["SupplementaryApplicantSignatureDate"] = _clean_value(supplementary_match.group(1))

    section9 = _section_text(lines, ranges, 9)
    if section9:
        verifier_date = _extract_date_value(section9)
        if verifier_date:
            values["VerifierDate"] = verifier_date

    section10 = _section_text(lines, ranges, 10)
    if section10:
        acc = re.search(r"ACCOUNT\s+NO\.?\s*[:\-]?\s*([A-Z0-9-]{5,})", section10, flags=re.IGNORECASE)
        if acc:
            values["PbCardAccountNumber"] = _clean_value(acc.group(1))

        credit_line = re.search(r"CREDIT\s+LINE\s+RM\s*[:\-]?\s*(RM\s*)?([0-9][0-9,]*(?:\.\d{2})?)", section10, flags=re.IGNORECASE)
        if credit_line:
            values["PbCardCreditLineRM"] = _clean_value(credit_line.group(2))

    return values


def _merge_missing(target: dict, incoming: dict):
    for key, value in incoming.items():
        if not target.get(key) and _clean_value(value):
            target[key] = _clean_value(value)


def _extract_keyword_specific_values(lines):
    specific = {}

    for idx, line in enumerate(lines):
        upper = line.upper()
        context_chunk = " ".join(lines[max(0, idx - 1): min(len(lines), idx + 2)])
        context_upper = context_chunk.upper()

        if "ISLAMIC" in upper and "CREDIT LINE" in upper:
            value = _extract_percent_value(line)
            if value:
                specific["CreditLineIslamicPercent"] = value

        if "CONVENTIONAL" in upper and "CREDIT LINE" in upper:
            value = _extract_percent_value(line)
            if value:
                specific["CreditLineConventionalPercent"] = value

        if "TAX IDENTIFICATION NUMBER" in upper and "TIN" in upper:
            match = re.search(r"TIN\)?\s*[:\-]?\s*([A-Z0-9-]{8,})", line, flags=re.IGNORECASE)
            if match:
                specific["TaxIdentificationNumberTIN"] = _clean_value(match.group(1))

        if "SST REGISTRATION NUMBER" in upper:
            match = re.search(r"SST[^:]*[:\-]?\s*([A-Z0-9-]{6,})", line, flags=re.IGNORECASE)
            if match:
                specific["SSTRegistrationNumber"] = _clean_value(match.group(1))

        if "COUNTRY OF TAX RESIDENCY" in upper and ":" in line:
            value = _clean_value(line.split(":", 1)[1])
            if _is_informative_value(value):
                specific["CountryOfTaxResidency"] = value

        if "TAX ID NUMBER" in upper and ":" in line:
            value = _clean_value(line.split(":", 1)[1])
            if _is_informative_value(value):
                specific["TaxIdNumber"] = value

        if "DATE FULL DOCUMENTS FURNISHED" in upper:
            date_value = _extract_date_value(line)
            if date_value:
                specific["FatcaDocumentsFurnishedDate"] = date_value

        if "PREFERENCE TO RECEIVE E-INVOICE" in context_upper:
            checkbox_value = _first_selected_checkbox_option(context_chunk)
            if checkbox_value:
                specific["EInvoicePreference"] = checkbox_value

        if "RESIDENT FOR PURPOSES IN ANY COUNTRY OTHER THAN MALAYSIA" in context_upper:
            checkbox_value = _first_selected_checkbox_option(context_chunk)
            if checkbox_value:
                specific["TaxResidentOutsideMalaysia"] = checkbox_value

        if "POLITICALLY EXPOSED PERSON" in context_upper:
            checkbox_value = _first_selected_checkbox_option(context_chunk)
            if checkbox_value:
                specific["IsPoliticallyExposedPerson"] = checkbox_value

        if "PRODUCTS AND SERVICES" in context_upper and "MARKETING" in context_upper:
            checkbox_value = _first_selected_checkbox_option(context_chunk)
            if checkbox_value:
                specific["MarketingPreference"] = checkbox_value

        if "OPT-IN FOR CNP AND OVERSEAS" in context_upper:
            if _is_checked_statement(line):
                specific["OptInCnpOverseas"] = "YES"

        if "TWO (2) OR MORE CREDIT CARD ISSUERS" in context_upper:
            if _is_checked_statement(line):
                specific["OwnsTwoOrMoreCreditCards"] = "YES"

        if "ONLY ONE (1) CREDIT CARD" in context_upper:
            if _is_checked_statement(line):
                specific["OwnsZeroOrOneCreditCard"] = "YES"

        if "FORM W-9" in context_upper and re.search(r"\[[Xx/✓✔☑☒]\]\s*FORM\s*W-?9|FORM\s*W-?9\s*\[[Xx/✓✔☑☒]\]", context_chunk, flags=re.IGNORECASE):
            specific["FatcaDocumentType"] = "FORM W-9"

        if "FORM W-8BEN" in context_upper and re.search(r"\[[Xx/✓✔☑☒]\]\s*FORM\s*W-?8BEN|FORM\s*W-?8BEN\s*\[[Xx/✓✔☑☒]\]", context_chunk, flags=re.IGNORECASE):
            specific["FatcaDocumentType"] = "FORM W-8BEN"

        if "OTHERS, PLEASE SPECIFY" in upper and ":" in line:
            value = _clean_value(line.split(":", 1)[1])
            if _is_informative_value(value):
                specific["FatcaOtherDocument"] = value

        if "SIGNATURE: PRINCIPAL CARD APPLICANT" in upper:
            date_value = _extract_date_value(line)
            if date_value:
                specific["PrincipalApplicantSignatureDate"] = date_value

        if "SIGNATURE: SUPPLEMENTARY" in upper:
            date_value = _extract_date_value(line)
            if date_value:
                specific["SupplementaryApplicantSignatureDate"] = date_value

        if "ACCOUNT NO" in upper and ":" in line:
            value = _clean_value(line.split(":", 1)[1])
            if _is_informative_value(value):
                specific["PbCardAccountNumber"] = value

        if "CREDIT LINE RM" in upper and ":" in line:
            value = _clean_value(line.split(":", 1)[1])
            if _is_informative_value(value):
                specific["PbCardCreditLineRM"] = value

    return specific


def _assess_field_value(keyword: str, value: str):
    cleaned = _clean_value(value)
    if not cleaned:
        return 0.0, "not found"

    upper = cleaned.upper()
    if upper == "BOTH":
        return 0.35, "ambiguous yes/no selection"

    if keyword == "FatcaDocumentType" and upper in {"FORM W-9", "FORM W-8BEN", "FORM W-BBEN"}:
        return 0.85, "selected FATCA document option"

    if upper in {"DATE", "YES NO", "FORM W-9", "FORM W-BBEN", "FORM W-8BEN"}:
        return 0.2, "looks like a label/option instead of a value"

    if keyword in {
        "PrincipalApplicantSignatureDate",
        "SupplementaryApplicantSignatureDate",
        "VerifierDate",
        "FatcaDocumentsFurnishedDate",
    }:
        return (0.95, "valid date format") if _extract_date_value(cleaned) else (0.25, "date format not recognized")

    if keyword in {"CreditLineIslamicPercent", "CreditLineConventionalPercent"}:
        return (0.95, "valid percent format") if re.search(r"\b\d+(?:\.\d+)?%\b", cleaned) else (0.2, "percent format not recognized")

    if keyword in {"TaxIdentificationNumberTIN", "SSTRegistrationNumber", "TaxIdNumber"}:
        if re.search(r"\b[A-Z0-9-]{6,}\b", cleaned, flags=re.IGNORECASE):
            return 0.9, "alphanumeric identifier pattern"
        return 0.35, "identifier pattern weak"

    if keyword in {"EInvoicePreference", "TaxResidentOutsideMalaysia", "IsPoliticallyExposedPerson", "MarketingPreference"}:
        if upper in {"YES", "NO", "BOTH"}:
            return 0.9, "checkbox selection"
        return 0.35, "non-standard checkbox value"

    if keyword in {"CountryOfTaxResidency", "NoTaxNumberReason", "PepPositionHeld", "VerifiedAndApprovedBy", "VerifierSignatureAndName"}:
        if len(cleaned) >= 3:
            return 0.8, "text value captured"
        return 0.3, "text value too short"

    return 0.75, "captured from matched label"


def _sanitize_field_value(keyword: str, value: str):
    cleaned = _clean_value(value)
    if not cleaned:
        return ""

    if keyword == "PbCardAccountNumber" and re.match(r"^PBBSTAFF\d+$", cleaned, flags=re.IGNORECASE):
        return ""

    if keyword == "SSTRegistrationNumber" and not re.search(r"\d", cleaned):
        return ""

    if keyword == "FatcaOtherDocument" and re.search(r"DATE\s+FULL\s+DOCUMENTS\s+FURNISHED", cleaned, flags=re.IGNORECASE):
        return ""

    if keyword == "PepPositionHeld" and cleaned.lower().startswith("please (/) tick your preference"):
        return ""

    invalid_patterns = {
        "CountryOfTaxResidency": [r"IF\s+NO\s+TAX\s+ID\s+NUMBER"],
        "TaxIdNumber": [r"^FORM\s+W-?9$", r"^FORM\s+W-?8BEN$", r"^FORM\s+W-?BBEN$", r"^☐$"],
        "VerifierSignatureAndName": [r"^DATE$"],
        "PbCardCreditLineRM": [r"^PBBSTAFF\d+$"],
    }

    for pattern in invalid_patterns.get(keyword, []):
        if re.search(pattern, cleaned, flags=re.IGNORECASE):
            return ""

    return cleaned


def _build_field_details(canonical_fields):
    # Resolve mutually exclusive checkbox contradictions before scoring.
    if (
        _clean_value(canonical_fields.get("OwnsTwoOrMoreCreditCards", "")).upper() == "YES"
        and _clean_value(canonical_fields.get("OwnsZeroOrOneCreditCard", "")).upper() == "YES"
    ):
        canonical_fields["OwnsTwoOrMoreCreditCards"] = ""
        canonical_fields["OwnsZeroOrOneCreditCard"] = ""

    for key in ["EInvoicePreference", "TaxResidentOutsideMalaysia", "MarketingPreference"]:
        if _clean_value(canonical_fields.get(key, "")).upper() == "BOTH":
            canonical_fields[key] = ""

    details = {}
    for keyword, raw_value in canonical_fields.items():
        sanitized = _sanitize_field_value(keyword, raw_value)
        confidence, reason = _assess_field_value(keyword, sanitized)
        source = "matched" if sanitized else "not-found"
        details[keyword] = {
            "value": sanitized,
            "confidence": round(confidence, 2),
            "reason": reason,
            "source": source,
        }

    return details


def _load_field_keyword_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Field config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    keywords = cfg.get("keywords")
    if not isinstance(keywords, list) or not keywords:
        raise ValueError("Field config must contain a non-empty 'keywords' list.")

    canonical = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
    if not canonical:
        raise ValueError("Field config 'keywords' list is empty after trimming.")

    aliases = cfg.get("aliases", {})
    if aliases is None:
        aliases = {}
    if not isinstance(aliases, dict):
        raise ValueError("Field config 'aliases' must be an object.")

    fuzzy_threshold = float(cfg.get("fuzzy_threshold", 0.82))
    if fuzzy_threshold < 0 or fuzzy_threshold > 1:
        raise ValueError("Field config 'fuzzy_threshold' must be between 0 and 1.")

    lookup = []
    contextual_lookup = []
    for keyword in canonical:
        lookup.append((keyword, _normalize_label(keyword)))
        alias_values = aliases.get(keyword, [])
        if alias_values is None:
            alias_values = []
        if not isinstance(alias_values, list):
            raise ValueError(f"Aliases for '{keyword}' must be a list.")
        for alias in alias_values:
            if isinstance(alias, dict):
                alias_text = str(alias.get("label", "")).strip()
                context_values = alias.get("context", [])
                if isinstance(context_values, str):
                    context_values = [context_values]
                if not isinstance(context_values, list):
                    raise ValueError(
                        f"Context alias for '{keyword}' must use list/string in 'context'."
                    )
                normalized_context = [
                    _normalize_label(item) for item in context_values if str(item).strip()
                ]
                if alias_text and normalized_context:
                    contextual_lookup.append(
                        {
                            "keyword": keyword,
                            "label": _normalize_label(alias_text),
                            "context": normalized_context,
                        }
                    )
                elif alias_text:
                    lookup.append((keyword, _normalize_label(alias_text)))
            else:
                alias_text = str(alias).strip()
                if alias_text:
                    lookup.append((keyword, _normalize_label(alias_text)))

    return {
        "keywords": canonical,
        "lookup": lookup,
        "contextual_lookup": contextual_lookup,
        "fuzzy_threshold": fuzzy_threshold,
    }


def _best_keyword_match(normalized_label: str, candidates, fuzzy_threshold: float):
    if not normalized_label or not candidates:
        return None

    for keyword, lookup_value in candidates:
        if normalized_label == lookup_value:
            return keyword

    best_keyword = None
    best_score = 0.0
    for keyword, lookup_value in candidates:
        score = SequenceMatcher(None, normalized_label, lookup_value).ratio()
        if score > best_score:
            best_score = score
            best_keyword = keyword

    if best_keyword and best_score >= fuzzy_threshold:
        return best_keyword
    return None


def _match_keyword(label_text: str, field_cfg, context_text: str = ""):
    normalized = _normalize_label(label_text)
    if not normalized:
        return None

    normalized_context = _normalize_label(context_text)
    contextual_candidates = []
    for entry in field_cfg.get("contextual_lookup", []):
        if any(ctx and ctx in normalized_context for ctx in entry["context"]):
            contextual_candidates.append((entry["keyword"], entry["label"]))

    contextual_match = _best_keyword_match(
        normalized,
        contextual_candidates,
        field_cfg["fuzzy_threshold"],
    )
    if contextual_match:
        return contextual_match

    return _best_keyword_match(normalized, field_cfg["lookup"], field_cfg["fuzzy_threshold"])


def _extract_fields_from_content_lines(content: str, field_cfg):
    found = {}
    lines = [_clean_value(line) for line in str(content or "").splitlines()]
    no_follow_value_keywords = {
        "CreditLineIslamicPercent",
        "CreditLineConventionalPercent",
        "EInvoicePreference",
        "TaxResidentOutsideMalaysia",
        "FatcaDocumentType",
        "IsPoliticallyExposedPerson",
        "MarketingPreference",
        "OptInCnpOverseas",
        "OwnsTwoOrMoreCreditCards",
        "OwnsZeroOrOneCreditCard",
    }

    # Handle common form rows like "___% ISLAMIC" and "___% CONVENTIONAL".
    for line in lines:
        normalized = _normalize_label(line)
        if not found.get("CreditLineIslamicPercent") and "islamic" in normalized and "%" in line:
            value = _extract_percent_value(line)
            if value:
                found["CreditLineIslamicPercent"] = value
        if not found.get("CreditLineConventionalPercent") and "conventional" in normalized and "%" in line:
            value = _extract_percent_value(line)
            if value:
                found["CreditLineConventionalPercent"] = value

    for idx, line in enumerate(lines):
        if not line:
            continue

        context_window = " ".join(lines[max(0, idx - 2): min(len(lines), idx + 3)])

        # Handle checkbox-based options (YES/NO) and checked statements.
        checkbox_value = _first_selected_checkbox_option(line)
        if checkbox_value:
            keyword = _match_keyword_from_line(line, field_cfg, context_window)
            if keyword and not found.get(keyword):
                found[keyword] = checkbox_value

        if _is_checked_statement(line):
            keyword = _match_keyword_from_line(line, field_cfg, context_window)
            if keyword and not found.get(keyword):
                found[keyword] = "YES"

        if ":" in line:
            left, right = line.split(":", 1)
            keyword = _match_keyword_from_line(left.strip(), field_cfg, context_window)
            value = _clean_value(right)
            if not _is_informative_value(value):
                value = _next_value_line(lines, idx + 1, field_cfg)
            if keyword and _is_informative_value(value) and not found.get(keyword):
                found[keyword] = value

        # Handle labels that are on one line with value on the next line.
        if ":" not in line:
            keyword = _match_keyword_from_line(line, field_cfg, context_window)
            if keyword and keyword not in no_follow_value_keywords and not found.get(keyword):
                next_value = _next_value_line(lines, idx + 1, field_cfg)
                if next_value:
                    found[keyword] = next_value

    return found


def _detect_table_header_rows(table):
    """Return the set of row indices that are header rows."""
    has_kind = any(getattr(c, "kind", None) == "columnHeader" for c in table.cells)
    if has_kind:
        return {c.row_index for c in table.cells if getattr(c, "kind", None) == "columnHeader"}

    # Fallback: scan up to 5 rows; stop when a row has mostly numeric cells.
    rows_by_idx: dict = {}
    for cell in table.cells:
        rows_by_idx.setdefault(cell.row_index, []).append(cell)

    last_header_row = 0
    for row_i in range(min(5, table.row_count)):
        cells_in_row = rows_by_idx.get(row_i, [])
        non_empty = [c for c in cells_in_row if _clean_value(c.content)]
        has_digits = sum(1 for c in non_empty if re.search(r"\d", c.content))
        if non_empty and has_digits / len(non_empty) >= 0.5:
            break
        last_header_row = row_i
    return set(range(last_header_row + 1))


def _build_table_header_map(table, header_row_set, field_cfg):
    """Map column index -> matched keyword using all header rows per column."""
    header_lines: dict = {}
    for cell in table.cells:
        if cell.row_index not in header_row_set:
            continue
        part = re.sub(r"^\*+\s*", "", _clean_value(cell.content)).rstrip("/").strip()
        if part:
            header_lines.setdefault(cell.column_index, []).append(part)

    header_by_col: dict = {}
    for col_idx, parts in header_lines.items():
        matched = None
        for part in parts:                          # try each line individually first
            matched = _match_keyword(part, field_cfg)
            if matched:
                break
        if not matched:                             # try combined label as fallback
            matched = _match_keyword(" ".join(parts), field_cfg)
        if matched:
            header_by_col[col_idx] = matched
    return header_by_col


def _extract_fields_from_tables(tables, field_cfg):
    found = {}
    for table in tables or []:
        header_row_set = _detect_table_header_rows(table)
        header_by_col  = _build_table_header_map(table, header_row_set, field_cfg)

        if not header_by_col:
            continue

        for cell in table.cells:
            if cell.row_index in header_row_set:
                continue
            keyword = header_by_col.get(cell.column_index)
            if not keyword:
                continue
            value = _clean_value(cell.content)
            if value and not found.get(keyword):
                found[keyword] = value

    return found


def _extract_table_row_data(tables, field_cfg):
    """Extract every data row from every table as structured row dicts.

    Returns:
        { "Table1": {"columns": [...], "rows": [{col: val, ...}, ...]}, ... }
    """
    table_data: dict = {}
    for table_idx, table in enumerate(tables or []):
        header_row_set = _detect_table_header_rows(table)
        header_by_col  = _build_table_header_map(table, header_row_set, field_cfg)

        if not header_by_col:
            continue

        # Group data cells by row index
        data_by_row: dict = {}
        for cell in table.cells:
            if cell.row_index in header_row_set:
                continue
            keyword = header_by_col.get(cell.column_index)
            if not keyword:
                continue
            data_by_row.setdefault(cell.row_index, {})[keyword] = _clean_value(cell.content)

        if not data_by_row:
            continue

        columns = [header_by_col[i] for i in sorted(header_by_col)]
        rows    = [data_by_row[r] for r in sorted(data_by_row)]
        table_data[f"Table{table_idx + 1}"] = {"columns": columns, "rows": rows}

    return table_data


def _build_canonical_field_map(result: AnalyzeResult, field_cfg, doc_path: str):
    lines = [_clean_value(line) for line in str(result.content or "").splitlines()]
    merged = {}
    _merge_missing(merged, _extract_checkbox_values_from_fixed_template(result, doc_path))
    _merge_missing(merged, _extract_checkbox_values_from_image(result, doc_path))
    _merge_missing(merged, _extract_selection_mark_values(result))
    _merge_missing(merged, _extract_keyword_specific_values(lines))
    _merge_missing(merged, _extract_section_specific_values(lines))
    _merge_missing(merged, _extract_fields_from_content_lines(result.content, field_cfg))
    _merge_missing(merged, _extract_fields_from_tables(result.tables, field_cfg))

    return {keyword: _clean_value(merged.get(keyword, "")) for keyword in field_cfg["keywords"]}


def _render_keyword_field_markdown(canonical_fields):
    lines = ["", "## Canonical Keywords and Fields", ""]
    for keyword, value in canonical_fields.items():
        display = value if value else "NOT FOUND"
        lines.append(f"- {keyword}: {display}")
    lines.append("")
    return "\n".join(lines)


def _ensure_parent_dir(file_path: str):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def analyze_full_context_and_tables(doc_path: str, md_out: str, excel_out: str, keyword_json_out: str):
    if not os.path.exists(doc_path):
        print(f"Error: '{doc_path}' not found.")
        return

    try:
        resolved_cfg_path = _resolve_field_config_path(FIELD_KEYWORD_CONFIG)
        field_cfg = _load_field_keyword_config(resolved_cfg_path)
    except Exception as exc:
        print(f"Error: failed to load field keyword config '{FIELD_KEYWORD_CONFIG}': {exc}")
        return

    client = DocumentIntelligenceClient(
        endpoint=ENDPOINT, 
        credential=AzureKeyCredential(KEY)
    )

    print(f"[*] Sending '{doc_path}' to Azure AI Document Intelligence...")
    start_time = time.time()

    with open(doc_path, "rb") as f:
        # We pass output_content_format="markdown" to extract full page context
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout", 
            body=f,
            output_content_format="markdown"
        )
        result: AnalyzeResult = poller.result()

    execution_time = time.time() - start_time
    print(f"[+] Processing completed in {execution_time:.2f} seconds.")

    # -----------------------------------------------------------------------
    # 1. EXPORT FULL CONTEXT (Paragraphs + Tables + Headings) AS MARKDOWN
    # -----------------------------------------------------------------------
    full_content = result.content if result.content else "No content extracted."
    canonical_fields = _build_canonical_field_map(result, field_cfg, doc_path)
    field_details = _build_field_details(canonical_fields)
    canonical_fields = {k: v["value"] for k, v in field_details.items()}
    keyword_section = _render_keyword_field_markdown(canonical_fields)

    _ensure_parent_dir(md_out)
    _ensure_parent_dir(excel_out)
    _ensure_parent_dir(keyword_json_out)
    
    with open(md_out, "w", encoding="utf-8") as f:
        f.write(f"<!-- Processing Time: {execution_time:.2f} seconds -->\n\n")
        f.write(full_content)
        f.write(keyword_section)

    print(f"[SUCCESS] Full document context saved to: {md_out}")

    unmatched = [k for k, v in canonical_fields.items() if not v]
    strict_fields = {
        k: info["value"]
        for k, info in field_details.items()
        if info["value"] and info["confidence"] >= STRICT_CONFIDENCE_THRESHOLD
    }
    strict_unmatched = [k for k in canonical_fields.keys() if k not in strict_fields]
    table_row_data = _extract_table_row_data(result.tables, field_cfg)

    keyword_payload = {
        "keywords": list(canonical_fields.keys()),
        "fields": canonical_fields,
        "fieldDetails": field_details,
        "strictConfidenceThreshold": STRICT_CONFIDENCE_THRESHOLD,
        "strictFields": strict_fields,
        "strictUnmatchedKeywords": strict_unmatched,
        "unmatchedKeywords": unmatched,
    }
    if table_row_data:
        keyword_payload["tableData"] = table_row_data

    with open(keyword_json_out, "w", encoding="utf-8") as f:
        json.dump(keyword_payload, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Keyword fields saved to: {keyword_json_out}")

    # -----------------------------------------------------------------------
    # 2. EXPORT EXPLICIT TABULAR DATA TO EXCEL
    # -----------------------------------------------------------------------
    tables = result.tables if result.tables else []
    print(f"[+] Total Tables Found: {len(tables)}")

    if tables:
        with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
            for idx, table in enumerate(tables):
                sheet_name = f"Table_{idx + 1}"
                
                # Reconstruct 2D matrix for Excel
                matrix = [["" for _ in range(table.column_count)] for _ in range(table.row_count)]
                for cell in table.cells:
                    matrix[cell.row_index][cell.column_index] = cell.content.replace("\n", " ").strip()

                df = pd.DataFrame(matrix[1:], columns=matrix[0]) if len(matrix) > 1 else pd.DataFrame(matrix)
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"[SUCCESS] Extracted tables saved to: {excel_out}")
    else:
        print("[-] No standalone tables found to export to Excel.")

if __name__ == "__main__":
    analyze_full_context_and_tables(INPUT_DOCUMENT, OUTPUT_MARKDOWN, OUTPUT_EXCEL, OUTPUT_KEYWORD_JSON)
