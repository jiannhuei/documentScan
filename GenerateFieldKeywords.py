"""
=============================================================================
GenerateFieldKeywords.py
=============================================================================
Sends a PDF to Azure AI Document Intelligence and auto-generates a
{filename}_field_keywords.json in the same format as CreditCard_field_keywords.json.

Handles:
  - Text / fill-in fields    →  "keywords" + "aliases" + field_types: "text"
  - Date fields              →  field_types: "date"   (detected from label)
  - Signature fields         →  field_types: "signature"
  - Number / amount fields   →  field_types: "number"
  - Checkbox (single tick)   →  field_types: "checkbox"
  - YES / NO radio pair      →  field_types: "yesno"
  - Multi-option group       →  field_types: "option"
  - Table column fields      →  "tables" + dotted "aliases"
                                 e.g. "TableAbc.Name": ["name", "username"]

Usage:
  python GenerateFieldKeywords.py sample/MyForm.pdf
  python GenerateFieldKeywords.py sample/MyForm.pdf --min-cols 2 --min-rows 1
Output:
  DocIntResult/MyForm.pdf_field_keywords.json
=============================================================================
"""

import os
import sys
import json
import re
import time
import argparse
import statistics
from collections import Counter

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeResult,
    DocumentAnalysisFeature,
)

# Field names are derived from the printed text, so OCR artefacts can carry
# characters the Windows console codepage cannot encode. Without this a
# successful run dies on its own summary line.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# CONFIGURATION  (reuse same credentials as TestOnTable.py)
# ---------------------------------------------------------------------------
ENDPOINT = "https://pbb-document-intelligence-poc.cognitiveservices.azure.com/"
KEY = ""
OUTPUT_DIR = "DocIntResult"


# ---------------------------------------------------------------------------
# GEOMETRY HELPERS
# ---------------------------------------------------------------------------

def _polygon_to_bbox(polygon):
    """Convert a polygon to a bbox dict.

    Document Intelligence returns polygons as a flat ``[x1, y1, x2, y2, ...]``
    float list; older SDKs returned point objects. Both are accepted.
    """
    if not polygon:
        return None

    xs, ys = [], []
    points = list(polygon)
    if all(isinstance(value, (int, float)) for value in points):
        if len(points) < 4:
            return None
        xs = [float(v) for v in points[0::2]]
        ys = [float(v) for v in points[1::2]]
    else:
        for point in points:
            try:
                xs.append(float(point.x)); ys.append(float(point.y))
            except Exception:
                try:
                    xs.append(float(point[0])); ys.append(float(point[1]))
                except Exception:
                    continue

    if not xs or not ys:
        return None
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return {"x0": x0, "x1": x1, "y0": y0, "y1": y1,
            "cx": (x0 + x1) / 2.0, "cy": (y0 + y1) / 2.0,
            "h": max(0.001, y1 - y0)}


def _page_words(page):
    """Return ``(words, median_word_height)`` for a page.

    The median height is the natural unit of scale for a page, so callers
    express tolerances as multiples of it instead of hard-coded inch values
    that break on other page sizes or scan resolutions.
    """
    words = []
    for word in getattr(page, "words", []) or []:
        bbox = _polygon_to_bbox(getattr(word, "polygon", None))
        content = str(getattr(word, "content", "") or "").strip()
        if bbox and content:
            words.append({"text": content, "x": bbox["x0"], "cy": bbox["cy"],
                          "h": bbox["h"], "bbox": bbox})
    if not words:
        return [], 0.01
    median_h = statistics.median([w["h"] for w in words])
    return words, max(0.001, median_h)


def _group_words_into_lines(words, median_h):
    """Group words into lines by vertical position, left-to-right within a line."""
    if not words:
        return []

    ordered = sorted(words, key=lambda w: (w["cy"], w["x"]))
    y_tol = max(0.01, median_h * 0.6)

    lines = []
    for word in ordered:
        if not lines or abs(word["cy"] - lines[-1]["cy"]) > y_tol:
            lines.append({"cy": word["cy"], "words": [word]})
        else:
            row = lines[-1]
            row["words"].append(word)
            row["cy"] = sum(w["cy"] for w in row["words"]) / len(row["words"])

    for line in lines:
        line["words"].sort(key=lambda w: w["x"])
        line["text"] = " ".join(w["text"] for w in line["words"])

    return lines


def _build_page_lines(page):
    """Group a page's words into text lines.

    Returns ``(lines, median_word_height)``.
    """
    words, median_h = _page_words(page)
    return _group_words_into_lines(words, median_h), median_h


# ---------------------------------------------------------------------------
# LABEL / TYPE HELPERS
# ---------------------------------------------------------------------------

# Beyond this the name stops being an identifier and becomes the sentence.
MAX_NAME_WORDS = 6

# Connectives carry no identifying meaning, so they are the first thing to drop
# when a caption has to be shortened into a name.
NAME_FILLER_WORDS = {
    "a", "an", "and", "or", "the", "to", "of", "for", "in", "on", "at", "by",
    "with", "your", "my", "our", "if", "is", "are", "be", "please", "from",
    "any", "am", "do", "not", "i", "we", "that", "this", "shall", "will",
}


def _to_camel_case(text: str) -> str:
    """
    'TAX IDENTIFICATION NUMBER (TIN)' -> 'TaxIdentificationNumberTin'
    'name/username'                    -> 'NameUsername'

    A checkbox caption is a whole printed sentence, so using it verbatim yields
    a 140-character identifier that is unusable as a field name. Keep the
    leading words that actually identify the field.
    """
    text = re.sub(r"\([^)]*\)", "", text)           # strip (parenthetical)
    parts = [p for p in re.split(r"[^a-zA-Z0-9]+", text) if p]
    if len(parts) > MAX_NAME_WORDS:
        meaningful = [p for p in parts if p.lower() not in NAME_FILLER_WORDS]
        parts = (meaningful or parts)[:MAX_NAME_WORDS]
    return "".join(p.capitalize() for p in parts)


def _register_field(detected: dict, label: str, entry: dict) -> str:
    """Store a field under a unique name derived from its caption.

    Shortened names can collide even where the captions differ, and a collision
    only means "already seen" when the caption itself is the same. Otherwise the
    second field needs a name of its own, or it is silently lost.

    ``_label`` records the text the name was built from, because the aliases
    alone can no longer answer the question - a contextual alias is a mapping,
    not the caption string.
    """
    base = _to_camel_case(label)
    if not base:
        return ""

    name = base
    suffix = 2
    while name in detected:
        if detected[name].get("_label") == label:
            return ""
        name = f"{base}{suffix}"
        suffix += 1

    entry["_label"] = label
    detected[name] = entry
    return name


def _clean_label(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _split_slash_aliases(text: str) -> list:
    """'name/username' -> ['name', 'username'],  'plain' -> ['plain']

    The full text leads, because that is what is actually printed. A fragment
    is only worth keeping when it can stand alone as a caption: a bare
    "DIVISION" left over from splitting "HEAD OF BRANCH/DIVISION/DEPARTMENT"
    matches half the form and drags unrelated text into the field.
    """
    parts = [p.strip() for p in text.split("/") if p.strip()]
    if len(parts) < 2:
        return [text]

    standalone = [p for p in parts if len(p.split()) >= 2]
    if standalone:
        return [text] + standalone
    # Every fragment is one word, so the split is the whole point - but only
    # for a short label, where the fragments really are alternatives.
    if len(text.split()) <= 3:
        return parts
    return [text]


def _is_selection_value(content: str) -> bool:
    """True when the KV value represents a checkbox mark."""
    return _clean_label(content) in (":selected:", ":unselected:")


def _detect_text_field_type(label: str) -> str:
    """Heuristic field-type detection for non-checkbox fields."""
    upper = label.upper()
    if re.search(r"\bDATE\b|DD[-/]MM|MM[-/]YY|\bDOB\b|\bBIRTH\b", upper):
        return "date"
    if re.search(r"\bSIGNATURE\b", upper):
        return "signature"
    if re.search(r"\bNO\.|NUMBER\b|\bAMOUNT\b|\bRM\b|\bCODE\b|%", upper):
        return "number"
    return "text"


def _looks_like_prose(text: str) -> bool:
    """True for sentences and paragraphs, which are never field labels."""
    words = text.split()
    if len(words) >= 8:
        return True
    if len(words) >= 5 and text.rstrip().endswith((".", ":", ";")):
        return True
    return False


# A printed statement beside a tick box can be long, but a whole paragraph is
# not a caption.
MAX_STATEMENT_WORDS = 28


def _is_option_noise(text: str) -> bool:
    """True for a line that is only option labels, never the question itself."""
    cleaned = _clean_label(text).strip(" .:-")
    if len(cleaned) < 3:
        return True
    stripped = re.sub(r"\b(YES|NO)\b", "", cleaned, flags=re.IGNORECASE)
    return len(re.sub(r"[^A-Za-z0-9]+", "", stripped)) < 3


def _is_value_like(text: str) -> bool:
    """True when the text is filled-in data rather than a label.

    Field names must come from labels. Without this check a value such as
    ``: 7881288`` ends up as a keyword called ``7881288``.
    """
    cleaned = _clean_label(text).strip(":").strip()
    if not cleaned:
        return True
    if not re.search(r"[A-Za-z]", cleaned):
        return True
    if re.fullmatch(r"[\d\s.,/()+-]+", cleaned):
        return True
    if re.fullmatch(r"(RM|MYR|USD|SGD|\$)\s*[\d.,]+", cleaned, flags=re.IGNORECASE):
        return True
    return False


# ---------------------------------------------------------------------------
# GEOMETRIC LABEL DETECTION  (fallback when Azure returns no key-value pairs)
# ---------------------------------------------------------------------------

def _inside_any(bbox, regions) -> bool:
    """True when a word's centre falls inside one of the given bboxes."""
    for region in regions:
        if (region["x0"] <= bbox["cx"] <= region["x1"]
                and region["y0"] <= bbox["cy"] <= region["y1"]):
            return True
    return False


def _split_line_segments(words, gap_tol):
    """Split a line into segments wherever a gap wider than ``gap_tol`` occurs.

    Forms routinely place several label/value pairs on one line; each segment
    is a candidate label/value pair on its own.
    """
    segments = [[words[0]]]
    for previous, word in zip(words, words[1:]):
        if word["bbox"]["x0"] - previous["bbox"]["x1"] > gap_tol:
            segments.append([])
        segments[-1].append(word)
    return segments


def _label_from_segment(segment, line_words, gap_tol):
    """Extract a field label from one line segment, or None.

    Recognises ``Label :``, ``Label ______`` / ``Label ......``, and a label
    followed by a wide blank area that a person writes into.
    """
    text = _clean_label(" ".join(w["text"] for w in segment))
    if not text or _looks_like_prose(text):
        return None

    if ":" in text:
        text = text.split(":", 1)[0]
    else:
        without_leader = re.split(r"_{3,}|\.{4,}|\u2026{2,}", text)[0]
        if without_leader != text:
            text = without_leader
        else:
            # No punctuation cue: only accept the segment when what follows it
            # on the line is a wide blank or a filled-in value.
            tail = [w for w in line_words if w["bbox"]["x0"] > segment[-1]["bbox"]["x1"]]
            if not tail:
                return None
            if tail[0]["bbox"]["x0"] - segment[-1]["bbox"]["x1"] < gap_tol:
                return None
            if not _is_value_like(" ".join(w["text"] for w in tail)):
                return None

    label = _clean_label(text).strip(" .-*_")
    if len(label) < 2 or len(label.split()) > 8:
        return None
    if _is_value_like(label) or _looks_like_prose(label):
        return None
    return label


def _extract_label_fields(result: AnalyzeResult, blocked_regions: dict) -> dict:
    """Derive fill-in fields from page geometry.

    ``blocked_regions`` maps page number -> list of bboxes already consumed as
    tables, so table content never leaks in as a standalone field.

    Returns ``{field_name: {"aliases": [...], "type": str}}``.
    """
    detected: dict = {}
    candidates: list = []

    for page in getattr(result, "pages", []) or []:
        lines, median_h = _build_page_lines(page)
        if not lines:
            continue
        blocked = blocked_regions.get(getattr(page, "page_number", 0), [])
        gap_tol = max(0.02, median_h * 3.0)

        for line_idx, line in enumerate(lines):
            words = [w for w in line["words"] if not _inside_any(w["bbox"], blocked)]
            if not words:
                continue
            for segment in _split_line_segments(words, gap_tol):
                label = _label_from_segment(segment, words, gap_tol)
                if label:
                    x_range = (
                        min(w["bbox"]["x0"] for w in segment),
                        max(w["bbox"]["x1"] for w in segment),
                    )
                    candidates.append((label, lines, line_idx, x_range, gap_tol))

    # A form puts a bare "DATE" under each signature box. Those are different
    # fields, but nothing in the caption itself says so, so a plain alias makes
    # them collapse into one and only the last value survives. The printed text
    # above each copy is what tells them apart.
    repeated = {
        label
        for label, count in Counter(label for label, *_ in candidates).items()
        if count > 1
    }

    for label, lines, line_idx, x_range, gap_tol in candidates:
        entry = {
            "aliases": _split_slash_aliases(label),
            "type": _detect_text_field_type(label),
        }
        name = label

        if label in repeated:
            context = _context_lines_above(
                lines, line_idx, label, x_range, gap_tol
            )
            if context:
                entry["aliases"] = [{"label": label, "context": context}]
                name = f"{context[0]} {label}"

        _register_field(detected, name, entry)

    return detected


# How far above a repeated caption to look for the text that identifies it.
MAX_CONTEXT_LOOKBACK = 3


def _context_lines_above(lines, line_idx, label, x_range=None, gap_tol=None):
    """The distinctive printed text just above a line, nearest first.

    ``x_range`` restricts each line to the caption's own column. That is what
    actually separates twin captions: a form prints "SIGNATURE: PRINCIPAL ..."
    and "SIGNATURE: SUPPLEMENTARY ..." side by side and puts a "DATE" under
    each, so the whole line above is identical for both.

    The column is taken as the whole segment overlapping the caption rather
    than the individual overlapping words, because a narrow "DATE" sits under
    only the first word of the caption that names it - clipping to overlap
    alone yields a bare "SIGNATURE:" for both twins and separates nothing.

    Short lines are skipped because they cannot anchor anything - a caption is
    only told apart from its twin by text that is unique on the page.
    """
    found = []
    target = label.strip().lower()
    for back in range(1, MAX_CONTEXT_LOOKBACK + 1):
        idx = line_idx - back
        if idx < 0:
            break
        words = lines[idx]["words"]
        if x_range and gap_tol and words:
            x0, x1 = x_range
            overlapping = [
                seg
                for seg in _split_line_segments(words, gap_tol)
                if seg
                and min(w["bbox"]["x0"] for w in seg) < x1
                and max(w["bbox"]["x1"] for w in seg) > x0
            ]
            # A caption with nothing above it in its own column is better
            # described by the whole line than by silence.
            if overlapping:
                words = [w for seg in overlapping for w in seg]
        text = _clean_label(" ".join(w["text"] for w in words))
        if len(text) < 8 or text.strip().lower() == target:
            continue
        found.append(text)
    return found


def _page_line_index(result: AnalyzeResult) -> dict:
    """Page number -> (lines, median word height), built once and reused."""
    index = {}
    for page in getattr(result, "pages", []) or []:
        lines, median_h = _build_page_lines(page)
        if lines:
            index[getattr(page, "page_number", 0)] = (lines, median_h)
    return index


def _context_for_region(page_index: dict, region, label: str) -> list:
    """The printed lines above a region, for telling repeated captions apart.

    The key-value extractor reports a caption's position but not which page
    line it is, so the line has to be recovered from the geometry before its
    neighbours can be read.
    """
    if not region:
        return []
    entry = page_index.get(getattr(region, "page_number", 0))
    bbox = _polygon_to_bbox(getattr(region, "polygon", None))
    if not entry or not bbox:
        return []

    lines, median_h = entry
    best_idx = None
    best_gap = None
    for idx, line in enumerate(lines):
        centres = [w["bbox"]["cy"] for w in line["words"]]
        if not centres:
            continue
        gap = abs(sum(centres) / len(centres) - bbox["cy"])
        if best_gap is None or gap < best_gap:
            best_gap, best_idx = gap, idx

    # A line further away than its own height is a different line.
    if best_idx is None or best_gap > max(median_h, 0.02):
        return []
    return _context_lines_above(
        lines,
        best_idx,
        label,
        (bbox["x0"], bbox["x1"]),
        max(0.02, median_h * 3.0),
    )


def _fields_from_rejected_table(table) -> dict:
    """Turn a table that failed the size gate into plain label fields.

    The layout model boxes simple ``LABEL : value`` pairs as one-row tables.
    Those are fields, not tables, so recover the label and drop the value.
    """
    detected: dict = {}
    rows: dict = {}
    for cell in table.cells:
        text = _clean_label(cell.content)
        if text:
            rows.setdefault(cell.row_index, []).append((cell.column_index, text))

    for cells in rows.values():
        cells.sort()
        texts = [text for _col, text in cells]
        if len(texts) == 1 and ":" in texts[0]:
            label = texts[0].split(":", 1)[0]
        elif len(texts) >= 2 and _is_value_like(texts[1]):
            label = texts[0]
        else:
            continue

        label = _clean_label(label).strip(" .-*:_")
        if len(label) < 2 or _is_value_like(label) or _looks_like_prose(label):
            continue
        _register_field(
            detected,
            label,
            {
                "aliases": _split_slash_aliases(label),
                "type": _detect_text_field_type(label),
            },
        )

    return detected


# ---------------------------------------------------------------------------
# SELECTION-MARK DETECTION
# ---------------------------------------------------------------------------

def _extract_selection_mark_fields(result: AnalyzeResult) -> dict:
    """
    Scan every page for selection marks (checkboxes / radio buttons).
    Groups spatially close marks on the same row, then classifies:
      - yesno   : group has a YES-labelled AND a NO-labelled mark
      - option  : group has 2+ distinct non-YES/NO option labels
      - checkbox: single isolated mark

    Returns:
      { field_name: {"type": str, "aliases": [str], "options": [str]|None} }
    """
    detected: dict = {}

    for page in getattr(result, "pages", []) or []:
        lines, median_h = _build_page_lines(page)
        marks = getattr(page, "selection_marks", []) or []
        if not marks or not lines:
            continue

        # All tolerances scale with the page's own text size so detection does
        # not depend on page dimensions or scan resolution.
        side_tol = max(0.005, median_h * 0.2)
        context_span = max(0.05, median_h * 4.0)
        row_tol = max(0.01, median_h * 1.5)

        # ---- collect mark metadata ----------------------------------------
        mark_infos = []
        for mark in marks:
            mark_bbox = _polygon_to_bbox(getattr(mark, "polygon", None))
            if not mark_bbox:
                continue
            state = str(getattr(mark, "state", "")).lower()

            nearest_idx = min(range(len(lines)),
                              key=lambda i: abs(lines[i]["cy"] - mark_bbox["cy"]))
            nearest_line = lines[nearest_idx]

            # local label = words immediately to the right of the mark on its line
            right_words = [w["text"] for w in nearest_line["words"]
                           if w["x"] >= mark_bbox["cx"] - side_tol]
            left_words  = [w["text"] for w in nearest_line["words"]
                           if w["x"] <  mark_bbox["cx"] - side_tol]
            # Two different lengths are needed. Deciding YES vs NO only needs
            # the first few words, but a "tick to agree" box is labelled with a
            # whole printed statement and that statement is the only thing that
            # identifies the field.
            local_label = " ".join(right_words[:5]).strip()
            if not local_label:
                local_label = " ".join(left_words[-3:]).strip()
            statement = _clean_label(" ".join(right_words[:MAX_STATEMENT_WORDS]))

            # Context label: the single printed line above the mark that asks
            # the question. Joining several lines builds a paragraph, and a
            # paragraph is never usable as an alias - it was rejected as prose
            # further down, which silently dropped almost every checkbox field.
            context_label = ""
            for i in range(nearest_idx - 1, -1, -1):
                if lines[nearest_idx]["cy"] - lines[i]["cy"] > context_span:
                    break
                candidate = _clean_label(lines[i]["text"])
                if _is_option_noise(candidate):
                    continue
                context_label = candidate
                break

            mark_infos.append({
                "state": state,
                "bbox": mark_bbox,
                "local_label": local_label.upper(),
                "statement": statement,
                "context_label": context_label,
                "line_cy": nearest_line["cy"],
            })

        if not mark_infos:
            continue

        # ---- group marks that share the same horizontal row ----------------
        mark_infos.sort(key=lambda m: (m["line_cy"], m["bbox"]["cx"]))
        groups: list[list] = []
        for m in mark_infos:
            placed = False
            for g in groups:
                if abs(m["line_cy"] - g[0]["line_cy"]) < row_tol:
                    g.append(m)
                    placed = True
                    break
            if not placed:
                groups.append([m])

        # ---- classify each group -------------------------------------------
        for group in groups:
            local_labels = [m["local_label"] for m in group]
            context = _clean_label(group[0]["context_label"])

            has_yes = any(re.search(r"\bYES\b", l) for l in local_labels)
            has_no  = any(re.search(r"\bNO\b",  l) for l in local_labels)

            if has_yes and has_no:
                field_type  = "yesno"
                # strip the YES/NO tokens themselves from the context to get the question
                field_label = re.sub(r"\bYES\b|\bNO\b", "", context,
                                     flags=re.IGNORECASE).strip()
                field_label = field_label or context
                options     = None
            elif len(group) == 1:
                field_type  = "checkbox"
                # A lone box is usually "tick to agree with this statement",
                # and the statement beside it names the field far better than
                # whatever line happens to sit above it.
                statement   = group[0]["statement"]
                field_label = statement if len(statement.split()) >= 5 else (
                    context or local_labels[0]
                )
                options     = None
            else:
                field_type  = "option"
                field_label = context
                options     = [l for l in local_labels if l]

            if not field_label:
                continue
            entry = {
                "type":    field_type,
                "aliases": _split_slash_aliases(field_label),
            }
            if options:
                entry["options"] = options
            _register_field(detected, field_label, entry)

    return detected


# ---------------------------------------------------------------------------
# TABLE NAME HELPER
# ---------------------------------------------------------------------------

def _get_table_name(table, result: AnalyzeResult, table_idx: int) -> str:
    """Return the closest paragraph heading above the table, else 'Table{N}'."""
    if not table.bounding_regions:
        return f"Table{table_idx + 1}"

    table_page = table.bounding_regions[0].page_number
    table_bbox = _polygon_to_bbox(table.bounding_regions[0].polygon)
    if not table_bbox:
        return f"Table{table_idx + 1}"
    table_top = table_bbox["y0"]

    best_para   = None
    best_bottom = -1.0

    for para in result.paragraphs or []:
        if not para.bounding_regions or not para.content:
            continue
        for region in para.bounding_regions:
            if region.page_number != table_page:
                continue
            pbbox = _polygon_to_bbox(region.polygon)
            if not pbbox:
                continue
            if pbbox["y1"] <= table_top and pbbox["y1"] > best_bottom:
                best_bottom = pbbox["y1"]
                best_para   = para

    if best_para:
        name = _clean_label(best_para.content)
        if name and len(name) <= 80:
            camel = _to_camel_case(name)
            if camel:
                return camel

    return f"Table{table_idx + 1}"


# ---------------------------------------------------------------------------
# TABLE EXTRACTION
# ---------------------------------------------------------------------------

MAX_HEADER_ROWS = 5


def _header_row_indices(table) -> set:
    """Return the row indices that make up the table's header.

    Prefers the cells DocInt explicitly tagged ``kind="columnHeader"``. When no
    cell is tagged, scan the leading rows and stop at the first row that looks
    like data (half or more of its non-empty cells contain digits).

    Returns an empty set when the very first row already looks like data, which
    means the table has no header and must not contribute field names.
    """
    tagged = {c.row_index for c in table.cells
              if getattr(c, "kind", None) == "columnHeader"}
    if tagged:
        return tagged

    rows_by_idx: dict = {}
    for cell in table.cells:
        rows_by_idx.setdefault(cell.row_index, []).append(cell)

    last_header_row = -1
    for row_i in range(min(MAX_HEADER_ROWS, table.row_count)):
        non_empty = [c for c in rows_by_idx.get(row_i, []) if _clean_label(c.content)]
        if not non_empty:
            continue
        with_digits = sum(1 for c in non_empty if re.search(r"\d", c.content))
        if with_digits / len(non_empty) >= 0.5:
            break
        last_header_row = row_i

    return set(range(last_header_row + 1))


def _cell_text_lines(cell, pages_words) -> list:
    """Return the physical text lines printed inside a table cell.

    Azure reports a header cell's content as one joined string, but the form
    prints it over several lines and a line-based matcher only ever sees those
    individual lines. Recovering them from the cell's geometry lets each line
    become an alias, so "NAME OF BORROWER/ CUSTOMER/ ACCOUNT NO." also matches
    a document that shows "NAME OF BORROWER/" on its own line.
    """
    regions = getattr(cell, "bounding_regions", None) or []
    if not regions:
        return []

    bbox = _polygon_to_bbox(regions[0].polygon)
    words, median_h = pages_words.get(regions[0].page_number, ([], 0.01))
    if not bbox or not words:
        return []

    inside = [w for w in words if _inside_any(w["bbox"], [bbox])]
    texts = []
    for line in _group_words_into_lines(inside, median_h):
        text = re.sub(r"^\*+\s*", "", _clean_label(line["text"])).strip()
        if text:
            texts.append(text)
    return texts


def _column_aliases(header_parts, alias_lines) -> list:
    """Build the alias list for a table column, most specific first."""
    candidates = list(header_parts)
    candidates.extend(alias_lines)
    for part in header_parts:
        candidates.extend(_split_slash_aliases(part))

    ordered = []
    seen_aliases = set()
    for candidate in candidates:
        cleaned = _clean_label(candidate).strip(" /*")
        if len(cleaned) < 2 or _is_value_like(cleaned):
            continue
        key = cleaned.upper()
        if key in seen_aliases:
            continue
        seen_aliases.add(key)
        ordered.append(cleaned)
    return ordered


def _extract_tables(result: AnalyzeResult, min_cols: int, min_rows: int):
    """Build the ``tables`` map from qualifying layout tables.

    A table qualifies when it exposes at least ``min_cols`` label-like header
    columns and has at least ``min_rows`` rows below its header. Layout boxes a
    lot of non-tabular content as tables; the gate keeps those out.

    Returns ``(tables_map, aliases, blocked_regions, rejected_tables)``.
    """
    tables_map: dict = {}
    aliases: dict = {}
    blocked_regions: dict = {}
    rejected_tables: list = []

    pages_words = {
        getattr(page, "page_number", idx + 1): _page_words(page)
        for idx, page in enumerate(getattr(result, "pages", []) or [])
    }

    for table_idx, table in enumerate(result.tables or []):
        header_rows = _header_row_indices(table)

        # Per column: the cell contents (used to build the field name) and the
        # physical printed lines (used to build the aliases).
        header_parts: dict = {}
        alias_lines: dict = {}
        for cell in table.cells:
            if cell.row_index not in header_rows:
                continue
            # Strip leading asterisks/spaces and trailing "/" that forms use
            part = re.sub(r"^\*+\s*", "", _clean_label(cell.content)).rstrip("/").strip()
            if not part or _is_value_like(part):
                continue
            header_parts.setdefault(cell.column_index, []).append(part)
            alias_lines.setdefault(cell.column_index, []).extend(
                _cell_text_lines(cell, pages_words)
            )

        data_row_count = max(0, table.row_count - len(header_rows))
        if len(header_parts) < min_cols or data_row_count < min_rows:
            rejected_tables.append(table)
            continue

        table_name = _get_table_name(table, result, table_idx)

        # Join a column's header lines into one label:
        #   ["NAME OF BORROWER", "CUSTOMER", "ACCOUNT NO."]
        #   -> label "NAME OF BORROWER/CUSTOMER/ACCOUNT NO."
        #   -> field "NameOfBorrowerCustomerAccountNo"
        col_fields: dict = {}
        for col_idx, parts in header_parts.items():
            raw_label = "/".join(parts)
            col_field_name = _to_camel_case(raw_label)
            if not col_field_name:
                continue
            col_fields[col_idx] = col_field_name
            dotted_key = f"{table_name}.{col_field_name}"
            if dotted_key not in aliases:
                aliases[dotted_key] = _column_aliases(
                    parts, alias_lines.get(col_idx, [])
                )

        if len(col_fields) < min_cols:
            rejected_tables.append(table)
            continue

        tables_map[table_name] = [col_fields[i] for i in sorted(col_fields)]

        for region in table.bounding_regions or []:
            bbox = _polygon_to_bbox(region.polygon)
            if bbox:
                blocked_regions.setdefault(region.page_number, []).append(bbox)

    return tables_map, aliases, blocked_regions, rejected_tables


# ---------------------------------------------------------------------------
# AZURE CALL
# ---------------------------------------------------------------------------

def _analyze_document(client: DocumentIntelligenceClient, doc_path: str) -> AnalyzeResult:
    """Analyze the document, requesting key-value pairs when available.

    ``prebuilt-layout`` only returns key-value pairs when the KEY_VALUE_PAIRS
    add-on is requested, and that add-on is not offered by every API version or
    region, so fall back to a plain layout call when it is rejected.
    """
    last_error = None

    for features in ([DocumentAnalysisFeature.KEY_VALUE_PAIRS], None):
        try:
            with open(doc_path, "rb") as handle:
                kwargs = {
                    "model_id": "prebuilt-layout",
                    "body": handle,
                    "output_content_format": "markdown",
                }
                if features:
                    kwargs["features"] = features
                poller = client.begin_analyze_document(**kwargs)
                return poller.result()
        except Exception as exc:  # noqa: BLE001 - re-raised below if both fail
            last_error = exc
            if features:
                print("[!] KEY_VALUE_PAIRS add-on unavailable, retrying without it.")
                print(f"    ({type(exc).__name__}: {exc})")

    raise RuntimeError(f"Document analysis failed: {last_error}")


# ---------------------------------------------------------------------------
# MAIN GENERATOR
# ---------------------------------------------------------------------------

def generate_field_keywords(doc_path: str, min_cols: int = 2, min_rows: int = 1):
    if not os.path.exists(doc_path):
        print(f"Error: '{doc_path}' not found.")
        sys.exit(1)

    base_name    = os.path.basename(doc_path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path  = os.path.join(OUTPUT_DIR, f"{base_name}_field_keywords.json")

    # --- Azure DocInt call ---------------------------------------------------
    client = DocumentIntelligenceClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY)
    )

    print(f"[*] Sending '{doc_path}' to Azure AI Document Intelligence...")
    start = time.time()
    result = _analyze_document(client, doc_path)
    print(f"[+] Analysis completed in {time.time() - start:.2f}s")

    keywords:    list[str]       = []
    aliases:     dict[str, list] = {}
    field_types: dict[str, str]  = {}
    seen:        set[str]        = set()
    # Labels already owned by a table column; a standalone field must never
    # duplicate one or it would compete with the column during matching.
    claimed_aliases: set[str] = set()

    def _add_field(field_name: str, field_aliases: list, field_type: str,
                   allow_statement: bool = False):
        if not field_name or field_name in seen:
            return
        # A contextual alias is a mapping, so the caption text has to be read
        # out of it before any of the string checks below can be applied.
        head = ""
        if field_aliases:
            first = field_aliases[0]
            head = str(first.get("label", "") if isinstance(first, dict) else first)
        if head.strip().upper() in claimed_aliases and head.strip():
            return
        # A paragraph is not a field. Selection-mark detection in particular can
        # latch onto whole blocks of declaration text. Checkbox captions are the
        # exception: "I presently do not hold any or am holding only one (1)
        # Credit Card..." is a printed sentence and is also the only thing that
        # names the field, so a single line of it is allowed through.
        if head:
            too_long = len(head.split()) > MAX_STATEMENT_WORDS
            if too_long or (not allow_statement and _looks_like_prose(head)):
                return
        seen.add(field_name)
        keywords.append(field_name)
        aliases[field_name]     = field_aliases
        field_types[field_name] = field_type

    # ---- 1. Tables ----------------------------------------------------------
    # Tables run first so their regions can be excluded from field detection.
    tables_map, table_aliases, blocked_regions, rejected_tables = _extract_tables(
        result, min_cols, min_rows
    )
    aliases.update(table_aliases)
    for alias_list in table_aliases.values():
        claimed_aliases.update(a.strip().upper() for a in alias_list if a.strip())

    # ---- 2. Text / fill-in fields from key-value pairs ----------------------
    kv_candidates = []
    for kv in result.key_value_pairs or []:
        if not kv.key or not kv.key.content:
            continue
        raw_label = _clean_label(kv.key.content)
        if not raw_label:
            continue

        # Skip pure checkbox entries — handled by selection-mark detection
        val_content = _clean_label(kv.value.content if kv.value else "")
        if _is_selection_value(val_content):
            continue
        if _is_value_like(raw_label) or _looks_like_prose(raw_label):
            continue

        kv_candidates.append((raw_label, kv))

    # A form puts a bare "DATE:" under each signature box. Those are different
    # fields, but nothing in the caption says so, so one name is derived twice
    # and the second field is dropped - taking its value with it. The printed
    # text above each copy is what tells them apart.
    repeated_kv = {
        label
        for label, count in Counter(label for label, _ in kv_candidates).items()
        if count > 1
    }
    page_index = _page_line_index(result) if repeated_kv else {}

    for raw_label, kv in kv_candidates:
        field_aliases = _split_slash_aliases(raw_label)
        name = raw_label

        if raw_label in repeated_kv:
            regions = getattr(kv.key, "bounding_regions", None) or []
            context = _context_for_region(
                page_index, regions[0] if regions else None, raw_label
            )
            if context:
                field_aliases = [{"label": raw_label, "context": context}]
                name = f"{context[0]} {raw_label}"

        _add_field(
            _to_camel_case(name),
            field_aliases,
            _detect_text_field_type(raw_label),
        )

    kv_field_count = len(keywords)

    # ---- 3. Checkbox / radio / YES-NO fields from selection marks -----------
    for field_name, info in _extract_selection_mark_fields(result).items():
        _add_field(field_name, info["aliases"], info["type"], allow_statement=True)
        # Carry through option labels when present
        if field_name in seen and info.get("options"):
            aliases[f"{field_name}.__options__"] = info["options"]

    # ---- 4. Geometric fallback ---------------------------------------------
    # prebuilt-layout returns key-value pairs only when the KEY_VALUE_PAIRS
    # add-on is available. When it is not, recover fill-in fields from the page
    # geometry so the config is never empty.
    if kv_field_count == 0:
        print("[*] No key-value pairs returned; deriving fields from page geometry.")
        for field_name, info in _extract_label_fields(result, blocked_regions).items():
            _add_field(field_name, info["aliases"], info["type"])

    # Tables that failed the size gate are usually boxed "LABEL : value" pairs.
    # Recover their labels as ordinary fields instead of discarding them.
    for table in rejected_tables:
        for field_name, info in _fields_from_rejected_table(table).items():
            _add_field(field_name, info["aliases"], info["type"])

    # ---- Promote table column dotted keys into keywords --------------------
    # TestOnTable.py requires a non-empty 'keywords' list to run.  For forms
    # that are table-only (no standalone KV fields), the table column fields
    # ARE the keywords — add them as "TableName.FieldName" entries so that
    # _load_field_keyword_config and _extract_fields_from_tables can match them.
    for table_name, col_field_names in tables_map.items():
        for col_field_name in col_field_names:
            dotted = f"{table_name}.{col_field_name}"
            if dotted not in seen:
                seen.add(dotted)
                keywords.append(dotted)
                field_types[dotted] = "text"   # default; refine manually if needed

    # ---- Write JSON ---------------------------------------------------------
    output: dict = {
        "keywords":    keywords,
        "field_types": field_types,
        "aliases":     aliases,
        # Scanned text never matches a caption character for character, so an
        # exact threshold silently drops fields whose caption OCR misread.
        "fuzzy_threshold": 0.9,
    }
    if tables_map:
        output["tables"] = tables_map

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Summary
    type_counts: dict[str, int] = {}
    for t in field_types.values():
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"[SUCCESS] Saved to: {output_path}")
    print(f"          {len(keywords)} field(s) detected: {type_counts}")
    if tables_map:
        print(f"          {len(tables_map)} table(s):")
        for tname, cols in tables_map.items():
            print(f"          - {tname}: {cols}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a *_field_keywords.json config from a form PDF."
    )
    parser.add_argument("document", help="Path to the PDF / image to analyse")
    parser.add_argument(
        "--min-cols", type=int, default=2,
        help="Minimum label-like header columns for a layout table to count "
             "as a real table (default: 2)",
    )
    parser.add_argument(
        "--min-rows", type=int, default=1,
        help="Minimum data rows below the header for a layout table to count "
             "as a real table (default: 1)",
    )
    args = parser.parse_args()
    generate_field_keywords(args.document, args.min_cols, args.min_rows)
