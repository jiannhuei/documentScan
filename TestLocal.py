"""
=============================================================================
STAGE 1 (Open-Source Fix): Table Extraction with PaddleOCR v3
=============================================================================
"""

import inspect
import json
import os
import re
import shutil
import statistics
import sys
import time
import tempfile
import types
import html
from difflib import SequenceMatcher
from io import StringIO

import pandas as pd
from PIL import Image, ImageFilter, ImageOps

# Force-disable oneDNN/PIR before any Paddle import to avoid the
# "ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute
# <pir::DoubleAttribute>]" crash in paddlepaddle 3.3.x on Windows CPU.
# setdefault() is insufficient here because PaddleX may override the values;
# direct assignment guarantees these are in place before paddle is imported.
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_new_executor"] = "0"


SUPPORTED_INPUT_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def _parse_csv_env(name: str):
    raw = os.environ.get(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _normalize_field_label(text: str):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _clean_field_value(text: str):
    if text is None:
        return ""
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    return cleaned


def _load_field_keyword_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Field config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    keywords = cfg.get("keywords")
    if not isinstance(keywords, list) or not keywords:
        raise ValueError("Field config must contain a non-empty 'keywords' list.")

    canonical_keywords = []
    for keyword in keywords:
        keyword_text = str(keyword).strip()
        if keyword_text:
            canonical_keywords.append(keyword_text)

    if not canonical_keywords:
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
    for keyword in canonical_keywords:
        lookup.append((keyword, _normalize_field_label(keyword)))
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
                    _normalize_field_label(item) for item in context_values if str(item).strip()
                ]
                if alias_text and normalized_context:
                    contextual_lookup.append(
                        {
                            "keyword": keyword,
                            "label": _normalize_field_label(alias_text),
                            "context": normalized_context,
                        }
                    )
                elif alias_text:
                    lookup.append((keyword, _normalize_field_label(alias_text)))
            else:
                alias_text = str(alias).strip()
                if alias_text:
                    lookup.append((keyword, _normalize_field_label(alias_text)))

    return {
        "keywords": canonical_keywords,
        "lookup": lookup,
        "contextual_lookup": contextual_lookup,
        "fuzzy_threshold": fuzzy_threshold,
        "field_types": cfg.get("field_types", {}),
        "tables": cfg.get("tables", {}),
    }


def _ocr_tolerant_threshold(configured: float, lookup_value: str) -> float:
    """How close a printed caption must be to a configured alias to count.

    An exact match is the wrong test for scanned text: OCR reliably turns
    "IDENTIFICATION" into "IDENTIRCATION", and demanding equality silently drops
    the whole field. The allowance scales with length because a fixed number of
    wrong characters matters far more in a short label - "FAX" and "TAX" differ
    by one character but mean different things, whereas a thirty-character
    caption is never mistaken for a different caption over one bad glyph.
    """
    if len(lookup_value) < 12:
        return configured
    return min(configured, max(0.88, 1.0 - 2.5 / len(lookup_value)))


def _best_keyword_match_local(normalized_label: str, candidates, fuzzy_threshold: float):
    if not normalized_label or not candidates:
        return None

    for keyword, lookup_value in candidates:
        if normalized_label == lookup_value:
            return keyword

    best_keyword = None
    best_score = 0.0
    best_threshold = 1.0
    for keyword, lookup_value in candidates:
        score = SequenceMatcher(None, normalized_label, lookup_value).ratio()
        if score > best_score:
            best_score = score
            best_keyword = keyword
            best_threshold = _ocr_tolerant_threshold(fuzzy_threshold, lookup_value)

    if best_keyword and best_score >= best_threshold:
        return best_keyword
    return None


def _context_present(context_value: str, normalized_context: str) -> bool:
    """Is a configured context phrase present in the surrounding text?

    Exact containment first, then a fuzzy test: OCR routinely mangles a
    character or two of the printed caption that anchors a context
    ("PRINCIPAL CARD APPLICANT" is read as "PRINCIPAL CARD APPLUCANT"), and an
    exact test would silently drop the whole field. The longest common run is
    used rather than a whole-string ratio because the caption is usually a
    small part of a much longer surrounding block.
    """
    if not context_value or not normalized_context:
        return False
    if context_value in normalized_context:
        return True
    if len(context_value) < 8:
        return False

    matcher = SequenceMatcher(None, context_value, normalized_context, autojunk=False)
    _, _, size = matcher.find_longest_match(
        0, len(context_value), 0, len(normalized_context)
    )
    return size / len(context_value) >= 0.7


def _has_contextual_anchor(field_cfg, normalized_context: str) -> bool:
    """Does this text anchor any contextual alias?"""
    for entry in field_cfg.get("contextual_lookup", []):
        if any(_context_present(ctx, normalized_context) for ctx in entry["context"]):
            return True
    return False


def _match_field_keyword(text: str, field_cfg, context_text: str = ""):
    normalized = _normalize_field_label(text)
    if not normalized:
        return None

    normalized_context = _normalize_field_label(context_text)
    contextual_candidates = []
    for entry in field_cfg.get("contextual_lookup", []):
        if any(_context_present(ctx, normalized_context) for ctx in entry["context"]):
            contextual_candidates.append((entry["keyword"], entry["label"]))

    contextual_match = _best_keyword_match_local(
        normalized, contextual_candidates, field_cfg["fuzzy_threshold"]
    )
    if contextual_match:
        return contextual_match

    return _best_keyword_match_local(normalized, field_cfg["lookup"], field_cfg["fuzzy_threshold"])


def _detect_grid_header_rows(grid):
    """Return the set of row indices that are header rows in a 2-D text grid."""
    last_header_row = 0
    for row_i in range(min(5, len(grid))):
        row = grid[row_i]
        non_empty = [c for c in row if str(c).strip()]
        has_digits = sum(
            1 for c in non_empty if re.search(r"^\d+(?:[.,]\d+)?$", str(c).strip())
        )
        if non_empty and has_digits / len(non_empty) >= 0.5:
            break
        last_header_row = row_i
    return set(range(last_header_row + 1))


def _build_grid_header_map(grid, header_row_set, field_cfg):
    """Map column index -> matched keyword using all header rows per column.

    Only keywords declared as table columns qualify. The layout engine wraps
    whole form sections in a <table>, so a form's captions and answers land in
    a grid even though they are not tabular data; matching scalar fields here
    would turn every such section into a table of its own captions.
    """
    header_lines = {}
    for row_i in sorted(header_row_set):
        if row_i >= len(grid):
            continue
        for col_i, cell in enumerate(grid[row_i]):
            part = re.sub(r"^\*+\s*", "", _clean_field_value(str(cell))).rstrip("/").strip()
            if part:
                header_lines.setdefault(col_i, []).append(part)

    def _column_keyword(text):
        matched = _match_field_keyword(text, field_cfg)
        return matched if matched and "." in matched else None

    header_by_col = {}
    for col_idx, parts in header_lines.items():
        matched = None
        for part in parts:
            matched = _column_keyword(part)
            if matched:
                break
        if not matched:
            matched = _column_keyword("/".join(parts))
        if not matched:
            matched = _column_keyword(" ".join(parts))
        if matched:
            header_by_col[col_idx] = matched
    return header_by_col


def _extract_records_from_table_grid(grid, field_cfg):
    if not grid or len(grid) < 2:
        return []

    header_row_set = _detect_grid_header_rows(grid)
    header_by_col = _build_grid_header_map(grid, header_row_set, field_cfg)

    if not header_by_col:
        return []

    records = []
    for row_i, row in enumerate(grid):
        if row_i in header_row_set:
            continue
        record = {}
        for col_idx, keyword in header_by_col.items():
            if col_idx >= len(row):
                continue
            value = _clean_field_value(row[col_idx])
            if value:
                record[keyword] = value
        if record:
            records.append(record)

    return records


def _extract_table_data_from_grid(grid, field_cfg):
    """Extract structured table data from a 2-D text grid with multi-row header support.

    Returns {"columns": [...], "rows": [{col: val, ...}, ...]} or None if no mapping found.
    """
    if not grid or len(grid) < 2:
        return None

    header_row_set = _detect_grid_header_rows(grid)
    header_by_col = _build_grid_header_map(grid, header_row_set, field_cfg)

    if not header_by_col:
        return None

    columns = [header_by_col[i] for i in sorted(header_by_col)]
    rows = []
    for row_i, row in enumerate(grid):
        if row_i in header_row_set:
            continue
        record = {}
        for col_idx, keyword in header_by_col.items():
            if col_idx >= len(row):
                continue
            value = _clean_field_value(row[col_idx])
            if value:
                record[keyword] = value
        if record:
            rows.append(record)

    if not rows:
        return None

    return {"columns": columns, "rows": rows}


# Characters that separate a label from its value, or that OCR picks up from
# the printed rule of a fill-in box. They are never part of the value itself.
_VALUE_EDGE_CHARS = " \t:;-_.\u2022\u2022%*\u2500\u2014\u2013"

# A fill-in value may wrap over several printed lines (a postal address is the
# common case). Beyond this the extractor is almost certainly running into
# unrelated content.
MAX_VALUE_LINES = 4

# How far back to look for the printed caption that anchors a contextual alias,
# e.g. the "SIGNATURE: PRINCIPAL CARD APPLICANT" above a bare "DATE".
MAX_CONTEXT_LINES = 4

# The only values a checkbox-like field can legitimately hold.
_SELECTION_TOKENS = {
    "YES", "NO", "TRUE", "FALSE", "SELECTED", "UNSELECTED",
    "- [X]", "- [ ]", "- (\u25CF)", "- ( )",
}

_SELECTION_TYPES = {"yesno", "checkbox", "option"}


def _strip_value_edges(text):
    """Remove label separators and box-rule artefacts from a candidate value."""
    return _clean_field_value(text).strip(_VALUE_EDGE_CHARS)


def _is_label_furniture(text, field_cfg, context_text=""):
    """Is this text part of the printed *label* rather than a value?

    Two cases, both of which mean "keep looking for the value":
      - the text is (mostly) a configured label itself
      - the text is a parenthetical qualifier such as "(MOBILE)" or
        "(MANDATORY FOR SST REGISTRANT)", which belongs to the label it follows
    """
    cleaned = text.strip()
    if not cleaned:
        return False

    if cleaned.startswith("(") and cleaned.endswith(")"):
        return True

    # A line that is a label outright, however short its alias ("NAME",
    # "EMAIL", "EPF NO."). The coverage rule below cannot see these because it
    # ignores aliases under 6 characters. The context lets a bare "DATE"
    # under a signature box be recognised as the label it is.
    if _match_field_keyword(cleaned, field_cfg, context_text):
        return True

    normalized = _normalize_field_label(cleaned)
    if not normalized:
        return False
    for _keyword, lookup_value in field_cfg["lookup"]:
        if len(lookup_value) < 6:
            continue
        if lookup_value in normalized and len(lookup_value) / len(normalized) >= 0.5:
            return True
        # The reverse case: a caption that spans several printed lines is
        # configured as one long alias, so each line it is built from is a
        # fragment of it. "POSITION HELD" is a caption, not an answer, even
        # though it is only a small part of the alias that contains it.
        if len(normalized) >= 8 and normalized in lookup_value:
            return True
    return False


def _looks_like_form_text(text, field_cfg, context_text=""):
    """Is this candidate printed form text rather than something a person wrote?

    A fill-in value is short and specific. Printed labels, parenthetical
    qualifiers, instructions and questions are none of those. Every rule here is
    document-agnostic - no hard-coded phrases.
    """
    cleaned = text.strip()
    if len(cleaned) < 2:
        return True

    # Questions are printed prompts, never answers.
    if cleaned.endswith("?"):
        return True

    # A bare one or two digit line is a section or page number. Real fill-in
    # numbers (account numbers, amounts, dates) are longer or punctuated.
    if cleaned.isdigit() and len(cleaned) <= 2:
        return True

    if _looks_like_prose(cleaned):
        return True

    if _is_label_furniture(cleaned, field_cfg, context_text):
        return True

    # Printed captions are upper-case running text. Handwriting that survives
    # OCR nearly always carries lower-case letters or digits, so upper-case
    # multi-word text with neither is form furniture.
    words = cleaned.split()
    # A leading section number belongs to the printed heading, so it must not
    # be the digit that makes a line look hand-filled.
    if len(words) >= 2 and words[0].isdigit() and len(words[0]) <= 2:
        words = words[1:]
    body = " ".join(words)
    if len(words) >= 3 and not any(ch.isdigit() for ch in body):
        letters = [ch for ch in body if ch.isalpha()]
        if letters and not any(ch.islower() for ch in letters):
            return True

    return False


def _extract_fields_from_text_lines(text_lines, field_cfg):
    fields = {}
    cleaned_lines = [_clean_field_value(line) for line in text_lines]
    field_types = field_cfg.get("field_types", {}) or {}
    contextual_keywords = {
        entry["keyword"] for entry in field_cfg.get("contextual_lookup", [])
    }

    def _context_for(idx):
        """Nearest preceding lines that anchor a contextual alias.

        The window grows one line at a time and stops at the first line that
        anchors a context phrase, so the *closest* caption wins when a form
        repeats a bare label like "DATE" under several signature boxes.
        """
        window = []
        for back in range(1, MAX_CONTEXT_LINES + 1):
            prev = idx - back
            if prev < 0:
                break
            window.insert(0, cleaned_lines[prev])
            joined = " ".join(window)
            if _has_contextual_anchor(field_cfg, _normalize_field_label(joined)):
                return joined
        return " ".join(window)

    def _assign(keyword, value):
        if not value:
            return
        # Selection fields carry a tick, not prose. Without a detected mark the
        # honest answer is "unknown", never the text of the options beside it.
        if field_types.get(keyword) in _SELECTION_TYPES:
            if value.strip().upper() not in _SELECTION_TOKENS:
                return
        # A contextual alias is anchored to one specific spot on the page, so
        # the first hit is the anchored one. Later repeats of the same bare
        # label elsewhere on the page must not overwrite it.
        if keyword in contextual_keywords and fields.get(keyword):
            return
        fields[keyword] = value

    def _max_lines_for(keyword):
        """Only free text wraps. A date, number or signature is one line."""
        return MAX_VALUE_LINES if field_types.get(keyword, "text") == "text" else 1

    def _continuation_value(start_idx, max_lines):
        """Collect the value line(s) printed below a label line."""
        parts = []
        for idx in range(start_idx, min(start_idx + max_lines, len(cleaned_lines))):
            candidate = _strip_value_edges(cleaned_lines[idx])
            if not candidate:
                break
            if _looks_like_form_text(candidate, field_cfg, _context_for(idx)):
                break
            parts.append(candidate)
        return " ".join(parts)

    def _resolve_value(remainder, label_idx, keyword, context_text):
        """Decide a field's value from the text left on the label line."""
        remainder = _strip_value_edges(remainder)
        # A parenthetical or a second label extends the label; the value is
        # printed further down.
        if remainder and _is_label_furniture(remainder, field_cfg, context_text):
            remainder = ""
        if remainder:
            if _looks_like_form_text(remainder, field_cfg, context_text):
                return ""
            return remainder
        return _continuation_value(label_idx + 1, _max_lines_for(keyword))

    for idx, line in enumerate(cleaned_lines):
        if not line:
            continue

        context_text = _context_for(idx)

        if ":" in line:
            left, right = line.split(":", 1)
            keyword = _match_field_keyword(left.strip(), field_cfg, context_text)
            if keyword:
                _assign(keyword, _resolve_value(right, idx, keyword, context_text))
                continue

        words = line.split()
        for prefix_len in range(min(6, len(words)), 0, -1):
            keyword = _match_field_keyword(
                " ".join(words[:prefix_len]), field_cfg, context_text
            )
            if not keyword:
                continue
            remainder = " ".join(words[prefix_len:])
            _assign(keyword, _resolve_value(remainder, idx, keyword, context_text))
            break

    return fields


def _assess_field_value(keyword: str, value: str, field_types: dict = None):
    """Score a field value based on its type hint and content quality."""
    cleaned = _clean_field_value(value)
    if not cleaned:
        return 0.0, "not found"

    if field_types is None:
        field_types = {}

    field_type = field_types.get(keyword, "text")

    if field_type == "date":
        if re.search(
            r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", cleaned
        ):
            return 0.95, "valid date format"
        return 0.3, "date format not recognized"

    if field_type == "number":
        if re.search(r"\b\d+(?:[.,]\d+)?\b", cleaned):
            return 0.9, "valid numeric format"
        return 0.3, "numeric format not recognized"

    if field_type in {"yesno", "checkbox", "option"}:
        upper = cleaned.upper()
        if upper in {"YES", "NO", "TRUE", "FALSE", "SELECTED", "UNSELECTED"}:
            return 0.9, "checkbox selection"
        if cleaned in {"- [x]", "- [ ]", "- (●)", "- ( )"}:
            return 0.9, "checkbox symbol"
        return 0.5, "non-standard checkbox value"

    if field_type == "signature":
        if len(cleaned) >= 2:
            return 0.8, "signature present"
        return 0.2, "signature too short"

    if len(cleaned) >= 3:
        return 0.8, "text value captured"
    return 0.4, "text value too short"


def _sanitize_field_value(keyword: str, value: str, field_types: dict = None):
    """Remove known-bad placeholder values for a field."""
    cleaned = _clean_field_value(value)
    if not cleaned:
        return ""
    return cleaned


def _build_field_details(canonical_fields, field_types: dict = None):
    """Build field detail dicts with confidence scoring for all keywords."""
    if field_types is None:
        field_types = {}

    details = {}
    for keyword, raw_value in canonical_fields.items():
        sanitized = _sanitize_field_value(keyword, raw_value, field_types)
        confidence, reason = _assess_field_value(keyword, sanitized, field_types)
        source = "matched" if sanitized else "not-found"
        details[keyword] = {
            "value": sanitized,
            "confidence": round(confidence, 2),
            "reason": reason,
            "source": source,
        }
    return details


def _match_field_keyword_by_substring(text: str, field_cfg):
    """Return a sorted list of (keyword, alias_length) where the alias appears
    as a substring of the normalized line.  Requires alias length >= 6 chars.
    Sorted longest alias first to give priority to more-specific matches.
    """
    normalized = _normalize_field_label(text)
    if not normalized:
        return []
    best = {}  # keyword → best matching alias length
    for keyword, lookup_value in field_cfg["lookup"]:
        if lookup_value and len(lookup_value) >= 6 and lookup_value in normalized:
            if keyword not in best or len(lookup_value) > best[keyword]:
                best[keyword] = len(lookup_value)
    return sorted(best.items(), key=lambda x: -x[1])


def _reconstruct_tables_from_text(text_lines, field_cfg, start_counter=0):
    """Reconstruct structured table data from OCR text when HTML tables are
    not detected by the layout engine.

    Identifies column-header lines using substring alias matching, but only
    considers aliases that are at least 12 characters long (normalized).  This
    prevents short, generic aliases such as "guarantee" (9) or "nameof" (6)
    from matching individual OCR words in footnotes or running text.

    Returns:
        tables       – {"Table1": {"columns":[...], "rows":[...]}, ...}
        header_idxs  – set of line indices belonging to detected header sequences
    """
    # Only aliases at least this long are used for header detection.
    # 6 chars is enough to exclude trivial 1–5 char tokens while still
    # matching short but unambiguous aliases like "nricno" (6) or "nameof" (6).
    MIN_ALIAS_LEN = 6

    def _header_kw_for_line(line, seen_kw):
        """Return the best unseen keyword whose alias (>= MIN_ALIAS_LEN chars)
        appears as a substring of the normalized line, or None.

        Only keywords declared as table columns qualify. A scalar field names a
        single answer on a form line, so treating one as a column header turns
        ordinary body text into an invented table.
        """
        normalized = _normalize_field_label(line)
        if not normalized:
            return None
        best = {}  # keyword -> best alias length
        for keyword, lookup_value in field_cfg["lookup"]:
            if "." not in keyword:
                continue
            if (
                lookup_value
                and len(lookup_value) >= MIN_ALIAS_LEN
                and lookup_value in normalized
            ):
                if keyword not in best or len(lookup_value) > best[keyword]:
                    best[keyword] = len(lookup_value)
        for kw, _ in sorted(best.items(), key=lambda x: -x[1]):
            if kw not in seen_kw:
                return kw
        return None

    cleaned = [_clean_field_value(l) for l in text_lines]
    tables = {}
    header_indices = set()
    table_idx = start_counter
    i = 0
    n = len(cleaned)

    while i < n:
        if not cleaned[i]:
            i += 1
            continue

        # Find a header sequence: consecutive lines where each line contains
        # the normalized form of some keyword alias as a substring, and each
        # matching keyword is distinct.
        header_seq = []
        header_idxs_local = []
        seen_kw = set()
        j = i

        while j < n:
            line = cleaned[j]
            if not line:
                j += 1
                continue
            kw = _header_kw_for_line(line, seen_kw)
            if kw:
                header_seq.append(kw)
                header_idxs_local.append(j)
                seen_kw.add(kw)
                j += 1
            else:
                break

        if len(header_seq) >= 2:
            # Filter header sequence to the dominant table prefix to avoid
            # mixing columns from Table1 and Table2 when aliases overlap.
            def _kw_table_prefix(kw):
                parts = kw.split(".")
                return parts[0] if len(parts) > 1 else ""

            from collections import Counter as _Counter
            _prefix_count = _Counter(_kw_table_prefix(k) for k in header_seq)
            _dominant = _prefix_count.most_common(1)[0][0]
            if _dominant:
                _filtered = [
                    (kw, idx)
                    for kw, idx in zip(header_seq, header_idxs_local)
                    if _kw_table_prefix(kw) == _dominant
                ]
                if len(_filtered) >= 2:
                    header_seq = [p[0] for p in _filtered]
                    # header_idxs_local kept intact: ALL detected header lines
                    # are excluded from field extraction even if dropped here.

            num_cols = len(header_seq)
            header_indices.update(header_idxs_local)
            rows = []
            k = j

            while True:
                # Collect next num_cols non-empty lines as one data row.
                row_vals = []
                m = k
                while m < n and len(row_vals) < num_cols:
                    line = cleaned[m]
                    m += 1
                    if not line:
                        continue
                    row_vals.append(line)

                if not row_vals:
                    k = m
                    break

                # If the first collected value maps to a keyword from the
                # current header set → we’ve hit a repeated/new header → stop.
                first_kw = _header_kw_for_line(row_vals[0], set())
                if first_kw and first_kw in seen_kw:
                    break

                record = {
                    header_seq[ci]: row_vals[ci]
                    for ci in range(min(num_cols, len(row_vals)))
                }
                if any(v for v in record.values()):
                    rows.append(record)
                k = m

            if rows:
                table_idx += 1
                tables[f"Table{table_idx}"] = {"columns": header_seq, "rows": rows}

            i = k if k > j else j
        else:
            i += 1

    return tables, header_indices


def _looks_like_prose(text):
    """Heuristic: is this OCR cell a sentence/paragraph rather than a table value?

    Table cell values are short. Narrative body text that happens to fall inside
    a table's x-range is long and sentence-like. This is document-agnostic --
    no hard-coded boilerplate phrases.
    """
    words = text.split()
    if len(words) >= 8:
        return True
    # Short but clearly a sentence fragment (ends with '.' and has a verb-ish
    # lowercase run) -- e.g. "shall vest in my estate."
    if len(words) >= 5 and text.rstrip().endswith((".", ":", ";")):
        return True
    return False


def _reconstruct_tables_geometric(page_output, field_cfg, start_counter=0):
    """Reconstruct table data using OCR cell x/y coordinates.

    Identifies column positions by finding cells whose text matches a keyword
    alias (header cells), then assigns data cells to columns by x-overlap.
    This is far more reliable than text-sequence matching for complex tables
    where PaddleOCR reads cells column-by-column or in non-sequential order.

    Returns:
        tables      – {"TableN": {"columns":[...], "rows":[...]}, ...}
        header_idxs – always empty set (geometry-based; index concept n/a)
    """
    MIN_ALIAS_LEN = 6

    cells = _extract_ocr_cells([page_output])
    if not cells:
        return {}, set()

    # Diagnostic: dump cell geometry when OCR_GEO_DEBUG=1 so column/row
    # assignment problems can be inspected for a given document.
    if os.environ.get("OCR_GEO_DEBUG") == "1":
        with open(os.path.join("DocIntResult", "geo_cells_debug.json"), "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "t": c["text"],
                        "x0": round(c["x0"], 1),
                        "x1": round(c["x1"], 1),
                        "y0": round(c["y0"], 1),
                    }
                    for c in cells
                ],
                f,
                indent=1,
                ensure_ascii=False,
            )

    # Require at least 25% of cells to have real (non-fallback) x > 1.
    if sum(1 for c in cells if c["x0"] > 1.0) < len(cells) * 0.25:
        return {}, set()

    rows = _group_cells_by_row(cells)
    if len(rows) < 2:
        return {}, set()

    # ------------------------------------------------------------------
    # Step 1: Find keyword → column x-range from cells that match aliases
    # ------------------------------------------------------------------
    # A header cell must be *mostly* the alias. Requiring high coverage stops
    # short aliases (e.g. "Proportion", "Relative") from matching body text
    # that merely mentions the word, which would otherwise stretch the
    # column's x-range and header band across the whole page.
    MIN_COVERAGE = 0.5

    def _best_kw(text):
        """Return (keyword, coverage) for the best alias match, else (None, 0.0).

        Only keywords declared as table columns are considered. A scalar field
        is a caption on a form line, and letting one define a column invents a
        table out of ordinary captions that happen to share a row - two
        unrelated fields side by side would otherwise become a one-row table
        and swallow the values belonging to the fields themselves.

        Coverage measures how strong the match is and is used to break ties
        between candidate header positions for the same keyword.
        """
        normalized = _normalize_field_label(text)
        if not normalized:
            return None, 0.0
        best_kw, best_len, best_cov = None, 0, 0.0
        for keyword, lookup_value in field_cfg["lookup"]:
            if "." not in keyword:
                continue
            if not lookup_value or len(lookup_value) < MIN_ALIAS_LEN:
                continue
            if lookup_value in normalized:
                # Alias inside the cell: how much of the cell is the alias?
                coverage = len(lookup_value) / len(normalized)
            elif len(normalized) >= MIN_ALIAS_LEN and normalized in lookup_value:
                # Cell is a fragment of a multi-line header: how much of the
                # alias does this fragment account for?
                coverage = len(normalized) / len(lookup_value)
            else:
                continue
            if coverage < MIN_COVERAGE:
                continue
            if len(lookup_value) > best_len:
                best_len = len(lookup_value)
                best_kw = keyword
                best_cov = coverage
        return best_kw, best_cov

    # Estimate a typical row height. It drives both the header-band clustering
    # tolerance below and the data-region thresholds in step 3.
    row_gaps = [
        rows[i]["cy"] - rows[i - 1]["cy"]
        for i in range(1, len(rows))
        if 5.0 < rows[i]["cy"] - rows[i - 1]["cy"] < 300.0
    ]
    typical_h = statistics.median(row_gaps) if row_gaps else 30.0

    # Collect every header match individually. The same wording often appears
    # in more than one table on a page, so matches must not be merged into one
    # x/y range up front - they are grouped into header bands first.
    matches = []
    for row in rows:
        for cell in row["cells"]:
            kw, coverage = _best_kw(cell["text"])
            if kw:
                matches.append(
                    {
                        "y": row["cy"],
                        "kw": kw,
                        "x0": cell["x0"],
                        "x1": cell["x1"],
                        "cov": coverage,
                    }
                )

    if not matches:
        return {}, set()

    # ------------------------------------------------------------------
    # Step 2: Cluster matches into header bands and give each band to one table
    # ------------------------------------------------------------------
    # A multi-line header spans a few consecutive rows; another table's header
    # sits far below. Split wherever the vertical gap exceeds a few row heights.
    matches.sort(key=lambda m: m["y"])
    band_gap = max(typical_h * 3.0, 60.0)
    bands = [[matches[0]]]
    for match in matches[1:]:
        if match["y"] - bands[-1][-1]["y"] > band_gap:
            bands.append([])
        bands[-1].append(match)

    def _prefix(kw):
        parts = kw.split(".")
        return parts[0] if len(parts) > 1 else "Table"

    # Each band belongs to the table prefix it matches most, and each prefix
    # keeps the single band where it is best represented.
    best_band: dict = {}
    for band in bands:
        by_prefix: dict = {}
        for match in band:
            by_prefix.setdefault(_prefix(match["kw"]), set()).add(match["kw"])
        owner = max(by_prefix.items(), key=lambda kv: len(kv[1]))[0]
        score = len(by_prefix[owner])
        if score > best_band.get(owner, (0, None))[0]:
            best_band[owner] = (score, band)

    table_groups: dict = {}
    for prefix, (_score, band) in best_band.items():
        per_kw: dict = {}
        for match in band:
            if _prefix(match["kw"]) == prefix:
                per_kw.setdefault(match["kw"], []).append(match)

        cols: dict = {}
        for kw, kw_matches in per_kw.items():
            # A header word such as "NAME OF" often repeats above a different
            # column. Cluster the matches by x-overlap and keep the dominant
            # cluster so one column's x-range cannot swallow its neighbour.
            clusters: list = []
            for match in sorted(kw_matches, key=lambda m: m["x0"]):
                if clusters and match["x0"] <= clusters[-1]["x1"]:
                    clusters[-1]["x1"] = max(clusters[-1]["x1"], match["x1"])
                    clusters[-1]["items"].append(match)
                else:
                    clusters.append(
                        {"x0": match["x0"], "x1": match["x1"], "items": [match]}
                    )
            # Score by total match strength: several confident header lines
            # stacked in one column beat a single weak fragment elsewhere.
            best = max(
                clusters,
                key=lambda c: (
                    sum(m["cov"] for m in c["items"]),
                    -(c["x1"] - c["x0"]),
                ),
            )
            cols[kw] = {
                "x0": best["x0"],
                "x1": best["x1"],
                "min_header_y": min(m["y"] for m in kw_matches),
                "max_header_y": max(m["y"] for m in kw_matches),
            }
        if len(cols) >= 2:
            table_groups[prefix] = cols

    if not table_groups:
        return {}, set()

    if os.environ.get("OCR_GEO_DEBUG") == "1":
        print("[geo] column_map:")
        for _p, _cols in sorted(table_groups.items()):
            for _k, _i in sorted(_cols.items(), key=lambda x: x[1]["x0"]):
                print(f"[geo]   {_k}: x=({_i['x0']:.0f},{_i['x1']:.0f}) "
                      f"hdr_y=({_i['min_header_y']:.0f},{_i['max_header_y']:.0f})")

    # ------------------------------------------------------------------
    # Step 3: For each table group assign data cells to columns by x-overlap
    # ------------------------------------------------------------------
    result_tables: dict = {}
    table_idx = start_counter

    # Each group's data region must not run into another group's header band.
    group_header_min = {
        prefix: min(info["min_header_y"] for info in cols.values())
        for prefix, cols in table_groups.items()
    }

    for prefix, cols in sorted(table_groups.items()):
        if len(cols) < 2:
            continue

        sorted_cols = sorted(cols.items(), key=lambda x: x[1]["x0"])
        col_keywords = [kw for kw, _ in sorted_cols]

        group_max_y = max(info["max_header_y"] for _, info in sorted_cols)
        data_y_start = group_max_y + typical_h * 0.5

        # Stop before the next table's header band begins.
        later_headers = [
            y for p, y in group_header_min.items() if p != prefix and y > data_y_start
        ]
        data_y_end = min(later_headers) if later_headers else float("inf")

        g_x_min = min(info["x0"] for _, info in sorted_cols)
        g_x_max = max(info["x1"] for _, info in sorted_cols)
        x_pad = max(10.0, (g_x_max - g_x_min) * 0.05)

        table_rows = []
        last_row_y = None
        for row in rows:
            if row["cy"] < data_y_start or row["cy"] >= data_y_end:
                continue

            # A gap far larger than a normal row means the table has ended.
            if last_row_y is not None and (row["cy"] - last_row_y) > typical_h * 2.0:
                break

            in_range = [
                c for c in row["cells"]
                if c["x1"] >= g_x_min - x_pad and c["x0"] <= g_x_max + x_pad
            ]
            if not in_range:
                continue

            record: dict = {}
            first_col_x0 = sorted_cols[0][1]["x0"]
            for cell in in_range:
                text = cell["text"].strip()
                # Strip leading row-labels like "a) ", "b) "
                text = re.sub(r"^[a-z]\)\s+", "", text)
                # Skip standalone markers ("a)", "b)")
                if re.match(r"^[a-z]\)$", text.lower()):
                    continue
                # Skip bare row numbers, but only in the gutter left of the
                # first column - inside a column a short number is real data
                # (a proportion, a count, a short reference).
                if re.match(r"^\d{1,3}$", text) and cell["x1"] <= first_col_x0:
                    continue
                if len(text) < 2:
                    continue
                if not text:
                    continue
                # Skip repeated header text and narrative body text.
                if _best_kw(text)[0] is not None:
                    continue
                if _looks_like_prose(text):
                    continue

                best_kw, best_overlap = None, -1.0
                for kw, info in sorted_cols:
                    overlap = max(
                        0.0,
                        min(cell["x1"], info["x1"] + x_pad)
                        - max(cell["x0"], info["x0"] - x_pad),
                    )
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_kw = kw

                if not best_kw or best_overlap <= 0:
                    continue

                # Skip narrative paragraphs: a data cell should not be much
                # wider than the column it belongs to.
                col_info = cols[best_kw]
                col_w = max(1.0, col_info["x1"] - col_info["x0"])
                if (cell["x1"] - cell["x0"]) > col_w * 2.5:
                    continue

                if best_kw in record:
                    record[best_kw] = f"{record[best_kw]} {text}".strip()
                else:
                    record[best_kw] = text

            # A genuine row of a multi-column table fills at least two of its
            # columns. Rows with a single value are stray page text (header
            # remnants, totals, narrative lines) that merely happen to sit
            # inside the column x-range, so they neither become rows nor
            # anchor the end-of-table gap detection.
            if sum(1 for v in record.values() if v) >= 2:
                table_rows.append(record)
                last_row_y = row["cy"]

        multi_col_rows = len(table_rows)
        if os.environ.get("OCR_GEO_DEBUG") == "1":
            print(f"[geo] {prefix}: y_range=({data_y_start:.0f},{data_y_end:.0f}) "
                  f"x_range=({g_x_min:.0f},{g_x_max:.0f}) typical_h={typical_h:.0f} "
                  f"rows={len(table_rows)} multi={multi_col_rows}")
            for _r in table_rows:
                print(f"[geo]     {_r}")
        if not table_rows:
            continue

        table_idx += 1
        result_tables[f"Table{table_idx}"] = {
            "columns": col_keywords,
            "rows": table_rows,
        }

    return result_tables, set()


def _build_table_keyword_set(field_cfg):
    """Return the set of full keywords that are declared as table columns.
    These keywords should be populated from table structure, not free text.
    """
    keyword_set = set(field_cfg["keywords"])
    table_kws = set()
    for table_name, columns in field_cfg.get("tables", {}).items():
        for col in columns:
            full_kw = f"{table_name}.{col}"
            if full_kw in keyword_set:
                table_kws.add(full_kw)
    return table_kws


def _extract_field_only_data(page_outputs, field_cfg):
    outputs = page_outputs if isinstance(page_outputs, list) else [page_outputs]
    aggregate = {
        "fields": {},
        "tableData": {},
        "html_table_names": set(),   # HTML-detected tables  (safe to promote)
        "geo_table_names": set(),    # geometric-reconstructed (also safe to promote)
        "unmatchedKeywords": list(field_cfg["keywords"]),
    }

    # Keywords that belong to tables should only come from table detection,
    # not from free-text extraction (avoids false matches from header text).
    table_kw_set = _build_table_keyword_set(field_cfg)

    table_counter = 0
    for output in outputs:
        blocks = _extract_layout_blocks([output])
        page_lines = []

        for label, content in blocks:
            if label == "table" and isinstance(content, str) and "<table" in content:
                grid = _expand_html_table_to_grid(content)
                if grid:
                    table_counter += 1
                    table_name = f"Table{table_counter}"
                    table_struct = _extract_table_data_from_grid(grid, field_cfg)
                    if table_struct:
                        aggregate["tableData"][table_name] = table_struct
                        aggregate["html_table_names"].add(table_name)
                continue

            if label != "table":
                normalized = _normalize_layout_text(content)
                if normalized:
                    # Split into individual lines so the reconstructor can
                    # inspect each line separately.
                    for subline in normalized.split("\n"):
                        subline = subline.strip()
                        if subline:
                            page_lines.append(subline)

        ocr_lines = _extract_ocr_lines([output])
        if ocr_lines:
            page_lines.extend(ocr_lines)

        # Debug dump: write page_lines to a JSON file when OCR_DEBUG_DUMP=1.
        if os.environ.get("OCR_DEBUG_DUMP") == "1":
            debug_path = os.path.join(
                os.path.dirname(FIELD_KEYWORD_CONFIG),
                os.path.basename(FIELD_KEYWORD_CONFIG).replace("_field_keywords.json", "_page_lines_debug.json"),
            )
            import json as _json
            with open(debug_path, "w", encoding="utf-8") as _f:
                _json.dump({"page_lines": page_lines}, _f, indent=2, ensure_ascii=False)

        # Geometric reconstruction (primary fallback when no HTML tables detected).
        # Uses OCR cell x/y coordinates to assign data to the correct column.
        geo_tables, _ = _reconstruct_tables_geometric(output, field_cfg, table_counter)
        for name, struct in geo_tables.items():
            if name not in aggregate["tableData"]:
                aggregate["tableData"][name] = struct
                aggregate["geo_table_names"].add(name)
        if geo_tables:
            table_counter += len(geo_tables)

        # Text-sequence reconstruction only as a last resort, when neither
        # HTML detection nor geometric reconstruction produced anything.
        if not geo_tables and not aggregate["html_table_names"]:
            text_tables, header_idxs = _reconstruct_tables_from_text(
                page_lines, field_cfg, table_counter
            )
            for name, struct in text_tables.items():
                if name not in aggregate["tableData"]:
                    aggregate["tableData"][name] = struct
            if text_tables:
                table_counter += len(text_tables)
        else:
            text_tables, header_idxs = {}, set()

        # Blank out header lines rather than dropping them (prevents table
        # header text from being captured as a field label, while keeping every
        # other line at its original index - a value is located by its position
        # relative to its label, so deleting lines would pair labels with
        # whatever text happened to close the gap).
        field_lines = [
            "" if idx in header_idxs else l for idx, l in enumerate(page_lines)
        ]
        text_fields = _extract_fields_from_text_lines(field_lines, field_cfg)
        # Remove any table-scoped keywords — those must come from table data.
        for kw in table_kw_set:
            text_fields.pop(kw, None)
        aggregate["fields"].update(text_fields)

    # Drop duplicate or subsumed tables. PaddleOCR often reports the same table
    # via both the layout parse and the dedicated table detector, and the HTML
    # parse frequently recovers only a fragment of a table that the geometric
    # reconstruction captured in full.
    def _row_items(row):
        return frozenset((k, v) for k, v in row.items() if v)

    def _subsumes(bigger, smaller):
        big_cols = set(bigger.get("columns", []))
        if not set(smaller.get("columns", [])) <= big_cols:
            return False
        big_rows = [_row_items(r) for r in bigger.get("rows", [])]
        matched = 0
        for row in smaller.get("rows", []):
            items = _row_items(row)
            # Single-value rows (totals, stray text) do not block subsumption.
            if len(items) < 2:
                continue
            if not any(items <= b for b in big_rows):
                return False
            matched += 1
        return matched > 0

    ordered = sorted(
        aggregate["tableData"].items(),
        key=lambda kv: len(kv[1].get("rows", [])),
        reverse=True,
    )
    kept: list = []
    for name, struct in ordered:
        if any(_subsumes(k_struct, struct) for _, k_struct in kept):
            aggregate["html_table_names"].discard(name)
            aggregate["geo_table_names"].discard(name)
            continue
        kept.append((name, struct))

    kept_names = {name for name, _ in kept}
    aggregate["tableData"] = {
        name: struct
        for name, struct in aggregate["tableData"].items()
        if name in kept_names
    }

    found = set(aggregate["fields"].keys())
    for table_struct in aggregate["tableData"].values():
        for row in table_struct.get("rows", []):
            found.update(row.keys())
    aggregate["unmatchedKeywords"] = [k for k in field_cfg["keywords"] if k not in found]
    return aggregate


def _render_field_only_markdown(field_data):
    lines = ["# Extracted Fields", ""]

    if field_data.get("fields"):
        lines.extend(["## Single Fields", ""])
        for key, value in field_data["fields"].items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    table_data = field_data.get("tableData", {})
    if table_data:
        for table_name, table_struct in table_data.items():
            lines.extend([f"## {table_name}", ""])
            columns = table_struct.get("columns", [])
            rows = table_struct.get("rows", [])
            if columns:
                lines.append("| " + " | ".join(columns) + " |")
                lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
                for row in rows:
                    cells = [str(row.get(col, "")).replace("|", "\\|") for col in columns]
                    lines.append("| " + " | ".join(cells) + " |")
                lines.append("")

    field_details = field_data.get("fieldDetails", {})
    if field_details:
        high_conf = {
            k: v for k, v in field_details.items()
            if v["confidence"] >= 0.7 and v["value"]
        }
        if high_conf:
            lines.extend(["## Field Details (High Confidence)", ""])
            for key, detail in high_conf.items():
                lines.append(
                    f"- **{key}**: {detail['value']} "
                    f"_(confidence: {detail['confidence']}, {detail['reason']})_"
                )
            lines.append("")

    if field_data.get("unmatchedKeywords"):
        lines.extend(["## Unmatched Keywords", ""])
        for keyword in field_data["unmatchedKeywords"]:
            lines.append(f"- {keyword}")
        lines.append("")

    if not field_data.get("fields") and not field_data.get("tableData"):
        lines.append("No configured fields were recognized.")

    return "\n".join(lines).strip() + "\n"


def _render_field_only_html(field_data):
    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset=\"UTF-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">",
        "<title>Extracted Fields</title>",
        "<style>",
        "body { font-family: Segoe UI, Arial, sans-serif; margin: 20px; }",
        "h1, h2, h3 { margin-bottom: 8px; }",
        "ul { margin-top: 0; }",
        "table { border-collapse: collapse; margin: 12px 0; width: 100%; max-width: 900px; }",
        "th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }",
        "th { background: #f4f4f4; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Extracted Fields</h1>",
    ]

    fields = field_data.get("fields", {})
    table_data = field_data.get("tableData", {})
    field_details = field_data.get("fieldDetails", {})
    unmatched = field_data.get("unmatchedKeywords", [])

    if fields:
        html_lines.extend(["<h2>Single Fields</h2>", "<ul>"])
        for key, value in fields.items():
            html_lines.append(f"<li><strong>{html.escape(str(key))}</strong>: {html.escape(str(value))}</li>")
        html_lines.append("</ul>")

    for table_name, table_struct in table_data.items():
        columns = table_struct.get("columns", [])
        rows = table_struct.get("rows", [])
        html_lines.extend([f"<h2>{html.escape(table_name)}</h2>", "<table>", "<thead><tr>"])
        for col in columns:
            html_lines.append(f"<th>{html.escape(str(col))}</th>")
        html_lines.append("</tr></thead><tbody>")
        for row in rows:
            html_lines.append("<tr>")
            for col in columns:
                html_lines.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
            html_lines.append("</tr>")
        html_lines.extend(["</tbody>", "</table>"])

    if field_details:
        high_conf = {
            k: v for k, v in field_details.items()
            if v["confidence"] >= 0.7 and v["value"]
        }
        if high_conf:
            html_lines.extend(["<h2>Field Details (High Confidence)</h2>", "<table>", "<thead><tr>"])
            for th in ("Field", "Value", "Confidence", "Reason"):
                html_lines.append(f"<th>{th}</th>")
            html_lines.append("</tr></thead><tbody>")
            for key, detail in high_conf.items():
                html_lines.append("<tr>")
                html_lines.append(f"<td>{html.escape(str(key))}</td>")
                html_lines.append(f"<td>{html.escape(str(detail['value']))}</td>")
                html_lines.append(f"<td>{detail['confidence']}</td>")
                html_lines.append(f"<td>{html.escape(str(detail['reason']))}</td>")
                html_lines.append("</tr>")
            html_lines.extend(["</tbody>", "</table>"])

    if unmatched:
        html_lines.extend(["<h2>Unmatched Keywords</h2>", "<ul>"])
        for keyword in unmatched:
            html_lines.append(f"<li>{html.escape(str(keyword))}</li>")
        html_lines.append("</ul>")

    if not fields and not table_data:
        html_lines.append("<p>No configured fields were recognized.</p>")

    html_lines.extend(["</body>", "</html>"])
    return "\n".join(html_lines)


def _discover_input_candidates(base_dir: str = "."):
    candidates = []
    try:
        for entry in os.listdir(base_dir):
            entry_path = os.path.join(base_dir, entry)
            if not os.path.isfile(entry_path):
                continue
            if os.path.splitext(entry)[1].lower() in SUPPORTED_INPUT_EXTENSIONS:
                candidates.append(entry)
    except Exception:
        return []

    return sorted(candidates)


def _runtime_ocr_config():
    """Load OCR runtime configuration from environment variables."""
    return {
        "det_model": os.environ.get("OCR_DET_MODEL", "PP-OCRv6_medium_det"),
        "rec_model": os.environ.get("OCR_REC_MODEL", "PP-OCRv6_medium_rec"),
        "rec_threshold": float(os.environ.get("OCR_REC_SCORE_THRESH", "0.5")),
        "device": os.environ.get("OCR_DEVICE", "cpu"),
        "enable_mkldnn": os.environ.get("OCR_ENABLE_MKLDNN", "0") == "1",
        "input_candidates": _parse_csv_env("OCR_INPUT_CANDIDATES"),
        "output_summary": os.environ.get("OCR_OUTPUT_SUMMARY", "PaddleLocalResult/opensource_table_extraction_summary.md"),
    }


def _runtime_text_config():
    """Load render/console text templates from environment variables."""
    return {
        "failure_title": os.environ.get("OCR_FAILURE_TITLE", "OCR processing failure"),
        "init_failure_prefix": os.environ.get("OCR_INIT_FAILURE_PREFIX", "Initialization failed"),
        "process_failure_prefix": os.environ.get("OCR_PROCESS_FAILURE_PREFIX", "Processing failed"),
        "file_not_found_prefix": os.environ.get("OCR_FILE_NOT_FOUND_PREFIX", "Input file not found"),
        "success_prefix": os.environ.get("OCR_SUCCESS_PREFIX", "OCR run complete"),
    }


def _build_failure_markdown(doc_path: str, execution_time, details: str, title: str):
    """Build a configurable markdown failure report."""
    lines = [
        title,
        "",
        f"Input document: `{doc_path}`",
        f"Execution time: {execution_time}",
        "",
        details,
        "",
    ]
    return "\n".join(lines)


class PaddleOCR:
    """Fallback placeholder used when PaddleOCR cannot be imported."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("PaddleOCR is not available in this environment.")

    def __call__(self, *args, **kwargs):
        raise RuntimeError("PaddleOCR is not available in this environment.")


def _patch_paddle_analysis_config_compat():
    """Patch older Paddle AnalysisConfig objects expected by newer PaddleX wrappers."""
    try:
        import paddle

        libpaddle = getattr(getattr(paddle, "base", None), "libpaddle", None)
        analysis_config_cls = getattr(libpaddle, "AnalysisConfig", None)
        if analysis_config_cls and not hasattr(analysis_config_cls, "set_optimization_level"):
            def _set_optimization_level_noop(self, *_args, **_kwargs):
                return None

            setattr(analysis_config_cls, "set_optimization_level", _set_optimization_level_noop)
    except Exception:
        # If Paddle is unavailable or internals changed, let normal initialization handle errors.
        return


def _ensure_modelscope_importable():
    """Install a minimal modelscope stub when local torch/modelscope binaries are unavailable."""
    try:
        import modelscope  # noqa: F401

        return
    except Exception:
        pass

    sys.modules.pop("modelscope", None)
    stub = types.ModuleType("modelscope")

    def _snapshot_download(*_args, **_kwargs):
        raise RuntimeError("modelscope snapshot download is unavailable in this environment.")

    stub.snapshot_download = _snapshot_download
    sys.modules["modelscope"] = stub


def get_structure_engine_class():
    """Return the strongest available local Paddle structure engine class."""
    try:
        _patch_paddle_analysis_config_compat()
        _ensure_modelscope_importable()
        from paddleocr import PPStructureV3 as StructureClass

        return StructureClass
    except Exception:
        pass

    try:
        _patch_paddle_analysis_config_compat()
        _ensure_modelscope_importable()
        from paddleocr import PPStructure as StructureClass

        return StructureClass
    except Exception:
        pass

    try:
        _patch_paddle_analysis_config_compat()
        _ensure_modelscope_importable()
        from paddleocr import PaddleOCR as StructureClass

        return StructureClass
    except Exception:
        return PaddleOCR


def get_ocr_engine_class():
    """Backward-compatible alias for older callers."""
    return get_structure_engine_class()


def build_structure_engine(engine_class):
    """Construct the PaddleOCR engine with only supported arguments."""
    runtime_cfg = _runtime_ocr_config()
    preferred_kwargs = {
        "text_detection_model_name": runtime_cfg["det_model"],
        "text_recognition_model_name": runtime_cfg["rec_model"],
        "enable_mkldnn": runtime_cfg["enable_mkldnn"],
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "text_rec_score_thresh": runtime_cfg["rec_threshold"],
        "device": runtime_cfg["device"],
    }

    try:
        signature = inspect.signature(engine_class)
        accepted_names = set(signature.parameters)
        supports_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

        if supports_var_kwargs:
            kwargs = dict(preferred_kwargs)
        else:
            kwargs = {k: v for k, v in preferred_kwargs.items() if k in accepted_names}

        return engine_class(**kwargs) if kwargs else engine_class()
    except TypeError:
        return engine_class()


def _extract_ocr_cells(output):
    """Extract OCR cells with geometry so fallback rendering can infer generic tables."""
    cells = []

    if not isinstance(output, list):
        output = [output]

    for item in output:
        item = _normalize_result_item(item)
        if not item:
            continue

        rec_texts = item.get("rec_texts") or []
        rec_boxes = item.get("rec_boxes")
        rec_polys = item.get("rec_polys")

        if not rec_texts:
            overall_ocr = item.get("overall_ocr_res")
            if isinstance(overall_ocr, dict):
                rec_texts = overall_ocr.get("rec_texts") or []
                rec_boxes = rec_boxes or overall_ocr.get("rec_boxes")
                rec_polys = rec_polys or overall_ocr.get("rec_polys")

        for idx, text in enumerate(rec_texts):
            normalized = _normalize_layout_text(text)
            if not normalized:
                continue

            x0 = x1 = y0 = y1 = None

            if rec_boxes is not None and idx < len(rec_boxes):
                box = rec_boxes[idx]
                if len(box) >= 4:
                    x0, y0, x1, y1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            elif rec_polys is not None and idx < len(rec_polys):
                poly = rec_polys[idx]
                xs = [float(p[0]) for p in poly]
                ys = [float(p[1]) for p in poly]
                if xs and ys:
                    x0, x1 = min(xs), max(xs)
                    y0, y1 = min(ys), max(ys)

            if None in (x0, y0, x1, y1):
                # Keep deterministic ordering when geometry is unavailable.
                y0 = y1 = float(len(cells) * 10)
                x0, x1 = 0.0, float(max(1, len(normalized)))

            cells.append(
                {
                    "text": normalized,
                    "x0": x0,
                    "x1": x1,
                    "y0": y0,
                    "y1": y1,
                    "cx": (x0 + x1) / 2.0,
                    "cy": (y0 + y1) / 2.0,
                    "h": max(1.0, y1 - y0),
                    "w": max(1.0, x1 - x0),
                }
            )

    return cells


def _group_cells_by_row(cells):
    """Group OCR cells into visual rows using y-center proximity."""
    if not cells:
        return []

    sorted_cells = sorted(cells, key=lambda c: (c["cy"], c["x0"]))
    median_h = statistics.median(c["h"] for c in sorted_cells)
    y_tol = max(8.0, median_h * 0.65)

    rows = []
    for cell in sorted_cells:
        if not rows:
            rows.append({"cy": cell["cy"], "cells": [cell]})
            continue

        if abs(cell["cy"] - rows[-1]["cy"]) <= y_tol:
            row = rows[-1]
            row["cells"].append(cell)
            row["cy"] = (row["cy"] * (len(row["cells"]) - 1) + cell["cy"]) / len(row["cells"])
        else:
            rows.append({"cy": cell["cy"], "cells": [cell]})

    for row in rows:
        row["cells"].sort(key=lambda c: c["x0"])

    return rows


# How far above a ticked box to look for the caption that names the field.
MAX_CHECKBOX_CONTEXT_ROWS = 3


def _detect_checkbox_boxes(image_path, text_height: float = 0.0):
    """Find every checkbox in a rendered page image and say which are ticked.

    This has to be pure image analysis because the local OCR engine never emits
    a checkbox glyph, so a tick is invisible to every text-based rule.

    Two different binarisations are needed, and using only one is why a naive
    version finds nothing on a scan. On a scanned form the printed box outline
    is light grey, so a global Otsu threshold throws it away and leaves only the
    tick floating in white space; an adaptive threshold keeps the outline. The
    tick, in contrast, is dark pen or toner, and separating it from the grey
    outline is exactly what the global threshold is good at. So: find boxes in
    the adaptive mask, then ask the dark mask whether anything is inside.

    ``text_height`` is the page's median OCR line height. It is the scale
    reference that rejects the two things that otherwise imitate a checkbox -
    enclosed glyph counters, and white-on-black text in section header bars,
    which inverts into small square blobs. Both are bounded by the text height;
    a checkbox is taller.

    Returns [(cx, cy, side, checked), ...] in pixels of the given image, i.e.
    the same coordinate space as the OCR cells for that page.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return []

    try:
        with Image.open(image_path) as opened:
            gray = np.array(opened.convert("L"))
    except Exception:
        return []

    h_img, w_img = gray.shape[:2]
    page_side = min(h_img, w_img)
    min_side = max(8.0, page_side * 0.008)
    max_side = page_side * 0.040
    if text_height > 0:
        min_side = max(min_side, text_height * 0.95)
        max_side = min(max_side, text_height * 3.0)
    if min_side >= max_side:
        return []

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] > 0
    block = max(3, int(page_side * 0.02) | 1)
    outline = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 8
    )
    contours, _ = cv2.findContours(outline, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (min_side <= w <= max_side and min_side <= h <= max_side):
            continue
        if not 0.75 <= w / float(h) <= 1.35:
            continue

        roi = outline[y : y + h, x : x + w] > 0
        # A checkbox is a printed square, so all four edges must carry a solid
        # run of ink. Testing every edge is what rejects ordinary glyphs.
        band = max(1, int(round(min(w, h) * 0.15)))
        edges = (
            roi[:band, :].mean(axis=1).max(),
            roi[-band:, :].mean(axis=1).max(),
            roi[:, :band].mean(axis=0).max(),
            roi[:, -band:].mean(axis=0).max(),
        )
        if min(edges) < 0.85:
            continue

        margin = max(1, int(round(min(w, h) * 0.28)))
        inner = dark[y + margin : y + h - margin, x + margin : x + w - margin]
        if inner.size == 0:
            continue
        boxes.append((x + w / 2.0, y + h / 2.0, float(max(w, h)), float(inner.mean())))

    boxes.sort(key=lambda b: (b[1], b[0]))
    deduped = []
    for box in boxes:
        if any(
            abs(box[0] - other[0]) < box[2] * 0.5
            and abs(box[1] - other[1]) < box[2] * 0.5
            for other in deduped
        ):
            continue
        deduped.append(box)

    combed = _boxes_in_character_combs(deduped)

    detected = []
    for idx, (cx, cy, side, ink) in enumerate(deduped):
        if idx in combed:
            continue
        # A tick is a thin stroke, so it covers part of the interior. Nothing
        # inside means an empty box; a nearly solid interior is a logo or a
        # printed character rather than a mark.
        detected.append((cx, cy, side, 0.10 <= ink <= 0.60))
    return detected


# A value written under its caption may sit a line or two below it.
MAX_COLUMN_VALUE_ROWS = 2


def _caption_keyword(text, field_cfg):
    """The field a cell names, whether it is a bare caption or "Caption: value".

    A line such as "DATE: 23 June 2026" is a caption in its own right, and
    treating it as free text lets it be claimed as the value of the caption
    above it.
    """
    keyword = _match_field_keyword(text, field_cfg)
    if keyword:
        return keyword
    head = text.split(":", 1)[0]
    if head != text and head.strip():
        return _match_field_keyword(head, field_cfg)
    return None


def _extract_column_aligned_fields(page_outputs, field_cfg):
    """Read values that are written underneath their printed caption.

    Reading order alone cannot do this. Forms put captions side by side -
    "TAX IDENTIFICATION NUMBER (TIN)" on the left and "SST REGISTRATION NUMBER"
    on the right - and the OCR line list flattens both captions and then the
    single value written beneath one of them. The last caption seen therefore
    always wins and the value lands in the wrong field. Horizontal overlap
    between caption and value is what actually identifies the owner.
    """
    values = {}

    for output in page_outputs:
        cells = _extract_ocr_cells([output])
        if not cells:
            continue
        rows = _group_cells_by_row(cells)

        row_labels = []
        for row in rows:
            found = []
            for cell in row["cells"]:
                keyword = _caption_keyword(cell["text"], field_cfg)
                if keyword:
                    found.append((keyword, cell))
            row_labels.append(found)

        # Only rows holding several captions are ambiguous in reading order,
        # and those are exactly the rows this pass exists to resolve.
        for row_idx, found in enumerate(row_labels):
            if len(found) < 2:
                continue
            for keyword, label_cell in found:
                value = _column_value_below(
                    rows, row_labels, row_idx, label_cell, field_cfg
                )
                if value:
                    values.setdefault(keyword, value)

    return values


def _column_value_below(rows, row_labels, row_idx, label_cell, field_cfg):
    """The text written under a caption, within that caption's column."""
    for offset in range(1, MAX_COLUMN_VALUE_ROWS + 1):
        idx = row_idx + offset
        if idx >= len(rows) or row_labels[idx]:
            # Another caption row ends this caption's column.
            return ""
        aligned = [
            cell
            for cell in rows[idx]["cells"]
            if cell["x0"] < label_cell["x1"] and cell["x1"] > label_cell["x0"]
        ]
        if not aligned:
            continue
        candidate = _clean_field_value(" ".join(cell["text"] for cell in aligned))
        if _looks_like_form_text(candidate, field_cfg, label_cell["text"]):
            return ""
        return candidate
    return ""


# A written answer sits on its caption's own row, or in the space just under
# it. Any further and there is printed text in between, which means the two
# belong to different parts of the form.
MAX_VALUE_DISTANCE_ROWS = MAX_COLUMN_VALUE_ROWS


def _reject_implausible_values(canonical_fields, field_cfg, page_outputs, protected):
    """Clear values that cannot belong to the caption they were paired with.

    Reading order pairs a caption with the next plausible line, and that line
    can be anywhere on the page: a caption in one section ends up owning a date
    written three sections lower, and the caption row directly beneath a
    caption gets read as its answer. Both are worth undoing because both
    produce a value that is confidently wrong rather than merely missing, and a
    blank field is visibly incomplete where a wrong one is not.
    """
    page_rows = []
    for output in page_outputs:
        cells = _extract_ocr_cells([output])
        if cells:
            page_rows.append(_group_cells_by_row(cells))

    for keyword, value in list(canonical_fields.items()):
        if not value or keyword in protected:
            continue

        # A line that names another field is that field's caption, never an
        # answer to this one.
        other = _caption_keyword(value, field_cfg)
        if other and other != keyword:
            canonical_fields[keyword] = ""
            continue

        gap = _rows_between_caption_and_value(page_rows, keyword, value, field_cfg)
        if gap > MAX_VALUE_DISTANCE_ROWS:
            canonical_fields[keyword] = ""


def _rows_between_caption_and_value(page_rows, keyword, value, field_cfg):
    """How many rows separate a field's caption from the text assigned to it.

    Returns 0 whenever either end cannot be located, so that values with no
    caption row of their own - table cells, ticked boxes, answers spanning
    several lines - are left untouched rather than guessed at.
    """
    needle = _normalize_field_label(value)
    if not needle:
        return 0

    best = None
    for rows in page_rows:
        caption_rows = []
        value_rows = []
        for idx, row in enumerate(rows):
            if any(
                _caption_keyword(cell["text"], field_cfg) == keyword
                for cell in row["cells"]
            ):
                caption_rows.append(idx)
            joined = _normalize_field_label(" ".join(c["text"] for c in row["cells"]))
            if joined and needle in joined:
                value_rows.append(idx)
        if not caption_rows or not value_rows:
            continue
        for caption_idx in caption_rows:
            for value_idx in value_rows:
                gap = abs(value_idx - caption_idx)
                best = gap if best is None else min(best, gap)

    return 0 if best is None else best


def _boxes_in_character_combs(boxes):
    """Indices of boxes that are cells of a character comb, not checkboxes.

    Forms write fixed-width values such as an account number into a row of
    touching equal-sized squares. Each cell is indistinguishable from a ticked
    checkbox on its own - a square outline with a mark inside - but a checkbox
    never appears in a tightly packed run, so the run itself is the signal.
    """
    combed = set()
    remaining = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))

    while remaining:
        seed = remaining[0]
        side = boxes[seed][2]
        band = [i for i in remaining if abs(boxes[i][1] - boxes[seed][1]) <= side * 0.5]
        remaining = [i for i in remaining if i not in set(band)]

        band.sort(key=lambda i: boxes[i][0])
        run = [band[0]]
        for idx in band[1:]:
            if boxes[idx][0] - boxes[run[-1]][0] <= boxes[idx][2] * 1.8:
                run.append(idx)
                continue
            if len(run) >= 3:
                combed.update(run)
            run = [idx]
        if len(run) >= 3:
            combed.update(run)

    return combed


def _extract_checkbox_fields(page_outputs, page_boxes, field_cfg):
    """Turn checkboxes into field values using only the field config.

    A tick is meaningful in one of two layouts, and the configured aliases tell
    them apart with no document-specific knowledge:

        "<statement>  [x]"  - the alias is the statement beside the box, so the
                              tick is itself the answer and the value is YES
        "<question> [x] YES [ ] NO"
                            - the alias is the question, and the short label
                              next to the tick is the answer

    Returns (values, unticked_labels). The labels belong to boxes the applicant
    left blank; they are printed text sitting where a handwritten answer would
    be, so the text extractor happily reports them as values, and knowing they
    are options is the only way to tell that they are not answers.
    """
    values = {}
    unticked_labels = []

    for output, boxes in zip(page_outputs, page_boxes):
        if not boxes:
            continue
        cells = _extract_ocr_cells([output])
        if not cells:
            continue
        rows = _group_cells_by_row(cells)
        if not rows:
            continue

        for cx, cy, side, checked in boxes:
            row_idx = min(range(len(rows)), key=lambda i: abs(rows[i]["cy"] - cy))
            if abs(rows[row_idx]["cy"] - cy) > side * 2.0:
                continue

            # The label runs from this box up to whatever the next box on the
            # row claims, so adjacent options do not swallow each other.
            next_x = min(
                (b[0] for b in boxes if b[0] > cx + side and abs(b[1] - cy) <= side),
                default=float("inf"),
            )
            right = [
                c
                for c in rows[row_idx]["cells"]
                if cx - side <= c["x0"] < next_x
            ]
            label = _clean_field_value(" ".join(c["text"] for c in right))

            if not checked:
                if label:
                    unticked_labels.append(_normalize_field_label(label))
                continue

            # The caption that names the field is on the tick's own line, or on
            # one of the few lines above it.
            keyword = ""
            for back in range(MAX_CHECKBOX_CONTEXT_ROWS + 1):
                idx = row_idx - back
                if idx < 0:
                    break
                text = " ".join(c["text"] for c in rows[idx]["cells"])
                matches = _match_field_keyword_by_substring(text, field_cfg)
                if matches:
                    keyword = matches[0][0]
                    break

            if os.environ.get("OCR_GEO_DEBUG") == "1":
                print(
                    f"[debug] tick @({cx:.0f},{cy:.0f}) side={side:.0f} "
                    f"label={label[:60]!r} keyword={keyword!r}"
                )

            if not keyword:
                continue

            option = _clean_field_value(right[0]["text"]) if right else ""
            upper = option.upper()
            if upper in {"YES", "NO"}:
                value = upper
            elif option and len(option.split()) <= 3:
                value = option
            else:
                # The label beside the tick is the statement itself, so the
                # tick alone carries the meaning.
                value = "YES"

            values.setdefault(keyword, value)

    return values, unticked_labels


def _clear_unticked_option_values(canonical_fields, protected, unticked_labels):
    """Drop values that are nothing more than the labels of unticked boxes.

    A row of options such as "[ ] FORM W-9  [ ] FORM W-8BEN" sits exactly where
    a written answer would sit, so the text extractor reports it as the value of
    the caption above it. None of it was chosen, so none of it is an answer.
    """
    labels = [label for label in unticked_labels if len(label) >= 4]
    if not labels:
        return

    for keyword, value in canonical_fields.items():
        if not value or keyword in protected:
            continue
        residue = _normalize_field_label(value)
        if not residue:
            continue
        for label in labels:
            residue = residue.replace(label, "")
        if len(residue) < 2:
            canonical_fields[keyword] = ""


def _build_markdown_table_from_rows(rows):
    """Build a markdown table from grouped OCR rows using column-center clustering."""
    if len(rows) < 2:
        return None

    all_cells = [c for row in rows for c in row["cells"]]
    if not all_cells:
        return None

    median_w = statistics.median(c["w"] for c in all_cells)
    x_tol = max(20.0, median_w * 0.8)

    centers = []
    for cell in sorted(all_cells, key=lambda c: c["cx"]):
        if not centers or abs(cell["cx"] - centers[-1]) > x_tol:
            centers.append(cell["cx"])
        else:
            centers[-1] = (centers[-1] + cell["cx"]) / 2.0

    if len(centers) < 2:
        return None

    grid = []
    for row in rows:
        values = [""] * len(centers)
        for cell in row["cells"]:
            col_idx = min(range(len(centers)), key=lambda i: abs(cell["cx"] - centers[i]))
            if values[col_idx]:
                values[col_idx] = f"{values[col_idx]} {cell['text']}".strip()
            else:
                values[col_idx] = cell["text"]
        if any(values):
            grid.append(values)

    if len(grid) < 2:
        return None

    header = [v if v else f"Col {idx + 1}" for idx, v in enumerate(grid[0])]
    markdown = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]

    for row_values in grid[1:]:
        markdown.append("| " + " | ".join(v.strip() for v in row_values) + " |")

    return markdown


def _segment_rows_into_blocks(rows):
    """Split rows into text and table blocks using generic row-density heuristics."""
    if not rows:
        return []

    blocks = []
    i = 0
    while i < len(rows):
        row = rows[i]
        row_len = len(row["cells"])

        if row_len >= 3:
            start = i
            j = i + 1
            while j < len(rows) and len(rows[j]["cells"]) >= 2:
                j += 1
            candidate = rows[start:j]
            table_md = _build_markdown_table_from_rows(candidate)
            if table_md:
                blocks.append({"type": "table", "content": table_md})
                i = j
                continue

        line_text = " ".join(cell["text"] for cell in row["cells"]).strip()
        blocks.append({"type": "text", "content": line_text})
        i += 1

    return blocks


def _normalize_result_item(item):
    """Normalize PaddleOCR Result/dict payload into a plain dict payload."""
    payload = item

    if not isinstance(payload, dict):
        if hasattr(payload, "json"):
            try:
                payload = payload.json() if callable(payload.json) else payload.json
            except Exception:
                payload = {}
        else:
            payload = {}

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}

    if isinstance(payload, dict):
        nested = payload.get("res")
        if isinstance(nested, dict):
            return nested

    return payload if isinstance(payload, dict) else {}


def _extract_html_tables(output):
    """Extract visible table HTML snippets from the PaddleOCR result."""
    tables = []

    if not isinstance(output, list):
        output = [output]

    for item in output:
        item = _normalize_result_item(item)
        if not item:
            continue

        for table_result in item.get("table_res_list", []):
            table_html = table_result.get("pred_html")
            if table_html and "<table" in table_html:
                tables.append(table_html)

        for parse_item in item.get("parsing_res_list", []):
            if isinstance(parse_item, dict) and parse_item.get("label") == "table":
                content = parse_item.get("content") or ""
                if "<table" in content:
                    tables.append(content)

    unique_tables = []
    seen = set()
    for table_html in tables:
        if table_html not in seen:
            seen.add(table_html)
            unique_tables.append(table_html)

    return unique_tables


def _expand_html_table_to_grid(table_html: str):
    """
    Parse an HTML table with BeautifulSoup, expanding colspan and rowspan
    into a flat 2-D grid of strings.
    Returns list-of-rows, each row being a list of cell strings.
    Returns None when parsing fails or no rows are found.

    Uses a single-pass expansion so max_cols is derived from the fully
    expanded grid rather than a naive per-row colspan sum that
    under-counts rows with rowspan carries.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None

    try:
        soup = BeautifulSoup(table_html, "html.parser")
    except Exception:
        return None

    table = soup.find("table")
    if not table:
        return None

    # Single pass: expand into sparse row dicts, respecting colspan + rowspan.
    sparse_rows = []  # list of {col_index: text}
    rowspan_carry = {}  # col_index -> (remaining_rows, text)

    for tr in table.find_all("tr"):
        row = {}
        # Absorb carried-forward rowspan cells.
        for col_idx, (remaining, text) in list(rowspan_carry.items()):
            row[col_idx] = text
            if remaining <= 1:
                del rowspan_carry[col_idx]
            else:
                rowspan_carry[col_idx] = (remaining - 1, text)

        col_cursor = 0
        for cell in tr.find_all(["td", "th"]):
            # Skip columns already occupied by rowspan carries.
            while col_cursor in row:
                col_cursor += 1

            text = cell.get_text(separator=" ").strip()
            text = _normalize_checkbox_symbols(text)
            colspan = max(1, int(cell.get("colspan", 1)))
            rowspan = max(1, int(cell.get("rowspan", 1)))

            for span_col in range(col_cursor, col_cursor + colspan):
                row[span_col] = text
                if rowspan > 1:
                    rowspan_carry[span_col] = (rowspan - 1, text)

            col_cursor += colspan

        if row:
            sparse_rows.append(row)

    if not sparse_rows:
        return None

    # Derive max_cols from the fully-expanded sparse rows (includes rowspan-carried cols).
    max_cols = max(max(row.keys()) + 1 for row in sparse_rows)
    if max_cols == 0:
        return None

    grid = [[row.get(c, "") for c in range(max_cols)] for row in sparse_rows]
    return grid if grid else None


def _html_to_markdown_table(table_html: str):
    """Convert the extracted HTML table into a Markdown table block, respecting colspan/rowspan."""
    grid = _expand_html_table_to_grid(table_html)

    # Fall back to pandas when BeautifulSoup is unavailable
    if grid is None:
        try:
            frames = pd.read_html(StringIO(table_html))
        except Exception:
            return None
        if not frames:
            return None
        frame = frames[0].fillna("").astype(str)
        if frame.empty:
            return None
        header_row = frame.iloc[0].astype(str).tolist()
        frame = frame.iloc[1:].copy()
        frame.columns = header_row
        grid = [[str(c).strip() for c in frame.columns]]
        for _, row in frame.iterrows():
            grid.append([str(c).strip() for c in row.tolist()])

    if len(grid) < 1:
        return None

    # Use first row as header
    header = [c if c else f"Col {i + 1}" for i, c in enumerate(grid[0])]
    separator = ["---"] * len(header)
    md_rows = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in grid[1:]:
        # Escape pipe characters that would break GFM table syntax
        cells = [c.replace("|", "\\|") for c in row]
        md_rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(md_rows)

def _preprocess_image_for_ocr(image_path: str):
    """Improve the document image before OCR so the local engine can read it more cleanly."""
    with Image.open(image_path) as image:
        if image.mode != "L":
            image = ImageOps.grayscale(image)
        image = ImageOps.autocontrast(image)
        image = image.filter(ImageFilter.SHARPEN)
        image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)

        temp_dir = tempfile.mkdtemp(prefix="ocr_preprocessed_")
        preprocessed_path = os.path.join(temp_dir, "preprocessed.png")
        image.save(preprocessed_path)
        return preprocessed_path, temp_dir


def _prepare_local_input_pages(doc_path: str):
    """Convert an input document into one or more locally preprocessed page images."""
    lower_path = doc_path.lower()
    if not lower_path.endswith(".pdf"):
        preprocessed_path, temp_dir = _preprocess_image_for_ocr(doc_path)
        return [preprocessed_path], [temp_dir]

    try:
        import pypdfium2 as pdfium
    except Exception:
        raise RuntimeError("PDF support requires pypdfium2 to be installed in the local environment.")

    pdf = pdfium.PdfDocument(doc_path)
    temp_dir = tempfile.mkdtemp(prefix="ocr_pdf_")
    page_paths = []
    cleanup_dirs = []

    try:
        for idx in range(len(pdf)):
            page = pdf[idx]
            bitmap = page.render(scale=2)
            pil_image = bitmap.to_pil()

            image_path = os.path.join(temp_dir, f"converted_page_{idx + 1}.png")
            pil_image.save(image_path)
            preprocessed_path, preprocess_dir = _preprocess_image_for_ocr(image_path)
            page_paths.append(preprocessed_path)
            cleanup_dirs.append(preprocess_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return page_paths, cleanup_dirs


_CHECKBOX_SYMBOL_MAP = [
    # Checked variants → markdown checked
    ("☑", "- [x]"),
    ("☒", "- [x]"),
    ("✓", "- [x]"),
    ("✔", "- [x]"),
    ("[x]", "- [x]"),
    ("[X]", "- [x]"),
    ("[V]", "- [x]"),
    ("[v]", "- [x]"),
    ("[/]", "- [x]"),
    ("\\\\", "- [x]"),
    ("(x)", "- (x)"),
    ("(X)", "- (x)"),
    # Unchecked variants → markdown unchecked
    ("☐", "- [ ]"),
    ("□", "- [ ]"),
    ("[ ]", "- [ ]"),
    # Radio button filled → selected
    ("●", "- (●)"),
    ("◉", "- (●)"),
    ("(●)", "- (●)"),
    ("(•)", "- (●)"),
    # Radio button empty → unselected
    ("○", "- ( )"),
    ("◯", "- ( )"),
    ("( )", "- ( )"),
    ("(○)", "- ( )"),
]

# Regex patterns for OCR-typical checkbox/radio variants not caught by exact-match map.
_CHECKBOX_REGEX_PATTERNS = [
    # [V], [v], [/], [\], [✓], [✔] inside brackets → checked
    (re.compile(r'\[(?:V|v|/|\\|\u2713|\u2714)\]'), '[x]'),
    # Completely empty or whitespace-only brackets → unchecked
    (re.compile(r'\[\s{0,4}\]'), '[ ]'),
    # Filled radio in parens: (●), (•), (◉)
    (re.compile(r'\(\s*[\u25CF\u2022\u25C9]\s*\)'), '(●)'),
    # Empty radio in parens: (○), (◯)
    (re.compile(r'\(\s*[\u25CB\u25EF]\s*\)'), '( )'),
]


def _normalize_checkbox_symbols(text: str) -> str:
    """Replace stray checkbox/radio symbols with GFM task-list equivalents."""
    if not text:
        return text
    stripped = text.strip()
    # Exact-match and prefix-match against the symbol map.
    for symbol, replacement in _CHECKBOX_SYMBOL_MAP:
        if stripped == symbol:
            return replacement
        if stripped.startswith(symbol + " "):
            return replacement + " " + stripped[len(symbol):].lstrip()
    # Regex-based inline replacement for OCR-typical patterns within longer text.
    for pattern, replacement in _CHECKBOX_REGEX_PATTERNS:
        if pattern.search(text):
            text = pattern.sub(replacement, text)
    return text


def _normalize_layout_text(text):
    """Normalize layout OCR text into Markdown-safe paragraph text."""
    if text is None:
        return ""

    repaired_text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines = []
    for line in repaired_text.split("\n"):
        candidate = line.strip()
        if candidate:
            cleaned_lines.append(candidate)

    return "\n".join(cleaned_lines).strip()


def _format_ocr_fallback_line(line: str):
    """Add lightweight Markdown structure for OCR fallback lines."""
    text = line.strip()
    if not text:
        return ""

    # Normalize checkbox/radio symbols before any other classification
    text = _normalize_checkbox_symbols(text)
    if text.startswith("- [") or text.startswith("- ("):
        return text

    # Promote short all-caps headings so section boundaries remain visible.
    alpha_count = sum(1 for ch in text if ch.isalpha())
    alpha_ratio = alpha_count / max(len(text), 1)
    looks_like_id = bool(re.search(r"\d{3,}[-–—]\d+", text))
    if (
        re.fullmatch(r"[A-Z0-9 /&'().,-]{6,}", text)
        and len(text.split()) <= 10
        and alpha_count >= 4
        and alpha_ratio >= 0.5
        and not looks_like_id
    ):
        return f"## {text}"

    # Keep numbered bullet points as a list.
    if re.match(r"^\d+[\)\.]\s*", text):
        return text

    return text


def _render_numbered_terms(lines):
    """Group numbered clauses into markdown bullets with wrapped text."""
    grouped = []
    i = 0
    bullet_start = re.compile(r"^(\d+)[\)\.]\s*(.*)$")

    while i < len(lines):
        current = lines[i].strip()
        match = bullet_start.match(current)
        if not match:
            grouped.append(current)
            i += 1
            continue

        label = match.group(1)
        body = match.group(2).strip()
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if bullet_start.match(nxt):
                break
            if not nxt:
                i += 1
                continue
            body = f"{body} {nxt}".strip()
            i += 1

        grouped.append(f"{label}. {body}".strip())

    return grouped


def _render_generic_fallback_markdown(output, ocr_lines):
    """Render OCR fallback using geometry-aware generic table/text segmentation."""
    rendered_lines = []
    cells = _extract_ocr_cells(output)
    rows = _group_cells_by_row(cells)
    blocks = _segment_rows_into_blocks(rows)

    if not blocks:
        body_started = False
        for raw_line in ocr_lines:
            formatted = _format_ocr_fallback_line(raw_line)
            if formatted:
                if formatted.startswith("## ") and body_started:
                    formatted = formatted[3:]
                if not formatted.startswith("## "):
                    body_started = True
                rendered_lines.append(formatted)
        return rendered_lines

    text_buffer = []
    body_started = False

    def flush_text_buffer():
        nonlocal text_buffer, body_started
        if not text_buffer:
            return
        entries = _render_numbered_terms(text_buffer)
        has_numbered_list = any(re.match(r"^\d+\.\s+", entry.strip()) for entry in entries)
        has_bullet_list = any(entry.strip().startswith(("- ", "* ")) for entry in entries)
        is_list_block = has_numbered_list or has_bullet_list
        # Ensure blank line before list block so MD renderers don't merge it into the preceding paragraph
        if is_list_block and rendered_lines and rendered_lines[-1] != "":
            rendered_lines.append("")
        previous_was_list = False
        for entry in entries:
            formatted = _format_ocr_fallback_line(entry)
            if not formatted:
                continue
            current_is_list = bool(re.match(r"^\d+\.\s+", formatted.strip())) or formatted.strip().startswith(("- ", "* "))
            if current_is_list and not previous_was_list and rendered_lines and rendered_lines[-1] != "":
                rendered_lines.append("")
            if formatted.startswith("## ") and body_started:
                formatted = formatted[3:]
            if not formatted.startswith("## "):
                body_started = True
            rendered_lines.append(formatted)
            previous_was_list = current_is_list
        if is_list_block:
            rendered_lines.append("")
        text_buffer = []

    for block in blocks:
        if block["type"] == "table":
            flush_text_buffer()
            if rendered_lines and rendered_lines[-1] != "":
                rendered_lines.append("")
            rendered_lines.extend(block["content"])
            rendered_lines.append("")
            continue

        content = (block.get("content") or "").strip()
        if content:
            text_buffer.append(content)

    flush_text_buffer()
    return rendered_lines



def _extract_layout_blocks(output):
    """Extract OCR layout blocks in document order, preserving page structure."""
    blocks = []

    if not isinstance(output, list):
        output = [output]

    for item in output:
        item = _normalize_result_item(item)
        if not item:
            continue

        for block in item.get("parsing_res_list", []):
            if isinstance(block, dict):
                label = block.get("label") or "text"
                content = block.get("content")
            else:
                label = getattr(block, "label", None) or "text"
                content = getattr(block, "content", None)

            if not content:
                continue

            if label == "table":
                content = content or ""
                if "<table" in content:
                    blocks.append((label, content))
                continue

            blocks.append((label, content))

        for table_result in item.get("table_res_list", []):
            table_html = table_result.get("pred_html") if isinstance(table_result, dict) else None
            if table_html and "<table" in table_html:
                blocks.append(("table", table_html))

    return blocks


def _extract_ocr_lines(output):
    """Extract the engine-level OCR text lines as a cleaner fallback source for paragraph content."""
    lines = []

    if not isinstance(output, list):
        output = [output]

    for item in output:
        item = _normalize_result_item(item)
        if not item:
            continue

        rec_texts = item.get("rec_texts") or []
        if rec_texts:
            for text in rec_texts:
                normalized = _normalize_layout_text(text)
                if normalized:
                    lines.append(normalized)
            continue

        overall_ocr = item.get("overall_ocr_res")

        if isinstance(overall_ocr, dict):
            rec_texts = overall_ocr.get("rec_texts") or []
        else:
            rec_texts = getattr(overall_ocr, "get", lambda *_args, **_kwargs: [])("rec_texts") or []

        for text in rec_texts:
            normalized = _normalize_layout_text(text)
            if normalized:
                lines.append(normalized)

    return lines


def _extract_confidence_scores(output):
    """Extract all recognized confidence scores from result payloads."""
    scores = []

    if not isinstance(output, list):
        output = [output]

    for item in output:
        item = _normalize_result_item(item)
        if not item:
            continue

        rec_scores = item.get("rec_scores") or []
        if not rec_scores:
            overall_ocr = item.get("overall_ocr_res")
            if isinstance(overall_ocr, dict):
                rec_scores = overall_ocr.get("rec_scores") or []

        for score in rec_scores:
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                continue

    return scores


def _render_confidence_summary(output):
    """Render confidence summary text for OCR results."""
    scores = _extract_confidence_scores(output)
    if not scores:
        return "Confidence score: unavailable"

    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)
    return f"Confidence score: avg={avg_score:.4f}, min={min_score:.4f}, max={max_score:.4f}, n={len(scores)}"


def _render_form_markdown(output):
    """Return a text-only Markdown transcription of the recognized form content."""
    lines = [_render_confidence_summary(output), ""]
    blocks = _extract_layout_blocks(output)
    ocr_lines = _extract_ocr_lines(output)

    if not blocks and not ocr_lines:
        lines.append("No page structure or table was recognized in the form image.")
        return "\n".join(lines)

    seen_tables = set()
    rendered_any = False

    for label, content in blocks:
        normalized_text = content
        if label != "table":
            normalized_text = _normalize_layout_text(content)
        if not normalized_text:
            continue

        if label == "table":
            if normalized_text in seen_tables:
                continue
            seen_tables.add(normalized_text)
            markdown_table = _html_to_markdown_table(normalized_text)
            if markdown_table:
                lines.extend([markdown_table, ""])
            else:
                lines.extend([normalized_text, ""])
            rendered_any = True
            continue

        if label in {"header", "doc_title"}:
            lines.append(f"## {normalized_text}")
        elif label == "paragraph_title":
            lines.append(f"### {normalized_text}")
        elif label == "vision_footnote":
            lines.append(f"> {normalized_text}")
        elif label in {"number", "page_number"}:
            continue
        else:
            lines.append(normalized_text)

        rendered_any = True

    if ocr_lines and not any(line.startswith("## ") for line in lines):
        lines.extend(_render_generic_fallback_markdown(output, ocr_lines))
        rendered_any = True

    if not rendered_any:
        lines.append("No page structure or table was recognized in the form image.")

    return "\n".join(lines) + "\n"


def _safe_percent(value, total):
    """Convert an absolute coordinate into a clamped percentage string."""
    if total <= 0:
        return "0.000"
    percentage = (float(value) / float(total)) * 100.0
    return f"{max(0.0, min(100.0, percentage)):.3f}"


def _render_high_fidelity_page_markdown(page_index: int, output, page_size):
    """Render one page using positioned HTML blocks embedded in Markdown."""
    width, height = page_size
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)

    lines = [f"Page {page_index}", ""]
    lines.append(
        (
            f'<div class="ocr-page" style="width: min(100%, {int(width)}px); '
            f'height: auto; aspect-ratio: {int(width)} / {int(height)};">'
        )
    )

    for cell in sorted(_extract_ocr_cells(output), key=lambda c: (c["y0"], c["x0"])):
        left = _safe_percent(cell["x0"], width)
        top = _safe_percent(cell["y0"], height)
        cell_width = _safe_percent(cell["x1"] - cell["x0"], width)
        cell_height = _safe_percent(cell["y1"] - cell["y0"], height)
        text = html.escape(cell["text"])
        lines.append(
            (
                f'<div class="ocr-cell" style="position:absolute;left:{left}%;top:{top}%;'
                f'width:{cell_width}%;height:{cell_height}%;">{text}</div>'
            )
        )

    lines.append("</div>")
    lines.append("")

    html_tables = _extract_html_tables(output)
    if html_tables:
        for table_html in html_tables:
            lines.append(table_html)
            lines.append("")

    return lines


def _render_high_fidelity_document_markdown(page_outputs, page_sizes):
    """Render a layout-preserving markdown document with embedded positioned HTML."""
    outputs = page_outputs if isinstance(page_outputs, list) else [page_outputs]
    lines = [
        _render_confidence_summary(outputs),
        "",
        "<style>",
        ".ocr-page { position: relative; border: 1px solid #d0d0d0; margin: 1rem 0; background: #ffffff; overflow: hidden; }",
        ".ocr-cell { font-family: 'Segoe UI', Tahoma, sans-serif; font-size: 12px; line-height: 1.2; white-space: pre-wrap; overflow: hidden; }",
        "table { border-collapse: collapse; margin: 0.75rem 0; }",
        "td, th { border: 1px solid #c0c0c0; padding: 0.3rem 0.5rem; }",
        "</style>",
        "",
    ]

    for idx, page_output in enumerate(outputs, start=1):
        page_size = page_sizes[idx - 1] if idx - 1 < len(page_sizes) else (1000, 1414)
        lines.extend(_render_high_fidelity_page_markdown(idx, page_output, page_size))

    for idx, page_output in enumerate(outputs, start=1):
        page_markdown = _render_form_markdown([page_output])
        body = page_markdown.strip()
        lines.append(f"Page {idx}")
        lines.append("")
        lines.append(body)
        lines.append("")

    return "\n".join(lines) + "\n"


def _render_readable_document_markdown(page_outputs):
    """Render a clean, human-readable semantic markdown variant."""
    outputs = page_outputs if isinstance(page_outputs, list) else [page_outputs]
    if len(outputs) == 1:
        return _render_form_markdown([outputs[0]])

    lines = [_render_confidence_summary(outputs), ""]
    for idx, page_output in enumerate(outputs, start=1):
        page_markdown = _render_form_markdown([page_output])
        body = page_markdown.strip()
        lines.extend([f"Page {idx}", "", body, ""])
    return "\n".join(lines) + "\n"


def _derive_readable_output_path(summary_out: str):
    """Derive the readable-output markdown path from the main summary output."""
    base, ext = os.path.splitext(summary_out)
    suffix = ext if ext else ".md"
    return f"{base}_readable{suffix}"


def _derive_json_output_path(summary_out: str):
    """Derive JSON output path from markdown output path."""
    base, ext = os.path.splitext(summary_out)
    return f"{base}.json"


def _derive_html_output_path(summary_out: str):
    """Derive HTML output path from markdown output path."""
    base, ext = os.path.splitext(summary_out)
    return f"{base}.html"


def _extract_word_level_data(output, page_num: int = 1):
    """
    Extract words with confidence scores and geometry from OCR output.
    Returns a list of word dicts with content, confidence, geometry (x0, y0, x1, y1), and computed source string.
    """
    words = []
    
    if not isinstance(output, list):
        output = [output]
    
    for item in output:
        item = _normalize_result_item(item)
        if not item:
            continue
        
        rec_texts = item.get("rec_texts") or []
        rec_scores = item.get("rec_scores") or []
        rec_boxes = item.get("rec_boxes")
        rec_polys = item.get("rec_polys")
        
        # Fallback to nested overall_ocr_res if not found at top level
        if not rec_texts:
            overall_ocr = item.get("overall_ocr_res")
            if isinstance(overall_ocr, dict):
                rec_texts = overall_ocr.get("rec_texts") or []
                rec_scores = rec_scores or overall_ocr.get("rec_scores") or []
                rec_boxes = rec_boxes or overall_ocr.get("rec_boxes")
                rec_polys = rec_polys or overall_ocr.get("rec_polys")
        
        # Process each recognized text with its confidence score
        for idx, text in enumerate(rec_texts):
            normalized = _normalize_layout_text(text)
            if not normalized:
                continue
            
            # Extract confidence score for this word
            confidence = 0.5
            if idx < len(rec_scores):
                try:
                    confidence = float(rec_scores[idx])
                except (TypeError, ValueError):
                    confidence = 0.5
            
            # Extract geometry
            x0 = x1 = y0 = y1 = None
            if rec_boxes is not None and idx < len(rec_boxes):
                box = rec_boxes[idx]
                if len(box) >= 4:
                    x0, y0, x1, y1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            elif rec_polys is not None and idx < len(rec_polys):
                poly = rec_polys[idx]
                try:
                    xs = [float(p[0]) for p in poly]
                    ys = [float(p[1]) for p in poly]
                    if xs and ys:
                        x0, x1 = min(xs), max(xs)
                        y0, y1 = min(ys), max(ys)
                except (TypeError, ValueError):
                    pass
            
            if None in (x0, y0, x1, y1):
                # Fallback geometry when not available
                x0, y0, x1, y1 = 0.0, float(idx * 10), float(max(1, len(normalized))), float((idx + 1) * 10)
            
            # Build source string encoding page, geometry, and confidence
            # Format: "D(page,x0,y0,x1,y1)" for Detection box, with confidence encoded
            source = f"D({page_num},{int(x0)},{int(y0)},{int(x1)},{int(y1)})"
            
            words.append({
                "content": normalized,
                "confidence": round(confidence, 6),
                "geometry": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "source": source,
            })
    
    return words


def _build_json_output(page_outputs, page_sizes):
    """
    Build JSON output structure with word-level confidence and source info.
    Returns dict with pages array containing word arrays.
    """
    pages = []
    
    for page_idx, (output, size) in enumerate(zip(page_outputs, page_sizes)):
        page_num = page_idx + 1
        width, height = size
        angle = 0  # We could detect rotation if available in output
        
        words = _extract_word_level_data(output, page_num)
        
        if not words:
            # If no words extracted, add empty page
            pages.append({
                "pageNumber": page_num,
                "angle": angle,
                "width": int(width),
                "height": int(height),
                "spans": [],
                "words": []
            })
            continue
        
        # Build spans: each span represents a contiguous text region
        # For simplicity, we'll group words into one or more spans based on proximity
        spans = []
        current_offset = 0
        words_content = ""
        
        # For each word, accumulate content and track offset
        for word in words:
            word_start = current_offset
            word_end = word_start + len(word["content"]) + 1  # +1 for space between words
            current_offset = word_end
            words_content += word["content"] + " "
        
        total_length = len(words_content.strip())
        if total_length > 0:
            spans.append({
                "offset": 0,
                "length": total_length
            })
        
        # Update word span offsets based on accumulated text
        current_offset = 0
        for word in words:
            word_len = len(word["content"])
            word["span"] = {
                "offset": current_offset,
                "length": word_len
            }
            current_offset += word_len + 1  # +1 for space
        
        pages.append({
            "pageNumber": page_num,
            "angle": angle,
            "width": int(width),
            "height": int(height),
            "spans": spans,
            "words": words
        })
    
    return {"pages": pages}


def _render_json_to_html(json_data):
    """
    Render JSON output to HTML with confidence-based color coding.
    High confidence (>0.8) = green, medium (0.5-0.8) = yellow, low (<0.5) = red.
    Layout is position-based using absolute positioning.
    """
    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset=\"UTF-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">",
        "<title>OCR Results with Confidence Scores</title>",
        "<style>",
        "  body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }",
        "  .page-container { margin: 20px 0; page-break-after: always; }",
        "  .page-header { font-size: 18px; font-weight: bold; margin: 10px 0; color: #333; }",
        "  .page-stats { font-size: 12px; color: #666; margin-bottom: 10px; }",
        "  .ocr-page-view {",
        "    position: relative;",
        "    background-color: white;",
        "    border: 1px solid #ddd;",
        "    box-shadow: 0 2px 8px rgba(0,0,0,0.1);",
        "    overflow: hidden;",
        "    margin: 10px 0;",
        "  }",
        "  .ocr-word {",
        "    position: absolute;",
        "    padding: 2px 4px;",
        "    border: 1px solid transparent;",
        "    font-size: 12px;",
        "    font-family: monospace;",
        "    cursor: pointer;",
        "    transition: background-color 0.2s;",
        "  }",
        "  .ocr-word:hover { border: 1px solid #333; }",
        "  .confidence-very-high { background-color: #90EE90; color: #000; }",
        "  .confidence-high { background-color: #FFEB3B; color: #000; }",
        "  .confidence-medium { background-color: #FFA500; color: #fff; }",
        "  .confidence-low { background-color: #FF6B6B; color: #fff; }",
        "  .legend {",
        "    margin: 20px 0;",
        "    padding: 10px;",
        "    background-color: #f9f9f9;",
        "    border-left: 4px solid #333;",
        "  }",
        "  .legend-item { margin: 5px 0; font-size: 12px; }",
        "  .legend-color { display: inline-block; width: 20px; height: 20px; margin-right: 8px; border: 1px solid #999; }",
        "</style>",
        "</head>",
        "<body>",
    ]
    
    # Add legend
    html_lines.extend([
        "<div class=\"legend\">",
        "<strong>Confidence Score Legend:</strong>",
        "<div class=\"legend-item\"><div class=\"legend-color\" style=\"background-color: #90EE90;\"></div>Very High (0.9+)</div>",
        "<div class=\"legend-item\"><div class=\"legend-color\" style=\"background-color: #FFEB3B;\"></div>High (0.8-0.9)</div>",
        "<div class=\"legend-item\"><div class=\"legend-color\" style=\"background-color: #FFA500;\"></div>Medium (0.5-0.8)</div>",
        "<div class=\"legend-item\"><div class=\"legend-color\" style=\"background-color: #FF6B6B;\"></div>Low (<0.5)</div>",
        "</div>",
    ])
    
    # Process each page
    for page_data in json_data.get("pages", []):
        page_num = page_data.get("pageNumber", 1)
        width = page_data.get("width", 2000)
        height = page_data.get("height", 2600)
        words = page_data.get("words", [])
        
        # Calculate confidence stats
        confidences = [w["confidence"] for w in words]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        
        html_lines.extend([
            f"<div class=\"page-container\">",
            f"<div class=\"page-header\">Page {page_num}</div>",
            f"<div class=\"page-stats\">Words: {len(words)}, Avg Confidence: {avg_conf:.4f}</div>",
            f"<div class=\"ocr-page-view\" style=\"width: {min(1000, width)}px; aspect-ratio: {width}/{height};\">",
        ])
        
        # Add words positioned absolutely
        for word in words:
            geo = word.get("geometry", {})
            x0 = geo.get("x0", 0)
            y0 = geo.get("y0", 0)
            x1 = geo.get("x1", 100)
            y1 = geo.get("y1", 100)
            
            # Convert absolute coordinates to percentages
            left_pct = (x0 / width * 100) if width > 0 else 0
            top_pct = (y0 / height * 100) if height > 0 else 0
            width_pct = ((x1 - x0) / width * 100) if width > 0 else 5
            height_pct = ((y1 - y0) / height * 100) if height > 0 else 3
            
            confidence = word.get("confidence", 0)
            content = html.escape(word.get("content", ""))
            source = html.escape(word.get("source", ""))
            
            # Determine color class based on confidence
            if confidence >= 0.9:
                color_class = "confidence-very-high"
            elif confidence >= 0.8:
                color_class = "confidence-high"
            elif confidence >= 0.5:
                color_class = "confidence-medium"
            else:
                color_class = "confidence-low"
            
            title = f"{content} (confidence: {confidence:.3f}, source: {source})"
            html_lines.append(
                f'<div class="ocr-word {color_class}" '
                f'style="left:{left_pct:.1f}%;top:{top_pct:.1f}%;width:{width_pct:.1f}%;height:{height_pct:.1f}%;" '
                f'title="{title}">{content}</div>'
            )
        
        html_lines.extend([
            "</div>",
            "</div>",
        ])
    
    html_lines.extend([
        "</body>",
        "</html>",
    ])
    
    return "\n".join(html_lines)


def _render_json_to_markdown(json_data):
    """
    Render JSON output to a layout-accurate Markdown file.
    Produces a positioned HTML block (Option A: HTML-in-Markdown) per page,
    mirroring _render_json_to_html, followed by a geometry-grouped text section.
    Confidence is shown as text colour rather than background so the file
    remains readable in plain-text viewers.
    """
    lines = [
        "<style>",
        "  .md-ocr-page { position: relative; border: 1px solid #ccc;",
        "    margin: 1.5rem 0; background: #fff; overflow: hidden; }",
        "  .md-ocr-word { position: absolute; font-family: monospace;",
        "    font-size: 0.75rem; white-space: nowrap; padding: 1px 2px; }",
        "  .md-conf-vhigh { color: #1a7f37; }",
        "  .md-conf-high  { color: #856404; }",
        "  .md-conf-med   { color: #cf6500; }",
        "  .md-conf-low   { color: #c00000; text-decoration: underline dotted; }",
        "</style>",
        "",
        "> **Legend:**"
        " <span style=\"color:#1a7f37\">&#9646; Very high (0.9+)</span> &nbsp;"
        " <span style=\"color:#856404\">&#9646; High (0.8&ndash;0.9)</span> &nbsp;"
        " <span style=\"color:#cf6500\">&#9646; Medium (0.5&ndash;0.8)</span> &nbsp;"
        " <span style=\"color:#c00000\">&#9646; Low (&lt;0.5)</span>",
        "",
    ]

    for page_data in json_data.get("pages", []):
        page_num = page_data.get("pageNumber", 1)
        width = page_data.get("width", 2000)
        height = page_data.get("height", 2600)
        words = page_data.get("words", [])

        confidences = [w["confidence"] for w in words]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        lines.append(f"## Page {page_num}")
        lines.append(f"*{len(words)} words &nbsp;&middot;&nbsp; avg confidence {avg_conf:.4f}*")
        lines.append("")

        # --- Positioned layout block (mirrors HTML renderer) ---
        view_width = min(1000, width)
        lines.append(
            f'<div class="md-ocr-page" '
            f'style="width:{view_width}px;aspect-ratio:{width}/{height};">'
        )
        for word in sorted(words, key=lambda w: (w["geometry"]["y0"], w["geometry"]["x0"])):
            geo = word["geometry"]
            left_pct = (geo["x0"] / width * 100) if width > 0 else 0
            top_pct  = (geo["y0"] / height * 100) if height > 0 else 0
            w_pct    = ((geo["x1"] - geo["x0"]) / width  * 100) if width  > 0 else 5
            h_pct    = ((geo["y1"] - geo["y0"]) / height * 100) if height > 0 else 3
            conf     = word.get("confidence", 0)
            content  = html.escape(word.get("content", ""))
            source   = html.escape(word.get("source", ""))
            if conf >= 0.9:
                cls = "md-conf-vhigh"
            elif conf >= 0.8:
                cls = "md-conf-high"
            elif conf >= 0.5:
                cls = "md-conf-med"
            else:
                cls = "md-conf-low"
            lines.append(
                f'<span class="md-ocr-word {cls}" '
                f'style="left:{left_pct:.1f}%;top:{top_pct:.1f}%;'
                f'width:{w_pct:.1f}%;height:{h_pct:.1f}%;" '
                f'title="{content} (conf:{conf:.3f}, {source})">{content}</span>'
            )
        lines.append("</div>")
        lines.append("")

        # --- Geometry-grouped text section ---
        if words:
            word_cells = [
                {
                    "text": _normalize_checkbox_symbols(w["content"]),
                    "x0": w["geometry"]["x0"],
                    "x1": w["geometry"]["x1"],
                    "y0": w["geometry"]["y0"],
                    "y1": w["geometry"]["y1"],
                    "cx": (w["geometry"]["x0"] + w["geometry"]["x1"]) / 2.0,
                    "cy": (w["geometry"]["y0"] + w["geometry"]["y1"]) / 2.0,
                    "h":  max(1.0, w["geometry"]["y1"] - w["geometry"]["y0"]),
                    "w":  max(1.0, w["geometry"]["x1"] - w["geometry"]["x0"]),
                }
                for w in words
            ]
            rows = _group_cells_by_row(word_cells)
            blocks = _segment_rows_into_blocks(rows)
            lines.append("### Extracted Text")
            lines.append("")
            for block in blocks:
                if block["type"] == "table":
                    lines.extend(block["content"])
                    lines.append("")
                else:
                    content = (block.get("content") or "").strip()
                    if content:
                        lines.append(_format_ocr_fallback_line(content))
            lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
INPUT_DOCUMENT = os.environ.get("OCR_INPUT_DOCUMENT", "sample/Test5.pdf")
OUTPUT_SUMMARY = _runtime_ocr_config()["output_summary"]
FIELD_KEYWORD_CONFIG = os.environ.get("OCR_FIELD_CONFIG", "DocIntResult/Test5.pdf_field_keywords.json")
STRICT_CONFIDENCE_THRESHOLD = float(os.environ.get("OCR_STRICT_CONFIDENCE", "0.8"))


def run_local_table_extraction(doc_path: str, summary_out: str):
    text_cfg = _runtime_text_config()

    if not os.path.exists(doc_path):
        print(f"[-] {text_cfg['file_not_found_prefix']}: '{doc_path}'")
        return

    print("[*] Initializing PaddleOCR local engine...")
    try:
        structure_engine_class = get_structure_engine_class()
        table_engine = build_structure_engine(structure_engine_class)
    except Exception as exc:
        print(f"[-] Failed to initialize PaddleOCR engine: {exc}")
        with open(summary_out, "w", encoding="utf-8") as f:
            f.write(
                _build_failure_markdown(
                    doc_path,
                    "unavailable",
                    f"{text_cfg['init_failure_prefix']}: `{exc}`",
                    text_cfg["failure_title"],
                )
            )
        print(f"[-] Script stopped after initialization failure. Summary file created: {summary_out}")
        return

    local_inputs, cleanup_dirs = _prepare_local_input_pages(doc_path)
    page_outputs = []
    page_sizes = []
    page_checkboxes = []

    print(f"[*] Processing {len(local_inputs)} page(s) locally...")
    start_time = time.time()

    try:
        for local_input in local_inputs:
            with Image.open(local_input) as page_image:
                page_sizes.append((page_image.width, page_image.height))

            if hasattr(table_engine, "predict"):
                output = table_engine.predict(
                    local_input,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            else:
                output = table_engine(local_input)

            # Must happen here, while the rendered page image still exists -
            # the temp directories are removed as soon as this loop ends.
            page_cells = _extract_ocr_cells(output if isinstance(output, list) else [output])
            text_height = (
                statistics.median(c["h"] for c in page_cells) if page_cells else 0.0
            )
            checkboxes = _detect_checkbox_boxes(local_input, text_height)
            if os.environ.get("OCR_GEO_DEBUG") == "1":
                ticked = sum(1 for b in checkboxes if b[3])
                print(
                    f"[debug] {local_input}: text_height={text_height:.1f} "
                    f"boxes={len(checkboxes)} ticked={ticked}"
                )

            if isinstance(output, list):
                page_outputs.extend(output)
                page_checkboxes.extend([checkboxes] * len(output))
            else:
                page_outputs.append(output)
                page_checkboxes.append(checkboxes)
    except Exception as exc:
        execution_time = time.time() - start_time
        print(f"[-] Failed during local OCR processing: {exc}")
        with open(summary_out, "w", encoding="utf-8") as f:
            f.write(
                _build_failure_markdown(
                    doc_path,
                    f"{execution_time:.2f} seconds",
                    f"{text_cfg['process_failure_prefix']}: `{exc}`",
                    text_cfg["failure_title"],
                )
            )
        print(f"[-] Script stopped during OCR processing. Summary file created: {summary_out}")
        return
    finally:
        for temp_dir in cleanup_dirs:
            if temp_dir and os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    execution_time = time.time() - start_time
    readable_out = _derive_readable_output_path(summary_out)
    json_out = _derive_json_output_path(summary_out)
    html_out = _derive_html_output_path(summary_out)

    try:
        field_cfg = _load_field_keyword_config(FIELD_KEYWORD_CONFIG)
    except Exception as exc:
        print(f"[-] Failed to load field config: {exc}")
        with open(summary_out, "w", encoding="utf-8") as f:
            f.write(
                _build_failure_markdown(
                    doc_path,
                    f"{execution_time:.2f} seconds",
                    f"Field config error: `{exc}`",
                    text_cfg["failure_title"],
                )
            )
        print(f"[-] Script stopped after field-config failure. Summary file created: {summary_out}")
        return

    field_data = _extract_field_only_data(page_outputs, field_cfg)
    field_types = field_cfg.get("field_types", {})

    # Build canonical_fields: ALL configured keywords → extracted value or empty string.
    # Only include keys that are in the configured keyword list (no accidental extras).
    text_fields = field_data["fields"]
    canonical_fields = {}
    for k in field_cfg["keywords"]:
        canonical_fields[k] = text_fields.get(k, "")
    # Promote first non-empty value from HTML-detected or geometric-reconstructed
    # tables.  Plain text-sequence reconstruction artifacts are NOT promoted.
    html_table_names = field_data.get("html_table_names", set())
    geo_table_names = field_data.get("geo_table_names", set())
    promotable = html_table_names | geo_table_names
    for table_name, table_struct in field_data["tableData"].items():
        if table_name not in promotable:
            continue
        for row in table_struct.get("rows", []):
            for k, v in row.items():
                if k not in canonical_fields or not v or canonical_fields[k]:
                    continue
                # A bare "Yes"/"No" in a reconstructed cell is the printed label
                # beside a tick box that happened to land near this column, not
                # an answer. Only a field declared as a selection may take one.
                if (
                    v.strip().upper() in _SELECTION_TOKENS
                    and field_types.get(k) not in _SELECTION_TYPES
                ):
                    continue
                canonical_fields[k] = v

    # A caption written above its value cannot be paired by reading order when
    # two captions share a line, so settle those cases geometrically.
    for keyword, value in _extract_column_aligned_fields(page_outputs, field_cfg).items():
        if keyword not in canonical_fields or canonical_fields[keyword]:
            continue
        # One piece of text cannot answer two captions, and the caption sitting
        # directly above it is the one that owns it.
        for other in canonical_fields:
            if other != keyword and canonical_fields[other] == value:
                canonical_fields[other] = ""
        canonical_fields[keyword] = value

    # Ticked boxes are invisible to every text rule because the local OCR
    # engine never emits a checkbox glyph, so read them from the page image.
    checkbox_values, unticked_labels = _extract_checkbox_fields(
        page_outputs, page_checkboxes, field_cfg
    )
    for k, v in checkbox_values.items():
        if k in canonical_fields and not canonical_fields[k]:
            canonical_fields[k] = v
    _clear_unticked_option_values(canonical_fields, set(checkbox_values), unticked_labels)
    _reject_implausible_values(
        canonical_fields, field_cfg, page_outputs, set(checkbox_values)
    )

    field_details = _build_field_details(canonical_fields, field_types)
    # Replace canonical_fields with sanitized values from details.
    canonical_fields = {k: v["value"] for k, v in field_details.items()}

    # Keep field_data["fields"] in sync with canonical values so renderers are consistent.
    field_data["fields"] = canonical_fields

    unmatched = [k for k, v in canonical_fields.items() if not v]
    strict_fields = {
        k: info["value"]
        for k, info in field_details.items()
        if info["value"] and info["confidence"] >= STRICT_CONFIDENCE_THRESHOLD
    }
    strict_unmatched = [k for k in field_cfg["keywords"] if k not in strict_fields]

    field_data["fieldDetails"] = field_details

    json_data = {
        "keywords": field_cfg["keywords"],
        "fields": canonical_fields,
        "fieldDetails": field_details,
        "strictConfidenceThreshold": STRICT_CONFIDENCE_THRESHOLD,
        "strictFields": strict_fields,
        "strictUnmatchedKeywords": strict_unmatched,
        "unmatchedKeywords": unmatched,
    }
    if field_data["tableData"]:
        json_data["tableData"] = field_data["tableData"]

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    markdown_layout = _render_field_only_markdown(field_data)
    with open(summary_out, "w", encoding="utf-8") as f:
        f.write(markdown_layout)

    readable_layout = _render_field_only_markdown(field_data)
    with open(readable_out, "w", encoding="utf-8") as f:
        f.write(readable_layout)

    html_content = _render_field_only_html(field_data)
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[SUCCESS] {text_cfg['success_prefix']}")
    print(f"  -> Field-only markdown saved to: {summary_out}")
    print(f"  -> Field-only readable markdown saved to: {readable_out}")
    print(f"  -> Field-only JSON output saved to: {json_out}")
    print(f"  -> Field-only HTML output saved to: {html_out}")
    print(f"  -> Execution Time: {execution_time:.2f} seconds")


if __name__ == "__main__":
    run_local_table_extraction(INPUT_DOCUMENT, OUTPUT_SUMMARY)
