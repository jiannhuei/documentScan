import os
import re
import math
import cv2
import numpy as np
import fitz  # PyMuPDF
import spacy
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
<<<<<<< HEAD

# Required on this stack: PaddleX may override defaults, so set flags explicitly.
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_new_executor"] = "0"

from paddleocr import PaddleOCR

CLEAN_MODE = os.getenv("SIGNATURE_CLEAN_MODE", "auto").strip().lower()

=======
from paddleocr import PaddleOCR

>>>>>>> 66f68420fec1941c9896d778fcfdc067697c4b7b
# -------------------------------------------------------------------------
# 1. Initialization & Pre-trained Model Loading (Zero Training Required)
# -------------------------------------------------------------------------
print("[*] Loading SpaCy NLP Model...")
nlp = spacy.load("en_core_web_sm")

print("[*] Fetching pre-trained signature weights from Hugging Face...")
# Automatically downloads fine-tuned YOLOv8 signature weights on first run
sig_model_path = hf_hub_download(
    repo_id="tech4humans/yolov8s-signature-detector", 
    filename="yolov8s.pt"
)
yolo_model = YOLO(sig_model_path)

print("[*] Initializing PaddleOCR Engine...")
<<<<<<< HEAD
ocr = PaddleOCR(
    use_textline_orientation=True,
    lang='en',
    device='cpu',
    engine='paddle',
    enable_mkldnn=False,
    enable_hpi=False,
    enable_cinn=False,
)
=======
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
>>>>>>> 66f68420fec1941c9896d778fcfdc067697c4b7b


# -------------------------------------------------------------------------
# 2. Image Processing & Overlap Cleanup
# -------------------------------------------------------------------------
def clean_signature_overlap(crop):
    """
    Isolates signature ink from background lines, printed text, and stamps.
    Uses HSV color filtering for colored ink and adaptive local thresholding for black ink.
    """
<<<<<<< HEAD
    if CLEAN_MODE == "auto":
        return _auto_select_clean_signature_overlap(crop)

=======
>>>>>>> 66f68420fec1941c9896d778fcfdc067697c4b7b
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Isolate Blue and Red ink
    lower_blue, upper_blue = np.array([90, 50, 50]), np.array([135, 255, 255])
    lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 50, 50]), np.array([180, 255, 255])
    
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    color_mask = cv2.bitwise_or(mask_blue, mask_red)

    # Local variance thresholding for dark/black ink
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
<<<<<<< HEAD
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive_thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 6
    )
    _, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Combine color and adaptive dark ink masks
    combined = cv2.bitwise_or(color_mask, adaptive_thresh)
    combined = cv2.bitwise_or(combined, otsu_thresh)
=======
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 10
    )

    # Combine color and adaptive dark ink masks
    combined = cv2.bitwise_or(color_mask, adaptive_thresh)
>>>>>>> 66f68420fec1941c9896d778fcfdc067697c4b7b

    # Morphological cleaning to erase isolated noise dots and thin lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned_mask = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
<<<<<<< HEAD
    base_mask = cleaned_mask.copy()

    h, w = cleaned_mask.shape

    # Detect long straight form lines and remove them from the signature mask.
    if CLEAN_MODE == "preserve":
        h_len = max(18, int(w * 0.55))
        v_len = max(18, int(h * 0.72))
    elif CLEAN_MODE == "strict":
        h_len = max(16, int(w * 0.35))
        v_len = max(16, int(h * 0.55))
    else:
        h_len = max(20, int(w * 0.45))
        v_len = max(20, int(h * 0.65))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    h_lines = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, v_kernel)
    line_candidates = cv2.bitwise_or(h_lines, v_lines)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(line_candidates, connectivity=8)
    line_mask = np.zeros_like(cleaned_mask)
    for i in range(1, n_labels):
        x, y, bw, bh, area = stats[i]
        if area <= 0:
            continue

        is_long_h = bw >= 0.60 * w and bh <= max(5, int(0.12 * h))
        is_long_v = bh >= 0.75 * h and bw <= max(5, int(0.12 * w))

        # Remove border-anchored baselines/box edges often found in signature fields.
        touches_bottom = (y + bh) >= (h - 2)
        border_baseline = touches_bottom and bw >= 0.45 * w and bh <= max(8, int(0.18 * h))

        if is_long_h or is_long_v or border_baseline:
            line_mask[labels == i] = 255

    cleaned_mask = cv2.bitwise_and(cleaned_mask, cv2.bitwise_not(line_mask))

    # Final pass: remove tiny edge artifacts and bottom thin fragments.
    # BUT: preserve signature ink that is connected to the main body
    n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    
    # First, find the main signature body (largest component)
    areas_all = stats2[1:, cv2.CC_STAT_AREA]
    if len(areas_all) > 0:
        main_body_idx = 1 + np.argmax(areas_all)
        main_body_label = main_body_idx
    else:
        main_body_label = -1
    
    artifact_mask = np.zeros_like(cleaned_mask)
    for i in range(1, n2):
        x, y, bw, bh, area = stats2[i]
        if area <= 0:
            continue

        touches_border = x <= 1 or y <= 1 or (x + bw) >= (w - 1) or (y + bh) >= (h - 1)
        tiny_edge_blob = touches_border and area <= 36

        # Only remove bottom fragments if they are NOT connected to the main signature body
        near_bottom = y >= int(0.78 * h)
        thin_bottom_fragment = near_bottom and bh <= 3 and bw >= int(0.10 * w)
        is_main_body = (i == main_body_label)
        
        # Remove thin bottom fragments ONLY if they're not part of the main signature
        should_remove_bottom = thin_bottom_fragment and not is_main_body

        if tiny_edge_blob or should_remove_bottom:
            artifact_mask[labels2 == i] = 255

    cleaned_mask = cv2.bitwise_and(cleaned_mask, cv2.bitwise_not(artifact_mask))

    # Keep tiny dots only when they are near the main signature cluster.
    n3, labels3, stats3, _ = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    if n3 > 1:
        main_idx = 1 + int(np.argmax(stats3[1:, cv2.CC_STAT_AREA]))
        mx = stats3[main_idx, cv2.CC_STAT_LEFT]
        my = stats3[main_idx, cv2.CC_STAT_TOP]
        mw = stats3[main_idx, cv2.CC_STAT_WIDTH]
        mh = stats3[main_idx, cv2.CC_STAT_HEIGHT]

        ex = int(max(8, 0.35 * mw))
        ey = int(max(8, 0.50 * mh))
        x1, y1 = max(0, mx - ex), max(0, my - ey)
        x2, y2 = min(w - 1, mx + mw + ex), min(h - 1, my + mh + ey)

        island_mask = np.zeros_like(cleaned_mask)
        for i in range(1, n3):
            if i == main_idx:
                continue

            x, y, bw, bh, area = stats3[i]
            if area > 45:
                continue

            cx = x + bw // 2
            cy = y + bh // 2
            inside_main_neighborhood = (x1 <= cx <= x2) and (y1 <= cy <= y2)
            if not inside_main_neighborhood:
                island_mask[labels3 == i] = 255

        cleaned_mask = cv2.bitwise_and(cleaned_mask, cv2.bitwise_not(island_mask))

    # Safety fallback: if aggressive cleanup erased most ink, keep a less aggressive result.
    base_pixels = cv2.countNonZero(base_mask)
    final_pixels = cv2.countNonZero(cleaned_mask)
    if base_pixels > 0 and final_pixels < max(30, int(base_pixels * 0.18)):
        cleaned_mask = cv2.bitwise_and(base_mask, cv2.bitwise_not(line_mask))

    cleaned_mask = _clip_bottom_baseline(cleaned_mask)

    # Light reconnect to reduce dotted/fragmented pen strokes.
    reconnect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, reconnect_kernel)

    # Remove any remaining long near-horizontal baseline segments.
    lines = cv2.HoughLinesP(cleaned_mask, 1, np.pi / 180, threshold=30,
                            minLineLength=max(20, int(0.45 * w)), maxLineGap=8)
    if lines is not None:
        line_arr = lines.reshape(-1, 4)
        for line in line_arr:
            x1, y1, x2, y2 = map(int, line)
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 1:
                continue
            slope = abs(dy / dx)
            length = math.hypot(dx, dy)
            y_mid = (y1 + y2) / 2.0

            if slope <= 0.10 and length >= 0.45 * w and y_mid >= 0.45 * h:
                cv2.line(cleaned_mask, (x1, y1), (x2, y2), 0, thickness=3)

    cleaned_mask = _remove_lower_horizontal_artifacts(cleaned_mask)

    cleaned_mask = _clip_bottom_baseline(cleaned_mask)

    # Preserve complete signature ink; do not aggressively cut off content below the main body.
    # If signature extends below form lines, that's legitimate signature ink that should be kept.
    # The line removal already happened in the morphological operations above.
=======
>>>>>>> 66f68420fec1941c9896d778fcfdc067697c4b7b

    # Return dark signature stroke on clean white background
    return cv2.cvtColor(cv2.bitwise_not(cleaned_mask), cv2.COLOR_GRAY2BGR)


<<<<<<< HEAD
def _score_cleaned_signature(clean_bgr):
    """Higher is better: preserve signature ink while penalizing residual form lines/noise."""
    gray = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY)
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = ink.shape

    fg = cv2.countNonZero(ink)
    if fg <= 0:
        return -1e9

    # Basic foreground adequacy.
    density = fg / float(max(1, h * w))
    if density < 0.004:
        return -1e6 * (0.004 - density)

    # Prefer a coherent main component with fewer tiny leftovers.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return -1e8

    areas = stats[1:, cv2.CC_STAT_AREA]
    main_area = float(np.max(areas))
    small_count = int(np.sum(areas < 12))
    coherence = main_area / float(max(1, fg))

    # Penalize lower long horizontal residuals (likely form baselines).
    line_penalty = 0.0
    lines = cv2.HoughLinesP(
        ink,
        1,
        np.pi / 180,
        threshold=24,
        minLineLength=max(18, int(0.35 * w)),
        maxLineGap=6,
    )
    if lines is not None:
        line_arr = lines.reshape(-1, 4)
        for line in line_arr:
            x1, y1, x2, y2 = map(int, line)
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) < 1:
                continue
            slope = abs(dy / dx)
            length = math.hypot(dx, dy)
            y_mid = (y1 + y2) * 0.5
            if slope <= 0.12 and y_mid >= 0.50 * h:
                line_penalty += (length / float(max(1, w)))

    # Reward enough ink and coherence; penalize tiny fragments and baseline-like leftovers.
    score = 6.0 * density + 2.2 * coherence - 0.03 * small_count - 1.8 * line_penalty
    return score


def _auto_select_clean_signature_overlap(crop):
    """Evaluate preserve/balanced/strict and keep the best cleaned signature variant."""
    global CLEAN_MODE
    original_mode = CLEAN_MODE

    modes = ["preserve", "balanced", "strict"]
    best_img = None
    best_score = -1e18

    try:
        for mode in modes:
            CLEAN_MODE = mode
            candidate = clean_signature_overlap(crop)
            score = _score_cleaned_signature(candidate)
            if score > best_score:
                best_score = score
                best_img = candidate
    finally:
        CLEAN_MODE = original_mode

    return best_img if best_img is not None else crop


def _poly_to_quad(poly):
    """Convert OCR polygon/box variants into 4-point [[x,y], ...] format."""
    if poly is None:
        return [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    arr = np.asarray(poly, dtype=float)

    # Common case: flattened polygon [x1,y1,x2,y2,x3,y3,x4,y4]
    if arr.ndim == 1 and arr.size == 8:
        arr = arr.reshape(4, 2)
        return arr.tolist()

    # Common case: bbox [x1,y1,x2,y2]
    if arr.ndim == 1 and arr.size == 4:
        x1, y1, x2, y2 = arr.tolist()
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    if arr.ndim == 2 and arr.shape[1] >= 2:
        if arr.shape[0] >= 4:
            return arr[:4, :2].tolist()
        xs, ys = arr[:, 0], arr[:, 1]
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    return [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]


def normalize_ocr_results(ocr_output):
    """Normalize PaddleOCR 3.x result objects into legacy block tuples."""
    if not ocr_output:
        return [[]]

    # Legacy shape from older PaddleOCR versions.
    first = ocr_output[0]
    if isinstance(first, list):
        return ocr_output

    if hasattr(first, "keys"):
        page_blocks = []
        rec_texts = first.get("rec_texts")
        rec_scores = first.get("rec_scores")
        rec_polys = first.get("rec_polys")
        rec_boxes = first.get("rec_boxes")

        rec_texts = [] if rec_texts is None else rec_texts
        rec_scores = [] if rec_scores is None else rec_scores
        rec_polys = [] if rec_polys is None else rec_polys
        rec_boxes = [] if rec_boxes is None else rec_boxes

        for i, text in enumerate(rec_texts):
            score = float(rec_scores[i]) if i < len(rec_scores) else 0.0
            poly = rec_polys[i] if i < len(rec_polys) else (rec_boxes[i] if i < len(rec_boxes) else None)
            box = _poly_to_quad(poly)
            page_blocks.append((box, (str(text), score)))

        return [page_blocks]

    return [[]]


def _rect_iou(a, b):
    """Compute IoU between two [x1,y1,x2,y2] rectangles."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    a_area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _box_to_rect(box):
    xs = [pt[0] for pt in box]
    ys = [pt[1] for pt in box]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]

def _clip_bottom_baseline(mask):
    """Remove form baseline artifacts while preserving complete signature ink.
    
    Strategy: Detect and remove thin isolated horizontal lines below the main
    signature body, but preserve ink that is connected to or part of the signature.
    """
    h, w = mask.shape
    if h < 20 or w < 40:
        return mask

    # Find the main signature body first
    n_main, labels_main, stats_main, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_main <= 1:
        return mask
    
    # Identify the largest component (main signature)
    areas = stats_main[1:, cv2.CC_STAT_AREA]
    main_idx = 1 + np.argmax(areas)
    main_y_bottom = stats_main[main_idx, cv2.CC_STAT_TOP] + stats_main[main_idx, cv2.CC_STAT_HEIGHT]
    
    # Only look for baselines significantly below the main body
    baseline_search_start = int(main_y_bottom + 0.05 * h)  # Start searching 5% below main body
    if baseline_search_start >= h:
        return mask
    
    # Detect thin horizontal lines using morphology
    h_len = max(20, int(w * 0.50))  # Need to span 50% of width to be a line
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, h_kernel)
    
    n, labels, stats, _ = cv2.connectedComponentsWithStats(h_lines, connectivity=8)
    line_mask = np.zeros_like(mask)
    
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area <= 0 or y < baseline_search_start:
            continue
        
        # Criteria for form baseline: thin, long, and isolated
        is_thin = bh <= max(3, int(0.10 * h))
        is_long = bw >= int(0.50 * w)
        is_isolated = area <= int(0.015 * (h * w))  # Small isolated line
        
        if is_thin and is_long and is_isolated:
            line_mask[labels == i] = 255
    
    # Remove detected lines
    out = cv2.bitwise_and(mask, cv2.bitwise_not(line_mask))
    return out


def _remove_lower_horizontal_artifacts(mask):
    """Remove thin horizontal form line artifacts in lower signature region.
    
    Targets form baselines and underlines that are clearly separate from
    the main signature ink based on thickness, length, and isolation.
    """
    h, w = mask.shape
    if h < 20 or w < 40:
        return mask

    k_len = max(18, int(0.45 * w))  # Kernel to detect long horizontal strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(horizontal, connectivity=8)
    artifact = np.zeros_like(mask)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area <= 0:
            continue

        # Remove form lines: must be thin, fairly long, and in lower half
        long_enough = bw >= int(0.40 * w)
        thin_enough = bh <= max(4, int(0.12 * h))  # Up to 12% of height (4-6px typically)
        in_lower_zone = y >= int(0.50 * h)
        line_like = area <= int(0.025 * (h * w))  # Looks like a line, not thick blob
        
        if long_enough and thin_enough and in_lower_zone and line_like:
            artifact[labels == i] = 255

    if cv2.countNonZero(artifact) == 0:
        return mask
    return cv2.bitwise_and(mask, cv2.bitwise_not(artifact))

def _is_suspiciously_small_box(sig_box, img):
    """
    Rejects very small boxes (< 0.5% of image area).
    These are almost always checkboxes, radio buttons, or form elements, not signatures.
    """
    x1, y1, x2, y2 = map(float, sig_box)
    box_area = (x2 - x1) * (y2 - y1)
    total_area = img.shape[0] * img.shape[1]
    area_ratio = box_area / total_area
    
    # Reject boxes smaller than 0.5% of image
    if area_ratio < 0.005:
        return True
    
    return False

def is_checkbox_or_form_element(sig_box, img_shape):
    """
    Rejects detector boxes that are checkboxes, radio buttons, or other small form elements.
    These are typically very small (< 1% of image) or square-ish with minimal area.
    """
    # Disabled: The checkbox filter was too aggressive even at 0.8% threshold.
    # Better to extract all signatures and let the user review.
    return False

def is_text_like_false_positive(sig_box, ocr_results, img):
    """
    Rejects detector boxes that are actually typed/printed text regions.
    Real signatures usually have little or no high-confidence OCR text overlap.
    """
    if not ocr_results or not ocr_results[0]:
        return False

    sx1, sy1, sx2, sy2 = map(float, sig_box)
    sig_rect = [sx1, sy1, sx2, sy2]

    for block in ocr_results[0]:
        box, (text, score) = block
        overlap = _rect_iou(sig_rect, _box_to_rect(box))
        if overlap < 0.20:  # Require at least 20% overlap
            continue

        t = str(text).strip()
        if not t:
            continue

        alnum = re.sub(r"[^A-Za-z0-9]", "", t)
        digit_ratio = (sum(ch.isdigit() for ch in alnum) / len(alnum)) if alnum else 0.0

        # High-confidence, long alphanumeric text is almost always printed text
        if float(score) >= 0.95 and len(alnum) >= 3:
            return True

    return False


=======
>>>>>>> 66f68420fec1941c9896d778fcfdc067697c4b7b
# -------------------------------------------------------------------------
# 3. Spatial Parsing & Named Entity Recognition (NER)
# -------------------------------------------------------------------------
def extract_document_title(ocr_results, fallback_name="Document"):
    """Extracts top header text to derive the document title context."""
    if not ocr_results or not ocr_results[0]:
        return fallback_name
    
    # Sort detected blocks by vertical position (Y-axis)
    sorted_blocks = sorted(ocr_results[0], key=lambda x: x[0][0][1])
    for block in sorted_blocks[:3]:
        text = block[1][0].strip()
        if len(text) > 4 and not re.match(r'^\d+$', text):
            clean_title = re.sub(r'[^a-zA-Z0-9_]', '', text.replace(" ", "_"))
            return clean_title[:25]
            
    return fallback_name

def find_signer_name(sig_box, ocr_results):
    """
    Locates the signer's name using dynamic spatial proximity search + SpaCy NER.
    """
    if not ocr_results or not ocr_results[0]:
        return "Unknown_Signer"

    sx1, sy1, sx2, sy2 = sig_box
    sig_center_x = (sx1 + sx2) / 2
    sig_bottom_y = sy2
    candidates = []

    for block in ocr_results[0]:
        box, (text, _) = block
        tx1, ty1 = box[0]
        tx2, _ = box[2]
        text_center_x = (tx1 + tx2) / 2

        # Filter text blocks within proximity window of the signature bounding box
        if (ty1 >= sy1 - 30) and (ty1 <= sy2 + 250) and (abs(text_center_x - sig_center_x) < 400):
            dist = math.hypot(text_center_x - sig_center_x, ty1 - sig_bottom_y)
            candidates.append((dist, text))

    candidates.sort(key=lambda x: x[0])

    # Method 1: SpaCy Named Entity Recognition for person names
    for _, text in candidates:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return re.sub(r'[^a-zA-Z0-9_]', '', ent.text.replace(" ", "_"))

    # Method 2: RegEx Anchor Keyphrases ("Name:", "By:", "Signer:")
    for _, text in candidates:
        match = re.search(r'(?:Name|By|Printed Name|Signer)[:\s]*([A-Za-z\s]{3,30})', text, re.IGNORECASE)
        if match:
            return re.sub(r'[^a-zA-Z0-9_]', '', match.group(1).strip().replace(" ", "_"))

    # Method 3: Fallback to closest non-keyword text
    for _, text in candidates:
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '', text.replace(" ", "_"))
        if len(cleaned) > 2 and not any(kw in cleaned.lower() for kw in ["date", "title", "signed", "page"]):
            return cleaned

    return "Unknown_Signer"


# -------------------------------------------------------------------------
# 4. Multi-threaded Rendering & Unified Execution Pipeline
# -------------------------------------------------------------------------
def render_pdf_page_task(pdf_path, page_idx, dpi=200):
    """Parallel CPU task for rendering PDF pages to OpenCV BGR images."""
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    doc.close()
    return page_idx + 1, img

def process_single_image(img, page_num, base_name, output_dir):
    """Core extraction workflow for a single image frame."""
    # Fast Pass: Detect signature bounding boxes with YOLO
<<<<<<< HEAD
    # Use 0.08 confidence for sensitive detection; filtering removes checkboxes and text
    yolo_results = yolo_model(img, conf=0.08, iou=0.45, verbose=False)
    sig_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    print(f"[*] Page {page_num}: Detected {len(sig_boxes)} signatures")
=======
    yolo_results = yolo_model(img, verbose=False)
    sig_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
>>>>>>> 66f68420fec1941c9896d778fcfdc067697c4b7b

    # Skip heavy OCR if no signatures are found on this page
    if len(sig_boxes) == 0:
        return 0

    # Run OCR only on pages where signatures exist
<<<<<<< HEAD
    ocr_results = normalize_ocr_results(ocr.predict(img))

    # Keep low-confidence candidates only if they do not look like printed text regions
    # and are not extremely small suspicious boxes
    filtered_sig_boxes = [
        box for box in sig_boxes
        if not _is_suspiciously_small_box(box[:4], img)
        and not is_text_like_false_positive(box[:4], ocr_results, img)
    ]
    print(f"[*] After filter: {len(filtered_sig_boxes)} signatures remain")

    if len(filtered_sig_boxes) == 0:
        return 0

=======
    ocr_results = ocr.ocr(img, cls=False)
>>>>>>> 66f68420fec1941c9896d778fcfdc067697c4b7b
    doc_title = extract_document_title(ocr_results, fallback_name=base_name)

    extracted_count = 0
    h, w, _ = img.shape

<<<<<<< HEAD
    for idx, box in enumerate(filtered_sig_boxes):
=======
    for idx, box in enumerate(sig_boxes):
>>>>>>> 66f68420fec1941c9896d778fcfdc067697c4b7b
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Add 5px padding around signature crop
        pad = 5
        crop_x1, crop_y1 = max(0, x1 - pad), max(0, y1 - pad)
        crop_x2, crop_y2 = min(w, x2 + pad), min(h, y2 + pad)
        
        raw_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

        # Clean overlap & restore ink
        clean_sig = clean_signature_overlap(raw_crop)

        # Spatial match to identify signer
        signer_name = find_signer_name((x1, y1, x2, y2), ocr_results)

        # Output filename: <DocName>_p<Page>_<PersonName>_sig<N>.png
        out_filename = f"{doc_title}_p{page_num}_{signer_name}_sig{idx+1}.png"
        out_path = os.path.join(output_dir, out_filename)
        cv2.imwrite(out_path, clean_sig)
        
        print(f"  [+] Extracted: {out_filename}")
        extracted_count += 1

    return extracted_count

def process_document(file_path, output_dir="extracted_signatures"):
    """
    Unified entry point. Automatically handles single-page images 
    or renders multi-page PDFs in parallel using all available CPU threads.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    ext = os.path.splitext(file_path)[1].lower()

    print(f"\n==========================================")
    print(f"Processing File: {file_path}")
    print(f"==========================================")

    # 1. Multi-Page PDF Handling
    if ext == ".pdf":
        doc = fitz.open(file_path)
        total_pages = len(doc)
        doc.close()

        # Automatically use all logical CPU cores available on the machine
        max_workers = os.cpu_count() or 4
        print(f"Rendering {total_pages} pages in parallel across {max_workers} threads...")

        rendered_pages = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(render_pdf_page_task, file_path, p_idx) 
                for p_idx in range(total_pages)
            ]
            for future in as_completed(futures):
                page_num, img = future.result()
                rendered_pages[page_num] = img

        total_sigs = 0
        for page_num in range(1, total_pages + 1):
            img = rendered_pages[page_num]
            sigs_found = process_single_image(img, page_num, base_name, output_dir)
            total_sigs += sigs_found

        print(f"\n[Completed] Extracted {total_sigs} signature(s) from PDF.")

    # 2. Single-Page Image Handling
    elif ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        img = cv2.imread(file_path)
        if img is None:
            print(f"Error: Could not load image file {file_path}")
            return
            
        sigs_found = process_single_image(img, 1, base_name, output_dir)
        print(f"\n[Completed] Extracted {sigs_found} signature(s) from Image.")

    else:
        print(f"Unsupported file format: {ext}")


# -------------------------------------------------------------------------
# 5. Pipeline Execution
# -------------------------------------------------------------------------
if __name__ == "__main__":
<<<<<<< HEAD
    # Manually assign documents to process
    documents = [
        "sample/signature/Test2.pdf",
        # "sample/Test3.pdf",
        # "sample/Test5.pdf",
        # Add or remove files as needed
    ]
    
    for doc_path in documents:
        if os.path.exists(doc_path):
            process_document(doc_path)
        else:
            print(f"File not found: {doc_path}")
=======
    # Example usage:
    # process_document("my_contract.pdf")
    # process_document("scanned_image.png")
    pass
>>>>>>> 66f68420fec1941c9896d778fcfdc067697c4b7b
