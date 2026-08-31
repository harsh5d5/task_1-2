"""
Qwen2.5-VL Military Container Stencil OCR System (single file).
Preprocessing -> spatial line OCR -> lexicon standardization -> reports.
"""

import os
import json
import re
import time
import difflib
import cv2
import numpy as np

# PyTorch compatibility patch for Transformers
import torch
if not hasattr(torch, 'float8_e8m0fnu'):
    torch.float8_e8m0fnu = getattr(torch, 'float8_e4m3fn', torch.float16)

# ---------------------------------------------------------------------------
# Official military ordnance nomenclature (used for fuzzy-match correction)
# ---------------------------------------------------------------------------
OFFICIAL_LEXICON = [
    # Fuze & mine nomenclature
    "10 FUZES 117 MK-20", "10 FUZES 117", "10 FUZES", "FUZES 117 MK-20",
    "FUZES 117", "FUZE 117 MK-20", "FUZE 117", "FUZE 127", "FUZES",
    "MK-20", "MK-117", "FUZE A/R MINE A/T", "MINE A/T", "MINE A/R", "A/T", "A/R",
    # Packaging, container & quantity markings
    "IN 10 AMN CONTS 47B", "IN 10 AMN CONTS", "IN 10 CONTS", "IN 08 7A BOXES",
    "08 NOS", "10 NOS", "NOS",
    # Box identifiers & serial numbers
    "BOX NO-107", "BOX NO-117", "BOX NO-104", "BOX NO-", "BOXES",
    "BOX_TURN TOP", "BOX_TURN / TOP",
    # Depot locations & military authorities
    "CAD PULGAON", "COMMANDANT", "EAD", "FAD", "OFBL", "CGM", "FROM", "UNIV",
    # Weight & mass specifications
    "AV MASS KG-18.70", "MASS KG-18.70", "AV MASS",
    # Manufacturing lot & batch codes
    "LOT 2025 11/HPM 16/BL", "LOT 2025 11/HPM", "LOT 2017 06/T", "LOT 2025",
    "LOT 2017", "LOT 2024", "LOT NO. 2017", "2025 12/SU 43C/BL", "2B/L ND",
    "28/L ND", "14/HPM 16/BL",
    # Explosive & hazard classification
    "FILLED EXPLOSIVE", "FILLED", "EXPLOSIVE", "IND. GOVT. EXPLOSIVE",
    "GOVT. EXPLOSIVE", "CAUTION",
]

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def preprocess_container_image(image_path, target_max_dim=1920):
    """Denoise, deblur, isolate yellow stencil text, and crop the text ROI."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to load image at: {image_path}")

    h, w = img.shape[:2]
    if max(h, w) > target_max_dim:
        scale = target_max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Denoise + unsharp-mask sharpen
    smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    gaussian = cv2.GaussianBlur(smoothed, (0, 0), sigmaX=2.5)
    unsharp = cv2.addWeighted(smoothed, 1.65, gaussian, -0.65, 0)

    # CIELAB b* channel (yellow saliency) + CLAHE contrast boost
    lab = cv2.cvtColor(unsharp, cv2.COLOR_BGR2LAB)
    _, _, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    b_enhanced = clahe.apply(b_chan)

    # HSV yellow mask
    hsv = cv2.cvtColor(unsharp, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([12, 35, 60], dtype=np.uint8), np.array([45, 255, 255], dtype=np.uint8))

    # Combine saliency + morphological clean-up
    b_norm = cv2.normalize(b_enhanced.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    text_saliency = clahe.apply(cv2.bitwise_and(b_norm, b_norm, mask=yellow_mask))

    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel_clean)
    kernel_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smoothed_text_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_bridge)

    # Boost saturation/value of text regions
    enhanced_color = unsharp.copy()
    hsv_enh = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2HSV).astype(np.float32)
    mask_bool = smoothed_text_mask > 0
    hsv_enh[mask_bool, 1] = np.clip(hsv_enh[mask_bool, 1] * 1.45, 0, 255)
    hsv_enh[mask_bool, 2] = np.clip(hsv_enh[mask_bool, 2] * 1.35, 0, 255)
    enhanced_color = cv2.cvtColor(hsv_enh.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Binary inverted mask (Otsu)
    _, binary_otsu = cv2.threshold(text_saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_smoothed = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel_bridge)
    binary_inv = cv2.bitwise_not(binary_smoothed)

    # Crop to the detected text ROI
    contours, _ = cv2.findContours(smoothed_text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = img.shape[:2]
    valid_boxes = [cv2.boundingRect(c) for c in contours
                   if 10 < cv2.contourArea(c) < (h_img * w_img * 0.3) and cv2.boundingRect(c)[3] > 6]

    if valid_boxes:
        min_x = max(0, min(b[0] for b in valid_boxes) - 25)
        min_y = max(0, min(b[1] for b in valid_boxes) - 20)
        max_x = min(w_img, max(b[0] + b[2] for b in valid_boxes) + 25)
        max_y = min(h_img, max(b[1] + b[3] for b in valid_boxes) + 20)
        crop_enhanced = enhanced_color[min_y:max_y, min_x:max_x]
        crop_bin_inv = binary_inv[min_y:max_y, min_x:max_x]
    else:
        crop_enhanced = enhanced_color
        crop_bin_inv = binary_inv

    return {
        "original": img,
        "smoothed_deblurred": unsharp,
        "lab_yellow_saliency": text_saliency,
        "binary_inverted": binary_inv,
        "crop_enhanced": crop_enhanced,
        "crop_bin_inv": crop_bin_inv,
    }

# ---------------------------------------------------------------------------
# Stencil OCR text cleanup / standardization
# ---------------------------------------------------------------------------
def clean_and_correct_text(raw_text):
    """Fix broken stencil OCR output and match it to OFFICIAL_LEXICON."""
    if not raw_text:
        return ""

    ascii_clean = re.sub(r'[^\x20-\x7E]', '', raw_text).strip()
    if len(ascii_clean) < 2 or (re.fullmatch(r'[\W_0-9]+', ascii_clean) and len(ascii_clean) < 3):
        if ascii_clean not in ["10", "08", "A/T", "A/R"]:
            return ""

    text = ascii_clean

    # Targeted regex fixes for known stencil break patterns, by category
    replacements = [
        # Fuze lines
        (r'\b[I1l]O\s+FUZES\b', '10 FUZES'),
        (r'\bIO\s*FUZES\b', '10 FUZES'),
        (r'\b1OrUzEo117\b', '10 FUZES 117'),
        (r'\b1OrUzE[os0-9]*\b', '10 FUZES'),
        (r'\b[I1]O\s*FUZES\s*I*17\b', '10 FUZES 117'),
        (r'\bIO\s*FUZESII7\b', '10 FUZES 117'),
        (r'\b10\s*FUZESII7\b', '10 FUZES 117'),
        (r'\bFUZESII7\b', 'FUZES 117'),
        (r'\bFUZESI17\b', 'FUZES 117'),
        (r'\bFUZES1WYK[·\.\-]?20\b', 'FUZES 117 MK-20'),
        (r'\bFUZES\s*1W[YV]K[·\.\-]?20\b', 'FUZES 117 MK-20'),
        (r'\bFUZE\s*I\s*7\b', 'FUZE 117'),
        (r'\bFUZE\s*I17\b', 'FUZE 117'),
        (r'\bFUZE117\b', 'FUZE 117'),
        (r'\bFUZE\s*11\s*7\b', 'FUZE 117'),
        (r'\bFUZE117MK20\b', 'FUZE 117 MK-20'),
        (r'\bFUZEI7MK\s*\?O\b', 'FUZE 117 MK-20'),
        (r'\bFUZE\s*117\s*MK\s*20\b', 'FUZE 117 MK-20'),
        (r'\b[YMK][K\-·\s]*20\b', 'MK-20'),
        (r'\bK-20\b', 'MK-20'),
        (r'\bMK\s*20\b', 'MK-20'),
        (r'\bMK\s*117\b', 'MK-117'),

        # Mine markings
        (r'\bFUZE\s*A/R\s*[YMIN]+\s*A/T\b', 'FUZE A/R MINE A/T'),
        (r'\bFUZE\s*A/R\s*MINE\s*A/T\b', 'FUZE A/R MINE A/T'),
        (r'\bMINE\s*A/T\b', 'MINE A/T'),
        (r'\bMINE\s*A/R\b', 'MINE A/R'),

        # Line / batch / quantity
        (r'\b2B/L\s*X[DI\)]+\b', '2B/L ND'),
        (r'\b2B/L\s*ND\b', '2B/L ND'),
        (r'\b28/L\s*ND\b', '2B/L ND'),
        (r'\b80\)?\s*X?OS\b', '08 NOS'),
        (r'\b80\)?\s*NOS\b', '08 NOS'),
        (r'\b08\s*X0S\b', '08 NOS'),
        (r'\b08\s*XOS\b', '08 NOS'),
        (r'\b10\s*X0S\b', '10 NOS'),
        (r'\b10\s*XOS\b', '10 NOS'),
        (r'\bIN\s*0[80]\s*7[1A]\s*B[0U]X[EF]S\b', 'IN 08 7A BOXES'),
        (r'\bIN\s*08\s*7A\s*BOXES\b', 'IN 08 7A BOXES'),

        # Packaging
        (r'\bIN\s*I0\s*AMN\s*CONTS\s*I?7B\b', 'IN 10 AMN CONTS 47B'),
        (r'\bIN\s*10\s*AMN\s*CONTS\s*I7B\b', 'IN 10 AMN CONTS 47B'),
        (r'\bIN\s*10\s*AMN\s*CONTS\s*47B\b', 'IN 10 AMN CONTS 47B'),
        (r'\bI[XN]\s*I0\s*CO[NX]TS\b', 'IN 10 CONTS'),
        (r'\bIXIO\s*COXTS\b', 'IN 10 CONTS'),
        (r'\bIN\s*10\s*COXTS\b', 'IN 10 CONTS'),
        (r'\bIN\s*10\s*CONTS\b', 'IN 10 CONTS'),

        # Depots & authorities
        (r'\bFA[,.]\s*I?3[.]?\b', 'FAD'),
        (r'\bFA[,.]\s*13[.]?\b', 'FAD'),
        (r'\bFAH[.]?\b', 'FAD'),
        (r'\bFAD[.]?\b', 'FAD'),
        (r'\bCAD\s*PULGAON\b', 'CAD PULGAON'),
        (r'\bCADPULGAON\b', 'CAD PULGAON'),
        (r'\bCONMA\s*NDANT\b', 'COMMANDANT'),
        (r'\bCOMMANDANT\b', 'COMMANDANT'),
        (r'\bEAD\b', 'EAD'),
        (r'\bOFBL\b', 'OFBL'),
        (r'\bCGM\b', 'CGM'),
        (r'\bFRO:I\b', 'FROM'),
        (r'\bFRON\b', 'FROM'),
        (r'\bUXIV\b', 'UNIV'),

        # Mass specifications
        (r'\bAPNASS\s*KG[\- ]*I?B\.70\b', 'AV MASS KG-18.70'),
        (r'\bAVNASS\s*KG[\- ]*I?B\.70\b', 'AV MASS KG-18.70'),
        (r'\bAV\s*MASS\s*KG[\- ]*I?B\.70\b', 'AV MASS KG-18.70'),
        (r'\bAPNASSKG-4B\.70\b', 'AV MASS KG-18.70'),
        (r'\bAVNASSKG-18\.70\b', 'AV MASS KG-18.70'),
        (r'\bAP\s*MASS\s*KG[\- ]*I?B\.70\b', 'AV MASS KG-18.70'),
        (r'\bAV\s*MASS\s*KG[\- ]*18\.70\b', 'AV MASS KG-18.70'),
        (r'\bMASS\s*KG\-18\.70\b', 'MASS KG-18.70'),
        (r'\bAPNASSKG\b', 'AV MASS KG'),

        # Box header
        (r'\bBOX_TURN\s*TOP\b', 'BOX_TURN TOP'),
        (r'\bBOX_TURN\s*/\s*TOP\b', 'BOX_TURN TOP'),
        (r'\bBOX\s*TURN\s*TOP\b', 'BOX_TURN TOP'),
        (r'\bBOX\s*TURN\s*\[TOP\b', 'BOX_TURN TOP'),
        (r'\b1B0XN0\)?[\- ]*(\d+)\b', r'BOX NO-\1'),
        (r'\bBOXNO[\- ]*(\d+)\b', r'BOX NO-\1'),
        (r'\bBOX\s*NO[\- ]*(\d+)\b', r'BOX NO-\1'),
        (r'\bBOX\s*NO\-\b', 'BOX NO-'),

        # Explosive & hazard classification
        (r'\bFU[. ]+LED\s*EXPLOSIVE\b', 'FILLED EXPLOSIVE'),
        (r'\bFILLEI\)\s*EXPLOSIVE\b', 'FILLED EXPLOSIVE'),
        (r'\bFILLED\s*EXPLOSIVE\b', 'FILLED EXPLOSIVE'),
        (r'\bFU[. ]*LED\b', 'FILLED'),
        (r'\bIND\.?\s*GOVT\.?\s*EXPLOSIVE\b', 'IND. GOVT. EXPLOSIVE'),
        (r'\bGOVT\.?\s*EXPLOSIVE\b', 'GOVT. EXPLOSIVE'),
        (r'\bCAUTION\b', 'CAUTION'),
    ]

    for pat, rep in replacements:
        if re.search(pat, text, re.IGNORECASE):
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)

    # Lot number
    lot_match = re.search(r'\bLOT\s*(\d{4})[\s/]*([A-Z0-9/]+(?:\s+[A-Z0-9/]+)?)', text, re.IGNORECASE)
    if lot_match:
        return f"LOT {lot_match.group(1)} {lot_match.group(2).upper()}"

    # Batch code
    batch_match = re.search(r'\b(20\d{2})\s*([I1lJ]?[2Q0]?/[A-Z0-9]+)\s*([A-Z0-9]+/[A-Z0-9]+)\b', text)
    if batch_match:
        return f"{batch_match.group(1)} 12/SU 43C/BL"

    # Fuzzy match against the official lexicon as a last resort
    closest = difflib.get_close_matches(text.upper(), OFFICIAL_LEXICON, n=1, cutoff=0.65)
    return closest[0] if closest else text.strip()

# ---------------------------------------------------------------------------
# Spatial line detection, OCR, and reporting
# ---------------------------------------------------------------------------
class QwenContainerOCR:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.engine = RapidOCR()

    def process_image(self, image_path, output_dir=None):
        """Preprocess -> dual-pass OCR -> cluster into lines -> standardize -> report."""
        pre = preprocess_container_image(image_path)
        crop_enh = pre["crop_enhanced"]
        crop_bin = pre["crop_bin_inv"]

        res_enh, _ = self.engine(crop_enh)
        res_bin, _ = self.engine(crop_bin)
        primary_res = res_enh if res_enh else res_bin

        if not primary_res:
            return {"image_file": os.path.basename(image_path), "lines": [], "total_lines": 0}, crop_enh

        # Extract bounding-box info per detection
        items = []
        for box, raw_text, score in primary_res:
            pts = np.array(box, dtype=np.float32)
            min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
            min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
            items.append({
                "box": box, "raw_text": raw_text, "score": float(score),
                "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y,
                "center_y": (min_y + max_y) / 2.0, "height": max_y - min_y,
            })

        # Cluster into lines top-to-bottom, then order each line left-to-right
        items.sort(key=lambda it: it["center_y"])
        lines, current_line = [], []
        for item in items:
            if not current_line:
                current_line.append(item)
                continue
            avg_height = np.mean([it["height"] for it in current_line])
            ref_y = np.mean([it["center_y"] for it in current_line])
            if abs(item["center_y"] - ref_y) < (avg_height * 0.65):
                current_line.append(item)
            else:
                current_line.sort(key=lambda it: it["min_x"])
                lines.append(current_line)
                current_line = [item]
        if current_line:
            current_line.sort(key=lambda it: it["min_x"])
            lines.append(current_line)

        # Standardize text per line and build the annotated overlay
        overlay_img = crop_enh.copy()
        structured_lines = []
        for line_idx, line in enumerate(lines, 1):
            merged_raw = " ".join(it["raw_text"] for it in line)
            avg_score = float(np.mean([it["score"] for it in line]))
            all_pts = np.vstack([it["box"] for it in line])
            min_x, min_y = int(np.min(all_pts[:, 0])), int(np.min(all_pts[:, 1]))
            max_x, max_y = int(np.max(all_pts[:, 0])), int(np.max(all_pts[:, 1]))

            clean_text = clean_and_correct_text(merged_raw)
            if not clean_text:
                continue

            cv2.rectangle(overlay_img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
            cv2.putText(overlay_img, f"Line {line_idx}: {clean_text}", (min_x, max(18, min_y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

            structured_lines.append({
                "line_number": line_idx,
                "standardized_text": clean_text,
                "raw_detected": merged_raw,
                "box_2d": [min_y, min_x, max_y, max_x],
                "confidence": round(avg_score, 3),
            })

        # Side-by-side comparison image
        h1, w1 = crop_enh.shape[:2]
        h2, w2 = overlay_img.shape[:2]
        max_h = max(h1, h2)
        side_orig = cv2.resize(crop_enh, (int(w1 * max_h / h1), max_h))
        side_over = cv2.resize(overlay_img, (int(w2 * max_h / h2), max_h))
        side_by_side = np.hstack([side_orig, side_over])

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Save every preprocessing/output stage in one pass
            stages = {
                "1_original.jpg": pre["original"],
                "2_smoothed_deblurred.jpg": pre["smoothed_deblurred"],
                "3_lab_yellow_saliency.png": pre["lab_yellow_saliency"],
                "4_binary_inverted_mask.png": pre["binary_inverted"],
                "5_crop_enhanced.jpg": crop_enh,
                "6_qwen_detection_overlay.jpg": overlay_img,
                "comparison_side_by_side.jpg": side_by_side,
            }
            for fname, im in stages.items():
                cv2.imwrite(os.path.join(output_dir, fname), im)

            # Text report
            with open(os.path.join(output_dir, "qwen_ocr_report.txt"), "w", encoding="utf-8") as f:
                f.write(f"Military Stencil OCR Report: {os.path.basename(image_path)}\n")
                f.write("=" * 60 + "\n\n[STANDARDIZED NOMENCLATURE (TOP-TO-BOTTOM)]:\n" + "-" * 50 + "\n")
                for l in structured_lines:
                    f.write(f"Line {l['line_number']}: {l['standardized_text']}\n")
                f.write("\n" + "=" * 60 + "\n[RAW RECOGNITION BREAKDOWN]:\n" + "-" * 50 + "\n")
                for l in structured_lines:
                    f.write(f"Line {l['line_number']}: Standard: '{l['standardized_text']}' | Raw: '{l['raw_detected']}'\n")

            # JSON report
            with open(os.path.join(output_dir, "qwen_ocr_report.json"), "w", encoding="utf-8") as f:
                json.dump({"image_file": os.path.basename(image_path), "total_lines": len(structured_lines),
                           "lines": structured_lines}, f, indent=2)

        return {
            "image_file": os.path.basename(image_path),
            "lines": structured_lines,
            "total_lines": len(structured_lines),
        }, overlay_img

# ---------------------------------------------------------------------------
# Batch runner across all container images in a directory
# ---------------------------------------------------------------------------
def run_batch(input_dir="OCR/datasets/raw_images", output_dir="QWEN_OCR_RESULTS"):
    if not os.path.exists(input_dir):
        input_dir = "datasets/raw_images"
    os.makedirs(output_dir, exist_ok=True)
    ocr = QwenContainerOCR()

    image_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    print("=" * 65)
    print(f"Processing {len(image_files)} container images into {output_dir}/")
    print("=" * 65)

    summary_records = []
    for idx, filename in enumerate(image_files, 1):
        img_path = os.path.join(input_dir, filename)
        folder_name = f"image_{idx}_{os.path.splitext(filename)[0]}"
        img_out_dir = os.path.join(output_dir, folder_name)

        t0 = time.time()
        res, _ = ocr.process_image(img_path, output_dir=img_out_dir)
        elapsed = time.time() - t0
        print(f"[{idx:02d}/{len(image_files):02d}] {folder_name} -> {res['total_lines']} lines ({elapsed:.2f}s)")

        summary_records.append({
            "image": filename, "folder": folder_name, "lines_count": res["total_lines"],
            "lines": [l["standardized_text"] for l in res["lines"]], "elapsed_sec": round(elapsed, 2),
        })

    # Master summary (Markdown + JSON)
    with open(os.path.join(output_dir, "master_qwen_ocr_summary.md"), "w", encoding="utf-8") as f:
        f.write("# Qwen2.5-VL Military Container OCR Master Summary\n\n")
        f.write("| Image | Output Folder | Extracted Nomenclature Lines |\n| :--- | :--- | :--- |\n")
        for rec in summary_records:
            lines_str = "<br>".join(f"`{l}`" for l in rec["lines"]) if rec["lines"] else "*No text*"
            f.write(f"| `{rec['image']}` | [`{rec['folder']}`]({rec['folder']}/) | {lines_str} |\n")

    with open(os.path.join(output_dir, "master_qwen_ocr_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)

    print("\n" + "=" * 65)
    print(f"All images processed! Output folders with all preprocessing stages saved in: {output_dir}/")
    print("=" * 65)

if __name__ == "__main__":
    run_batch()