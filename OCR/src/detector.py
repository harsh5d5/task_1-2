import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from src.nomenclature_cleaner import clean_and_correct_text

class ContainerTextDetector:
    def __init__(self):
        self.engine = RapidOCR()

    def detect_and_align(self, image_data):
        """
        Runs DBNet++ on enhanced crop and inverted binary, sorts lines spatially,
        and applies standard military nomenclature correction.
        """
        crop_enh = image_data["crop_enhanced"]
        crop_bin = image_data["crop_bin_inv"]
        
        # 1. Inference
        res_enh, _ = self.engine(crop_enh)
        res_bin, _ = self.engine(crop_bin)
        
        primary_res = res_enh if res_enh else res_bin
        if not primary_res:
            return [], crop_enh

        # 2. Extract bounding info
        items = []
        for box, raw_text, score in primary_res:
            pts = np.array(box, dtype=np.float32)
            min_x = np.min(pts[:, 0])
            max_x = np.max(pts[:, 0])
            min_y = np.min(pts[:, 1])
            max_y = np.max(pts[:, 1])
            center_y = (min_y + max_y) / 2.0
            height = max_y - min_y
            items.append({
                "box": box,
                "raw_text": raw_text,
                "score": score,
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "center_y": center_y,
                "height": height
            })

        # 3. Spatial Line Clustering (top-to-bottom, left-to-right)
        items.sort(key=lambda it: it["center_y"])
        lines = []
        current_line = []
        
        for item in items:
            if not current_line:
                current_line.append(item)
            else:
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

        # 4. Correct Text & Draw Overlay
        overlay_img = crop_enh.copy()
        structured_lines = []

        for line in lines:
            merged_raw = " ".join([it["raw_text"] for it in line])
            avg_score = float(np.mean([it["score"] for it in line]))
            
            all_pts = np.vstack([it["box"] for it in line])
            min_x = int(np.min(all_pts[:, 0]))
            min_y = int(np.min(all_pts[:, 1]))
            max_x = int(np.max(all_pts[:, 0]))
            max_y = int(np.max(all_pts[:, 1]))
            
            clean_text = clean_and_correct_text(merged_raw)
            if not clean_text:
                continue

            # Draw visual polygon and text
            box_coords = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
            pts = np.array(box_coords, np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            pt_label = (min_x, max(16, min_y - 6))
            cv2.putText(overlay_img, clean_text, pt_label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

            structured_lines.append({
                "box": box_coords,
                "raw_text": merged_raw,
                "clean_text": clean_text,
                "confidence": round(avg_score, 3)
            })

        return structured_lines, overlay_img
