import cv2
import numpy as np

def draw_numbered_boxes(image_bgr, boxes, scores, score_thresh=0.55):
    """
    Renders high-contrast bounding boxes with sequential numbered badges (#1, #2, ...)
    and a top summary banner.
    """
    img_h, img_w = image_bgr.shape[:2]
    img_area = img_h * img_w
    
    # 1. Filter out giant outlier bounding boxes (>35% of total image area)
    valid_detections = []
    for box, score in zip(boxes, scores):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        bw, bh = x2 - x1, y2 - y1
        area = bw * bh
        if 100 < area < (0.35 * img_area):
            valid_detections.append((x1, y1, x2, y2, float(score)))

    # 2. Sort spatially (top-to-bottom, left-to-right) for natural sequential numbering
    valid_detections.sort(key=lambda d: (d[1] // 100, d[0]))

    annotated = image_bgr.copy()
    
    for idx, (x1, y1, x2, y2, score) in enumerate(valid_detections, start=1):
        # Green bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Numbered Badge (e.g. "#1", "#2")
        badge_text = f"#{idx}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thickness = 2
        (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, font_thickness)

        bx1 = x1
        by1 = max(0, y1 - th - 10)
        bx2 = bx1 + tw + 10
        by2 = by1 + th + 8

        # Badge pill background
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 100, 0), -1)
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 1)

        # Badge white text
        cv2.putText(
            annotated,
            badge_text,
            (bx1 + 5, by2 - 4),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )

    # 3. Top Summary Banner
    total_count = len(valid_detections)
    banner_text = f"TOTAL GREEN BOXES COUNTED: {total_count}"
    
    cv2.rectangle(annotated, (20, 20), (520, 80), (0, 0, 0), -1)
    cv2.rectangle(annotated, (20, 20), (520, 80), (0, 255, 0), 2)
    
    cv2.putText(
        annotated,
        banner_text,
        (35, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return annotated, total_count, valid_detections
