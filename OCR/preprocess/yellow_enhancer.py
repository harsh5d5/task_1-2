import cv2
import numpy as np
import os

def find_text_region(mask, img_shape):
    """
    Finds bounding cluster of yellow stencil text.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = img_shape[:2]
    
    valid_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area > 15 and h > 8 and y > h_img * 0.4:
            valid_boxes.append((x, y, w, h))
            
    if not valid_boxes:
        valid_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 20]
        
    if valid_boxes:
        min_x = max(0, min([b[0] for b in valid_boxes]) - 30)
        min_y = max(0, min([b[1] for b in valid_boxes]) - 25)
        max_x = min(w_img, max([b[0] + b[2] for b in valid_boxes]) + 30)
        max_y = min(h_img, max([b[1] + b[3] for b in valid_boxes]) + 25)
        return min_x, min_y, max_x, max_y
    return 0, int(h_img * 0.4), w_img, h_img

def preprocess_yellow_text(image_path, output_dir="output_preprocessed"):
    """
    Applies bilateral smoothing, unsharp deblurring, LAB b* yellow isolation,
    and morphological stencil bridging to enhance yellow text on metal boxes.
    """
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    # 1. Bilateral Filtering for edge-preserving noise smoothing
    smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # 2. Deblurring / Unsharp Masking for crisp stencil text
    gaussian = cv2.GaussianBlur(smoothed, (0, 0), sigmaX=2.5)
    unsharp = cv2.addWeighted(smoothed, 1.7, gaussian, -0.7, 0)

    # 3. LAB Color Space - Extract b* channel (yellow-blue opponent axis)
    lab = cv2.cvtColor(unsharp, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    b_enhanced = clahe.apply(b_chan)
    l_enhanced = clahe.apply(l_chan)

    # 4. HSV Color Space - Precise Yellow Extraction
    hsv = cv2.cvtColor(unsharp, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([14, 45, 70], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 5. Combine LAB yellow response with HSV mask
    b_norm = cv2.normalize(b_enhanced.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    text_saliency = cv2.bitwise_and(b_norm, b_norm, mask=yellow_mask)
    text_saliency = clahe.apply(text_saliency)

    # 6. Morphological refinement & Stencil character smoothing
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel_clean)
    
    # Bridge stencil gaps gently
    kernel_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smoothed_text_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_bridge)

    # 7. Enhanced Crisp Color Image
    enhanced_color = unsharp.copy()
    hsv_enh = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2HSV).astype(np.float32)
    mask_bool = smoothed_text_mask > 0
    hsv_enh[mask_bool, 1] = np.clip(hsv_enh[mask_bool, 1] * 1.5, 0, 255)
    hsv_enh[mask_bool, 2] = np.clip(hsv_enh[mask_bool, 2] * 1.35, 0, 255)
    enhanced_color = cv2.cvtColor(hsv_enh.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 8. Clean Binary OCR Ready Image
    _, binary_otsu = cv2.threshold(text_saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_smoothed = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel_bridge)
    binary_inv = cv2.bitwise_not(binary_smoothed)

    # 9. ROI Crop around the text panel
    x1, y1, x2, y2 = find_text_region(smoothed_text_mask, img.shape)
    
    crop_orig = img[y1:y2, x1:x2]
    crop_enhanced = enhanced_color[y1:y2, x1:x2]
    crop_saliency = text_saliency[y1:y2, x1:x2]
    crop_binary = binary_smoothed[y1:y2, x1:x2]
    crop_binary_inv = binary_inv[y1:y2, x1:x2]

    # Save cropped & full results
    cv2.imwrite(os.path.join(output_dir, "crop_original.jpg"), crop_orig)
    cv2.imwrite(os.path.join(output_dir, "crop_enhanced_color.jpg"), crop_enhanced)
    cv2.imwrite(os.path.join(output_dir, "crop_text_saliency.jpg"), crop_saliency)
    cv2.imwrite(os.path.join(output_dir, "crop_binary_white_on_black.png"), crop_binary)
    cv2.imwrite(os.path.join(output_dir, "crop_binary_black_on_white.png"), crop_binary_inv)

    # Create a side-by-side comparison
    crop_bin_bgr = cv2.cvtColor(crop_binary_inv, cv2.COLOR_GRAY2BGR)
    crop_sal_bgr = cv2.cvtColor(crop_saliency, cv2.COLOR_GRAY2BGR)
    
    side_by_side = np.hstack([crop_orig, crop_enhanced, crop_sal_bgr, crop_bin_bgr])
    cv2.imwrite(os.path.join(output_dir, "comparison_side_by_side.jpg"), side_by_side)

    # Full frame outputs
    cv2.imwrite(os.path.join(output_dir, "1_smoothed_deblurred.jpg"), unsharp)
    cv2.imwrite(os.path.join(output_dir, "2_yellow_enhanced_color.jpg"), enhanced_color)
    cv2.imwrite(os.path.join(output_dir, "3_yellow_saliency_gray.jpg"), text_saliency)
    cv2.imwrite(os.path.join(output_dir, "4_binary_clean.png"), binary_smoothed)
    cv2.imwrite(os.path.join(output_dir, "5_binary_inverted.png"), binary_inv)

    print(f"Preprocessing completed! All files saved to '{output_dir}/'")
    return crop_orig, crop_enhanced, crop_binary_inv
