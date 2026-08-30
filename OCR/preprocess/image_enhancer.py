import cv2
import numpy as np
import os

def enhance_and_isolate(image_path, target_max_dim=1920):
    """
    Modular image preprocessor:
    - Bilateral edge-preserving filter
    - Gaussian subtraction unsharp deblurring
    - LAB b* yellow color isolation
    - Stencil gap closing and ROI text cropping
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image at: {image_path}")

    h, w = img.shape[:2]
    if max(h, w) > target_max_dim:
        scale = target_max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 1. Bilateral Filter
    smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # 2. Deblurring / Unsharp Masking
    gaussian = cv2.GaussianBlur(smoothed, (0, 0), sigmaX=2.5)
    unsharp = cv2.addWeighted(smoothed, 1.65, gaussian, -0.65, 0)

    # 3. LAB Color Space - Yellow Isolation (b* channel)
    lab = cv2.cvtColor(unsharp, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    b_enhanced = clahe.apply(b_chan)

    # 4. HSV Yellow Mask
    hsv = cv2.cvtColor(unsharp, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([12, 35, 60], dtype=np.uint8), np.array([45, 255, 255], dtype=np.uint8))

    # 5. Combined Saliency
    b_norm = cv2.normalize(b_enhanced.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    text_saliency = cv2.bitwise_and(b_norm, b_norm, mask=yellow_mask)
    text_saliency = clahe.apply(text_saliency)

    # 6. Morphological Stencil Bridging
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel_clean)
    kernel_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smoothed_text_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_bridge)

    # 7. Enhanced Crisp Color
    enhanced_color = unsharp.copy()
    hsv_enh = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2HSV).astype(np.float32)
    mask_bool = smoothed_text_mask > 0
    hsv_enh[mask_bool, 1] = np.clip(hsv_enh[mask_bool, 1] * 1.45, 0, 255)
    hsv_enh[mask_bool, 2] = np.clip(hsv_enh[mask_bool, 2] * 1.35, 0, 255)
    enhanced_color = cv2.cvtColor(hsv_enh.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 8. Clean Binary Mask
    _, binary_otsu = cv2.threshold(text_saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_smoothed = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel_bridge)
    binary_inv = cv2.bitwise_not(binary_smoothed)

    # 9. ROI Crop around text cluster
    contours, _ = cv2.findContours(smoothed_text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = img.shape[:2]
    valid_boxes = [cv2.boundingRect(c) for c in contours if 10 < cv2.contourArea(c) < (h_img * w_img * 0.3) and cv2.boundingRect(c)[3] > 6]
    
    if valid_boxes:
        min_x = max(0, min([b[0] for b in valid_boxes]) - 25)
        min_y = max(0, min([b[1] for b in valid_boxes]) - 20)
        max_x = min(w_img, max([b[0] + b[2] for b in valid_boxes]) + 25)
        max_y = min(h_img, max([b[1] + b[3] for b in valid_boxes]) + 20)
        crop_enhanced = enhanced_color[min_y:max_y, min_x:max_x]
        crop_bin_inv = binary_inv[min_y:max_y, min_x:max_x]
    else:
        crop_enhanced = enhanced_color
        crop_bin_inv = binary_inv

    return {
        "original": img,
        "enhanced_color": enhanced_color,
        "binary_mask": binary_smoothed,
        "binary_inverted": binary_inv,
        "crop_enhanced": crop_enhanced,
        "crop_bin_inv": crop_bin_inv
    }
