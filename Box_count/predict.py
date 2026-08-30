"""
Green Box Counting & Visual Prediction
Usage:
    python predict.py
"""
import os
import sys
import time
import json
import torch
import cv2
import numpy as np
from PIL import Image

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.model import build_faster_rcnn
from src.visualizer import draw_numbered_boxes

def run_predictions(
    model_path=os.path.join(current_dir, "models", "faster_rcnn_best.pth"),
    image_dir=os.path.join(current_dir, "data", "raw_images"),
    output_dir=os.path.join(current_dir, "outputs", "predictions"),
    score_thresh=0.55,
    max_images=10
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")

    # 1. Load Model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = build_faster_rcnn(num_classes=2, nms_thresh=0.5, pretrained_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    best_epoch = checkpoint.get("epoch", "N/A")
    print(f"[OK] Faster R-CNN Model loaded (Trained Epoch: {best_epoch})")

    # 2. Collect Images
    valid_exts = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)]
    if max_images:
        image_files = image_files[:max_images]

    print(f"\nProcessing {len(image_files)} test images...")
    print("=" * 65)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    results_summary = []

    for filename in image_files:
        img_path = os.path.join(image_dir, filename)
        orig_pil = Image.open(img_path).convert("RGB")
        
        img_np = np.array(orig_pil, dtype=np.float32) / 255.0
        norm_img = (img_np - mean) / std
        tensor = torch.from_numpy(norm_img).permute(2, 0, 1).float().unsqueeze(0).to(device)

        t0 = time.time()
        with torch.no_grad():
            preds = model(tensor)[0]
        elapsed_ms = (time.time() - t0) * 1000

        scores = preds["scores"].cpu().numpy()
        boxes = preds["boxes"].cpu().numpy()

        # Render numbered bounding boxes and summary banner
        bgr_img = cv2.imread(img_path)
        annotated_img, count, detections = draw_numbered_boxes(bgr_img, boxes, scores, score_thresh=score_thresh)

        out_path = os.path.join(output_dir, f"pred_{filename}")
        cv2.imwrite(out_path, annotated_img)

        print(f"Image: {filename:30s} | Boxes Counted: {count:3d} (#1..#{count}) | Latency: {elapsed_ms:.1f}ms")

        results_summary.append({
            "image": filename,
            "count": count,
            "latency_ms": round(elapsed_ms, 1),
            "output_file": f"pred_{filename}"
        })

    # Save summary json
    summary_path = os.path.join(output_dir, "..", "box_count_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print("=" * 65)
    print(f"Predictions complete! Outputs saved to: {output_dir}")

if __name__ == "__main__":
    run_predictions()
