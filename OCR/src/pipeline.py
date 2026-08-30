import os
import glob
import json
import cv2
from .preprocessor import preprocess_image
from .detector import ContainerTextDetector
from .nomenclature_cleaner import clean_and_correct_text

def run_pipeline(input_dir="datasets/raw_images", output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Gather all images
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    image_paths = sorted(list(set(image_paths)))

    print(f"Found {len(image_paths)} container images in '{input_dir}'.")
    detector = ContainerTextDetector()

    master_summary = []

    for idx, img_path in enumerate(image_paths, start=1):
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        
        # Preprocess
        img_data = preprocess_image(img_path)
        
        # Detect & Correct
        lines, overlay_img = detector.detect_and_align(img_data)
        
        # Build descriptive folder name based on prominent extracted text
        clean_text_list = [l["clean_text"] for l in lines]
        tag = clean_text_list[0].replace(" ", "_").replace("/", "-") if clean_text_list else "Unknown"
        tag = "".join([c for c in tag if c.isalnum() or c in ("_", "-")])[:20]
        
        folder_name = f"image_{idx:02d}_{tag}"
        target_folder = os.path.join(output_dir, folder_name)
        os.makedirs(target_folder, exist_ok=True)

        # Save organized assets
        cv2.imwrite(os.path.join(target_folder, "1_original.jpg"), img_data["original"])
        cv2.imwrite(os.path.join(target_folder, "2_enhanced.jpg"), img_data["enhanced_color"])
        cv2.imwrite(os.path.join(target_folder, "3_binary_mask.png"), img_data["binary_mask"])
        cv2.imwrite(os.path.join(target_folder, "4_detection_overlay.jpg"), overlay_img)

        # Write clean report
        report_path = os.path.join(target_folder, "ocr_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"CONTAINER STENCIL OCR REPORT: {folder_name}\n")
            f.write(f"Source: {filename}\n")
            f.write("=" * 60 + "\n\n")
            f.write("[EXTRACTED NOMENCLATURE LINES (TOP-TO-BOTTOM)]:\n")
            f.write("-" * 45 + "\n")
            for line_idx, l in enumerate(lines, start=1):
                f.write(f"Line {line_idx}: {l['clean_text']}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("[CONFIDENCE & RAW DETECTIONS]:\n")
            f.write("-" * 45 + "\n")
            for line_idx, l in enumerate(lines, start=1):
                f.write(f"Line {line_idx} [Conf: {l['confidence']:.2f}] Raw: {l['raw_text']}\n")

        print(f"[{idx:02d}/{len(image_paths)}] Processed: {folder_name} -> {len(clean_text_list)} lines extracted.")

        master_summary.append({
            "index": idx,
            "source_file": filename,
            "folder": folder_name,
            "extracted_lines": clean_text_list,
            "details": lines
        })

    # Save master JSON and Markdown summaries
    json_path = os.path.join(output_dir, "master_ocr_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    md_path = os.path.join(output_dir, "master_ocr_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Master Military Container Stencil OCR Report\n\n")
        f.write("| # | Folder | Source File | Extracted Nomenclature Lines |\n")
        f.write("| :-: | :--- | :--- | :--- |\n")
        for item in master_summary:
            lines_str = "<br>".join([f"• `{l}`" for l in item["extracted_lines"]]) if item["extracted_lines"] else "*(None)*"
            f.write(f"| {item['index']} | **[{item['folder']}](file:///{os.path.abspath(os.path.join(output_dir, item['folder'])).replace('\\', '/')})** | `{item['source_file']}` | {lines_str} |\n")

    print("\n" + "=" * 60)
    print(f"Pipeline complete! Output organized in: {output_dir}/")
    print(f"Master Summary Markdown: {md_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()
