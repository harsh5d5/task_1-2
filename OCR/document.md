# Military Container Stencil OCR System

A computer vision and deep learning pipeline to enhance, detect, and recognize yellow stencil text on military ammunition containers and fuze boxes.

---

## 1. Pipeline Overview

```text
Raw Image ➔ Preprocessing (Deblur + Yellow Isolation) ➔ DBNet++ Text Detection ➔ Spatial Line Sorting ➔ Nomenclature Correction ➔ Final Output
```

---

## 2. Key Processing Stages

### A. Image Preprocessing & Enhancement (`preprocess/yellow_enhancer.py`, `src/preprocessor.py`)
1. **Bilateral Denoising**: Smooths rough metal texture while keeping text edges sharp.
2. **Unsharp Deblurring**: Restores crisp character borders ($1.65 \times \text{Original} - 0.65 \times \text{Blurred}$).
3. **Yellow Color Isolation**:
   - **CIELAB $b^*$ Channel**: Isolates yellow paint against dark green metal.
   - **HSV Filter ($12^\circ - 45^\circ$) + CLAHE**: Equalizes uneven warehouse lighting.
4. **Morphological Closing**: Gently bridges physical stencil cuts in characters.
5. **ROI Crop**: Automatically pinpoints the text box region on the container.

---

### B. DBNet++ Text Detection (`src/detector.py`)
- **Model**: Uses **DBNet++** (*Differentiable Binarization*), optimized for arbitrary-shaped, elongated, multi-line scene text.
- **Why DBNet++ over YOLO**: YOLO is generic object detection; DBNet++ detects pixel-level text boundaries and handles broken stencil typography without fragmenting words.
- **Dual-Pass Inference**: Runs on both the enhanced color image and the inverted binary text mask.

---

### C. Spatial Line Ordering
Reconstructs natural top-to-bottom reading order:
- **Vertical Sorting**: Orders lines by $y_{\text{center}} = (y_{\text{min}} + y_{\text{max}}) / 2$.
- **Horizontal Grouping**: Merges words on the same line ($|y_1 - y_2| < 0.65 \times \text{Height}$) from left to right.

---

### D. Military Nomenclature Correction (`src/nomenclature_cleaner.py`)
Automatically resolves common stencil OCR misreads against the official ordnance dictionary:

| Raw Misread | Corrected Standard Text | Category |
| :--- | :--- | :--- |
| `IO FUZESII7` / `1OrUzEo117` | **`10 FUZES 117`** | Fuze Nomenclature |
| `FUZES1WYK·20` | **`FUZES 117 MK-20`** | Fuze Variant |
| `IXIO COXTS` | **`IN 10 CONTS`** | Packaging |
| `FA,13.` / `FAH.` | **`FAD`** | Depot / Authority |
| `2B/L XI)` | **`2B/L ND`** | Line / Batch Code |
| `AVNASSKG-18.70` | **`AV MASS KG-18.70`** | Mass Specification |

*Also filters out non-ASCII noise glyphs, stray punctuation, and unrecognized symbols.*

---

## 3. Project Structure

```text
TEST/
├── datasets/raw_images/     # Input container photos
├── preprocess/              # Standalone enhancement scripts (yellow_enhancer.py)
├── src/                     # Core pipeline (preprocessor.py, detector.py, nomenclature_cleaner.py)
├── results/                 # Per-container results with images & text reports
└── run.py                   # 1-command batch runner
```

---

## 4. How to Run

### Run Complete Batch Pipeline:
```bash
python run.py
```

### Run Single-Image Preprocessing:
```python
from preprocess.yellow_enhancer import preprocess_yellow_text

preprocess_yellow_text("datasets/raw_images/img01_fuze_box.jpg", output_dir="my_output")
```

---

## 5. Output Files (Generated in `results/`)

Every container folder contains:
- `1_original.jpg` — Input image.
- `2_enhanced.jpg` — Denoised, deblurred, yellow-boosted image.
- `3_binary_mask.png` — Clean binary stencil mask.
- `4_detection_overlay.jpg` — Bounding boxes and labels on image.
- `ocr_report.txt` — Final line-by-line standardized text.
- `master_ocr_summary.md` / `.json` — Consolidated master report across all images.
