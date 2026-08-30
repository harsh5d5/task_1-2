# Military Computer Vision Assessment (Task 1 & Task 2)

This repository contains implementations for two computer vision tasks:

1. **Task 1: Green Box Counting (`Box_count/`)** — Deep learning-based automated counting of military ammunition containers in dense warehouse stacks using **Faster R-CNN (ResNet-50-FPN V2)** with spatial raster ordering and numbered badges.
2. **Task 2: Military Container Stencil OCR (`OCR/`)** — An image preprocessing, deblurring, yellow-paint isolation, and **DBNet++** text detection pipeline integrated with a domain-specific military nomenclature dictionary.

---

## 📁 Repository Structure

```text
task_1-2/
├── 📁 Box_count/                           # Task 1: Green Box Counting
│   ├── 📁 models/                          # Metrics, evaluation JSON, and training curves
│   ├── 📁 outputs/                         # Prediction visuals & count summaries
│   ├── 📁 src/                             # Faster R-CNN model builder & badge visualizer
│   ├── 📄 predict.py                       # Local test inference script
│   ├── 📄 train_colab.py                   # Google Colab / GPU training script
│   └── 📄 README.md                        # Task 1 Technical Documentation
│
├── 📁 OCR/                                 # Task 2: Container Stencil OCR
│   ├── 📁 preprocess/                      # Bilateral filter & LAB yellow color isolation tools
│   ├── 📁 src/                             # DBNet++ detector & military nomenclature cleaner
│   ├── 📄 run.py                           # Batch OCR pipeline launcher
│   └── 📄 document.md                      # Task 2 Technical Documentation
│
├── 📄 .gitignore                           # Excludes raw dataset images & heavy weight files
└── 📄 README.md                            # Main project overview
```

---

## 🚀 Quick Start

### Task 1: Box Counting Inference
```bash
cd Box_count
python predict.py
```

### Task 2: Stencil OCR Pipeline
```bash
cd OCR
python run.py
```
