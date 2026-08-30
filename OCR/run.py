"""
Military & Container Stencil Text Detection and OCR Pipeline
Usage:
    python run.py
"""
import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_pipeline

if __name__ == "__main__":
    print("=" * 65)
    print("Starting Military Container Stencil OCR Pipeline...")
    print("=" * 65)
    run_pipeline(input_dir="datasets/raw_images", output_dir="results")
