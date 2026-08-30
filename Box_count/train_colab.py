# Green Box Detection with Faster R-CNN (ResNet-50-FPN V2)
# Dataset: Roboflow COCO | Framework: PyTorch + Albumentations

import os, json, time, random, zipfile, shutil, subprocess, sys
from collections import defaultdict

for pkg in ("pycocotools", "albumentations"):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# -------------------- Setup --------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch: {torch.__version__} | Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# -------------------- Dataset --------------------

from google.colab import files

print("Upload the Roboflow COCO ZIP dataset.")
uploaded = files.upload()
zip_name = next(iter(uploaded))

DATASET_DIR = "/content/dataset"
if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)

with zipfile.ZipFile(zip_name) as z:
    z.extractall(DATASET_DIR)

# Find the directory containing train/valid folders.
for root, dirs, _ in os.walk(DATASET_DIR):
    if "train" in dirs and "valid" in dirs:
        DATASET_DIR = root
        break

print(f"Dataset: {DATASET_DIR}")


def validate_coco(dataset_dir, split):
    """Check categories and invalid bounding boxes before training."""
    path = os.path.join(dataset_dir, split, "_annotations.coco.json")
    if not os.path.exists(path):
        return

    with open(path) as f:
        data = json.load(f)

    images = {x["id"]: x for x in data["images"]}
    print(f"{split}: {len(images)} images, {len(data['annotations'])} annotations")

    for ann in data["annotations"]:
        if ann["image_id"] not in images:
            raise ValueError(f"Missing image for annotation {ann['id']}")
        x, y, w, h = ann["bbox"]
        if w < 1 or h < 1:
            raise ValueError(f"Invalid bbox: {ann['bbox']}")

    print(f"{split}: annotations OK")


for split in ("train", "valid", "test"):
    validate_coco(DATASET_DIR, split)


class GreenBoxDataset(Dataset):
    """COCO dataset loader for green-box detection."""

    def __init__(self, root, split="train", transforms=None):
        self.root = os.path.join(root, split)
        self.transforms = transforms

        with open(os.path.join(self.root, "_annotations.coco.json")) as f:
            data = json.load(f)

        self.images = {x["id"]: x for x in data["images"]}
        self.anns = defaultdict(list)

        for ann in data["annotations"]:
            self.anns[ann["image_id"]].append(ann)

        # Keep negative images; they teach the detector what background looks like.
        self.ids = list(self.images)

        positives = sum(bool(self.anns[i]) for i in self.ids)
        print(f"{split}: {len(self.ids)} images | "
              f"{positives} positive | {len(self.ids) - positives} negative")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        info = self.images[image_id]
        path = os.path.join(self.root, info["file_name"])

        try:
            image = np.array(Image.open(path).convert("RGB"))
        except Exception:
            # Return an empty sample if an image cannot be opened.
            image = np.zeros((64, 64, 3), dtype=np.uint8)

        h, w = image.shape[:2]
        boxes, labels = [], []

        for ann in self.anns[image_id]:
            x, y, bw, bh = ann["bbox"]
            if bw < 2 or bh < 2:
                continue

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)

            if x2 - x1 >= 2 and y2 - y1 >= 2:
                boxes.append([x1, y1, x2, y2])
                labels.append(1)

        if self.transforms:
            t = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = t["image"]
            boxes, labels = list(t["bboxes"]), list(t["labels"])
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255

        # Faster R-CNN accepts empty tensors for images without objects.
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64)
        area = ((boxes[:, 2] - boxes[:, 0]) *
                (boxes[:, 3] - boxes[:, 1]))

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
        }
        return image, target


def train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.4),
        A.HueSaturationValue(10, 30, 20, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(max_holes=3, max_height=30, max_width=30,
                        fill_value=0, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format="pascal_voc", label_fields=["labels"],
        min_area=100, min_visibility=0.3
    ))


def val_transforms():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format="pascal_voc", label_fields=["labels"],
        min_area=1, min_visibility=0.1
    ))


def collate_fn(batch):
    # Detection images can contain different numbers of boxes.
    return tuple(zip(*batch))


# -------------------- Model --------------------

def build_model(num_classes=2, nms=0.5):
    """Create COCO-pretrained Faster R-CNN and replace its classifier."""
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, num_classes)
    model.roi_heads.nms_thresh = nms
    return model


NUM_EPOCHS = 15
BATCH_SIZE = 4
LR = 0.005
NMS = 0.5
CONF = 0.5
SAVE_DIR = "/content/models"
os.makedirs(SAVE_DIR, exist_ok=True)

model = build_model(nms=NMS).to(DEVICE)

train_set = GreenBoxDataset(DATASET_DIR, "train", train_transforms())
val_set = GreenBoxDataset(DATASET_DIR, "valid", val_transforms())

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True, collate_fn=collate_fn
)

optimizer = torch.optim.SGD(
    model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=5, gamma=0.1
)

# -------------------- Evaluation --------------------

def iou(box1, box2):
    """Calculate IoU for [x1, y1, x2, y2] boxes."""
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union else 0


@torch.no_grad()
def evaluate(model, loader, conf=0.3, iou_thr=0.5):
    """Calculate precision, recall, F1, COCO mAP and count metrics."""
    model.eval()
    tp = fp = fn = 0
    count_errors, exact = [], 0
    gt_data = {
        "images": [], "annotations": [],
        "categories": [{"id": 1, "name": "green_box"}]
    }
    dt_data = []
    ann_id = image_id = 1

    for images, targets in tqdm(loader, desc="Evaluating", leave=False):
        predictions = model([x.to(DEVICE) for x in images])

        for pred, target in zip(predictions, targets):
            keep = pred["scores"].cpu().numpy() >= conf
            pboxes = pred["boxes"].cpu().numpy()[keep]
            pscores = pred["scores"].cpu().numpy()[keep]
            gboxes = target["boxes"].numpy()

            # Use the real image dimensions in COCO evaluation.
            ih, iw = images[0].shape[-2:]
            gt_data["images"].append({
                "id": image_id, "width": int(iw), "height": int(ih)
            })

            for b in gboxes:
                x1, y1, x2, y2 = b
                gt_data["annotations"].append({
                    "id": ann_id, "image_id": image_id, "category_id": 1,
                    "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    "area": float((x2-x1)*(y2-y1)), "iscrowd": 0
                })
                ann_id += 1

            for b, score in zip(pboxes, pscores):
                x1, y1, x2, y2 = b
                dt_data.append({
                    "image_id": image_id, "category_id": 1,
                    "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    "score": float(score)
                })

            matched = set()
            for pi in np.argsort(-pscores):
                best, best_i = 0, -1
                for gi, gb in enumerate(gboxes):
                    if gi in matched:
                        continue
                    score = iou(pboxes[pi], gb)
                    if score > best:
                        best, best_i = score, gi

                if best >= iou_thr:
                    tp += 1
                    matched.add(best_i)
                else:
                    fp += 1

            fn += len(gboxes) - len(matched)
            count_errors.append(abs(len(pboxes) - len(gboxes)))
            exact += len(pboxes) == len(gboxes)
            image_id += 1

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    mAP50 = mAP5095 = 0.0
    if gt_data["annotations"] and dt_data:
        gt_path, dt_path = "/content/gt.json", "/content/dt.json"
        with open(gt_path, "w") as f:
            json.dump(gt_data, f)
        with open(dt_path, "w") as f:
            json.dump(dt_data, f)

        try:
            coco_gt = COCO(gt_path)
            coco_dt = coco_gt.loadRes(dt_path)
            ev = COCOeval(coco_gt, coco_dt, "bbox")
            ev.evaluate()
            ev.accumulate()
            mAP5095, mAP50 = ev.stats[0], ev.stats[1]
        finally:
            for p in (gt_path, dt_path):
                if os.path.exists(p):
                    os.remove(p)

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "mAP@0.5": mAP50, "mAP@0.5:0.95": mAP5095,
        "count_mae": np.mean(count_errors) if count_errors else 0,
        "count_exact_acc": exact / image_id if image_id > 1 else 0,
        "tp": tp, "fp": fp, "fn": fn,
    }


# -------------------- Training --------------------

history = {k: [] for k in (
    "loss", "precision", "recall", "f1", "map50", "count_mae", "count_acc"
)}
best_f1, best_epoch = -1, 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for images, targets in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"
    ):
        images = [x.to(DEVICE) for x in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        losses = sum(model(images, targets).values())
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += losses.item()

    scheduler.step()
    metrics = evaluate(model, val_loader)
    avg_loss = total_loss / max(1, len(train_loader))

    history["loss"].append(avg_loss)
    history["precision"].append(metrics["precision"])
    history["recall"].append(metrics["recall"])
    history["f1"].append(metrics["f1"])
    history["map50"].append(metrics["mAP@0.5"])
    history["count_mae"].append(metrics["count_mae"])
    history["count_acc"].append(metrics["count_exact_acc"])

    print(
        f"Epoch {epoch+1}: loss={avg_loss:.4f}, "
        f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
        f"F1={metrics['f1']:.4f}, mAP50={metrics['mAP@0.5']:.4f}"
    )

    # Save the best model based on validation F1.
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_f1": max(best_f1, metrics["f1"]),
        "val_metrics": metrics,
        "num_classes": 2,
        "architecture": "Faster R-CNN ResNet-50-FPN V2",
        "nms_thresh": NMS,
        "seed": SEED,
    }

    torch.save(checkpoint, os.path.join(SAVE_DIR, "faster_rcnn_latest.pth"))

    if metrics["f1"] > best_f1:
        best_f1, best_epoch = metrics["f1"], epoch + 1
        checkpoint["best_f1"] = best_f1
        torch.save(checkpoint, os.path.join(SAVE_DIR, "faster_rcnn_best.pth"))

print(f"Training complete. Best F1: {best_f1:.4f} at epoch {best_epoch}")

# -------------------- Curves --------------------

epochs = range(1, NUM_EPOCHS + 1)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

ax[0].plot(epochs, history["loss"], marker="o")
ax[0].set_title("Training Loss")
ax[0].set_xlabel("Epoch")

ax[1].plot(epochs, history["precision"], label="Precision")
ax[1].plot(epochs, history["recall"], label="Recall")
ax[1].plot(epochs, history["f1"], label="F1")
ax[1].set_ylim(0, 1.05)
ax[1].set_title("Validation Metrics")
ax[1].legend()

ax[2].plot(epochs, history["map50"], label="mAP@0.5")
ax[2].plot(epochs, history["count_acc"], label="Count Accuracy")
ax[2].set_ylim(0, 1.05)
ax[2].set_title("Detection / Counting")
ax[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"), dpi=150)
plt.show()

# -------------------- Final Evaluation --------------------

best_path = os.path.join(SAVE_DIR, "faster_rcnn_best.pth")
model.load_state_dict(torch.load(best_path, map_location=DEVICE)["model_state_dict"])

eval_split = "test" if os.path.exists(
    os.path.join(DATASET_DIR, "test", "_annotations.coco.json")
) else "valid"

eval_set = GreenBoxDataset(DATASET_DIR, eval_split, val_transforms())
eval_loader = DataLoader(
    eval_set, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True, collate_fn=collate_fn
)

final_metrics = evaluate(model, eval_loader, conf=CONF)

print(f"\nFinal evaluation: {eval_split.upper()}")
for key, value in final_metrics.items():
    print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

with open(os.path.join(SAVE_DIR, "evaluation_results.json"), "w") as f:
    json.dump(final_metrics, f, indent=2)

# -------------------- Sample Predictions --------------------

model.eval()
sample = GreenBoxDataset(DATASET_DIR, eval_split, val_transforms())
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

for i, ax in enumerate(axes.flat):
    if i >= len(sample):
        ax.axis("off")
        continue

    image, target = sample[i]
    start = time.time()

    with torch.no_grad():
        pred = model([image.to(DEVICE)])[0]

    elapsed = (time.time() - start) * 1000
    display = (image.cpu() * std + mean).permute(1, 2, 0).numpy()
    ax.imshow(np.clip(display, 0, 1))

    keep = pred["scores"].cpu() >= CONF
    boxes = pred["boxes"].cpu().numpy()[keep]
    scores = pred["scores"].cpu().numpy()[keep]

    for b, score in zip(boxes, scores):
        x1, y1, x2, y2 = b
        rect = plt.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            fill=False, linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x1, max(0, y1-5), f"{score:.2f}", fontsize=8)

    # Ground truth is shown as dashed boxes for comparison.
    for x1, y1, x2, y2 in target["boxes"].numpy():
        ax.add_patch(plt.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            fill=False, linestyle="--", linewidth=1.5
        ))

    ax.set_title(
        f"GT: {len(target['boxes'])} | Pred: {len(boxes)} | {elapsed:.0f} ms"
    )
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "sample_predictions.png"), dpi=150)
plt.show()

# -------------------- Package Model --------------------

zip_path = "/content/green_box_faster_rcnn_model.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for name in os.listdir(SAVE_DIR):
        z.write(os.path.join(SAVE_DIR, name), name)

print(f"Model package: {zip_path}")
files.download(zip_path)
