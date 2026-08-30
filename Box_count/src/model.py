import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_faster_rcnn(num_classes=2, nms_thresh=0.5, pretrained_backbone=True):
    """
    Builds Faster R-CNN with ResNet-50-FPN V2 backbone.
    Args:
        num_classes (int): Number of classes including background (default: 2 for green_box + background).
        nms_thresh (float): Non-Maximum Suppression threshold (default: 0.5).
        pretrained_backbone (bool): Whether to load COCO pretrained weights.
    """
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained_backbone else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.nms_thresh = nms_thresh
    return model
