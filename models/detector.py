import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_fasterrcnn_model(num_classes: int) -> torch.nn.Module:
    """Create a Faster R-CNN model with a ResNet-50 backbone.

    num_classes includes background (class 0).
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

