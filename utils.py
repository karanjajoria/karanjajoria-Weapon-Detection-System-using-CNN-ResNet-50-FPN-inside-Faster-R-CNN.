import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import json

def get_model_instance(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)
