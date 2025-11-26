import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def load_efficientnet_b0(num_classes=2):
    """
    Loads pretrained EfficientNet-B0 and replaces classifier head.
    """
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    # Replace classification head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
