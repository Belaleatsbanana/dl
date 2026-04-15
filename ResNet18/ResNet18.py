"""
ResNet-18 for Food-101 Classification
Pre-trained on ImageNet from torchvision
https://arxiv.org/abs/1512.03385
"""
import torch
import torch.nn as nn
import torchvision.models as models

def resnet18(num_classes=5, **kwargs):
    drop_rate = kwargs.pop('drop_path_rate', 0.1)
    if 'drop_rate' in kwargs:
        drop_rate = kwargs.pop('drop_rate', drop_rate)
    
    # Load pre-trained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Get classifier input dimension
    num_features = model.fc.in_features
    
    # Build classifier with dropout regularization
    if drop_rate > 0:
        classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(num_features, num_classes)
        )
    else:
        classifier = nn.Linear(num_features, num_classes)
    
    # Replace ImageNet classifier with Food-101 classifier
    model.fc = classifier
    
    return model