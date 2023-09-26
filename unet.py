import torch
from torch import nn
from torchvision import models
torch.cuda.init()
torch.autograd.set_detect_anomaly(True)


def load_unet():
    """
    Load Unet with pretrained weights
    """
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=3, out_channels=1, init_features=32, pretrained=True)
    return model

