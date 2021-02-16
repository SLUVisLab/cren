from .model import Model
from utils import seed_torch

from torchvision import models
import torch
seed_torch()


class ResNet50(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        resnet = models.resnet50(pretrained=kwargs['pretrained'])

        self.trunk = torch.nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.trunk = torch.nn.DataParallel(self.trunk).to(kwargs['device'])
        self.embedder = torch.nn.Linear(2048, kwargs['output_size']).to(kwargs['device'])
