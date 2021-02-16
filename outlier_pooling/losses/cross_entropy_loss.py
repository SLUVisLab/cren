from utils import seed_torch

import torch
seed_torch()

from .base_loss import BaseLoss


class CrossEntropyLoss(BaseLoss):
    def __init__(self, **kwargs):
        assert kwargs['train_type'] == 'classification', "Cross Entropy Loss can only be used for classification."
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        return self.criterion(outputs, labels)
