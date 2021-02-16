from utils import seed_torch

import torch
seed_torch()


def classification_accuracy(**kwargs) -> float:
    predictions = torch.max(kwargs['outputs'], dim=1).indices
    accuracy = sum(predictions == kwargs['labels']) / len(predictions)
    return accuracy
