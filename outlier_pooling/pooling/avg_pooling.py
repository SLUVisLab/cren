from utils import seed_torch

import torch
seed_torch()


def avg_pooling(conv_feats):
    return torch.mean(conv_feats, -1)
