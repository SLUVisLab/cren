from utils import seed_torch

import torch
seed_torch()


def max_pooling(conv_feats):
    return torch.max(conv_feats, -1)[0]
