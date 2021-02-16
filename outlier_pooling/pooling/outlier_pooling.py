from utils import seed_torch

import torch
seed_torch()


def outlier_pooling(conv_feats, threshold_parameter):
    threshold = torch.mean(conv_feats, dim=-1, keepdim=True) + threshold_parameter* torch.std(conv_feats, dim=-1, keepdim=True)

    above_thresh = (conv_feats >= threshold)
    conv_feats = torch.sum(above_thresh * conv_feats, -1) / (torch.sum(above_thresh * 1.0, -1) + 0.01)
    
    return conv_feats
