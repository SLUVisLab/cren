from utils import seed_torch

import torch
seed_torch()


def dynamic_outlier_pooling(conv_feats, current_epoch, max_epochs, threshold_parameter):
    threshold = torch.mean(conv_feats, dim=-1, keepdim=True) + threshold_parameter*torch.std(conv_feats, dim=-1, keepdim=True)
    weight_multiplier = current_epoch*1.0 / max_epochs

    above_thresh = (conv_feats >= threshold)
    below_thresh = (conv_feats < threshold)
    
    weights = (above_thresh * (1.0 + weight_multiplier)) + (below_thresh * (1.0 - weight_multiplier))

    conv_feats = torch.mean(weights*conv_feats, -1)
    
    return conv_feats
