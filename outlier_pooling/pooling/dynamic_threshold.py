from utils import seed_torch

import torch

seed_torch()


def dynamic_threshold(conv_feats, current_epoch, max_epochs, threshold_parameter):
    threshold = torch.mean(conv_feats, dim=-1, keepdim=True) + (current_epoch*threshold_parameter/max_epochs)*torch.std(conv_feats, dim=-1, keepdim=True)

    above_thresh = (conv_feats >= threshold)
    conv_feats = torch.sum(above_thresh * conv_feats, -1) / (torch.sum(above_thresh * 1.0, -1) + 0.01)

    return conv_feats