from pooling import avg_pooling, max_pooling, dynamic_outlier_pooling, outlier_pooling, dynamic_threshold
from utils import seed_torch

import torch
import torch.nn.functional as f
seed_torch()


class Model(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.dropout = kwargs['dropout']
        self.max_epochs = kwargs['max_epochs']
        self.train_type = kwargs['train_type']
        self.switch_epoch = kwargs['switch_epoch']
        self.threshold_parameter= kwargs['threshold_parameter']

        self.current_epoch = 0
        self.pooling_method = kwargs['pooling_type']

    def forward(self, data):
        batch_size, c, h, w = data.size()
        features = self.trunk(data)
        
        features = self.pooling(features)
        features[torch.isnan(features)] = 0.

        if self.training:
            features = f.dropout(features, self.dropout)

        features = self.embedder(features.view(batch_size, -1))
        return features
    
    def pooling(self, conv_feats):
        batch_size, num_features, h, w = conv_feats.size()
        conv_feats = conv_feats.reshape(batch_size, num_features, h*w)

        if self.pooling_method.lower() == 'avg':
            conv_feats = avg_pooling(conv_feats)
        elif self.pooling_method.lower() == 'max':
            conv_feats = max_pooling(conv_feats)
        elif self.pooling_method.lower() == 'dynamic_outlier':
            conv_feats = dynamic_outlier_pooling(conv_feats, self.current_epoch, self.max_epochs, self.threshold_parameter)
        elif self.pooling_method.lower() == 'switch_outlier':
            conv_feats = avg_pooling(conv_feats) if self.current_epoch < self.switch_epoch else dynamic_threshold(conv_feats,self.current_epoch, self.max_epochs,self.threshold_parameter)
        elif self.pooling_method.lower() == 'outlier':
            conv_feats = dynamic_threshold(conv_feats,self.current_epoch, self.max_epochs,self.threshold_parameter)
        else:
            conv_feats = dynamic_outlier_pooling(conv_feats, self.current_epoch, self.max_epochs, self.threshold_parameter)
            
        return conv_feats
