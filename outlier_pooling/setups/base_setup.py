from utils import seed_torch

import torch
seed_torch()

class BaseSetup:
    def __init__(self, setup: dict):
        super().__init__()
        self.model = setup['model']
        self.optimizer = setup['optimizer']
        self.criterion = setup['criterion']
        self.loaders = setup['loaders']
        self.device = setup['device']
        self._train = True

    def run(self, batch, current_epoch=None, pooling_method=None) -> (torch.Tensor, torch.Tensor):
        self.model.current_epoch = current_epoch if current_epoch is not None else self.model.current_epoch
        self.model.pooling_method = pooling_method if pooling_method is not None else self.model.pooling_method

        if self._train:
            self.optimizer.zero_grad()
        images, labels, _ = batch

        outputs = self.model(images.to(self.device))

        loss = self.criterion(outputs, labels.to(self.device))
        if self._train:
            loss.backward()
            self.optimizer.step()

        return outputs.detach().cpu(), loss.detach().cpu()

    def set_train(self):
        self.model.train()
        self.model.training = True
        self._train = True

    def set_eval(self):
        self.model.eval()
        self.model.training = False
        self._train = False
