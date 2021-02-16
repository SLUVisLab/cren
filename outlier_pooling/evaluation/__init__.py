from setups import BaseSetup

from .classification_accuracy import classification_accuracy
from .retrieval import retrieval_at_1, retrieval_at_10, retrieval_at_100

import torch

_factory = {
    'Classification Accuracy': classification_accuracy,
    'Retrieval @1': retrieval_at_1,
    'Retrieval @10': retrieval_at_10,
    'Retrieval @100': retrieval_at_100
}


def names():
    return sorted(_factory.keys())


def run(name, *args, **kwargs):
    if name not in _factory:
        raise KeyError("Unknown evaluation:", name)
    return _factory[name](*args, **kwargs)


def evaluate(setup: BaseSetup, logger, loader='validation', include=[]) -> None:
    loss = 0.0
    outputs = torch.Tensor()
    labels = torch.Tensor()
    paths = []
    setup.set_eval()
    batch_idx = 0

    for batch in setup.loaders[loader]:
        _, batch_labels, batch_paths = batch
        batch_outputs, batch_loss = setup.run(batch)

        outputs = torch.cat((outputs, batch_outputs))
        labels = torch.cat((labels, batch_labels))
        paths = paths + list(batch_paths)

        loss += batch_loss.item()
        batch_idx += 1

    logger.log_metric('{} Loss'.format(loader.capitalize()), loss / batch_idx)
    for evaluation in include:
        metric = run(evaluation, outputs=outputs, labels=labels, paths=paths)
        logger.log_metric('{} {}'.format(loader.capitalize(), evaluation), metric)
