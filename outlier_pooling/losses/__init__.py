from .triplet_margin_loss import TripletMarginLoss
from .ephn_loss import EPHNLoss
from .epshn_loss import EPSHNLoss
from .batchall_loss import BatchAllLoss
from .cross_entropy_loss import CrossEntropyLoss

_factory = {
    'triplet_margin_loss': TripletMarginLoss,
    'ephn_loss': EPHNLoss,
    'epshn_loss': EPSHNLoss,
    'batchall_loss': BatchAllLoss,
    'cross_entropy_loss': CrossEntropyLoss,
}


def names():
    return sorted(_factory.keys())


def create(name, *args, **kwargs):
    if name not in _factory:
        raise KeyError("Unknown loss:", name)
    return _factory[name](*args, **kwargs)