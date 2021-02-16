from .model import Model
from .resnet50 import ResNet50

_factory = {
    'resnet50': ResNet50,
}


def names():
    return sorted(_factory.keys())


def create(name, *args, **kwargs):
    if name not in _factory:
        raise KeyError("Unknown model:", name)
    return _factory[name](*args, **kwargs)
