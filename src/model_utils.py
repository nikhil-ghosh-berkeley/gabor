import torch.optim as optim
import torch.nn as nn
from functools import partial


class Optimizer:
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.__dict__.update(kwargs)
        self.partial = partial(getattr(optim, name), **kwargs)


class LR_Scheduler:
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.__dict__.update(kwargs)
        if 'decay_time' in kwargs:
            del kwargs['decay_time']
        self.partial = partial(getattr(optim.lr_scheduler, name), **kwargs)

class Activation:
    def __init__(self, name: str) -> None:
        self.name = name
        activation_class = getattr(nn, name)
        self.activation = activation_class()

    def __call__(self, x):
        return self.activation(x)
