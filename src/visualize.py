import torchvision
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


def visualize_fc_weights(tensor, patch_size: Tuple[int, int], nrow: int = 20):
    n, wh = tensor.shape
    w, h = patch_size
    assert wh == w * h

    kernels = tensor.view(n, 1, w, h)
    grid = torchvision.utils.make_grid(kernels, nrow=nrow, normalize=True)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.show()
