import torch


class GaussianNoise:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        return x + self.sigma * torch.randn_like(x)
