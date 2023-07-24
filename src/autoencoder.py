from typing import Callable, List, Optional
import math
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from src.model_utils import Optimizer, LR_Scheduler
from src.utils import dict_get

import warnings

warnings.filterwarnings(
    "ignore", message="Setting attributes on ParameterDict is not supported."
)


class PerturbedInitializer:
    def __init__(self, path: str, delta: float = 0.0):
        self.delta = delta
        self.path = path

    def __call__(self, in_features, out_features, tied_weights) -> nn.ParameterDict:
        params = dict()
        k = 1 / math.sqrt(in_features)

        def rescaled_unif(*size):
            # return x ~ Unif(-k, k)
            return 2 * k * torch.rand(size) - k

        pdict = torch.load(self.path, map_location=torch.device("cpu"))
        # import pdb; pdb.set_trace()
        W0 = pdict["W"]
        n, m = W0.shape
        params["W"] = Parameter(W0 + self.delta * torch.randn(n, m) / math.pow(m, 0.25))
        params["b_enc"] = Parameter(rescaled_unif(out_features))
        params["b_dec"] = Parameter(rescaled_unif(in_features))

        if tied_weights is False:
            params["W_dec"] = Parameter(rescaled_unif(in_features, out_features))

        return nn.ParameterDict(params)


class SymmetricInitializer:
    def __init__(self, init_scale: float = 1.0):
        self.init_scale = init_scale

    def __call__(self, in_features, out_features, tied_weights) -> nn.ParameterDict:
        params = dict()

        def rescaled_ones(*size):
            # return x ~ Unif(-k, k)
            return self.init_scale * torch.ones(size)

        params["W"] = Parameter(rescaled_ones(out_features, in_features))
        params["b_enc"] = Parameter(rescaled_ones(out_features))
        params["b_dec"] = Parameter(rescaled_ones(in_features))

        if tied_weights is False:
            params["W_dec"] = Parameter(rescaled_ones(in_features, out_features))

        return nn.ParameterDict(params)


class RandomInitializer:
    def __init__(
        self,
        init_scale: float = 1.0,
        init_pow: int = 1,
        b_enc: bool = True,
        b_dec: bool = True,
    ):
        self.init_scale = init_scale
        self.init_pow = init_pow
        self.b_enc = b_enc
        self.b_dec = b_dec

    def __call__(self, in_features, out_features, tied_weights) -> nn.ParameterDict:
        params = dict()
        k = self.init_scale / math.pow(math.sqrt(in_features), self.init_pow)

        def rescaled_unif(*size):
            # return x ~ Unif(-k, k)
            return 2 * k * torch.rand(size) - k

        params["W"] = Parameter(rescaled_unif(out_features, in_features))

        if self.b_enc:
            params["b_enc"] = Parameter(rescaled_unif(out_features))

        if self.b_dec:
            params["b_dec"] = Parameter(rescaled_unif(in_features))

        if tied_weights is False:
            params["W_dec"] = Parameter(rescaled_unif(in_features, out_features))

        return nn.ParameterDict(params)


class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        width,
        activation: nn.Module,
        img_dims: List[int],
        optimizer: Optimizer,
        initializer: Callable,
        threshold: float = 0.001,
        lr_scheduler: Optional[LR_Scheduler] = None,
        tied_weights: bool = False,
        corruption: Optional[Callable] = None,
        seed: Optional[int] = None,
        weight_norm: bool = False,
    ):
        super().__init__()
        self.width = width
        self.activation = activation
        self.img_dims = img_dims
        self.optimizer = optimizer
        self.threshold = threshold
        self.lr_scheduler = lr_scheduler
        self.tied_weights = tied_weights
        self.corruption = corruption
        self.seed = seed
        self.weight_norm = weight_norm

        img_size = np.prod(img_dims)
        self.params = initializer(img_size, width, tied_weights)

    def forward(self, x):
        if self.weight_norm:
            with torch.no_grad():
                self.params["W"] = Parameter(F.normalize(self.params["W"], dim=1))

        batch_size = x.size(0)
        x = self.encoder(x)

        if self.tied_weights:
            x = F.linear(x, self.params["W"].t(), dict_get(self.params, "b_dec"))
        else:
            x = F.linear(x, self.params["W_dec"], dict_get(self.params, "b_dec"))

        x = x.view([batch_size] + self.img_dims)

        return x

    def encoder(self, x):
        batch_size = x.size(0)
        x = x.view([batch_size, -1])
        x = F.linear(x, self.params["W"], dict_get(self.params, "b_enc"))
        x = self.activation(x)
        return x

    def configure_optimizers(self):
        optimizer = self.optimizer.partial(self.parameters())

        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler.partial(optimizer)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        loss = self._shared_eval(batch, batch_idx)
        self.log("train_mse", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval(batch, batch_idx)
        self.log("val_mse", loss)

        # with torch.no_grad():
        #     encoded = self.encoder(batch)
        #     encoded[torch.abs(encoded) < self.threshold] = 0
        #     sparsity = torch.sum(torch.eq(encoded, torch.zeros_like(encoded))) / torch.numel(encoded)
        #     self.log("activation_sparsity", sparsity)

    def _shared_eval(self, batch, batch_idx):
        x = batch
        x_corr = x if self.corruption is None else self.corruption(x)
        x_rec = self(x_corr)
        loss = F.mse_loss(x, x_rec)
        return loss
