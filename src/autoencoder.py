from typing import Callable, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F


class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        width,
        activation: nn.Module,
        img_dims: List[int],
        optimizer_partial: Callable,
        lr_scheduler_partial: Callable,
        corruption: Optional[Callable] = None,
    ):
        super().__init__()
        self.width = width
        self.activation = activation
        self.img_dims = img_dims
        self.optimizer_partial = optimizer_partial
        self.lr_scheduler_partial = lr_scheduler_partial
        self.corruption = corruption

        img_size = np.prod(img_dims)
        self.encoder = nn.Linear(img_size, width)
        self.decoder = nn.Linear(width, img_size)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view([batch_size, -1])
        x = self.encoder(x)
        x = self.activation(x)
        x = self.decoder(x)
        x = x.view([batch_size] + self.img_dims)
        return x

    def configure_optimizers(self):
        optimizer = self.optimizer_partial(self.parameters())
        lr_scheduler = self.lr_scheduler_partial(optimizer)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        loss = self._shared_eval(batch, batch_idx)
        self.log("train_mse", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval(batch, batch_idx)
        self.log("val_mse", loss)

    def _shared_eval(self, batch, batch_idx):
        x = batch
        x_corr = x if self.corruption is None else self.corruption(x)
        x_rec = self(x_corr)
        loss = F.mse_loss(x, x_rec)
        return loss
