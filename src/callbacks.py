import os
from os.path import join as pjoin
from typing import Dict

import numpy as np
import torch
import wandb
from pytorch_lightning import Callback, LightningModule, Trainer
from torchvision.utils import make_grid
import torch.nn.functional as F
from src.metrics import cos_sim


class VisualizeFilters(Callback):
    def __init__(
        self, patch_width: int, patch_height: int, nrow: int = 20, padding: int = 2
    ):
        super().__init__()
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.nrow = nrow
        self.padding = padding

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        tensor = pl_module.params["W"].detach().cpu().clone()
        n, wh = tensor.shape
        assert wh == self.patch_width * self.patch_height

        kernels = tensor.view(n, 1, self.patch_width, self.patch_height)
        grid = make_grid(kernels, nrow=self.nrow, padding=self.padding, normalize=True)
        img = grid.numpy().transpose((1, 2, 0))
        trainer.logger.log_image(
            f"epoch {pl_module.current_epoch}", images=[wandb.Image(img)]
        )


class DistanceToReference(Callback):
    def __init__(
        self, save_dir: str, fname: str, ext: str = ".pt", param_key: str = "W"
    ):
        self.path = pjoin(save_dir, fname) + ext
        self.param_key = param_key
        self.saved = None

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.saved is None:
            self.saved = torch.load(self.path, map_location=pl_module.device)

        if isinstance(self.saved, dict):
            W0 = self.saved[self.param_key]
        else:
            W0 = self.saved

        W = pl_module.params["W"].detach()
        all_pairs = cos_sim(W0, W).cpu().numpy()
        max_sim = np.max(np.abs(all_pairs), axis=0)
        trainer.logger.log_metrics({"min_max_sim": np.min(max_sim)})
        trainer.logger.experiment.log({"max_sim": wandb.Histogram(max_sim)})


class SaveWeights(Callback):
    def __init__(self, save_dirs: Dict[str, str], batch_size: int) -> None:
        super().__init__()
        self.save_dir = pjoin(
            save_dirs["top_dir"],
            save_dirs["data_dir"],
            save_dirs["arch_dir"],
            save_dirs["exp_dir"],
        )
        self.batch_size = batch_size
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        fname = (
            f"fc_{pl_module.optimizer.name.lower()}"
            f"_lr={pl_module.optimizer.lr}"
            f"_bs={self.batch_size}"
            f"_sigma={pl_module.corruption.sigma}"
            f"_epoch={pl_module.current_epoch}"
            f"_seed={pl_module.seed}.pt"
        )

        if pl_module.current_epoch == (trainer.max_epochs - 1):
            pdict = pl_module.params
            torch.save(
                {p: pdict[p].detach() for p in pdict}, pjoin(self.save_dir, fname)
            )
