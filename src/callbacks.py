import os
from os.path import join as pjoin
from typing import Dict

import numpy as np
import torch
import wandb
from pytorch_lightning import Callback, LightningModule, Trainer
from torchvision.utils import make_grid
import torch.nn.functional as F
from src.metrics import cos_sim, comp_incoherence


class DictionaryIncoherence(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        inco = comp_incoherence(trainer.datamodule.get_dictionary())
        trainer.logger.log_hyperparams({"dict_incoherence": inco})


class VisualizeFilters(Callback):
    def __init__(
        self, patch_width: int, patch_height: int, nrow: int = 20, padding: int = 2
    ):
        super().__init__()
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.nrow = nrow
        self.padding = padding

    def on_validation_epoch_end(
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


class WeightNorm(Callback):
    def __init__(self):
        super().__init__()

    def save_weight_norm(self, trainer: Trainer, pl_module: LightningModule):
        W = pl_module.params["W"].detach()
        trainer.logger.log_metrics({"weight_norm": torch.norm(W)})

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.save_weight_norm(trainer, pl_module)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.save_weight_norm(trainer, pl_module)


class DistanceToReference(Callback):
    def __init__(
        self, save_dir: str, fname: str, param_key: str = "W", reco_thresh: float = 0.8, signed: bool = True,
    ):
        super().__init__()
        self.path = pjoin(save_dir, fname)
        self.param_key = param_key
        self.reco_thresh = reco_thresh
        self.signed = signed
        self.saved = None
    
    def save_distance(self, trainer: Trainer, pl_module: LightningModule):
        # load saved parameters
        if self.saved is None:
            self.saved = torch.load(self.path, map_location=pl_module.device)

        # get first layer weights
        if isinstance(self.saved, dict):
            W0 = self.saved[self.param_key]
        else:
            W0 = self.saved

        W = pl_module.params["W"].detach()
        # W0 is m x n, W is p x n
        all_pairs = cos_sim(W0, W).cpu().numpy()
        # all_pairs is m x p with all_pairs[i,j] = cossim(W0[i,:], W[j,:])
        if self.signed:
            dict_reco = np.max(all_pairs, axis=1)
            max_sim = np.max(all_pairs, axis=0)
        else:
            dict_reco = np.max(np.abs(all_pairs), axis=1)
            max_sim = np.max(np.abs(all_pairs), axis=0)
        trainer.logger.log_metrics(
            {
                "min_max_sim": np.min(max_sim),
                f"reco_frac_{self.reco_thresh}": np.mean(dict_reco >= self.reco_thresh),
            }
        )
        trainer.logger.experiment.log(
            {
                "max_sim": wandb.Histogram(max_sim),
                "dict_reco": wandb.Histogram(dict_reco),
            }
        )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        self.save_distance(trainer, pl_module)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.save_distance(trainer, pl_module)


class SaveWeights(Callback):
    def __init__(self, save_dirs: Dict[str, str], save_last_epoch_only=False) -> None:
        super().__init__()
        self.save_dir = pjoin(
            save_dirs["top_dir"],
            save_dirs["data_dir"],
            save_dirs["arch_dir"],
            save_dirs["exp_dir"],
        )
        self.save_last_epoch_only = save_last_epoch_only

    def save_weights(self, trainer: Trainer, pl_module: LightningModule):
        fname = (
            f"fc_{pl_module.optimizer.name.lower()}"
            f"_lr={pl_module.optimizer.lr}"
            f"_bs={trainer.datamodule.batch_size}"
            f"_sigma={pl_module.corruption.sigma}"
            f"_epoch={pl_module.current_epoch}"
            f"_step={trainer.global_step}"
        )

        is_last_epoch = pl_module.current_epoch == (trainer.max_epochs - 1)
        save = (not self.save_last_epoch_only) or (
            self.save_last_epoch_only and is_last_epoch
        )

        if save:
            pdict = pl_module.params
            os.makedirs(self.save_dir, exist_ok=True)
            torch.save(
                {p: pdict[p].detach() for p in pdict}, pjoin(self.save_dir, fname)
            )

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.save_weights(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        self.save_weights(trainer, pl_module)

        
