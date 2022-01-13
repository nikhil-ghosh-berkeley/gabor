from typing import Tuple

import wandb
from pytorch_lightning import Callback, LightningModule, Trainer
from torchvision.utils import make_grid


class VisualizeFilters(Callback):
    def __init__(self, patch_size: Tuple[int, int], nrow: int = 20, padding: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.nrow = nrow
        self.padding = padding

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        tensor = pl_module.encoder.weight.detach().cpu().clone()
        n, wh = tensor.shape
        w, h = self.patch_size
        assert wh == w * h

        kernels = tensor.view(n, 1, w, h)
        grid = make_grid(kernels, nrow=self.nrow, padding=self.padding, normalize=True)
        img = grid.numpy().transpose((1, 2, 0))
        trainer.logger.log_image(
            f"epoch {pl_module.current_epoch}", images=[wandb.Image(img)]
        )
