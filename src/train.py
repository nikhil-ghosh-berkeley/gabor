import logging
from typing import List
import hydra

import wandb
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, Callback
from pytorch_lightning.loggers import WandbLogger

from src.utils import log_hyperparams
from src.callbacks import VisualizeFilters, SaveWeights

log = logging.getLogger(__name__)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def train(config: DictConfig):
    log.info(f"Instantiating logger <{config.logger._target_}>")
    logger: WandbLogger = hydra.utils.instantiate(config.logger)

    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    log.info(f"Instantiating trainer <{config.trainer._target_}>")

    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=False
    )

    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model, _convert_="all")

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    log.info("Logging hyperparameters!")
    log_hyperparams(config=config, trainer=trainer)

    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    wandb.finish()
