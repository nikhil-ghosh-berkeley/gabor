from omegaconf import DictConfig
from pytorch_lightning import Trainer


def log_hyperparams(config: DictConfig, trainer: Trainer) -> None:
    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    # if "callbacks" in config:
    #     hparams["callbacks"] = config["callbacks"]

    trainer.logger.log_hyperparams(hparams)
