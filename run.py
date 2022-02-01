import hydra
from omegaconf import DictConfig, OmegaConf

from src.train import train
from pytorch_lightning.utilities.seed import seed_everything

OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
OmegaConf.register_new_resolver("int_mult", lambda x, y: int(x * y))

@hydra.main(config_path="conf/", config_name="config.yaml")
def main(config: DictConfig):
    config.seed = seed_everything(config.seed)
    train(config)

if __name__ == "__main__":
    main()
