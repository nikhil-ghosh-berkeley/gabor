from typing import Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.synth_dataset import OrthoDataset


class OrthoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        save_dir: str,
        patch_width: int,
        patch_height: int,
        m: int,
        k: int,
        noise: float,
        n_train: int,
        n_val: int,
        batch_size: int,
        gamma: float,
        rho: float,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.batch_size = batch_size
        self.m = m
        self.k = k
        self.noise = noise
        self.n_train = n_train
        self.n_val = n_val
        self.gamma = gamma
        self.rho = rho

    def setup(self, stage: Optional[str] = None):
        dataset = OrthoDataset(
            self.save_dir,
            (self.patch_width, self.patch_height),
            (self.n_train + self.n_val),
            m=self.m,
            k=self.k,
            noise=self.noise,
            gamma=self.gamma,
            rho=self.rho
        )

        self.dictionary = dataset.get_dictionary()

        self.train_set, self.val_set = random_split(
            dataset, [self.n_train, (len(dataset) - self.n_train)]
        )

    def get_dictionary(self):
        return self.dictionary

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
