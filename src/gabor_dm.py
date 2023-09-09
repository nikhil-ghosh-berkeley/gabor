from typing import Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.synth_dataset import GaborDataset


class GaborDataModule(pl.LightningDataModule):
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
        L: int,
        inc_bound: float,
        seed: int
    ):
        super().__init__()
        self.save_dir = save_dir
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.batch_size = batch_size
        self.m = m
        self.k = k
        self.noise = noise
        self.L = L
        self.inc_bound = inc_bound
        self.seed = seed
        self.n_train = n_train
        self.n_val = n_val

    def setup(self, stage: Optional[str] = None):
        dataset = GaborDataset(
            self.save_dir,
            (self.patch_width, self.patch_height),
            (self.n_train + self.n_val),
            m=self.m,
            k=self.k,
            noise=self.noise,
            L=self.L,
            inc_bound=self.inc_bound,
            seed=self.seed
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
