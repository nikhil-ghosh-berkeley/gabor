from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.patch_dataset import ImagePatchesDataset


class PatchesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        patch_width: int,
        patch_height: int,
        use_rawimage: bool,
        batch_size: int,
        n_train: int,
        n_val: int,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.use_rawimage = use_rawimage
        self.batch_size = batch_size
        self.n_train = n_train
        self.n_val = n_val

    def setup(self, stage: Optional[str] = None):
        dataset = ImagePatchesDataset(
            self.data_dir,
            (self.patch_width, self.patch_height),
            self.use_rawimage,
            (self.n_train + self.n_val),
        )

        self.train_set, self.val_set = random_split(
            dataset, [self.n_train, (len(dataset) - self.n_train)]
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
