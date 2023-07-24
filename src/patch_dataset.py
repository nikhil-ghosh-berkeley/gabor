from os.path import join as pjoin
from typing import Tuple

import numpy as np
import scipy.io as sio
import torch
from sklearn.feature_extraction import image
from torch.utils.data import Dataset


def get_image_patches(
    data_dir: str, patch_size: Tuple[int, int], use_rawimage: bool, num_patches: int
):
    assert len(patch_size) == 2

    if use_rawimage:
        images = sio.loadmat(pjoin(data_dir, "IMAGES_RAW.mat"))["IMAGESr"]
    else:
        images = sio.loadmat(pjoin(data_dir, "IMAGES.mat"))["IMAGES"]

    images = images.astype("float32")

    patches = []
    num_images = images.shape[2]
    patches_per_image = num_patches // num_images

    for i in range(num_images):
        img = images[:, :, i]
        img_patches = image.extract_patches_2d(
            img,
            patch_size,
            max_patches=patches_per_image,
        )
        patches.append(img_patches)

    # unsqueeze for NCHW
    all_patches = torch.from_numpy(np.concatenate(patches)).unsqueeze(1)
    return all_patches


class ImagePatchesDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        patch_size: Tuple[int, int],
        use_rawimage: bool,
        num_patches: int,
    ) -> None:

        self.data_x = get_image_patches(data_dir, patch_size, use_rawimage, num_patches)

    def __getitem__(self, index):
        img = self.data_x[index]
        return img

    def __len__(self):
        return self.data_x.shape[0]
