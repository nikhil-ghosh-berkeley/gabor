import math
import os
from itertools import product
from os.path import join as pjoin
from random import sample
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def gabor_2d(M, N, lamb, theta, sigma, gamma=1.0, offset_x=0, offset_y=0):
    """
    (partly from kymatio package)
    Computes a 2D Gabor filter.
    See https://en.wikipedia.org/wiki/Gabor_filter for the formula of a gabor filter

    Parameters
    ----------
    M, N : int
        spatial sizes
    lamb : float
        central frequency
    theta : float
        angle in [0, pi]
    sigma : float
        bandwidth parameter of the gaussian envelope
    gamma : float
        parameter which guides the elipsoidal shape of the gabor
    offset_x : int, optional
        offset by which the signal starts in col
    offset_y : int, optional
        offset by which the signal starts in row

    Returns
    -------
    gab : ndarray
        numpy array of size (M, N)
    """
    R = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float64
    )
    R_inv = np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float64
    )
    D = np.array([[1, 0], [0, gamma * gamma]])
    curv = np.matmul(R, np.matmul(D, R_inv)) / (2 * sigma * sigma)

    gab = np.zeros((M, N), np.complex128)
    xx, yy = np.mgrid[
        offset_x - (M // 2):offset_x + M - (M // 2),
        offset_y - (N // 2):offset_y + N - (N // 2),
    ]

    arg = -(
        curv[0, 0] * xx * xx
        + (curv[0, 1] + curv[1, 0]) * xx * yy
        + curv[1, 1] * yy * yy
    ) + 1.0j * (
        xx * 2 * np.pi / lamb * np.cos(theta) + yy * 2 * np.pi / lamb * np.sin(theta)
    )
    gab = np.exp(arg)

    norm_factor = 2 * np.pi * sigma * sigma / gamma
    gab = gab / norm_factor

    return gab


def gabor_filterbank(
    num_filters: int, patch_size: Tuple[int, int], L: int, inc_bound: float
):
    """
    generate gabor filters uniformly sampled from a gabor filterbank

    Parameters
    ----------
    M, N : int
        spatial sizes
    L : int
        number of orientations to create for gabor filterbank
    num_filters : int
        number of gabor filters to sample
    inc_bound : float
        upper bound on the incoherence of the generated dictionary
        max value is sqrt(M*N)

    Returns
    -------
    A : torch.Tensor
        n x m tensor array of gabor dictionaries
    """
    M, N = patch_size
    Theta = [(int(L - L / 2 - 1) - theta) * np.pi / L for theta in range(L)]
    Sigma = np.linspace(0.3, 1, num=20)
    Gamma = [4 / L]
    Offset_x = np.arange(-3, 4)
    Offset_y = np.arange(-3, 4)

    param_comb = list(product(Theta, Sigma, Gamma, Offset_x, Offset_y))
    A = torch.randn(M * N, num_filters)

    i = 0
    max_iter = 0
    while i < num_filters:
        theta, sigma, gamma, offset_x, offset_y = sample(param_comb, 1)[0]
        lamb = 10 / 3 * sigma
        wavelet = gabor_2d(
            M=M,
            N=N,
            lamb=lamb,
            theta=theta,
            sigma=sigma,
            gamma=gamma,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        r = np.random.binomial(1, 0.5)
        if r == 0:
            psi = torch.from_numpy(wavelet.real.astype("float32")).reshape(-1)
        else:
            psi = torch.from_numpy(wavelet.imag.astype("float32")).reshape(-1)
        psi = psi / torch.norm(psi)

        if i == 0:
            A[:, i] = psi
            i += 1
        elif i > 0:
            inner = A[:, :i] * psi[:, None]
            incoherence = torch.max(abs(inner.sum(axis=0))) * math.sqrt(A.shape[0])
            if incoherence < inc_bound:
                A[:, i] = psi
                i += 1
        max_iter += 1
        if max_iter >= 50000:
            raise ValueError("The upper bound on the incoherence is too low.")
    return A


def generate_from_dict(A: torch.Tensor, n_samples: int, k: int, noise: float):
    n, m = A.shape

    # train data
    Z = abs(torch.randn(m, n_samples))
    mask = 1.0 * (torch.randn(m, n_samples).argsort(axis=0) < k)
    Z *= mask
    X = torch.matmul(A, Z) + noise * torch.randn(n, n_samples) / np.sqrt(n)

    return X.t()


class GaborDataset(Dataset):
    def __init__(
        self,
        save_dir: str,
        patch_size: Tuple[int, int],
        num_samples: int,
        m: int,
        k: int,
        noise: float,
        L: int,
        inc_bound: float,
    ) -> None:
        w, h = patch_size
        dict_fname = f"gabor_{w}x{h}_m={m}_L={L}_inc={inc_bound}"
        dict_path = pjoin(save_dir, f"{dict_fname}.pt")
        if os.path.exists(dict_path):
            A = torch.load(dict_path)
            A = A.t()
        else:
            A = gabor_filterbank(
                num_filters=m, patch_size=patch_size, L=L, inc_bound=inc_bound
            )
            os.makedirs(save_dir, exist_ok=True)
            torch.save(A.t(), dict_path)

        # A is size n x m
        self.dictionary = A
        self.data_x = generate_from_dict(A, num_samples, k, noise)
        self.data_x = self.data_x.reshape(num_samples, 1, w, h)

    def get_dictionary(self):
        return self.dictionary

    def __getitem__(self, index):
        img = self.data_x[index]
        return img

    def __len__(self):
        return self.data_x.shape[0]


class GaussianDataset(Dataset):
    def __init__(
        self,
        save_dir: str,
        patch_size: Tuple[int, int],
        num_samples: int,
        m: int,
        k: int,
        noise: float,
    ) -> None:
        w, h = patch_size
        n = w * h
        dict_fname = f"gaussian_{w}x{h}_m={m}"

        dict_path = pjoin(save_dir, f"{dict_fname}.pt")
        if os.path.exists(dict_path):
            A = torch.load(dict_path)
            A = A.t()
        else:
            A = torch.randn(n, m) / np.sqrt(n)
            A = A / torch.norm(A, dim=0)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(A.t(), dict_path)

        self.dictionary = A
        self.data_x = generate_from_dict(A, num_samples, k, noise)
        self.data_x = self.data_x.reshape(num_samples, 1, w, h)

    def get_dictionary(self):
        return self.dictionary

    def __getitem__(self, index):
        img = self.data_x[index]
        return img

    def __len__(self):
        return self.data_x.shape[0]
