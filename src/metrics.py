import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def standardize(X):
    X = X - torch.min(X, dim=1, keepdim=True)[0]
    X = X / torch.max(X, dim=1, keepdim=True)[0]
    return X

def cos_sim(A, B):
    '''
    A: tensor of size m x n
    B: tensor of size p x n
    '''

    return F.normalize(A) @ F.normalize(B).T

def match_min_row_cost(all_pairs):
    row_ind, col_ind = linear_sum_assignment(all_pairs)
    return all_pairs[row_ind, col_ind].sum()

def comp_incoherence(A):
    """
    Computes incoherence of dictionary

    Parameters
    ----------
    A : torch.Tensor
        n x m tensor array

    """
    A_norm = torch.sqrt(torch.sum(A ** 2, axis=0))
    A_normalize = A / A_norm[None, :]
    inner = (A_normalize[:, :, None] * A_normalize[:, None, :]).sum(axis=0)
    inner = torch.triu(inner, diagonal=1)
    inc = torch.max(abs(inner)) * math.sqrt(A.shape[0])
    return inc.item()
