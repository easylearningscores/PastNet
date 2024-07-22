import torch
from torch import nn
from modules import ConvSC, Inception
import torch.nn.functional as F
import torch.fft
import numpy as np
import torch.optim as optimizer
from functools import partial
from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.utils.checkpoint import checkpoint_sequential
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from sklearn.neighbors import NearestNeighbors


def kNN(X, n_neighbors, n_jobs):
    X = X.cpu().detach().numpy()
    neigh = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs).fit(X)
    dists, inds = neigh.kneighbors(X)
    return dists, inds

def Levina_Bickel(X, dists, k):
    m = np.log(dists[:, k:k+1] / dists[:, 1:k])
    m = (k-2) / np.sum(m, axis=1)
    dim = np.mean(m)
    return dim

def estimate_dimension(latent_embedding, k=1000):
    B, T, C_, H_, W_ = latent_embedding.shape
    latent_embedding = latent_embedding.permute(0, 3, 4, 1, 2)
    X = latent_embedding.reshape(B*H_*W_, T*C_)
    dists, _ = kNN(X, k+1, n_jobs=-1)
    dim_estimate = Levina_Bickel(X, dists, k)
    return dim_estimate