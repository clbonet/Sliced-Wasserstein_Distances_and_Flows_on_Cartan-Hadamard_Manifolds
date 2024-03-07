import torch
import ot
import time

import numpy as np

from tqdm import trange
from hswfs.manifold.euclidean import Mahalanobis
from hswfs.sw import sliced_wasserstein


def compute_matrix_w(X, w, A, alpha, idx_docs, device="cpu", get_time=False):
    """
        Inputs
        - X: List of documents, X[i] of shape (300, len(doc))
        - w: Weights of each document
        - A: Matrix projection (size (300, d))
        - alpha: reweighting of importance of words (size # of words in the vocabulary)
        - idx_docs: idx in each doc
        
        Output:
        - d_w: matrix of distances of size (len(docs), len(docs))
    """
    d_w = np.zeros((len(X), len(X)))
    ts = []
    
    pbar = trange(len(X))

    for i in pbar:
        x1 = (A.T@torch.tensor(X[i], device=device, dtype=torch.float64)).T
        w1 = torch.tensor(w[i], device=device, dtype=torch.float64)[0] / np.sum(w[i])
        w1 = w1 * alpha[idx_docs[i]] / torch.sum(w1 * alpha[idx_docs[i]])

        for j in range(i+1, len(X)):
            x2 = (A.T@torch.tensor(X[j], device=device, dtype=torch.float64)).T
            w2 = torch.tensor(w[j], device=device, dtype=torch.float64)[0] / np.sum(w[j])
            w2 = w2 * alpha[idx_docs[j]] / torch.sum(w2 * alpha[idx_docs[j]])

            t0 = time.time()
            M = ot.dist(x1, x2, metric="sqeuclidean")
            d_w[i, j] = ot.emd2(w1, w2, M)
            ts.append(time.time() - t0)
            d_w[j, i] = d_w[i, j]
    
    if get_time:
        return d_w, ts
    
    return d_w



def compute_matrix_sw(X, w, A, alpha, idx_docs, n_projs=500, device="cpu", get_time=False):
    """
        Inputs
        - X: List of documents, X[i] of shape (300, len(doc))
        - w: Weights of each document
        - A: Matrix projection (size (300, 300))
        - alpha: reweighting of importance of words (size # of words in the vocabulary)
        - n_projs: Number of projections to compute SW
        - idx_docs: idx in each doc
        
        Output:
        - d_w: matrix of distances of size (len(docs), len(docs))
    """
    manifold = Mahalanobis(X[0].shape[0], A)

    d_sw = np.zeros((len(X), len(X)))
    ts = []

    pbar = trange(len(X))

    for i in pbar:
        x1 = torch.tensor(X[i], device=device, dtype=torch.float64).T
        w1 = torch.tensor(w[i], device=device, dtype=torch.float64)[0] / np.sum(w[i])
        w1 = w1 * alpha[idx_docs[i]] / torch.sum(w1 * alpha[idx_docs[i]])

        for j in range(i+1, len(X)):
            x2 = torch.tensor(X[j], device=device, dtype=torch.float64).T
            w2 = torch.tensor(w[j], device=device, dtype=torch.float64)[0] / np.sum(w[j])
            w2 = w2 * alpha[idx_docs[j]] / torch.sum(w2 * alpha[idx_docs[j]])

            t0 = time.time()
            loss = sliced_wasserstein(x1, x2, n_projs, manifold, w1, w2)
            ts.append(time.time() - t0)

            d_sw[i, j] = loss.item()
            d_sw[j, i] = loss.item()
            
    if get_time:
        return d_sw, ts
    
    return d_sw
