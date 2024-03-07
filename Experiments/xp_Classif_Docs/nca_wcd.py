import torch

import numpy as np

from tqdm import trange
from pytorch_metric_learning import losses


def train_nca_wcd(X, y, w, idx_train, d=30, n_epochs=10000, lr=1e-3, device="cpu"):
    """
        Inputs:
        - X: List of documents, X[i] of shape (300, len(doc))
        - y: Labels of documents
        - w: Weights of each document
        - idx_train: indices of train set
        - d: dimension of A (can be rectangular)
        - n_epochs
        - lr: lr for A using Adam (default 1e-3)
        - device: default on cpu
        
        Outputs:
        - A: matrix of the Mahalanobis distance of size (300, d) (d(x,y) = (x-y)^T A A^T (x-y))
        - L: loss
    """
    pbar = trange(n_epochs)
    
    X_transformed = np.concatenate([(X[i] @ w[i][0])[None] / np.sum(w[i]) for i in range(len(X))], axis=0)
    X_transformed = torch.from_numpy(X_transformed[idx_train]).to(device)

    labels = torch.from_numpy(y[idx_train])

    loss_func = losses.NCALoss(softmax_scale=1)

    A = torch.rand((300, d), device=device, requires_grad=True, dtype=torch.float64)
    optimizer_A = torch.optim.Adam([A], lr=lr)

    L = []

    for e in pbar:
        optimizer_A.zero_grad()

        embeddings = X_transformed@A
        loss = loss_func(embeddings, labels)
        loss.backward()
        optimizer_A.step()

        L.append(loss.item())

    return A, L
