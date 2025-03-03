# Sliced-Wasserstein Distances and Flows on Cartan-Hadamard Manifolds

This repository contains the code to reproduce the experiments of the paper [Sliced-Wasserstein Distances and Flows on Cartan-Hadamard Manifolds](https://arxiv.org/abs/2403.06560). We derive in this paper a general construction allowing to define Sliced-Wasserstein distances on Cartan-Hadamard manifolds. In this repository, we provide the code to compute the Sliced-Wasserstein distances and Wasserstein gradient flows of these Sliced-Wasserstein distances on Cartan-Hadamard manifolds. The function require an object of the class BaseManifold. We provide the code for the Euclidean manifold, the Mahalanobis manifold, the Hyperbolic space through the Lorentz model and the Poincaré ball, the space of SPD matrices with Log-Euclidean metric, and a product of these manifolds.

## Abstract

While many Machine Learning methods were developed or transposed on Riemannian manifolds to tackle data with known non Euclidean geometry, Optimal Transport (OT) methods on such spaces have not received much attention. The main OT tool on these spaces is the Wasserstein distance which suffers from a heavy computational burden. On Euclidean spaces, a popular alternative is the Sliced-Wasserstein distance, which leverages a closed-form solution of the Wasserstein distance in one dimension, but which is not readily available on manifolds. In this work, we derive general constructions of Sliced-Wasserstein distances on Cartan-Hadamard manifolds, Riemannian manifolds with non-positive curvature, which include among others Hyperbolic spaces or the space of Symmetric Positive Definite matrices. Then, we propose different applications such as classification of documents with a suitably learned ground cost on a manifold, and data set comparison on a product manifold. Additionally, we derive non-parametric schemes to minimize these new distances by approximating their Wasserstein gradient flows.

## Citation

```
@article{bonet2025sliced,
    title={{Sliced-Wasserstein Distances and Flows on Cartan-Hadamard Manifolds}},
    author={Clément Bonet and Lucas Drumetz and Nicolas Courty},
    year={2025},
    journal={Journal of Machine Learning Research},
    volume={26},
    number={32},
    pages={1--76}
}
```

## Install the package

```
$ python setup.py install
```

## Description of the library

This library contains mainly two functions: `sliced_wasserstein` and `chswf`. For both functions, we need to specify on which manifold to run it by giving it in argument an object from the `BaseManifold` class. Here is an example to compute the Euclidean Sliced-Wasserstein distance:

```
from hswfs.sw import sliced_wasserstein
from hswfs.manifold.euclidean import Euclidean

n, d = 100, 2

x1 = torch.rand((n, d))
x2 = torch.rand((n, d))

manifold = Euclidean(d)

sw = sliced_wasserstein(x1, x2, 500, manifold)
```
Then, here is an example to compute the Wasserstein gradient flow minimizing the Euclidean Sliced-Wasserstein distance:
```
from hswfs.chswf import chswf
from hswfs.manifold.euclidean import Euclidean
from itertools import cycle
from torch.utils import data

n, d = 100, 2
n_epochs = 501

X = 10 + torch.randn((n, d))
y = torch.zeros(len(X))
train_dataset = data.TensorDataset(X, y)
rand_sampler = torch.utils.data.RandomSampler(train_dataset) #, replacement=True)
train_sampler = torch.utils.data.DataLoader(train_dataset, batch_size=500, sampler=rand_sampler)
dataiter = iter(cycle(train_sampler))

x0 = torch.randn(n, d)
L_particles = chswf(x0, n_epochs, dataiter, manifold, tauk=0.1, n_projs=500)
```



## Experiments

- In Experiments/Euclidean, you can find the trajectories of gradient flows for the Euclidean Sliced-Wasserstein distance and the Mahalanobis Sliced-Wasserstein.
- In Experiments/Hyperbolic, you can find the code to reproduce the gradient flow experiment of Section 7.3
- In Experiments/xp_Classif_Docs, you can find the code to reproduce the experiment of Section 6.1 on Document Classification.
- In Experiments/otdd, we provide the code to reproduce the experiment of Section 6.2 on the comparison of datasets.


## Package requirements

* numpy
* torch
* pot
* matplotlib
* tqdm
* [torchinterp1d](https://github.com/aliutkus/torchinterp1d)

## Credits

* For the OTDD experiment, code from [otdd](https://github.com/microsoft/otdd/tree/main) and [Wasserstein-Task-Embedding](https://github.com/xinranliueva/Wasserstein-Task-Embedding) were used.
