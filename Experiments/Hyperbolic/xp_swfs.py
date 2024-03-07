import torch
import ot
import argparse

import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from itertools import cycle
from tqdm.auto import trange

from hswfs.manifold.euclidean import Euclidean
from hswfs.manifold.lorentz import Lorentz
from hswfs.manifold.poincare import Poincare
from hswfs.chswf import chswf
from hswfs.utils_swf import utils_plot_poincare


parser = argparse.ArgumentParser()
parser.add_argument("--type_target", type=str, default="wnd", help="wnd or mwnd")
parser.add_argument("--target", type=str, default="center", help="Which target to use")
parser.add_argument("--pbar", action="store_true", help="If yes, plot pbar")
parser.add_argument("--ntry", type=int, default=5, help="number of restart")
parser.add_argument("--nproj", type=int, default=50, help="Number of projections")
parser.add_argument("--lr", type=float, default=1, help="Learning rate")
parser.add_argument("--n_epochs", type=int, default=1001, help="Number of epochs")
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    K = -1
    d = 2

    manifold_lorentz_geod = Lorentz(d=d+1, K=K, projection="geodesic",
                                    device=device)
    manifold_lorentz_horo = Lorentz(d=d+1, K=K, projection="horospheric",
                                    device=device)
    manifold_poincare_horo = Poincare(d=d, K=K, projection="horospheric",
                                      device=device)
    manifold_euclidean = Euclidean(d=d)

    if args.type_target == "wnd":
        if args.target == "center":
            mu = torch.tensor([1.5, np.sqrt(1.5**2-1), 0], dtype=torch.float64,
                              device=device)
        elif args.target == "border":
            mu = torch.tensor([8, np.sqrt(63), 0], dtype=torch.float64,
                              device=device)
        Sigma = 0.1 * torch.tensor([[1, 0], [0, 1]], dtype=torch.float,
                                   device=device)

    elif args.type_target == "mwnd":
        ps = np.ones(5)/5
        if args.target == "center":
            mus_lorentz = torch.tensor([[0, 0.5], [0.5, 0],
                                        [0, -0.5], [-0.5, 0],
                                        [0, 0.1]], dtype=torch.float, device=device)
        elif args.target == "border":
            mus_lorentz = torch.tensor([[0, 0.9], [0.9, 0], [0, -0.9],
                                        [-0.9, 0], [0, 0.1]],
                                       dtype=torch.float, device=device)

        mus = manifold_poincare_horo.to_lorentz(mus_lorentz)
        sigma = 0.01 * torch.tensor([[1, 0], [0, 1]], dtype=torch.float, device=device)

    num_projections = args.nproj
    lr = args.lr
    n_epochs = args.n_epochs
    n_try = args.ntry

    mu0 = torch.tensor([1, 0, 0], dtype=torch.float64, device=device)
    Sigma0 = torch.eye(2, dtype=torch.float, device=device)

    L_swp = np.zeros((n_try, n_epochs))
    L_hhsw = np.zeros((n_try, n_epochs))
    L_ghsw = np.zeros((n_try, n_epochs))
    L_hhswp = np.zeros((n_try, n_epochs))

    for k in range(n_try):
        if args.type_target == "wnd":
            X_target = manifold_lorentz_horo.sample_wrapped_normal(10000,
                                                                   mu,
                                                                   Sigma)
        elif args.type_target == "mwnd":
            Z = np.random.multinomial(10000, ps)
            X = []
            for l in range(len(Z)):
                if Z[l] > 0:
                    samples = manifold_lorentz_horo.sample_wrapped_normal(int(Z[l]), mus[l], sigma).cpu().numpy()
                    X += list(samples)

            X_target = torch.tensor(X, device=device, dtype=torch.float64)

        train_dl = DataLoader(X_target, batch_size=500, shuffle=True)
        dataiter_lorentz = iter(cycle(train_dl))

        X_target_poincare = manifold_lorentz_horo.to_poincare(X_target)
        train_dl = DataLoader(X_target_poincare, batch_size=500, shuffle=True)
        dataiter_poincare = iter(cycle(train_dl))

        x0 = manifold_lorentz_horo.sample_wrapped_normal(500, mu0, Sigma0)
        x0_p = manifold_lorentz_horo.to_poincare(x0)

        particles_swp = chswf(x0_p, n_epochs, dataiter_poincare,
                              manifold_euclidean, tauk=lr)
        particles_ghsw = chswf(x0, n_epochs, dataiter_lorentz,
                               manifold_lorentz_geod, tauk=lr)
        particles_hhsw = chswf(x0, n_epochs, dataiter_lorentz,
                               manifold_lorentz_horo, tauk=lr)
        particles_hhsw_p = chswf(x0_p, n_epochs, dataiter_poincare,
                                 manifold_poincare_horo, tauk=lr)

        for e in range(n_epochs):
            n = 500
            if args.type_target == "wnd":
                x_test = manifold_lorentz_horo.sample_wrapped_normal(n, mu, Sigma)
            elif args.type_target == "mwnd":
                Z = np.random.multinomial(n, ps)
                X = []
                for l in range(len(Z)):
                    if Z[l] > 0:
                        samples = manifold_lorentz_horo.sample_wrapped_normal(int(Z[l]), mus[l],
                                                                              sigma).cpu().numpy()
                        X += list(samples)
                x_test = torch.tensor(X, device=device, dtype=torch.float64)

            x_swp = particles_swp[e].to(device)
            x_swp_l = manifold_poincare_horo.to_lorentz(x_swp).to(device)

            x_ghsw = particles_ghsw[e].to(device)
            x_hhsw = particles_hhsw[e].to(device)

            x_hhswp = particles_hhsw_p[e].to(device)
            x_hhswp_l = manifold_poincare_horo.to_lorentz(x_hhswp).to(device)

            a = torch.ones((n,), device=device)/n
            b = torch.ones((n,), device=device)/n

            ip = manifold_lorentz_horo.minkowski_ip2(x_swp_l, x_test)
            M = torch.arccosh(torch.clamp(-ip, min=1+1e-15))**2
            w = ot.emd2(a, b, M)
            L_swp[k, e] = w.item()

            ip = manifold_lorentz_horo.minkowski_ip2(x_ghsw, x_test)
            M = torch.arccosh(torch.clamp(-ip, min=1+1e-15))**2
            w = ot.emd2(a, b, M)
            L_ghsw[k, e] = w.item()

            ip = manifold_lorentz_horo.minkowski_ip2(x_hhsw, x_test)
            M = torch.arccosh(torch.clamp(-ip, min=1+1e-15))**2
            w = ot.emd2(a, b, M)
            L_hhsw[k, e] = w.item()

            ip = manifold_lorentz_horo.minkowski_ip2(x_hhswp_l, x_test)
            M = torch.arccosh(torch.clamp(-ip, min=1+1e-15))**2
            w = ot.emd2(a, b, M)
            L_hhswp[k, e] = w.item()

    np.savetxt("./Results/sw_loss_"+args.type_target+"_"+args.target, L_swp)
    np.savetxt("./Results/ghsw_loss_"+args.type_target+"_"+args.target, L_ghsw)
    np.savetxt("./Results/hhsw_loss_"+args.type_target+"_"+args.target, L_hhsw)
    np.savetxt("./Results/hhswp_loss_"+args.type_target+"_"+args.target,
               L_hhswp)
