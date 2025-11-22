"""Bayesian optimization core functions.

This module provides utility functions and the core BO_step function
for fitting Gaussian processes, optimizing acquisition functions,
and plotting results during Bayesian optimization.

Functions:
    unit_bounds: Generate unit hypercube bounds for acquisition optimization.
    plot_iter: Visualize GP posterior and acquisition function at each iteration.
    BO_step: Execute a single Bayesian optimization step with timing and logging.
"""
from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import torch
from botorch.acquisition import (
    LogExpectedImprovement,
    UpperConfidenceBound,
    qLogExpectedImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from proteus.inference.utils import get_kernel_w_prior

# Set tensor dtype for consistent precision
dtype = torch.double

def unit_bounds(d):
    """Generate unit hypercube bounds for d-dimensional input.

    Args:
        d (int): Dimensionality of the input space.

    Returns:
        torch.Tensor: Tensor of shape (2, d) where the first row is zeros and the second row is ones.
    """
    # Build bounds [[0,...,0], [1,...,1]]
    bounds = torch.tensor([[0] * d, [1] * d], dtype=dtype)
    return bounds


def BO_step(D, B, f, k, acqf, lock, worker_id, x_in = None):

    """Perform a single Bayesian optimization step.

    Fits a GP to current data, optimizes an acquisition function,
    updates busy points, evaluates the objective, and logs timing.

    Args:
        D (dict): Shared dict containing 'X' and 'Y' lists of data.
        B (dict): Shared dict mapping worker IDs to busy input points.
        f (callable): Objective function to evaluate.
        k (Kernel): GPyTorch kernel for the GP covariance.
        acqf (str): Acquisition function.
        lock (multiprocessing.Lock): Lock for synchronizing shared state.
        worker_id (int): ID of the calling worker.
        x_in (torch.Tensor, optional): Initial input for the first iteration.

    Returns:
        tuple: (x_next, y_next, bo_duration, eval_duration,
                lock_duration, fit_duration, acq_duration, min_dist)
    """

    t_0_bo = time.perf_counter()
    noise = 1e-4

    if x_in is None:

        t_0_lock = time.perf_counter()
        with lock:
            X = D["X"]
            Y = D["Y"]
            busys = list(B.values())

        t_1_lock = time.perf_counter()

        busys = torch.cat(busys, dim=0)

        d = X.shape[-1]

        best = Y.max().item()

        t_0_fit = time.perf_counter()
        gp = SingleTaskGP(  train_X=X,
                            train_Y=Y,
                            covar_module=k,
                            train_Yvar=torch.full_like(Y, noise),
                            input_transform=Normalize(d=d),
                            outcome_transform=Standardize(m=1),
                            )

        lik = gp.likelihood
        mll = ExactMarginalLogLikelihood(lik, gp)
        fit_gpytorch_mll(mll, optimizer=fit_gpytorch_mll_torch,
                         kwargs={"pick_best_of_all_attempts" : True, "max_attempts" : 10})

        t_1_fit = time.perf_counter()

        t_0_ac = time.perf_counter()

        # build acquisition function
        if acqf == "UCB":
            acqf = UpperConfidenceBound(gp, beta = 2.)
        elif acqf =="LogEI":
            acqf = LogExpectedImprovement(gp, best_f=best)
        elif acqf =="E-LogEI":
            acqf = qLogExpectedImprovement(gp, best_f=best, X_pending=busys)
        else:
            raise ValueError("Unknown acquisition function, choices are UCB or LogEI")

        x, _ = optimize_acqf(   acq_function=acqf, # expects outputs shape (N)
                                bounds=unit_bounds(d),
                                q=1,
                                num_restarts=10,
                                raw_samples=1000*d,
                                options={"maxiter": 1000},
                            )

        t_1_ac = time.perf_counter()

        mask = torch.ones(busys.size(0), dtype=torch.bool)
        mask[worker_id] = False
        b = busys[mask]
        dist = torch.min(torch.cdist(b, x)).item()

        if d == 1:
            plot_iter(gp=gp, acqf=acqf, X=X, Y=Y, next_x=x,
                    busys=b, dir = "plots/iters/", name=f"BO_{len(X)}_.png")

    else:
        x = x_in

        dist = None

        t_0_lock = 0
        t_1_lock = 0
        t_0_fit = 0
        t_1_fit = 0
        t_0_ac = 0
        t_1_ac = 0

    t_1_bo = time.perf_counter()

    t_2_lock = time.perf_counter()
    with lock:
        B[worker_id] = x
    t_3_lock = time.perf_counter()

    t_0_ev = time.perf_counter()
    y = f(x)
    t_1_ev = time.perf_counter()

    return x, y, t_1_bo - t_0_bo, t_1_ev - t_0_ev, t_3_lock-t_2_lock+t_1_lock-t_0_lock, t_1_fit-t_0_fit, t_1_ac-t_0_ac, dist

def init_locs(n_workers: int, D: dict) -> torch.Tensor:
    """Generate initial sample locations using batch acqf.

    Args:
        n_workers (int): Number of initial points to generate.
        D (dict): Shared dict with key 'X' to infer problem dimension.

    Returns:
        torch.Tensor: Tensor of shape (n_workers, d) in [0,1]^d for initial sampling.
    """

    X, Y = D["X"], D["Y"]

    d = X.shape[-1]

    best = Y.max().item()

    noise = 1e-4
    kernel = get_kernel_w_prior(d, use_rbf_kernel=False, nu=1.5)
    gp = SingleTaskGP(  train_X=X,
                        train_Y=Y,
                        covar_module=kernel,
                        input_transform=Normalize(d=d),
                        outcome_transform=Standardize(m=1),
                        train_Yvar=torch.full_like(Y, noise)
                        )

    lik = gp.likelihood
    mll = ExactMarginalLogLikelihood(lik, gp)
    fit_gpytorch_mll(mll, optimizer=fit_gpytorch_mll_torch,
                     kwargs={"pick_best_of_all_attempts" : True, "max_attempts" : 10})

    # build acquisition function
    acqf = qLogExpectedImprovement(gp, best_f=best)

    x_batch, _ = optimize_acqf( acq_function=acqf,
                                bounds=unit_bounds(d),
                                q=n_workers,
                                num_restarts=10,
                                raw_samples=1000*d,
                                options={"maxiter": 1000},
                                )

    return x_batch # (n_workers, d)


def plot_iter(gp, acqf, X, Y, next_x, busys, dir, name):

    """Plot the GP posterior and acquisition function for the current iteration.

    Args:
        gp (SingleTaskGP): Fitted Gaussian Process model.
        acqf (AcquisitionFunction): Acquisition function instance.
        X (torch.Tensor): Observed inputs, shape (n, 1).
        Y (torch.Tensor): Observed outputs, shape (n, 1).
        next_x (torch.Tensor): Next query point, shape (1, 1).
        busys (list of torch.Tensor): Busy points to overlay on acquisition plot.
        dir (str): Directory to save the plot.
        name (str): Filename for the saved plot.
    """

    fig, ax = plt.subplots(2,1)

    xs = (torch.linspace(0,1, 400) ).reshape(-1,1,1) # batch_shape x q=1 x d for acqf

    post = gp.posterior(xs)
    mu = post.mean.detach().flatten()
    sig = post.variance.sqrt().detach().flatten()

    ac = acqf(xs).detach().flatten()

    ax[0].plot(xs.detach().flatten(), mu, color = "black", linewidth = 0.5, label = "post mean")
    ax[0].fill_between(xs.detach().flatten(), mu-2*sig, mu+2*sig, alpha = 0.25 )
    ax[0].scatter(X.flatten(), Y.flatten(), s = 5, label = "data", color = "blue")
    # fs = objective(xs).flatten()
    # ax[0].plot(xs.detach().flatten(), fs, color = "grey", label = "f", linewidth = 0.5)

    ax[1].plot(xs.detach().flatten(), ac, color = "cornflowerblue", label = "acqf")
    a = acqf(next_x).detach().flatten()
    ax[1].scatter(next_x, a, label = "next query", color = "green", marker = "*")
    if busys is not None:
        y_min, y_max = ax[1].get_ylim()
        ax[1].scatter(busys, torch.full_like(busys, y_min),
                        label = "busy", color = "orange", marker = "*")

    ax[0].legend(loc = "upper left")
    ax[1].legend(loc = "upper left")

    last_x = float(X.flatten()[-1])
    last_y = float(Y.flatten()[-1])

    ax[0].text(
        0.98, 0.98,
        f"({last_x:.2f}, {last_y:.2f})",
        va="top", ha="right",
        transform=ax[0].transAxes,
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    ax[1].text(
        0.98, 0.98,
        f"({float(next_x):.2f}, {float(a):.2f})",
        va="top", ha="right",
        transform=ax[1].transAxes,
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    path = dir + name
    os.makedirs(dir, exist_ok=True)
    fig.savefig(path, dpi = 300)
    plt.close(fig)
