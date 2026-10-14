"""Posterior samples of the parameters from a GP surrogate of the objective.

With `[sigma]`, Y = -log10(S + EPS_CLIP) and S = -2 ln L + const, so ln L = -10**(-Y) / 2
up to a constant. The GP mean stands in for Y; the GP's own uncertainty is ignored. The
prior is uniform in the unit cube, so log-uniform for log10-scaled parameters.
"""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from proteus.inference.objective import BAD_OBJ_VALUE
from proteus.inference.plot import dpi, fmt
from proteus.inference.transforms import unnormalize_parameters
from proteus.inference.utils import get_kernel
from proteus.utils.coupler import variable_is_logarithmic

log = logging.getLogger('fwl.' + __name__)

dtype = torch.double

DEFAULT_POSTERIOR_SAMPLES = 2000
N_PROPOSAL = 1_000_000  # uniform proposal draws
BATCH = 50_000  # proposals per GP call
GP_NOISE = 1e-4  # as in BO.py


def validate_posterior_samples(n, sigma: dict | None) -> int:
    """Number of posterior samples; None gives 2000 with [sigma] and 0 (off) without."""
    if n is None:
        return DEFAULT_POSTERIOR_SAMPLES if sigma is not None else 0
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f'posterior_samples must be a non-negative integer, got {n!r}')
    if n > 0 and sigma is None:
        raise ValueError('posterior_samples needs [sigma] to make the objective a likelihood')
    return n


def fit_surrogate(X: torch.Tensor, Y: torch.Tensor, kernel: str, seed: int) -> SingleTaskGP:
    """Fit a GP to the scored evaluations, as BO_step does."""
    scored = Y[:, 0] > BAD_OBJ_VALUE
    X, Y = X[scored], Y[scored]
    d = X.shape[-1]
    if len(X) < d + 1:
        raise ValueError(f'{len(X)} scored evaluations cannot constrain a GP in {d} parameters')

    torch.manual_seed(seed)
    gp = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        covar_module=get_kernel(kernel, d),
        train_Yvar=torch.full_like(Y, GP_NOISE),
        input_transform=Normalize(d=d),
        outcome_transform=Standardize(m=1),
    )
    fit_gpytorch_mll(
        ExactMarginalLogLikelihood(gp.likelihood, gp),
        optimizer=fit_gpytorch_mll_torch,
        kwargs={'pick_best_of_all_attempts': True, 'max_attempts': 10},
    )
    return gp.eval()


def gp_mean(gp: SingleTaskGP, U: torch.Tensor) -> torch.Tensor:
    """GP mean at each row of U, one point per batch so no joint covariance is formed."""
    with torch.no_grad():
        return torch.cat(
            [
                gp.posterior(U[i : i + BATCH].unsqueeze(1)).mean.reshape(-1)
                for i in range(0, len(U), BATCH)
            ]
        )


def importance_resample(
    objective, d: int, n_samples: int, n_proposal: int, seed: int
) -> tuple[torch.Tensor, float]:
    """Resample uniform draws in the unit cube by L = exp(-10**(-Y) / 2).

    Returns the samples (n_samples, d) and the effective sample size of the weights.
    """
    g = torch.Generator().manual_seed(seed)
    U = torch.rand(n_proposal, d, generator=g, dtype=dtype)
    lnl = -0.5 * 10.0 ** (-objective(U))
    w = torch.exp(lnl - lnl.max())
    idx = torch.multinomial(w, n_samples, replacement=True, generator=g)
    return U[idx], float(w.sum() ** 2 / (w**2).sum())


def sample_posterior(
    D: dict, parameters: dict, kernel: str, n_samples: int, seed: int, n_proposal: int
) -> tuple[pd.DataFrame, float]:
    """Posterior samples in config units, one column per parameter, and the ESS."""
    keys = list(parameters)
    gp = fit_surrogate(D['X'], D['Y'], kernel, seed)
    U, ess = importance_resample(
        lambda Z: gp_mean(gp, Z), len(keys), n_samples, n_proposal, seed
    )
    bounds = torch.tensor([parameters[k] for k in keys], dtype=dtype).T
    return pd.DataFrame(unnormalize_parameters(U, bounds, keys).numpy(), columns=keys), ess


def plot_posterior(samples: pd.DataFrame, parameters: dict, truth: dict | None, directory: str):
    """Corner plot of the samples, with the truth if given."""
    keys = list(parameters)
    d = len(keys)
    log_ax = {k: variable_is_logarithmic(k) for k in keys}
    edges = {k: (np.geomspace if log_ax[k] else np.linspace)(*parameters[k], 31) for k in keys}

    style = {'font.family': 'sans-serif', 'xtick.direction': 'in', 'ytick.direction': 'in'}
    style |= {'xtick.top': True, 'ytick.right': True}
    with plt.rc_context(style):
        fig, axs = plt.subplots(d, d, figsize=(2.2 * d, 2.2 * d), squeeze=False)
        for i, ki in enumerate(keys):
            for j, kj in enumerate(keys):
                ax = axs[i, j]
                if j > i:
                    ax.set_visible(False)
                    continue
                if i == j:
                    ax.hist(samples[ki], bins=edges[ki], color='#0072B2', histtype='stepfilled')
                    ax.set_yticks([])
                else:
                    ax.hist2d(
                        samples[kj], samples[ki], bins=[edges[kj], edges[ki]], cmap='Blues'
                    )
                    if truth is not None:
                        ax.plot(truth[kj], truth[ki], 'x', color='#D55E00', ms=6)
                    ax.set_yscale('log' if log_ax[ki] else 'linear')
                    ax.set_ylim(parameters[ki])
                if truth is not None:
                    ax.axvline(truth[kj], color='#D55E00', lw=1)
                ax.set_xscale('log' if log_ax[kj] else 'linear')
                ax.set_xlim(parameters[kj])
                if i == d - 1:
                    ax.set_xlabel(kj, fontsize=8)
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(ki, fontsize=8)
                elif j > 0:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=7, which='both')
        fig.tight_layout()
        fig.savefig(os.path.join(directory, 'plots', f'result_posterior.{fmt}'), dpi=dpi)
        plt.close(fig)


def write_posterior(config: dict, D: dict, directory: str) -> pd.DataFrame:
    """Sample the posterior, log its quantiles, and write `posterior.csv` and its plot."""
    n = config['posterior_samples']
    log.info(f'Sampling {n} posterior samples from the GP surrogate')
    samples, ess = sample_posterior(
        D, config['parameters'], config['kernel'], n, config['seed'], N_PROPOSAL
    )
    log.info(f'    effective sample size of {N_PROPOSAL} proposals: {ess:.0f}')
    if ess < n:
        log.warning(
            f'Effective sample size {ess:.0f} is below the {n} samples, so samples repeat'
        )

    q = samples.quantile([0.05, 0.5, 0.95])
    for k in samples.columns:
        log.info(f'    {k:30s} 5/50/95 %: ' + '  '.join(f'{v:.4g}' for v in q[k]))

    samples.to_csv(os.path.join(directory, 'posterior.csv'), index=False)
    plot_posterior(samples, config['parameters'], config.get('truth'), directory)
    return samples
