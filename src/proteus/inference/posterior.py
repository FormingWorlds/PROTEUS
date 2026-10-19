"""Posterior samples of the parameters from a GP surrogate of the objective.

With `[sigma]`, Y = -log10(S + EPS_CLIP) and S = -2 ln L + const, so ln L = -10**(-Y) / 2
up to a constant. The GP mean stands in for Y; the GP's own uncertainty is ignored. The
prior is uniform in the unit cube, so log-uniform for log10-scaled parameters.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from proteus.inference.objective import BAD_OBJ_VALUE
from proteus.inference.transforms import unnormalize_parameters
from proteus.inference.utils import get_kernel

log = logging.getLogger('fwl.' + __name__)

dtype = torch.double

DEFAULT_POSTERIOR_SAMPLES = 2000
N_PROPOSAL = 1_000_000  # uniform proposal draws
BATCH = 50_000  # proposals per GP call
GP_NOISE = 1e-4  # as in BO.py

# Proposal refinement while the effective sample size is below the samples asked for
ADAPT_ROUNDS = 5
ADAPT_PROPOSAL = 200_000  # draws per refined round
ADAPT_CENTRES = 200  # Gaussian components, centred on draws picked by weight
DEFENSIVE = 0.1  # uniform share of a refined proposal, so that no weight can blow up
SD_FLOOR = 0.01  # its square is added to the proposal variance, in the unit cube


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


def _weights(lnw: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Importance weights scaled to a maximum of 1, and their effective sample size."""
    w = torch.exp(lnw - lnw.max())
    return w, float(w.sum() ** 2 / (w**2).sum())


def _log_mixture(U: torch.Tensor, centres: torch.Tensor, chol: torch.Tensor) -> torch.Tensor:
    """Log density in the unit cube of the refined proposal: DEFENSIVE uniform plus the
    equal-weight Gaussian components at `centres` with covariance chol @ chol.T."""
    d = U.shape[-1]
    inv = torch.linalg.inv(chol).T
    log_norm = -0.5 * d * np.log(2 * np.pi) - torch.log(torch.diagonal(chol)).sum()
    log_uniform = torch.tensor(np.log(DEFENSIVE), dtype=dtype)
    out = []
    for i in range(0, len(U), 10_000):
        z = (U[i : i + 10_000, None, :] - centres[None]) @ inv
        gauss = torch.logsumexp(log_norm - 0.5 * (z**2).sum(-1), dim=1) - np.log(len(centres))
        out.append(torch.logaddexp(gauss + np.log(1 - DEFENSIVE), log_uniform))
    return torch.cat(out)


def _refined_draws(U, w, g) -> tuple[torch.Tensor, torch.Tensor]:
    """Draws in the unit cube from a Gaussian mixture on the weighted draws U, with their
    log proposal density."""
    d = U.shape[-1]
    p = w / w.sum()
    mu = p @ U
    # Silverman's bandwidth for a kernel density estimate on ADAPT_CENTRES points
    h2 = (4 / ((d + 2) * ADAPT_CENTRES)) ** (2 / (d + 4))
    cov = h2 * (U - mu).T @ ((U - mu) * p[:, None]) + SD_FLOOR**2 * torch.eye(d, dtype=dtype)
    chol = torch.linalg.cholesky(cov)
    centres = U[torch.multinomial(w, ADAPT_CENTRES, replacement=True, generator=g)]

    n_uniform = int(DEFENSIVE * ADAPT_PROPOSAL)
    pick = torch.randint(ADAPT_CENTRES, (ADAPT_PROPOSAL - n_uniform,), generator=g)
    z = torch.randn(len(pick), d, generator=g, dtype=dtype)
    V = torch.cat(
        [centres[pick] + z @ chol.T, torch.rand(n_uniform, d, generator=g, dtype=dtype)]
    )
    V = V[((V >= 0) & (V <= 1)).all(dim=1)]  # outside the cube the prior is zero
    return V, _log_mixture(V, centres, chol)


def importance_resample(
    objective, d: int, n_samples: int, n_proposal: int, seed: int
) -> tuple[torch.Tensor, float]:
    """Resample draws in the unit cube by L = exp(-10**(-Y) / 2): n_proposal uniform draws,
    then up to ADAPT_ROUNDS refined rounds while the effective sample size is below
    n_samples. Returns the samples (n_samples, d) and the final ESS.
    """
    g = torch.Generator().manual_seed(seed)
    U = torch.rand(n_proposal, d, generator=g, dtype=dtype)
    w, ess = _weights(-0.5 * 10.0 ** (-objective(U)))
    for r in range(ADAPT_ROUNDS):
        if ess >= n_samples:
            break
        log.info(f'    effective sample size {ess:.0f}; refining the proposal (round {r + 1})')
        U, lnq = _refined_draws(U, w, g)
        w, ess = _weights(-0.5 * 10.0 ** (-objective(U)) - lnq)
    idx = torch.multinomial(w, n_samples, replacement=True, generator=g)
    return U[idx], ess


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


def write_posterior(config: dict, D: dict, directory: str) -> pd.DataFrame:
    """Sample the posterior, log its quantiles, and write `posterior.csv`."""
    n = config['posterior_samples']
    log.info(f'Sampling {n} posterior samples from the GP surrogate')
    samples, ess = sample_posterior(
        D, config['parameters'], config['kernel'], n, config['seed'], N_PROPOSAL
    )
    log.info(f'    effective sample size: {ess:.0f}')
    if ess < n:
        log.warning(
            f'Effective sample size {ess:.0f} is below the {n} samples, so samples repeat'
        )

    q = samples.quantile([0.05, 0.5, 0.95])
    for k in samples.columns:
        log.info(f'    {k:30s} 5/50/95%: ' + '  '.join(f'{v:.4g}' for v in q[k]))

    samples.to_csv(os.path.join(directory, 'posterior.csv'), index=False)
    return samples
