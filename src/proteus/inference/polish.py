"""Local refinement of the best-fitting islands after the Bayesian optimisation.

An island starts at a scored evaluation within DELTA_CHI2 of the best chi-squared and
below MAX_CHI2, at least ISLAND_SEPARATION from the starts already chosen. Each island
gets a bounded Nelder-Mead search from its start with a fixed budget of PROTEUS runs, so
the islands are compared at their own minima rather than where the optimiser left them.
"""

from __future__ import annotations

import logging
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from proteus.inference.objective import close_worker_runner, prot_builder

log = logging.getLogger('fwl.' + __name__)

DELTA_CHI2 = 10.0  # likelihood ratio exp(-5) to the best fit
MAX_CHI2 = 15.0  # about the 95% limit of chi-squared with 7 degrees of freedom
ISLAND_SEPARATION = 0.15  # between island starts, in the unit cube
MAX_ISLANDS = 3
SIMPLEX_STEP = 0.05  # first Nelder-Mead step, in the unit cube


def validate_polish_runs(n, sigma: dict | None) -> int:
    """PROTEUS runs per island; None or 0 turns the polish off. Needs [sigma]."""
    if n is None:
        return 0
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f'polish_runs must be a non-negative integer, got {n!r}')
    if n > 0 and sigma is None:
        raise ValueError('polish_runs needs [sigma]: islands are chosen by chi-squared')
    return n


def find_islands(X: np.ndarray, y: np.ndarray) -> list[int]:
    """Indices of the island starts, best first; y is the objective -log10(chi2)."""
    chi2 = 10.0 ** (-y)
    limit = min(chi2.min() + DELTA_CHI2, MAX_CHI2)
    starts = []
    for i in np.argsort(chi2):
        if chi2[i] > limit or len(starts) == MAX_ISLANDS:
            break
        if all(np.linalg.norm(X[i] - X[j]) >= ISLAND_SEPARATION for j in starts):
            starts.append(int(i))
    return starts


def polish_island(k: int, x0: np.ndarray, y0: float, n_runs: int, builder_args: dict) -> list:
    """Nelder-Mead from x0 in the unit cube, maximising y over at most n_runs PROTEUS runs.

    Runs go to workers/w_{-2 - k}. Returns the (x, y) of each run.
    """
    runs = []

    def neg_y(x):
        if np.array_equal(x, x0):
            return -y0  # the start is already evaluated
        if len(runs) == n_runs:
            return np.inf  # budget spent; Nelder-Mead may overrun maxfev within a step
        f = prot_builder(**builder_args, worker=-2 - k, iter=len(runs))
        y = float(f(torch.tensor(x[None, :], dtype=torch.double)))
        runs.append((x.copy(), y))
        return -y

    simplex = [x0]
    for i in range(len(x0)):
        v = x0.copy()
        v[i] += SIMPLEX_STEP if x0[i] + SIMPLEX_STEP <= 1 else -SIMPLEX_STEP
        simplex.append(v)
    try:
        minimize(
            neg_y,
            x0,
            method='Nelder-Mead',
            bounds=[(0.0, 1.0)] * len(x0),
            options={'maxfev': n_runs + 1, 'initial_simplex': np.array(simplex)},
        )
    finally:
        close_worker_runner()
    return runs


def polish(D: dict, n_runs: int, builder_args: dict, directory: str) -> dict:
    """Polish every island in parallel, write `polish.csv`, and return D with the polish runs."""
    X, y = D['X'].numpy(), D['Y'].numpy().reshape(-1)
    starts = find_islands(X, y)
    if not starts:
        log.info(f'No evaluation with chi2 below {MAX_CHI2}; nothing to polish')
        return D
    log.info(f'Polishing {len(starts)} island(s) with {n_runs} PROTEUS runs each')
    with Pool(len(starts)) as pool:
        results = pool.starmap(
            polish_island, [(k, X[i], y[i], n_runs, builder_args) for k, i in enumerate(starts)]
        )

    rows = []
    for k, (i, runs) in enumerate(zip(starts, results)):
        best = max([y[i]] + [yr for _, yr in runs])
        log.info(
            f'    island {k} (workers/w_{-2 - k}): chi2 {10 ** -y[i]:.3g} -> {10**-best:.3g}'
        )
        rows += [
            {'island': k, **{f'x_{j}': v for j, v in enumerate(x)}, 'y': yr} for x, yr in runs
        ]
    if not rows:
        return D
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(directory, 'polish.csv'), index=False)
    Xp = torch.tensor(P[[f'x_{j}' for j in range(X.shape[1])]].to_numpy(), dtype=D['X'].dtype)
    Yp = torch.tensor(P[['y']].to_numpy(), dtype=D['Y'].dtype)
    return {'X': torch.cat([D['X'], Xp]), 'Y': torch.cat([D['Y'], Yp])}
