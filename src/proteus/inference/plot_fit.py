"""Plots comparing the best fit with the targets and, if known, the true parameters, and
the posterior corner plot.

`collect_case_observables` reads each case's helpfile once; both best-fit plots take its frame.
"""

from __future__ import annotations

import itertools
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
import torch
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

from proteus.inference.correlation import CorrelationWhitener
from proteus.inference.failures import read_failure_records
from proteus.inference.objective import BAD_OBJ_VALUE, EPS_CLIP, eval_obj
from proteus.inference.plot import dpi, fmt
from proteus.inference.transforms import normalize_parameters, unnormalize_parameters
from proteus.inference.utils import get_obs
from proteus.utils.coupler import variable_is_logarithmic
from proteus.utils.helper import recursive_get

log = logging.getLogger('fwl.' + __name__)

# Colourblind-friendly palette (Wong).
WONG = {
    'orange': '#E69F00',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
}

# Plot standards for the observable comparison
OBS_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.axisbelow': True,
}

# Range of the simulated/target ratio axis.
RATIO_CLIP = (1e-2, 1e2)

# Posterior corner plot: fills of the 95 and 68 % regions, and KDE grid points per axis
POST_FILLS = ['#cfe3f1', '#7fb3d9']
POST_GRID = 120
PARAM_LABELS = {
    'interior_struct.core_frac': 'core radius fraction',
    'outgas.fO2_shift_IW': 'fO$_2$ (ΔIW, log$_{10}$)',
    'planet.elements.H_budget': 'H budget (ppmw)',
    'planet.elements.C_budget': 'C/H (mass ratio)',
    'planet.elements.S_budget': 'S/H (mass ratio)',
}


def collect_case_observables(
    directory,
    obs: dict,
    sigma: dict | None = None,
    correlation: dict | None = None,
    student_t_dof: float | None = None,
) -> pd.DataFrame:
    """Read the final observables and fit quality of every case on disk.

    The objective is recomputed from the stored helpfile rather than read back
    from the optimiser.

    Returns one row per readable case, with `worker`, `iter`, `case`,
    `excluded`, one column per observable, and `J`.
    """
    obs_keys = list(obs.keys())
    whitener = None if correlation is None else CorrelationWhitener(obs_keys, correlation)

    excluded = {
        (int(rec['worker']), int(rec['iter'])) for rec in read_failure_records(directory)
    }

    # Filtered to directories only
    cases = sorted(p for p in (Path(directory) / 'workers').glob('w_*/i_*') if p.is_dir())

    rows = []
    for case in cases:
        hf_path = case / 'runtime_helpfile.csv'
        if not hf_path.is_file():
            log.warning(f'Missing helpfile for {case}')
            continue
        sim = get_obs(hf_path, obs_keys)
        sim = {k: float(sim[k]) for k in obs_keys}
        w, i = int(case.parent.name.split('_')[1]), int(case.name.split('_')[1])
        rows.append(
            {
                'worker': w,
                'iter': i,
                'case': f'w{w}_i{i}',
                'excluded': (w, i) in excluded,
                **sim,
                'J': float(eval_obj(sim, obs, sigma, whitener, student_t_dof)),
            }
        )

    return pd.DataFrame(rows)


def _best_case(ok: pd.DataFrame, best_config=None) -> pd.Series:
    """Pick the row of the case the results summary reported as the best fit."""
    if best_config is not None:
        case = Path(best_config).parent
        try:
            w, i = int(case.parent.name.split('_')[1]), int(case.name.split('_')[1])
        except (IndexError, ValueError):
            log.warning(f'Cannot read case indices from {best_config}; using the best J')
        else:
            match = ok[(ok['worker'] == w) & (ok['iter'] == i)]
            if len(match):
                return match.iloc[0]
            log.warning(f'Case w_{w}/i_{i} is not among the scored cases; using the best J')
    return ok.loc[ok['J'].idxmax()]


def _panel_ratio(ax, ok: pd.DataFrame, obs: dict, fit: np.ndarray, best_case: str) -> None:
    """Draw every scored case as its ratio to the target observable.

    A target of zero or below has no meaningful ratio so such a row is left empty.
    """
    obs_keys = list(obs.keys())
    y = np.arange(len(obs_keys))[::-1]
    lo, hi = RATIO_CLIP
    # One generator for the whole panel
    rng = np.random.default_rng(42)
    drawn: set = set()

    for j, k in enumerate(obs_keys):
        with np.errstate(divide='ignore', invalid='ignore'):
            r = ok[k].to_numpy(dtype=float) / float(obs[k])
        yy = np.full(r.shape, float(y[j])) + rng.normal(0, 0.07, r.size)
        finite = np.isfinite(r)
        # Three groups per row.
        # Each group takes its legend entry from the first row to populate it.
        for sel, edge, marker, size, shade, text in (
            (finite & (r >= lo) & (r <= hi), None, 'o', 10, '0.72', 'completed cases'),
            (finite & (r < lo), lo * 1.12, '<', 26, '0.45', f'below {lo:g}x target'),
            (finite & (r > hi), hi / 1.12, '>', 26, '0.45', f'above {hi:g}x target'),
        ):
            if not sel.any():
                continue
            ax.scatter(
                r[sel] if edge is None else np.full(int(sel.sum()), edge),
                yy[sel],
                s=size,
                marker=marker,
                color=shade,
                edgecolors='none',
                zorder=2,
                label=None if text in drawn else text,
            )
            drawn.add(text)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_best = fit / np.array([obs[k] for k in obs_keys], dtype=float)

    ax.axvline(1.0, color='k', lw=1.3, zorder=3, label='target')
    ax.scatter(
        ratio_best,
        y,
        s=100,
        marker='D',
        color=WONG['vermillion'],
        edgecolors='k',
        linewidths=0.6,
        zorder=4,
        label=f'best fit ({best_case})',
    )
    ax.set_xscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(y.min() - 0.7, y.max() + 0.7)
    ax.set_yticks(y, obs_keys)
    ax.set_xlabel('simulated / target  (dimensionless)')
    ax.set_title('Final observables normalised by target', fontsize=11)
    ax.grid(axis='x', color='0.9', lw=0.6)
    ax.legend(fontsize=8, loc='upper left', framealpha=0.92)


def _panel_residual(ax, obs_keys: list[str], pct: np.ndarray, best_J: float) -> None:
    """Draw the signed percent residual of the best-fit case, one bar per row."""
    y = np.arange(len(obs_keys))[::-1]
    ax.barh(
        y,
        pct,
        color=[WONG['blue'] if p < 0 else WONG['orange'] for p in pct],
        edgecolor='k',
        linewidth=0.5,
        height=0.6,
        zorder=3,
    )
    ax.axvline(0.0, color='k', lw=1.3, zorder=4)

    # A single observable leaves no spread to scale the label offsets and the axis limits by.
    span = float(pct.max() - pct.min())
    if span <= 0.0:
        span = max(float(np.abs(pct).max()), 1.0)

    for j, p in enumerate(pct):
        ax.text(
            p + np.sign(p) * 0.025 * span,
            y[j],
            f'{p:+.1f}%',
            va='center',
            ha='left' if p >= 0 else 'right',
            fontsize=9,
            zorder=5,
        )

    ax.set_xlim(pct.min() - 0.30 * span, pct.max() + 0.22 * span)
    ax.set_ylim(y.min() - 0.7, y.max() + 0.7)
    ax.set_yticks(y, obs_keys)
    ax.set_xlabel('(best fit - target) / target  [%]')
    ax.set_title(f'Best-fit residuals, J = {best_J:+.4f}', fontsize=11)
    ax.grid(axis='x', color='0.9', lw=0.6)


def plot_result_observables(cases: pd.DataFrame, obs: dict, directory, best_config=None):
    """Plot the best-fit final observables against the target observables.

    Two panels share the observable rows: every scored case as its ratio to
    the target, and the signed percent residual of the best-fit case.

    Parameters
    ----------
    - cases (pd.DataFrame): Scored cases from `collect_case_observables`.
    - obs (dict): Observable names and target values.
    - directory (str): Base dir where the inference was performed.
    - best_config (str | Path | None): Path to the best fitting case's config
      TOML.

    Returns
    ----------
    - None
    """
    obs_keys = list(obs.keys())

    if cases.empty:
        log.warning('No case produced observables; skipping the observable comparison')
        return

    ok = cases[~cases['excluded']].copy()
    if ok.empty:
        log.warning('Every case carries the failure score; skipping the observable comparison')
        return

    best = _best_case(ok, best_config)

    tgt = np.array([obs[k] for k in obs_keys], dtype=float)
    fit = np.array([best[k] for k in obs_keys], dtype=float)

    # Same denominator convention as the printed results table
    denom = np.where(tgt >= 0, tgt + EPS_CLIP, tgt - EPS_CLIP)
    pct = 100.0 * (fit - tgt) / denom

    height = max(3.2, 0.7 * len(obs_keys) + 1.4)
    with plt.rc_context(OBS_STYLE):
        fig, axs = plt.subplots(1, 2, figsize=(12.0, height), width_ratios=[1.15, 1.0])

        _panel_ratio(axs[0], ok, obs, fit, str(best['case']))
        _panel_residual(axs[1], obs_keys, pct, float(best['J']))

        fig.suptitle(f'{Path(directory).name}: best-fit vs target observables', fontsize=12.5)
        fig.tight_layout()

    fig.savefig(
        os.path.join(directory, 'plots', f'result_observables.{fmt}'),
        dpi=dpi,
        bbox_inches='tight',
    )
    plt.close(fig)


def _position_in_range(x, key: str, bounds) -> np.ndarray:
    """Place values of parameter `key` within `bounds`, 0 at the lower and 1 at the upper.

    Log-scaled parameters are placed in log10, as the optimiser normalises them.
    """
    f = np.log10 if variable_is_logarithmic(key) else np.asarray
    lo, hi = f(float(bounds[0])), f(float(bounds[1]))
    with np.errstate(divide='ignore', invalid='ignore'):
        return (f(np.asarray(x, dtype=float)) - lo) / (hi - lo)


def _panel_position(ax, pos, truth_pos, best_pos, labels, best_case: str) -> None:
    """Draw every scored case (one column of `pos` per row), the truth and the best fit."""
    y = np.arange(len(labels))[::-1]
    jitter = np.random.default_rng(42).normal(0, 0.07, pos.shape)
    ax.scatter(
        pos,
        y + jitter,
        s=10,
        color='0.72',
        edgecolors='none',
        zorder=2,
        label='completed cases',
    )
    ax.scatter(
        truth_pos, y, s=300, marker='|', color='k', linewidths=2.2, zorder=3, label='truth'
    )
    ax.scatter(
        best_pos,
        y,
        s=100,
        marker='D',
        color=WONG['vermillion'],
        edgecolors='k',
        linewidths=0.6,
        zorder=4,
        label=f'best fit ({best_case})',
    )

    # Widen past [0, 1] only as far as an out-of-range truth needs
    ax.set_xlim(min(0.0, truth_pos.min()) - 0.04, max(1.0, truth_pos.max()) + 0.04)
    for edge in (0.0, 1.0):
        ax.axvline(edge, color='0.5', lw=0.8, ls='--', zorder=1)
    ax.set_ylim(y.min() - 0.7, y.max() + 0.7)
    ax.set_yticks(y, labels)
    ax.set_xlabel('position within sampled range  (log10 for log-scaled parameters)')
    ax.set_title('Parameters within their sampled range', fontsize=11)
    ax.grid(axis='x', color='0.9', lw=0.6)
    # Below the axis, since every row spans the full width and would be covered
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3)


def plot_result_parameters(
    cases: pd.DataFrame, pars: dict, truth: dict, directory, best_config=None
):
    """Plot the best-fit parameters against the true parameters, if known.

    Parameters
    ----------
    - cases (pd.DataFrame): Scored cases from `collect_case_observables`.
    - pars (dict): Parameter names and bounds.
    - truth (dict): True value of each parameter.
    - directory (str): Base dir where the inference was performed.
    - best_config (str | Path | None): Path to the best fitting case's config TOML.

    Returns
    ----------
    - None
    """
    par_keys = list(pars.keys())

    ok = cases[~cases['excluded']].copy() if not cases.empty else cases
    if ok.empty:
        log.warning('No scored case to compare with the true parameters; skipping')
        return

    # Parameter values each case actually ran with
    workers = Path(directory) / 'workers'
    confs = [
        toml.load(workers / f'w_{w}' / f'i_{i}' / 'init_coupler.toml')
        for w, i in zip(ok['worker'], ok['iter'])
    ]
    for k in par_keys:
        ok[k] = [float(recursive_get(c, k.split('.'))) for c in confs]
    best = _best_case(ok, best_config)

    pos = np.column_stack([_position_in_range(ok[k], k, pars[k]) for k in par_keys])
    truth_pos = np.array([_position_in_range(truth[k], k, pars[k]) for k in par_keys])
    best_pos = np.array([_position_in_range(best[k], k, pars[k]) for k in par_keys])
    labels = [
        f'{k}\n[{pars[k][0]:g}, {pars[k][1]:g}]'
        + ('  log' if variable_is_logarithmic(k) else '')
        for k in par_keys
    ]

    with plt.rc_context(OBS_STYLE):
        fig, axs = plt.subplots(
            1, 2, figsize=(12.0, max(3.2, 0.8 * len(par_keys) + 1.4)), width_ratios=[1.15, 1.0]
        )
        _panel_position(axs[0], pos, truth_pos, best_pos, labels, str(best['case']))
        _panel_residual(axs[1], labels, 100.0 * (best_pos - truth_pos), float(best['J']))
        axs[1].set_xlabel('(best fit - truth) / sampled range  [%]')
        axs[1].set_title(f'Best-fit parameter error, J = {best["J"]:+.4f}', fontsize=11)
        fig.suptitle(f'{Path(directory).name}: best-fit vs true parameters', fontsize=12.5)
        fig.tight_layout()

    fig.savefig(
        os.path.join(directory, 'plots', f'result_parameters.{fmt}'),
        dpi=dpi,
        bbox_inches='tight',
    )
    plt.close(fig)


def _reflected_kde(U: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Gaussian KDE of unit-cube samples U (n, k) at points G (m, k), reflected at 0 and 1
    in every dimension so that mass against a bound stays inside the cube."""
    kde = gaussian_kde(U.T)
    maps = (lambda x: x, lambda x: -x, lambda x: 2 - x)
    return sum(
        kde(np.column_stack([f(G[:, i]) for i, f in enumerate(fs)]).T)
        for fs in itertools.product(maps, repeat=U.shape[1])
    )


def _mass_levels(Z: np.ndarray, masses=(0.95, 0.68)) -> list[float]:
    """Density thresholds that enclose the given probability masses of a gridded density."""
    z = np.sort(Z.ravel())[::-1]
    c = np.cumsum(z) / z.sum()
    return [z[np.searchsorted(c, m)] for m in masses]


def _padded(bounds, log_axis: bool, frac: float = 0.03) -> tuple[float, float]:
    """Axis limits a few percent beyond the bounds, so points on a bound stay visible."""
    lo, hi = bounds
    if log_axis:
        f = (hi / lo) ** frac
        return lo / f, hi * f
    return lo - frac * (hi - lo), hi + frac * (hi - lo)


def _posterior_axes(ax, i: int, j: int, keys: list, pars: dict):
    """Limits a little past the bounds (dotted), scales, ticks and labels of one corner panel."""
    n = len(keys)
    for k, lim, scale, axline, axis in (
        (keys[j], ax.set_xlim, ax.set_xscale, ax.axvline, ax.xaxis),
        (keys[i], ax.set_ylim, ax.set_yscale, ax.axhline, ax.yaxis),
    ):
        if axis is ax.yaxis and i == j:
            continue
        log_axis = variable_is_logarithmic(k)
        lim(*_padded(pars[k], log_axis))
        for v in pars[k]:
            axline(v, color='0.75', lw=0.5, ls=':', zorder=0)
        if log_axis:
            scale('log')
        else:
            axis.set_major_locator(MaxNLocator(4, prune='both'))
    if i == n - 1:
        ax.set_xlabel(PARAM_LABELS.get(keys[j], keys[j]))
    else:
        ax.tick_params(labelbottom=False)
    if j == 0 and i > 0:
        ax.set_ylabel(PARAM_LABELS.get(keys[i], keys[i]))
    elif i != j:
        ax.tick_params(labelleft=False)


def _posterior_legend(fig, n_init: int, n_bo: int, chi2_best: float, truth: bool):
    """Legend of the posterior corner plot."""
    blue = WONG['blue']
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=POST_FILLS[1], ec=blue, label='posterior, 68 % region'),
        plt.Rectangle(
            (0, 0), 1, 1, fc=POST_FILLS[0], ec=blue, lw=0.6, label='posterior, 95 % region'
        ),
        plt.Line2D(
            [],
            [],
            marker='s',
            ls='',
            mfc='none',
            mec='0.55',
            label=f'initial samples ({n_init})',
        ),
        plt.Line2D(
            [], [], marker='o', ls='', ms=3, color='0.15', alpha=0.6, label=f'BO steps ({n_bo})'
        ),
        plt.Line2D(
            [],
            [],
            marker='D',
            ls='',
            ms=6,
            mfc='white',
            mec='k',
            label=f'best fit (χ² = {chi2_best:.3g}), dotted on diagonal',
        ),
    ]
    if truth:
        handles.append(
            plt.Line2D(
                [],
                [],
                marker='*',
                ls='',
                ms=12,
                color=WONG['vermillion'],
                mec='k',
                label='truth',
            )
        )
    fig.legend(
        handles=handles,
        loc='upper right',
        bbox_to_anchor=(0.95, 0.93),
        frameon=False,
        fontsize=10,
    )


def plot_result_posterior(
    samples: pd.DataFrame, D: dict, n_init: int, pars: dict, truth: dict | None, directory
):
    """Corner plot of the posterior as 68 and 95 % credible regions, with every scored
    evaluation, the best fit and, if known, the truth drawn on top.

    Parameters
    ----------
    - samples (pd.DataFrame): Posterior samples in config units, one column per parameter.
    - D (dict): Study data, 'X' in the unit cube and 'Y'; the first n_init rows are initial.
    - n_init (int): Number of initial samples.
    - pars (dict): Parameter names and bounds.
    - truth (dict | None): True value of each parameter.
    - directory (str): Base dir where the inference was performed.
    """
    keys = list(pars)
    n = len(keys)
    bounds = torch.tensor([pars[k] for k in keys], dtype=torch.double).T
    Up = normalize_parameters(torch.tensor(samples[keys].to_numpy()), bounds, keys).numpy()
    X = unnormalize_parameters(D['X'], bounds, keys).numpy()
    y = D['Y'].numpy().ravel()
    scored = y > BAD_OBJ_VALUE
    init = np.arange(len(y)) < n_init
    best = X[np.argmax(y)]
    g = np.linspace(0, 1, POST_GRID)
    grid = unnormalize_parameters(
        torch.tensor(np.tile(g[:, None], (1, n))), bounds, keys
    ).numpy()
    tv = [truth[k] for k in keys] if truth is not None else None
    red = WONG['vermillion']

    with plt.rc_context(OBS_STYLE | {'font.size': 9}):
        fig, axs = plt.subplots(n, n, figsize=(11, 11), squeeze=False)
        for i, j in itertools.product(range(n), range(n)):
            ax = axs[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                dens = _reflected_kde(Up[:, [j]], g[:, None])
                ax.fill_between(grid[:, j], dens, color=POST_FILLS[1], lw=0)
                ax.plot(grid[:, j], dens, color=WONG['blue'], lw=1.2)
                ax.axvline(best[j], color='k', lw=0.8, ls=':')
                if tv is not None:
                    ax.axvline(tv[j], color=red, lw=1.2)
                ax.set_ylim(0, dens.max() * 1.1)
                ax.set_yticks([])
            else:
                G0, G1 = np.meshgrid(g, g)
                Z = _reflected_kde(Up[:, [j, i]], np.column_stack([G0.ravel(), G1.ravel()]))
                Z = Z.reshape(G0.shape)
                l95, l68 = _mass_levels(Z)
                RX, RY = np.meshgrid(grid[:, j], grid[:, i])
                ax.contourf(
                    RX, RY, Z, levels=[l95, l68, Z.max() * 1.01], colors=POST_FILLS, zorder=1
                )
                ax.contour(
                    RX,
                    RY,
                    Z,
                    levels=[l95, l68],
                    colors=WONG['blue'],
                    linewidths=[0.6, 0.9],
                    zorder=2,
                )
                s = scored & init
                ax.scatter(
                    X[s, j], X[s, i], s=10, marker='s', fc='none', ec='0.55', lw=0.5, zorder=3
                )
                s = scored & ~init
                ax.scatter(X[s, j], X[s, i], s=4, color='0.15', alpha=0.45, lw=0, zorder=3)
                if tv is not None:
                    ax.axvline(tv[j], color=red, lw=0.7, ls='--', zorder=4)
                    ax.axhline(tv[i], color=red, lw=0.7, ls='--', zorder=4)
                    ax.scatter(
                        tv[j], tv[i], marker='*', s=150, color=red, ec='k', lw=0.6, zorder=6
                    )
                ax.scatter(
                    best[j], best[i], marker='D', s=30, fc='white', ec='k', lw=0.9, zorder=7
                )
            _posterior_axes(ax, i, j, keys, pars)

        _posterior_legend(fig, n_init, len(y) - n_init, 10 ** -y.max(), tv is not None)
        fig.subplots_adjust(
            left=0.08, right=0.97, bottom=0.07, top=0.97, wspace=0.07, hspace=0.07
        )
        fig.savefig(os.path.join(directory, 'plots', f'result_posterior.{fmt}'), dpi=dpi)
        plt.close(fig)
