# Status-summary and ECDF grid plots for a post-processed PROTEUS grid.
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proteus.utils.plot import get_colourmap, get_label, get_log_scale, get_scale

log = logging.getLogger('fwl.' + __name__)

# Wong colour-blind-safe qualitative palette (Wong 2011, Nat. Methods 8, 441),
# used for the categorical status bars.
WONG_PALETTE = (
    '#000000',  # black
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#F0E442',  # yellow
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#CC79A7',  # reddish purple
)


def _apply_axis_style(ax) -> None:
    """Apply the PROTEUS tick style: inward ticks on all four sides."""
    ax.tick_params(direction='in', which='both', top=True, right=True)


def plot_status_summary(
    df: pd.DataFrame, out_dir: str | Path, grid_name: str, plot_format: str = 'png'
) -> Path:
    """Plot the number of grid cases per final status.

    Parameters
    ----------
    df : pandas.DataFrame
        The 'all' summary CSV; must contain a ``status`` column.
    out_dir : str or Path
        Directory to write the plot into.
    grid_name : str
        Grid name, used in the plot title.
    plot_format : str
        Image format extension (for example 'png' or 'pdf').

    Returns
    -------
    pathlib.Path
        The path written.
    """
    if 'status' not in df.columns:
        raise ValueError("Summary CSV must contain a 'status' column")

    counts = df['status'].astype(str).value_counts().sort_values(ascending=False)
    total = int(counts.sum())
    labels = [s.replace(' (', '\n(') for s in counts.index]
    colours = [WONG_PALETTE[i % len(WONG_PALETTE)] for i in range(len(counts))]

    fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * len(counts)), 6.0))
    bars = ax.bar(labels, counts.values, color=colours, edgecolor='black')

    for bar, count in zip(bars, counts.values):
        pct = 100.0 * count / total if total else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count,
            f'{count} ({pct:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=11,
        )

    ax.set_title(f'Status summary for grid {grid_name} ({total} cases)')
    ax.set_xlabel('Final simulation status')
    ax.set_ylabel('Number of simulations')
    ax.margins(y=0.12)
    ax.grid(alpha=0.2, axis='y')
    _apply_axis_style(ax)
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'summary_grid_statuses.{plot_format}'
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    log.info('Wrote status summary plot to %s', out_file.name)
    return out_file


def clean_series(series: pd.Series, log_axis: bool) -> pd.Series:
    """Drop NaN/inf; on a log axis also drop non-positive values, logging the count."""
    cleaned = series.replace([np.inf, -np.inf], np.nan).dropna()
    if log_axis:
        n_before = len(cleaned)
        cleaned = cleaned[cleaned > 0]
        dropped = n_before - len(cleaned)
        if dropped:
            log.warning('Dropped %d non-positive value(s) before a log-axis ECDF', dropped)
    return cleaned


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) of the empirical CDF of ``values`` as a percentage in [0, 100].

    The step function starts at 0 % just below the smallest value, so the curve is
    drawn with ``drawstyle='steps-post'``.
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n == 0:
        return np.array([]), np.array([])
    y = np.arange(1, n + 1) / n * 100.0
    return np.concatenate([x[:1], x]), np.concatenate([[0.0], y])


def _value_colours(values, log_norm: bool, cmap):
    """Return (colour_of, is_numeric, norm) for colouring one parameter's values.

    For numeric values a continuous norm drives a colourbar; for string values a
    discrete colour per value drives a legend (norm is None).
    """
    is_numeric = np.issubdtype(np.asarray(values).dtype, np.number)
    if is_numeric:
        vmin, vmax = float(min(values)), float(max(values))
        if vmin == vmax:
            vmin, vmax = vmin - 1e-9, vmax + 1e-9
        norm = (
            mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
            if log_norm and vmin > 0
            else mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        )
        return (lambda v: cmap(norm(v))), True, norm

    uniques = sorted({str(v) for v in values})
    discrete = cmap.resampled(max(len(uniques), 1))
    lookup = {val: discrete(i) for i, val in enumerate(uniques)}
    return (lambda v: lookup[str(v)]), False, None


def _plot_cell(ax, df, param, value, output, colour, log_axis):
    """Draw one ECDF curve (cases with ``param == value``) on a cell, return success."""
    subset = df[df[param] == value]
    series = clean_series(subset[output] * get_scale(output), log_axis)
    if series.empty:
        return False
    x, y = ecdf(series.to_numpy())
    ax.plot(x, y, drawstyle='steps-post', color=colour, linewidth=2.0)
    return True


def plot_ecdf_grid(
    df: pd.DataFrame,
    tested_params: dict,
    outputs: list[str],
    out_dir: str | Path,
    grid_name: str,
    colourmap: str | None = None,
    plot_format: str = 'png',
) -> Path:
    """Plot a grid of ECDF panels: one row per tested parameter, one column per output.

    Within a panel, one ECDF curve is drawn per tested value of the row parameter,
    coloured by that value (a colourbar for numeric parameters, a legend for string
    parameters).

    Parameters
    ----------
    df : pandas.DataFrame
        The 'completed' summary CSV.
    tested_params : dict
        Mapping of parameter path to its swept values.
    outputs : list of str
        Output columns to plot (one per column), already validated as present.
    out_dir : str or Path
        Directory to write the plot into.
    grid_name : str
        Grid name, used in the output filename.
    colourmap : str or None
        Colour map name for numeric parameter values (scientific map via
        :func:`proteus.utils.plot.get_colourmap`).
    plot_format : str
        Image format extension.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    params = [p for p in tested_params if p in df.columns]
    skipped = [p for p in tested_params if p not in df.columns]
    for p in skipped:
        log.warning("Parameter '%s' absent from summary data; skipping its ECDF row", p)
    if not params or not outputs:
        raise ValueError('Need at least one tested parameter and one output to plot')

    cmap = get_colourmap(colourmap)
    n_rows, n_cols = len(params), len(outputs)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.6 * n_cols, 2.6 * n_rows),
        squeeze=False,
        sharey=True,
    )

    for i, param in enumerate(params):
        values = tested_params[param]
        colour_of, is_numeric, norm = _value_colours(values, get_log_scale(param), cmap)

        for j, output in enumerate(outputs):
            ax = axes[i][j]
            for value in values:
                if value in df[param].values:
                    _plot_cell(
                        ax, df, param, value, output, colour_of(value), get_log_scale(output)
                    )
            if get_log_scale(output):
                ax.set_xscale('log')
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.3)
            _apply_axis_style(ax)
            if i == n_rows - 1:
                ax.set_xlabel(get_label(output))
            if j == 0:
                ax.set_ylabel('ECDF [%]')

        _add_row_key(fig, axes[i][-1], param, values, is_numeric, norm, cmap, colour_of)

    fig.suptitle(f'ECDF summary for grid {grid_name}')
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'ecdf_grid_plot.{plot_format}'
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    log.info('Wrote ECDF grid plot to %s', out_file.name)
    return out_file


def _add_row_key(fig, ax, param, values, is_numeric, norm, cmap, colour_of) -> None:
    """Attach a colourbar (numeric) or legend (string) labelling a parameter row."""
    if is_numeric:
        mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(mappable, ax=ax, pad=0.02, aspect=12)
        cbar.set_label(get_label(param))
        cbar.set_ticks(sorted(set(values)))
    else:
        handles = [
            mpl.lines.Line2D([0], [0], color=colour_of(v), lw=2.0, label=str(v))
            for v in sorted({str(v) for v in values})
        ]
        ax.legend(
            handles=handles,
            title=get_label(param),
            fontsize=8,
            bbox_to_anchor=(1.01, 1.0),
            loc='upper left',
        )
