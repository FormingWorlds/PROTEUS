# Status-summary and ECDF grid plots for a post-processed PROTEUS grid.
from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from proteus.utils.plot import get_colourmap, get_label, get_log_scale, get_scale

log = logging.getLogger('fwl.' + __name__)

# Image formats the grid plots may be saved as.
SUPPORTED_PLOT_FORMATS = ('png', 'pdf')

# Status-bar colours, fixed per status category so a category keeps its colour from
# grid to grid. Wong colour-blind-safe palette (Wong 2011, Nat. Methods 8, 441),
# with neutral greys for cases whose outcome is unknown or that never started.
CATEGORY_COLOURS = {
    'Completed': '#009E73',  # bluish green
    'Running': '#56B4E9',  # sky blue
    'Error': '#D55E00',  # vermillion
    'Unknown': '#999999',  # grey
    'Not started': '#DDDDDD',  # light grey
}
# Bar order of the status categories, left to right.
CATEGORY_ORDER = tuple(CATEGORY_COLOURS)

# ECDF curve colours must stand out on white: the lighter end of a colour map is
# trimmed until its colour has a contrast of at least 2:1 against white (relative
# luminance <= 0.475), trimming at most MAX_COLOURMAP_TRIM of the map.
MAX_CURVE_LUMINANCE = 0.475
MAX_COLOURMAP_TRIM = 0.35


def _luminance(rgba) -> float:
    """Return the relative luminance (sRGB, Rec. 709 weights) of an RGBA colour."""
    rgb = np.asarray(rgba[:3], dtype=float)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(linear @ np.array([0.2126, 0.7152, 0.0722]))


def _trim_colourmap(cmap):
    """Return ``cmap`` without its too-light end, so every curve colour is visible.

    The lighter of the two ends is cut back to the first colour with luminance
    <= MAX_CURVE_LUMINANCE (at most MAX_COLOURMAP_TRIM of the map); a map whose
    ends are both dark enough is returned unchanged.
    """
    light_top = _luminance(cmap(1.0)) >= _luminance(cmap(0.0))
    steps = np.linspace(0.0, MAX_COLOURMAP_TRIM, 71)
    trim = next(
        (
            t
            for t in steps
            if _luminance(cmap(1.0 - t if light_top else t)) <= MAX_CURVE_LUMINANCE
        ),
        MAX_COLOURMAP_TRIM,
    )
    base = np.linspace(0.0, 1.0 - trim, 256) if light_top else np.linspace(trim, 1.0, 256)
    return mpl.colors.LinearSegmentedColormap.from_list(f'{cmap.name}_trim', cmap(base))


def _apply_axis_style(ax) -> None:
    """Apply the PROTEUS tick style: inward ticks on all four sides."""
    ax.tick_params(direction='in', which='both', top=True, right=True)


def status_counts(df: pd.DataFrame, n_not_started: int = 0) -> pd.DataFrame:
    """Count cases per final status, ordered by category then by count.

    Parameters
    ----------
    df : pandas.DataFrame
        The 'all' summary CSV, with ``status`` and ``status_category`` columns
        (a CSV without ``status_category`` counts every status as 'Unknown').
    n_not_started : int
        Number of grid cases with no case folder yet, added as a 'Not started' row.

    Returns
    -------
    pandas.DataFrame
        Columns ``status``, ``category`` and ``count``, one row per status.
    """
    if 'status' not in df.columns:
        raise ValueError("Summary CSV must contain a 'status' column")
    category = df.get('status_category', pd.Series('Unknown', index=df.index))
    counts = (
        pd.DataFrame({'status': df['status'].astype(str), 'category': category.astype(str)})
        .value_counts()
        .rename('count')
        .reset_index()
    )
    if n_not_started > 0:
        extra = {'status': 'Not started', 'category': 'Not started', 'count': n_not_started}
        counts = pd.concat([counts, pd.DataFrame([extra])], ignore_index=True)
    rank = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    counts['_rank'] = counts['category'].map(rank).fillna(len(rank))
    counts = counts.sort_values(['_rank', 'count'], ascending=[True, False])
    return counts.drop(columns='_rank').reset_index(drop=True)


def plot_status_summary(
    df: pd.DataFrame,
    out_dir: str | Path,
    grid_name: str,
    plot_format: str = 'png',
    n_not_started: int = 0,
) -> Path:
    """Plot the number of grid cases per final status, coloured by status category.

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
    n_not_started : int
        Number of grid cases with no case folder yet (shown as a 'Not started' bar).

    Returns
    -------
    pathlib.Path
        The path written.
    """
    counts = status_counts(df, n_not_started)
    total = int(counts['count'].sum())
    labels = [s.replace(' (', '\n(') for s in counts['status']]
    colours = [CATEGORY_COLOURS.get(c, CATEGORY_COLOURS['Unknown']) for c in counts['category']]

    fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * len(counts)), 6.0))
    bars = ax.bar(labels, counts['count'], color=colours, edgecolor='black')

    for bar, count in zip(bars, counts['count']):
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
    try:
        fig.savefig(out_file, dpi=200, bbox_inches='tight')
    finally:
        plt.close(fig)
    log.info('Wrote status summary plot to %s', out_file.name)
    return out_file


def clean_series(series: pd.Series) -> pd.Series:
    """Drop NaN and infinite values (a case without a finite value has no step)."""
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def ecdf(values: np.ndarray, n_total: int | None = None):
    """Return (x, y) of the empirical CDF of ``values`` as a percentage of ``n_total``.

    ``n_total`` (default ``len(values)``) counts every case the percentage refers
    to; cases without a value (for example never solidified) have no step, so the
    curve levels off at the fraction that have one. Draw with ``steps-post``.
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n_total is None:
        n_total = n
    if n == 0 or n_total <= 0:
        return np.array([]), np.array([])
    y = np.arange(1, n + 1) / n_total * 100.0
    return np.concatenate([x[:1], x]), np.concatenate([[0.0], y])


def axis_scale(values: pd.Series, preset_log: bool) -> tuple[str, float | None]:
    """Return the x-axis scale ``(name, linthresh)`` for one output.

    A log-preset quantity gets ``'log'`` if all values are positive and span a
    decade or more, ``'symlog'`` (a 0 + log axis, linear up to the decade below the
    smallest positive value) if some are <= 0, else ``'linear'`` (a narrower log
    axis adds nothing and its minor-tick labels overlap). Others are ``'linear'``.
    """
    if not preset_log:
        return 'linear', None
    finite = clean_series(values)
    positive = finite[finite > 0]
    if positive.empty:
        return 'linear', None
    if (finite <= 0).any():
        return 'symlog', float(10.0 ** np.floor(np.log10(positive.min())))
    if positive.max() / positive.min() >= 10.0:
        return 'log', None
    return 'linear', None


def _set_axis_scale(ax, scale: str, linthresh: float | None, values: pd.Series) -> None:
    """Apply an :func:`axis_scale` result; a 0 + log axis starts at 0 if nothing is negative."""
    if scale == 'log':
        ax.set_xscale('log')
    elif scale == 'symlog':
        ax.set_xscale('symlog', linthresh=linthresh, linscale=0.5)
        if clean_series(values).min() >= 0:
            # Start just below 0 so a step at 0 is not hidden under the axis frame.
            ax.set_xlim(left=-0.2 * linthresh)


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
    # Grey fallback for a value not among the tested set (should not happen).
    return (lambda v: lookup.get(str(v), (0.5, 0.5, 0.5, 1.0))), False, None


def _plot_cell(ax, df, param, value, output, colour):
    """Draw one ECDF curve (cases with ``param == value``) on a cell, return success."""
    subset = df[df[param] == value]
    series = clean_series(subset[output] * get_scale(output))
    if series.empty:
        return False
    # Normalise by every case in the subset, so cases without a value (for example
    # never solidified) keep the curve below 100 %.
    x, y = ecdf(series.to_numpy(), n_total=len(subset))
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
    parameters). Each curve is a percentage of all cases with that value, so it
    levels off below 100 % when some cases have no value for the output.

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

    # Trim the palest end of the colour map so a top-of-range curve stays visible.
    cmap = _trim_colourmap(get_colourmap(colourmap))
    n_rows, n_cols = len(params), len(outputs)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.6 * n_cols, 2.6 * n_rows),
        squeeze=False,
        sharey=True,
    )

    scaled = {out: df[out] * get_scale(out) for out in outputs}
    scales = {out: axis_scale(scaled[out], get_log_scale(out)) for out in outputs}
    row_keys = []
    for i, param in enumerate(params):
        values = tested_params[param]
        colour_of, is_numeric, norm = _value_colours(values, get_log_scale(param), cmap)

        # Iterate the values actually present in the column (not the in-memory
        # swept list) so a float that changed on the CSV round-trip still matches.
        present_values = sorted(df[param].dropna().unique())

        for j, output in enumerate(outputs):
            ax = axes[i][j]
            for value in present_values:
                _plot_cell(ax, df, param, value, output, colour_of(value))
            _set_axis_scale(ax, *scales[output], scaled[output])
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.3)
            _apply_axis_style(ax)
            if i == n_rows - 1:
                ax.set_xlabel(get_label(output))

        row_keys.append((axes[i][-1], param, values, is_numeric, norm, colour_of))

    fig.suptitle(f'ECDF summary for grid {grid_name}')
    fig.supylabel('Empirical cumulative fraction of simulations [%]', x=-0.005)
    fig.tight_layout()

    # Add the colourbar or legend AFTER the layout, in an axes anchored outside the
    # rightmost panel, so every data panel keeps an identical size (a colourbar added
    # with ax=... would shrink only the panel it sits beside).
    for ax, param, values, is_numeric, norm, colour_of in row_keys:
        _add_row_key(fig, ax, param, values, is_numeric, norm, cmap, colour_of)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'ecdf_grid_plot.{plot_format}'
    try:
        fig.savefig(out_file, dpi=200, bbox_inches='tight')
    finally:
        plt.close(fig)
    log.info('Wrote ECDF grid plot to %s', out_file.name)
    return out_file


def _add_row_key(fig, ax, param, values, is_numeric, norm, cmap, colour_of) -> None:
    """Attach a colourbar (numeric) or legend (string) labelling a parameter row.

    The key sits in an axes anchored just outside ``ax`` (not stolen from it), so
    the data panels all keep the same size whether a row uses a colourbar or a legend.
    """
    if is_numeric:
        cax = inset_axes(
            ax,
            width='5%',
            height='100%',
            loc='center left',
            bbox_to_anchor=(1.03, 0.0, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(mappable, cax=cax)
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
            bbox_to_anchor=(1.03, 1.0),
            loc='upper left',
            borderaxespad=0.0,
        )


def validate_output_variables(df: pd.DataFrame, requested_outputs: list) -> list:
    """Return the requested outputs present in ``df``; warn (never substitute) for missing.

    Parameters
    ----------
    df : pandas.DataFrame
        The summary CSV loaded for plotting.
    requested_outputs : list
        Output variable names requested in the grid config.
    """
    available = list(df.columns)
    present = [v for v in requested_outputs if v in available]
    missing = [v for v in requested_outputs if v not in available]
    if missing:
        log.warning(
            'Output variable(s) %s not found in summary data; skipping them. '
            'Available columns: %s',
            missing,
            available,
        )
    return present


def resolve_plot_format(requested: str | None, base_fmt: str | None = None) -> str:
    """Return a validated plot format for the grid plots.

    Uses ``[post_processing].plot_format`` when set, else ``base_fmt`` (the base
    run's ``params.out.plot_fmt`` from ref_config.toml), else 'png'. Raises on an unsupported value rather than writing a ``.None`` file.
    """
    fmt = requested if requested is not None else base_fmt
    fmt = (fmt or 'png').lower()
    if fmt not in SUPPORTED_PLOT_FORMATS:
        raise ValueError(
            f"Unsupported plot_format '{fmt}'; choose one of {SUPPORTED_PLOT_FORMATS}"
        )
    return fmt


def count_not_started(df_all: pd.DataFrame, tested_params: dict) -> int:
    """Return how many grid cases have no case folder (not started yet).

    The grid has one case per combination of swept values, numbered from 0, so the
    expected size is the product of the axis lengths.
    """
    expected = math.prod(len(v) for v in tested_params.values())
    present = set(df_all['case_number']) & set(range(expected))
    missing = expected - len(present)
    if missing > 0:
        log.warning('%d of %d grid cases have no case folder (not started)', missing, expected)
    return max(missing, 0)


def make_grid_plots(opts, csv_paths, tested_params, grid_dir, base_fmt=None) -> None:
    """Draw the status summary and ECDF grid plots requested in ``opts``.

    Parameters
    ----------
    opts : dict
        Validated ``[post_processing]`` options (``plot_status``, ``plot_ecdf``,
        ``output_variables``, ``colormap``, ``plot_format``).
    csv_paths : dict
        Paths of the 'all' and 'completed' summary CSVs.
    tested_params : dict
        Mapping of parameter path to its swept values.
    grid_dir : Path
        Grid output directory; plots go to ``post_processing/grid_plots`` inside it.
    base_fmt : str or None
        The base run's ``params.out.plot_fmt``, used when ``plot_format`` is unset.
    """
    grid_name = Path(grid_dir).name
    plots_dir = Path(grid_dir) / 'post_processing' / 'grid_plots'
    plot_format = resolve_plot_format(opts['plot_format'], base_fmt)

    if opts['plot_status']:
        df_all = pd.read_csv(csv_paths['all'], sep='\t')
        n_not_started = count_not_started(df_all, tested_params)
        plot_status_summary(df_all, plots_dir, grid_name, plot_format, n_not_started)

    if not opts['plot_ecdf']:
        return
    df_completed = pd.read_csv(csv_paths['completed'], sep='\t')
    if df_completed.empty:
        log.warning('No completed cases; skipping the ECDF plot')
        return
    outputs = validate_output_variables(df_completed, opts['output_variables'])
    if not outputs:
        raise ValueError(
            'No valid output_variables for the ECDF plot. Set output_variables in the '
            '[post_processing] table to columns present in the summary CSV.'
        )
    plot_ecdf_grid(
        df_completed,
        tested_params,
        outputs,
        plots_dir,
        grid_name,
        colourmap=opts['colormap'],
        plot_format=plot_format,
    )
