# Status-summary and ECDF grid plots for a post-processed PROTEUS grid.
from __future__ import annotations

import colorsys
import logging
import math
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from proteus.grid.analyse_options import SUPPORTED_PLOT_FORMATS
from proteus.utils.plot import (
    FADE_MID_GREY,
    get_colourmap,
    get_label,
    get_log_scale,
    get_scale,
    style_neutral,
)

log = logging.getLogger('fwl.' + __name__)

# Hue [degrees] each status category looks for in the colour cycle of the active style,
# so Completed stays green, Running blue and Error red in any style.
CATEGORY_HUES = {
    'Completed': 120.0,
    'Running': 210.0,
    'Error': 0.0,
}
# Bar order of the status categories, top to bottom.
CATEGORY_ORDER = (*CATEGORY_HUES, 'Unknown', 'Not started')
# A cycle colour below this HSV saturation or value is a neutral, not a hue.
MIN_HUE_SATURATION = 0.25
MIN_HUE_VALUE = 0.25
# Grey of the 'Not started' bar, between the text (0) and face (1) colour of the style.
FADE_LIGHT_GREY = 0.75

# An ECDF curve colour keeps a contrast of at least 2:1 against the axes face colour
# of the style: each end of a colour map is trimmed (at most MAX_COLOURMAP_TRIM of
# the map) until it does.
MIN_CURVE_CONTRAST = 2.0
MAX_COLOURMAP_TRIM = 0.35

# Line width [pt] of the ECDF curves and of their legend entries.
ECDF_LINEWIDTH = 3.0

# Least ECDF figure height [inch per pt of font size]: the 48-character shared y label
# (about 0.5 em per character) fits beside one row.
MIN_ECDF_HEIGHT_PER_PT = 0.4

# Most labelled ticks on a colourbar; with more swept values every k-th one is labelled.
MAX_COLOURBAR_LABELS = 8


def _import_proteus_mpl(required: bool):
    """Return the proteus_mpl module, or None when it is absent and not ``required``."""
    try:
        import proteus_mpl
    except ImportError as exc:
        if required:
            raise ImportError(
                '[post_processing].style needs the proteus-mpl package '
                '(pip install proteus-mpl)'
            ) from exc
        log.info('proteus-mpl is not installed; the grid plots use the matplotlib style')
        return None
    return proteus_mpl


@contextmanager
def plot_style(style: str | None):
    """Draw inside the proteus-mpl ``style`` ('light' or 'dark') or the active style.

    ``'matplotlib'`` keeps the active matplotlib style. ``None`` (the key unset)
    means 'light' when proteus-mpl is installed and the active style otherwise;
    an explicit 'light' or 'dark' without proteus-mpl raises ImportError. The
    matplotlib settings are restored on exit.
    """
    with plt.rc_context():
        if style != 'matplotlib':
            proteus_mpl = _import_proteus_mpl(required=style is not None)
            if proteus_mpl is not None:
                proteus_mpl.use(style or 'light')
        yield


def _cycle_colours() -> list:
    """Return the colours of the active style's colour cycle as RGB tuples."""
    cycle = mpl.rcParams['axes.prop_cycle'].by_key().get('color', [])
    return [mpl.colors.to_rgb(c) for c in cycle] or [style_neutral()]


def _luminance(rgba) -> float:
    """Return the relative luminance (sRGB, Rec. 709 weights) of an RGBA colour."""
    rgb = np.asarray(rgba[:3], dtype=float)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(linear @ np.array([0.2126, 0.7152, 0.0722]))


def _contrast(a, b) -> float:
    """Return the WCAG contrast ratio (1 to 21) between two colours."""
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _trim_colourmap(cmap, face=None):
    """Return ``cmap`` without the ends that are too close to the ``face`` colour.

    Each end is cut back to the first colour with contrast >= MIN_CURVE_CONTRAST
    against ``face`` (default the axes face colour of the active style), at most
    MAX_COLOURMAP_TRIM of the map; a map whose ends both stand out is unchanged.
    """
    face = style_neutral(1.0) if face is None else face
    steps = np.linspace(0.0, MAX_COLOURMAP_TRIM, 71)

    def trim(end):
        position = (lambda t: t) if end == 0 else (lambda t: 1.0 - t)
        return next(
            (t for t in steps if _contrast(cmap(position(t)), face) >= MIN_CURVE_CONTRAST),
            MAX_COLOURMAP_TRIM,
        )

    base = np.linspace(trim(0), 1.0 - trim(1), 256)
    return mpl.colors.LinearSegmentedColormap.from_list(f'{cmap.name}_trim', cmap(base))


def _hue_distance(rgb, hue: float) -> float:
    """Return the angle [degrees] between the hue of ``rgb`` and ``hue``."""
    delta = abs(colorsys.rgb_to_hsv(*rgb)[0] * 360.0 - hue) % 360.0
    return min(delta, 360.0 - delta)


def category_colours() -> dict:
    """Return the bar colour of each status category in the active style.

    Completed, Running and Error take the cycle colour nearest in hue to green,
    blue and red (each a different one); Unknown and Not started take greys
    between the text and face colours of the style.
    """
    cycle = _cycle_colours()
    chromatic = [
        c
        for c in cycle
        if colorsys.rgb_to_hsv(*c)[1] >= MIN_HUE_SATURATION
        and colorsys.rgb_to_hsv(*c)[2] >= MIN_HUE_VALUE
    ]
    colours = {}
    for category, hue in CATEGORY_HUES.items():
        free = [c for c in chromatic if c not in colours.values()] or cycle
        colours[category] = min(free, key=lambda c: _hue_distance(c, hue))
    colours['Unknown'] = style_neutral(FADE_MID_GREY)
    colours['Not started'] = style_neutral(FADE_LIGHT_GREY)
    return colours


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
    """Plot the number of grid cases per final status as horizontal bars.

    Bars are coloured by status category (:func:`category_colours`) and listed top
    to bottom in category order, so long status names never overlap.

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
    palette = category_colours()
    colours = [palette.get(c, palette['Unknown']) for c in counts['category']]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'summary_grid_statuses.{plot_format}'

    fig, ax = plt.subplots(figsize=(9.0, 1.5 + 0.55 * len(counts)))
    try:
        bars = ax.barh(counts['status'], counts['count'], color=colours)
        for bar, count in zip(bars, counts['count']):
            pct = 100.0 * count / total if total else 0.0
            ax.annotate(
                f'{count} ({pct:.1f}%)',
                (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0),
                textcoords='offset points',
                va='center',
            )
        ax.invert_yaxis()
        ax.set_title(f'Status summary for grid {grid_name} ({total} cases)')
        ax.set_xlabel('Number of simulations')
        ax.margins(x=0.2)
        ax.grid(False, axis='y')
        fig.tight_layout()
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


def value_label(value) -> str:
    """Return the label of a swept value: 4 significant digits for a number."""
    if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
        return f'{float(value):.4g}'
    return str(value)


def _is_numeric(values) -> bool:
    """Return whether swept values are numbers (booleans count as labels)."""
    return np.issubdtype(np.asarray(values).dtype, np.number)


def _value_colours(values, log_norm: bool, cmap):
    """Return (colour_of, norm) for colouring the curves of one parameter's values.

    Up to the length of the style's colour cycle, each value takes a cycle colour
    and the row gets a legend (norm is None). More numeric values take the colour
    map through a continuous norm that drives a colourbar; more string values take
    evenly spaced colours of the map and a legend.
    """
    numeric = _is_numeric(values)
    uniques = sorted(set(values)) if numeric else sorted({str(v) for v in values})
    cycle = _cycle_colours()
    if len(uniques) <= len(cycle) or not numeric:
        if len(uniques) <= len(cycle):
            palette = cycle[: len(uniques)]
        else:
            palette = [cmap(t) for t in np.linspace(0.0, 1.0, len(uniques))]
        if numeric:
            # Nearest swept value, so a float changed on the CSV round-trip still matches.
            grid = np.asarray(uniques, dtype=float)
            return (lambda v: palette[int(np.argmin(np.abs(grid - float(v))))]), None
        lookup = dict(zip(uniques, palette))
        return (lambda v: lookup.get(str(v), style_neutral(FADE_MID_GREY))), None

    vmin, vmax = float(uniques[0]), float(uniques[-1])
    norm = (
        mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        if log_norm and vmin > 0
        else mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    )
    return (lambda v: cmap(norm(v))), norm


def _plot_cell(ax, df, param, value, output, colour):
    """Draw one ECDF curve (cases with ``param == value``) on a cell, return success."""
    subset = df[df[param] == value]
    series = clean_series(subset[output] * get_scale(output))
    if series.empty:
        return False
    # Normalise by every case in the subset, so cases without a value (for example
    # never solidified) keep the curve below 100 %.
    x, y = ecdf(series.to_numpy(), n_total=len(subset))
    ax.plot(x, y, drawstyle='steps-post', color=colour, linewidth=ECDF_LINEWIDTH)
    return True


def ecdf_figsize(n_rows: int, n_cols: int) -> tuple[float, float]:
    """Return the ECDF figure size [inch], with panels that grow with the style's font.

    A panel is 3.6 x 2.6 inch at a 10 pt font and scales with ``font.size``; the
    height is at least MIN_ECDF_HEIGHT_PER_PT times the font size, so the shared
    y label fits beside a single row.
    """
    scale = float(mpl.rcParams['font.size']) / 10.0
    height = max(2.6 * scale * n_rows, MIN_ECDF_HEIGHT_PER_PT * mpl.rcParams['font.size'])
    return 3.6 * scale * n_cols, height


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
    coloured by that value (see :func:`_value_colours`: a legend for a few values
    or string values, a colourbar labelled at the swept values otherwise). Each
    curve is a percentage of all cases with that value, so it levels off below
    100 % when some cases have no value for the output.

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
        Colour map name for parameters with more values than the colour cycle
        (scientific map via :func:`proteus.utils.plot.get_colourmap`).
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

    # Trim the ends of the colour map that would vanish against the axes face colour.
    cmap = _trim_colourmap(get_colourmap(colourmap))
    n_rows, n_cols = len(params), len(outputs)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=ecdf_figsize(n_rows, n_cols),
        squeeze=False,
        sharey=True,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'ecdf_grid_plot.{plot_format}'
    try:
        row_keys = _draw_ecdf_rows(axes, df, params, tested_params, outputs, cmap)
        fig.suptitle(f'ECDF summary for grid {grid_name}')
        fig.supylabel('Empirical cumulative fraction of simulations [%]', x=-0.005)
        fig.tight_layout()
        # Add the colourbar or legend after the layout, outside the rightmost panel,
        # so every data panel keeps an identical size.
        for ax, param, values, norm, colour_of in row_keys:
            _add_row_key(fig, ax, param, values, norm, cmap, colour_of)
        fig.savefig(out_file, dpi=200, bbox_inches='tight')
    finally:
        plt.close(fig)
    log.info('Wrote ECDF grid plot to %s', out_file.name)
    return out_file


def _draw_ecdf_rows(axes, df, params, tested_params, outputs, cmap) -> list:
    """Draw every ECDF panel and return the key data of each parameter row.

    Each returned tuple is (rightmost axes, param, swept values, norm, colour_of),
    the arguments :func:`_add_row_key` needs after the layout.
    """
    n_rows = len(params)
    scaled = {out: df[out] * get_scale(out) for out in outputs}
    scales = {out: axis_scale(scaled[out], get_log_scale(out)) for out in outputs}
    row_keys = []
    for i, param in enumerate(params):
        values = tested_params[param]
        # A parameter is coloured on a log scale by the same rule as a log axis.
        log_norm = _is_numeric(values) and (
            axis_scale(pd.Series(values, dtype=float), get_log_scale(param))[0] == 'log'
        )
        colour_of, norm = _value_colours(values, log_norm, cmap)

        # Iterate the values actually present in the column (not the in-memory
        # swept list) so a float that changed on the CSV round-trip still matches.
        present_values = sorted(df[param].dropna().unique())

        for j, output in enumerate(outputs):
            ax = axes[i][j]
            for value in present_values:
                _plot_cell(ax, df, param, value, output, colour_of(value))
            _set_axis_scale(ax, *scales[output], scaled[output])
            ax.set_ylim(0, 100)
            if i == n_rows - 1:
                ax.set_xlabel(get_label(output))

        row_keys.append((axes[i][-1], param, values, norm, colour_of))
    return row_keys


def label_colourbar(cbar, values) -> None:
    """Put the colourbar ticks at the swept ``values``, labelled, with no minor ticks.

    With more than MAX_COLOURBAR_LABELS values every k-th tick is labelled, so the
    labels never collide; every tick still marks a swept value.
    """
    ticks = sorted({float(v) for v in values})
    step = max(1, math.ceil(len(ticks) / MAX_COLOURBAR_LABELS))
    labels = [value_label(t) if k % step == 0 else '' for k, t in enumerate(ticks)]
    cbar.set_ticks(ticks, labels=labels)
    cbar.minorticks_off()


def _add_row_key(fig, ax, param, values, norm, cmap, colour_of) -> None:
    """Attach a colourbar (``norm`` set) or legend labelling a parameter row.

    The key sits in an axes anchored just outside ``ax`` (not stolen from it), so
    the data panels all keep the same size whether a row uses a colourbar or a legend.
    """
    if norm is not None:
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
        label_colourbar(cbar, values)
        return

    uniques = sorted(set(values)) if _is_numeric(values) else sorted({str(v) for v in values})
    handles = [
        mpl.lines.Line2D([0], [0], color=colour_of(v), lw=ECDF_LINEWIDTH, label=value_label(v))
        for v in uniques
    ]
    ax.legend(
        handles=handles,
        title=get_label(param),
        fontsize='small',
        bbox_to_anchor=(1.03, 1.0),
        loc='upper left',
        borderaxespad=0.0,
    )


def validate_output_variables(df: pd.DataFrame, requested_outputs: list) -> list:
    """Return the requested outputs present in ``df`` with numeric values.

    A missing or non-numeric output (for example ``status``) is dropped with a
    warning, never substituted.

    Parameters
    ----------
    df : pandas.DataFrame
        The summary CSV loaded for plotting.
    requested_outputs : list
        Output variable names requested in the grid config.
    """
    available = list(df.columns)
    missing = [v for v in requested_outputs if v not in available]
    if missing:
        log.warning(
            'Output variable(s) %s not found in summary data; skipping them. '
            'Available columns: %s',
            missing,
            available,
        )
    present = [v for v in requested_outputs if v in available]
    non_numeric = [v for v in present if not pd.api.types.is_numeric_dtype(df[v])]
    if non_numeric:
        log.warning('Output variable(s) %s are not numeric; skipping them', non_numeric)
    return [v for v in present if v not in non_numeric]


def resolve_plot_format(requested: str | None, base_fmt: str | None = None) -> str:
    """Return a validated plot format for the grid plots.

    Uses ``[post_processing].plot_format`` when set, else ``base_fmt`` (the base
    run's ``params.out.plot_fmt`` from ref_config.toml), else 'png'. Raises on an
    unsupported value rather than writing a ``.None`` file.
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
        ``output_variables``, ``colormap``, ``plot_format``, ``style``).
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

    with plot_style(opts.get('style')):
        if opts['plot_status']:
            df_all = pd.read_csv(csv_paths['all'], sep='\t')
            n_not_started = count_not_started(df_all, tested_params)
            plot_status_summary(df_all, plots_dir, grid_name, plot_format, n_not_started)
        if opts['plot_ecdf']:
            _make_ecdf_plot(opts, csv_paths, tested_params, plots_dir, grid_name, plot_format)


def _make_ecdf_plot(opts, csv_paths, tested_params, plots_dir, grid_name, plot_format):
    """Draw the ECDF grid plot from the 'completed' CSV, skipping a grid with none."""
    df_completed = pd.read_csv(csv_paths['completed'], sep='\t')
    if df_completed.empty:
        log.warning('No completed cases; skipping the ECDF plot')
        return
    outputs = validate_output_variables(df_completed, opts['output_variables'])
    if not outputs:
        raise ValueError(
            'No valid output_variables for the ECDF plot. Set output_variables in the '
            '[post_processing] table to numeric columns of the summary CSV.'
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
