# Extract per-case final-row results from a PROTEUS grid into summary CSV files.
from __future__ import annotations

import logging
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from proteus.grid.manage import dimension_values, grid_dimension_keys
from proteus.grid.summarise import status_category
from proteus.utils.helper import CommentFromStatus, ReadStatus, get_proteus_dir

log = logging.getLogger('fwl.' + __name__)

# Fallback critical melt fraction, used only when neither a case config nor the
# base config records params.stop.solid.phi_crit (every normal run does). Mirrors
# the config schema default (proteus.config._params.StopSolid.phi_crit).
DEFAULT_PHI_CRIT = 0.01

# CSV filter modes: name -> status categories kept (None keeps every case).
CSV_MODES = {
    'all': None,
    'completed': ('Completed',),
    'running_error': ('Running', 'Error'),
}

# Image formats the grid plots may be saved as.
SUPPORTED_PLOT_FORMATS = ('png', 'pdf')


def get_grid_name(grid_dir: str | Path) -> str:
    """Return the grid name (final component of the grid directory path)."""
    grid_dir = Path(grid_dir)
    return grid_dir.name


def resolve_grid_dir(config: dict) -> Path:
    """Return the absolute grid output directory named by a grid config's ``output`` key.

    Raises
    ------
    ValueError
        If ``output`` is ``'auto'``: the grid manager turns that into a unique
        timestamped folder at run time, so the name cannot be recovered from the
        config. Point grid-analyse at a grid config with a fixed ``output``.
    """
    output = str(config['output']).strip().strip('/')
    if output.lower() == 'auto':
        raise ValueError(
            "Cannot analyse a grid whose output is 'auto': the run folder name is "
            'generated at run time. Set output to the actual grid run folder name.'
        )
    return (Path(get_proteus_dir()) / 'output' / output).resolve()


def _dig(mapping, dotted_path: str, default=None):
    """Return a nested value addressed by a dotted path, or ``default`` if absent."""
    value = mapping
    for key in dotted_path.split('.'):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def load_grid_cases(grid_dir: str | Path) -> list[dict]:
    """Load status, config and final results for every case in a grid.

    Parameters
    ----------
    grid_dir : str or Path
        Grid output directory containing the ``case_*`` folders.

    Returns
    -------
    list of dict
        One dict per case (ascending case index), each with keys
        ``index`` (int), ``status_code`` (int), ``status`` (str comment),
        ``category`` (str), ``init_params`` (dict from init_coupler.toml) and
        ``output`` (pandas.DataFrame from runtime_helpfile.csv, or None).
    """
    grid_dir = Path(grid_dir)
    case_dirs = sorted(grid_dir.glob('case_*'), key=lambda p: int(p.name.split('_')[-1]))

    cases = []
    for case_dir in case_dirs:
        index = int(case_dir.name.split('_')[-1])

        # Status code is read tolerantly (ReadStatus returns STATUS_MISSING if absent).
        code = ReadStatus(case_dir)
        comment = CommentFromStatus(code) if code >= 0 else 'Missing status file'
        if code < 0:
            log.warning('Case %06d has no readable status file', index)

        # Case config, as written by the run.
        init_params = {}
        init_file = case_dir / 'init_coupler.toml'
        if init_file.exists():
            try:
                with init_file.open('rb') as f:
                    init_params = tomllib.load(f)
            except (OSError, tomllib.TOMLDecodeError) as exc:
                log.warning('Case %06d: cannot read init_coupler.toml: %s', index, exc)

        # Final-row results from the runtime helpfile (may be absent for a failed case).
        output = None
        runtime_file = case_dir / 'runtime_helpfile.csv'
        if runtime_file.exists():
            try:
                output = pd.read_csv(runtime_file, sep='\t')
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                log.warning('Case %06d: cannot read runtime_helpfile.csv: %s', index, exc)
        else:
            log.warning('Case %06d has no runtime_helpfile.csv', index)

        cases.append(
            {
                'index': index,
                'status_code': code,
                'status': comment,
                'category': status_category(code),
                'init_params': init_params,
                'output': output,
            }
        )

    log.info("Loaded %d cases from '%s'", len(cases), grid_dir)
    return cases


def get_tested_parameters(config: dict, config_fpath: str | Path) -> dict:
    """Return each swept grid parameter and the values it takes across the grid.

    The swept values come from ``proteus.grid.manage.dimension_values``, the same
    construction ``proteus grid`` uses, so the analysis axes match the grid exactly.

    Parameters
    ----------
    config : dict
        The grid config parsed from TOML.
    config_fpath : str or Path
        Path of the grid file, quoted back in any validation error.

    Returns
    -------
    dict
        Mapping of parameter path (for example ``'planet.mass_tot'``) to its list
        of swept values.
    """
    dim_keys = grid_dimension_keys(config, str(config_fpath))
    return {key: dimension_values(config[key]) for key in dim_keys}


def extract_solidification_time(output: pd.DataFrame | None, phi_crit: float) -> float:
    """Return the first time at which ``Phi_global`` drops below ``phi_crit``.

    Solidification time is the FIRST crossing of ``Phi_global < phi_crit``, not the
    last row. Returns NaN when the case never solidifies, has no results, or lacks
    the required columns.

    Parameters
    ----------
    output : pandas.DataFrame or None
        A case's runtime helpfile.
    phi_crit : float
        Critical melt fraction below which the mantle counts as solidified.
    """
    if output is None or output.empty:
        return np.nan
    if 'Phi_global' not in output.columns or 'Time' not in output.columns:
        log.warning('Missing Phi_global or Time column; cannot find solidification time')
        return np.nan

    solidified = output['Phi_global'] < phi_crit
    if not solidified.any():
        return np.nan
    # idxmax returns the first index where the boolean is True (the first crossing).
    return float(output.loc[solidified.idxmax(), 'Time'])


def build_summary_dataframe(
    cases: list[dict], param_paths: list[str], fallback_phi_crit: float = DEFAULT_PHI_CRIT
) -> pd.DataFrame:
    """Assemble one summary row per case: status, tested parameters, final outputs.

    Parameters
    ----------
    cases : list of dict
        Output of :func:`load_grid_cases`.
    param_paths : list of str
        Swept grid-parameter paths to record per case (from init_coupler.toml).
    fallback_phi_crit : float
        phi_crit to use for a case whose init_coupler.toml omits
        ``params.stop.solid.phi_crit`` (normally the base config's value).

    Returns
    -------
    pandas.DataFrame
        One row per case. Columns: ``case_number``, ``status``, the tested
        parameters, every runtime-helpfile column at the final row, and
        ``solidification_time``.
    """
    records = []
    for case in cases:
        phi_crit = _dig(case['init_params'], 'params.stop.solid.phi_crit', fallback_phi_crit)
        record = {'case_number': case['index'], 'status': case['status']}

        # Tested parameter values actually used by this case.
        for path in param_paths:
            record[path] = _dig(case['init_params'], path, np.nan)

        # Final-row outputs (all helpfile columns), if the case produced any.
        output = case['output']
        if output is not None and not output.empty:
            record.update(output.iloc[-1].to_dict())

        record['solidification_time'] = extract_solidification_time(output, phi_crit)
        records.append(record)

    return pd.DataFrame(records)


def write_summary_csvs(
    summary: pd.DataFrame, cases: list[dict], out_dir: str | Path
) -> dict[str, Path]:
    """Write the all/completed/running_error summary CSVs (tab-separated).

    Parameters
    ----------
    summary : pandas.DataFrame
        Full per-case summary from :func:`build_summary_dataframe`.
    cases : list of dict
        Output of :func:`load_grid_cases`, used for the status category of each row.
    out_dir : str or Path
        Directory to write the CSVs into (created if needed).

    Returns
    -------
    dict
        Mapping of mode name to the path written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    category_by_case = {case['index']: case['category'] for case in cases}
    categories = summary['case_number'].map(category_by_case)

    written = {}
    for mode, kept in CSV_MODES.items():
        subset = summary if kept is None else summary[categories.isin(kept)]
        path = out_dir / f'final_extracted_data_{mode}.csv'
        subset.to_csv(path, sep='\t', index=False)
        log.info('Wrote %d cases to %s', len(subset), path.name)
        written[mode] = path
    return written


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


def _read_post_processing_options(config: dict) -> dict:
    """Return the ``[post_processing]`` options with defaults filled in."""
    opts = dict(config.get('post_processing', {}))
    opts.setdefault('update_csv', True)
    opts.setdefault('plot_status', True)
    opts.setdefault('plot_ecdf', True)
    # Accept both spellings; 'colormap' matches matplotlib, 'colourmap' the UK style.
    opts['colormap'] = opts.get('colormap', opts.get('colourmap'))
    opts.setdefault('output_variables', [])
    opts.setdefault('plot_format', None)
    return opts


def _resolve_base_phi_crit(grid_dir: Path) -> float:
    """Return the base config's params.stop.solid.phi_crit, else DEFAULT_PHI_CRIT."""
    ref_file = grid_dir / 'ref_config.toml'
    if ref_file.exists():
        with ref_file.open('rb') as f:
            ref = tomllib.load(f)
        value = _dig(ref, 'params.stop.solid.phi_crit')
        if value is not None:
            return float(value)
    return DEFAULT_PHI_CRIT


def _resolve_plot_format(requested: str | None, grid_dir: Path) -> str:
    """Return a validated plot format for the grid plots.

    Uses ``[post_processing].plot_format`` when set, else the base run's
    ``params.out.plot_fmt`` (from ref_config.toml in the grid directory), else
    'png'. Raises on an unsupported value rather than writing a ``.None`` file.
    """
    fmt = requested
    if fmt is None:
        ref_file = grid_dir / 'ref_config.toml'
        if ref_file.exists():
            with ref_file.open('rb') as f:
                ref = tomllib.load(f)
            fmt = _dig(ref, 'params.out.plot_fmt')
    fmt = (fmt or 'png').lower()
    if fmt not in SUPPORTED_PLOT_FORMATS:
        raise ValueError(
            f"Unsupported plot_format '{fmt}'; choose one of {SUPPORTED_PLOT_FORMATS}"
        )
    return fmt


def analyse_grid(config_path: str | Path) -> None:
    """Post-process a PROTEUS grid from its ``.grid.toml`` config.

    Reads the optional ``[post_processing]`` table for its options, extracts
    per-case final-row results into summary CSVs, and (optionally) makes the
    status summary and ECDF grid plots.

    Parameters
    ----------
    config_path : str or Path
        Path to the grid config file (the same ``.grid.toml`` used to run the grid).
    """
    config_path = Path(config_path)
    with config_path.open('rb') as f:
        config = tomllib.load(f)

    opts = _read_post_processing_options(config)
    grid_dir = resolve_grid_dir(config)
    if not grid_dir.is_dir():
        raise FileNotFoundError(f"Grid output directory not found: '{grid_dir}'")
    grid_name = get_grid_name(grid_dir)
    log.info("Analysing grid '%s' at '%s'", grid_name, grid_dir)

    # Prefer the grid's own copy.grid.toml (the authoritative record of what was
    # run) for the swept axes, so the analysis matches the grid even if the config
    # file passed on the CLI was edited after the run.
    run_config, run_config_path = config, config_path
    copy_grid = grid_dir / 'copy.grid.toml'
    if copy_grid.exists():
        with copy_grid.open('rb') as f:
            run_config = tomllib.load(f)
        run_config_path = copy_grid
    tested_params = get_tested_parameters(run_config, run_config_path)

    # Base config phi_crit, used only for a case whose own config omits it.
    fallback_phi_crit = _resolve_base_phi_crit(grid_dir)

    extracted_dir = grid_dir / 'post_processing' / 'extracted_data'
    csv_paths = {mode: extracted_dir / f'final_extracted_data_{mode}.csv' for mode in CSV_MODES}

    if opts['update_csv']:
        cases = load_grid_cases(grid_dir)
        summary = build_summary_dataframe(cases, list(tested_params), fallback_phi_crit)
        csv_paths = write_summary_csvs(summary, cases, extracted_dir)
    else:
        missing = [p.name for p in csv_paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f'update_csv is false but these CSVs are missing from {extracted_dir}: '
                f'{missing}. Set update_csv = true to generate them.'
            )

    if not (opts['plot_status'] or opts['plot_ecdf']):
        return

    # Import plotting lazily so the CSV-only path needs no matplotlib.
    from proteus.grid import plot_grid

    plots_dir = grid_dir / 'post_processing' / 'grid_plots'
    plot_format = _resolve_plot_format(opts['plot_format'], grid_dir)

    if opts['plot_status']:
        df_all = pd.read_csv(csv_paths['all'], sep='\t')
        plot_grid.plot_status_summary(df_all, plots_dir, grid_name, plot_format)

    if opts['plot_ecdf']:
        df_completed = pd.read_csv(csv_paths['completed'], sep='\t')
        outputs = validate_output_variables(df_completed, opts['output_variables'])
        if not outputs:
            raise ValueError(
                'No valid output_variables for the ECDF plot. Set output_variables in the '
                '[post_processing] table to columns present in the summary CSV.'
            )
        plot_grid.plot_ecdf_grid(
            df_completed,
            tested_params,
            outputs,
            plots_dir,
            grid_name,
            colourmap=opts['colormap'],
            plot_format=plot_format,
        )
