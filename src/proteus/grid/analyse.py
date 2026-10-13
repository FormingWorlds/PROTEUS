# Extract per-case final-row results from a PROTEUS grid into summary CSV files.
from __future__ import annotations

import logging
import tomllib
from pathlib import Path

import attrs
import numpy as np
import pandas as pd

from proteus.config._params import StopSolid
from proteus.grid.dimensions import dimension_values
from proteus.grid.manage import grid_dimension_keys
from proteus.grid.summarise import status_category
from proteus.utils.coupler import HelpfileFormatError, helpfile_path, read_helpfile_table
from proteus.utils.helper import CommentFromStatus, ReadStatus, get_proteus_dir

log = logging.getLogger('fwl.' + __name__)

# Schema defaults of params.stop.solid, used only when neither a case config nor the
# base config records them (every run writes both into init_coupler.toml).
DEFAULT_PHI_CRIT = attrs.fields(StopSolid).phi_crit.default
DEFAULT_SOLID_ENABLED = attrs.fields(StopSolid).enabled.default

# CSV filter modes: name -> status categories kept (None keeps every case). A case
# whose status is missing or unrecognised ('Unknown') did not finish normally.
CSV_MODES = {
    'all': None,
    'completed': ('Completed',),
    'running_error': ('Running', 'Error', 'Unknown'),
}

# Outputs drawn in the ECDF plot when [post_processing] sets no output_variables.
DEFAULT_OUTPUT_VARIABLES = (
    'solidification_time',
    'Phi_global',
    'T_surf',
    'P_surf',
    'atm_kg_per_mol',
)

# Keys accepted in the [post_processing] table, with their expected types.
POST_PROCESSING_OPTIONS = {
    'update_csv': bool,
    'plot_status': bool,
    'plot_ecdf': bool,
    'colormap': str,
    'colourmap': str,
    'output_variables': list,
    'plot_format': str,
}


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


def _case_folders(grid_dir: Path) -> list[tuple[int, Path]]:
    """Return (index, path) of every ``case_<N>`` folder, ascending by index.

    Entries that are not folders or whose suffix is not an integer are skipped with
    a warning, as ``proteus grid-summarise`` does.
    """
    found = []
    for path in grid_dir.glob('case_*'):
        try:
            index = int(path.name.rsplit('_', 1)[-1])
        except ValueError:
            index = None
        if index is None or not path.is_dir():
            log.warning("Ignoring '%s': not a case_<N> folder", path.name)
            continue
        found.append((index, path))
    return sorted(found)


def _load_case(index: int, case_dir: Path) -> dict:
    """Load one case's status, init_coupler.toml and runtime helpfile (tolerantly)."""
    # Status code is read tolerantly (ReadStatus returns STATUS_MISSING if absent).
    code = ReadStatus(case_dir)
    comment = CommentFromStatus(code) if code >= 0 else 'Missing status file'
    if code < 0:
        log.warning('Case %06d has no readable status file', index)

    init_params = {}
    init_file = case_dir / 'init_coupler.toml'
    if init_file.exists():
        try:
            with init_file.open('rb') as f:
                init_params = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            log.warning('Case %06d: cannot read init_coupler.toml: %s', index, exc)

    # Read through the coupler's exact reader, which refuses a cut or malformed file
    # (for example a run killed mid-write) instead of returning a partial last row.
    output = None
    runtime_file = Path(helpfile_path(case_dir))
    if runtime_file.exists():
        try:
            output = read_helpfile_table(runtime_file)
        except (OSError, HelpfileFormatError) as exc:
            log.warning('Case %06d: cannot read runtime_helpfile.csv: %s', index, exc)
    else:
        log.warning('Case %06d has no runtime_helpfile.csv', index)

    return {
        'index': index,
        'status_code': code,
        'status': comment,
        'category': status_category(code),
        'init_params': init_params,
        'output': output,
    }


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
    cases = [_load_case(index, path) for index, path in _case_folders(grid_dir)]
    log.info("Loaded %d cases from '%s'", len(cases), grid_dir)
    return cases


def get_tested_parameters(config: dict, config_fpath: str | Path) -> dict:
    """Return each swept grid parameter and the values it takes across the grid.

    The swept values come from ``proteus.grid.dimensions.dimension_values``, the
    same construction ``proteus grid`` uses, so the analysis axes match the grid.

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
    """Return the first time at which ``Phi_global`` reaches ``phi_crit`` or below.

    Solidification time is the FIRST row with ``Phi_global <= phi_crit``, the
    solidification stop criterion of ``proteus.utils.terminate``, not the last row.
    Returns NaN when the case never solidifies, has no results, or lacks the
    required columns.

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

    solidified = output['Phi_global'] <= phi_crit
    if not solidified.any():
        return np.nan
    # idxmax returns the first index where the boolean is True (the first crossing).
    return float(output.loc[solidified.idxmax(), 'Time'])


def _case_solidification_time(case: dict, base_solid: dict) -> float:
    """Return a case's solidification time, or NaN if its solid stop is disabled."""
    solid = _dig(case['init_params'], 'params.stop.solid', {})
    enabled = solid.get('enabled', base_solid['enabled'])
    if not enabled:
        return np.nan
    phi_crit = solid.get('phi_crit', base_solid['phi_crit'])
    return extract_solidification_time(case['output'], phi_crit)


def build_summary_dataframe(
    cases: list[dict], param_paths: list[str], base_solid: dict | None = None
) -> pd.DataFrame:
    """Assemble one summary row per case: status, tested parameters, final outputs.

    Parameters
    ----------
    cases : list of dict
        Output of :func:`load_grid_cases`.
    param_paths : list of str
        Swept grid-parameter paths to record per case (from init_coupler.toml).
    base_solid : dict or None
        ``{'enabled': bool, 'phi_crit': float}`` for a case whose init_coupler.toml
        omits ``params.stop.solid`` (normally the base config's values; default
        the schema defaults).

    Returns
    -------
    pandas.DataFrame
        One row per case. Columns: ``case_number``, ``status``,
        ``status_category``, the tested parameters, every runtime-helpfile column
        at the final row, and ``solidification_time`` (NaN when the case never
        solidified or its solidification stop criterion was disabled).
    """
    if base_solid is None:
        base_solid = {'enabled': DEFAULT_SOLID_ENABLED, 'phi_crit': DEFAULT_PHI_CRIT}
    records = []
    for case in cases:
        record = {
            'case_number': case['index'],
            'status': case['status'],
            'status_category': case['category'],
        }

        # Tested parameter values actually used by this case.
        for path in param_paths:
            record[path] = _dig(case['init_params'], path, np.nan)

        # Final-row outputs (all helpfile columns), if the case produced any.
        output = case['output']
        if output is not None and not output.empty:
            record.update(output.iloc[-1].to_dict())

        record['solidification_time'] = _case_solidification_time(case, base_solid)
        records.append(record)

    return pd.DataFrame(records)


def write_summary_csvs(summary: pd.DataFrame, out_dir: str | Path) -> dict[str, Path]:
    """Write the all/completed/running_error summary CSVs (tab-separated).

    Parameters
    ----------
    summary : pandas.DataFrame
        Full per-case summary from :func:`build_summary_dataframe`.
    out_dir : str or Path
        Directory to write the CSVs into (created if needed).

    Returns
    -------
    dict
        Mapping of mode name to the path written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for mode, kept in CSV_MODES.items():
        subset = summary if kept is None else summary[summary['status_category'].isin(kept)]
        path = out_dir / f'final_extracted_data_{mode}.csv'
        subset.to_csv(path, sep='\t', index=False)
        log.info('Wrote %d cases to %s', len(subset), path.name)
        written[mode] = path
    return written


def _check_post_processing_table(table) -> None:
    """Raise ValueError for a [post_processing] table with unknown keys or bad types."""
    if not isinstance(table, dict):
        raise ValueError(f'[post_processing] must be a table, got {type(table).__name__}')
    unknown = sorted(set(table) - set(POST_PROCESSING_OPTIONS))
    if unknown:
        raise ValueError(
            f'Unknown key(s) {unknown} in [post_processing]; accepted keys are '
            f'{sorted(POST_PROCESSING_OPTIONS)}'
        )
    for key, value in table.items():
        expected = POST_PROCESSING_OPTIONS[key]
        if not isinstance(value, expected):
            raise ValueError(
                f'[post_processing].{key} must be a {expected.__name__}, '
                f'got {value!r} ({type(value).__name__})'
            )
    outputs = table.get('output_variables', [])
    if not all(isinstance(v, str) for v in outputs):
        raise ValueError('[post_processing].output_variables must be a list of strings')
    if {'colormap', 'colourmap'} <= set(table) and table['colormap'] != table['colourmap']:
        raise ValueError('[post_processing] sets both colormap and colourmap differently')


def _read_post_processing_options(config: dict) -> dict:
    """Return the validated ``[post_processing]`` options with defaults filled in."""
    table = config.get('post_processing', {})
    _check_post_processing_table(table)
    opts = dict(table)
    opts.setdefault('update_csv', True)
    opts.setdefault('plot_status', True)
    opts.setdefault('plot_ecdf', True)
    # Accept both spellings; 'colormap' matches matplotlib, 'colourmap' the UK style.
    opts['colormap'] = opts.get('colormap', opts.pop('colourmap', None))
    opts.setdefault('output_variables', list(DEFAULT_OUTPUT_VARIABLES))
    opts.setdefault('plot_format', None)
    return opts


def _read_ref_config(grid_dir: Path) -> dict:
    """Return the grid's ref_config.toml (the base config of every case), or {}."""
    ref_file = grid_dir / 'ref_config.toml'
    if not ref_file.exists():
        return {}
    with ref_file.open('rb') as f:
        return tomllib.load(f)


def _resolve_base_solid(ref: dict) -> dict:
    """Return the base config's params.stop.solid enabled/phi_crit, else schema defaults."""
    return {
        'enabled': _dig(ref, 'params.stop.solid.enabled', DEFAULT_SOLID_ENABLED),
        'phi_crit': float(_dig(ref, 'params.stop.solid.phi_crit', DEFAULT_PHI_CRIT)),
    }


def _run_tested_parameters(config: dict, config_path: Path, grid_dir: Path) -> dict:
    """Return the swept axes, from the grid's own copy.grid.toml when it exists.

    copy.grid.toml is the record of what was run, so it is preferred over the config
    passed on the command line (which may have been edited since the run).

    Raises
    ------
    ValueError
        If copy.grid.toml no longer passes the current grid-file checks (for
        example a grid launched with an older PROTEUS).
    """
    copy_grid = grid_dir / 'copy.grid.toml'
    if not copy_grid.exists():
        return get_tested_parameters(config, config_path)
    with copy_grid.open('rb') as f:
        run_config = tomllib.load(f)
    try:
        return get_tested_parameters(run_config, copy_grid)
    except ValueError as exc:
        raise ValueError(
            f"Cannot read the swept axes from '{copy_grid}', the copy of the grid file "
            'saved when the grid was launched: it does not pass the current grid-file '
            'checks. Update the rejected keys in that file to the current names (or '
            f"remove it to use the axes in '{config_path}').\n{exc}"
        ) from exc


def _extract_csvs(grid_dir: Path, tested_params: dict, ref: dict, update: bool) -> dict:
    """Write (or, when ``update`` is false, locate) the three summary CSVs."""
    extracted_dir = grid_dir / 'post_processing' / 'extracted_data'
    if update:
        cases = load_grid_cases(grid_dir)
        if not cases:
            raise FileNotFoundError(
                f"No case_<N> folders in '{grid_dir}': no grid case has started yet."
            )
        summary = build_summary_dataframe(cases, list(tested_params), _resolve_base_solid(ref))
        return write_summary_csvs(summary, extracted_dir)

    csv_paths = {m: extracted_dir / f'final_extracted_data_{m}.csv' for m in CSV_MODES}
    missing = [p.name for p in csv_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f'update_csv is false but these CSVs are missing from {extracted_dir}: '
            f'{missing}. Set update_csv = true to generate them.'
        )
    return csv_paths


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
    log.info("Analysing grid '%s' at '%s'", get_grid_name(grid_dir), grid_dir)

    tested_params = _run_tested_parameters(config, config_path, grid_dir)
    ref = _read_ref_config(grid_dir)
    csv_paths = _extract_csvs(grid_dir, tested_params, ref, opts['update_csv'])

    if opts['plot_status'] or opts['plot_ecdf']:
        # Import plotting lazily so the CSV-only path needs no matplotlib.
        from proteus.grid.plot_grid import make_grid_plots

        base_fmt = _dig(ref, 'params.out.plot_fmt')
        make_grid_plots(opts, csv_paths, tested_params, grid_dir, base_fmt)
