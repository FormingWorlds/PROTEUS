# Extract per-case final-row results from a PROTEUS grid into summary CSV files.
from __future__ import annotations

import logging
import tomllib
from pathlib import Path

import attrs
import numpy as np
import pandas as pd

from proteus.config._params import StopSolid
from proteus.grid.analyse_options import read_post_processing_options
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
        config. Pass the grid folder with ``-o`` instead.
    """
    output = str(config['output']).strip().strip('/')
    if output.lower() == 'auto':
        raise ValueError(
            "Cannot find the grid folder from a config whose output is 'auto': the "
            'folder name is generated at run time. Pass the grid folder instead, as in '
            "'proteus grid-analyse -o output/<grid folder>'."
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


def _read_toml(path: Path, index: int) -> dict | None:
    """Return a case's TOML file as a dict, or None if it is absent or unreadable."""
    if not path.exists():
        return None
    try:
        with path.open('rb') as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        log.warning('Case %06d: cannot read %s: %s', index, path.name, exc)
        return None


def _load_case_config(index: int, case_dir: Path) -> tuple[dict, str]:
    """Return a case's config and the file it came from.

    The run writes ``init_coupler.toml`` only after its start-up checks; a case
    that failed before that still has the config the grid manager wrote to
    ``<grid>/cfgs/case_<N>.toml``, so its swept values are not lost.
    """
    init_params = _read_toml(case_dir / 'init_coupler.toml', index)
    if init_params is not None:
        return init_params, 'init_coupler.toml'
    grid_cfg = case_dir.parent / 'cfgs' / f'{case_dir.name}.toml'
    grid_params = _read_toml(grid_cfg, index)
    if grid_params is not None:
        log.warning('Case %06d has no init_coupler.toml; reading cfgs/%s', index, grid_cfg.name)
        return grid_params, f'cfgs/{grid_cfg.name}'
    log.warning('Case %06d has no readable config', index)
    return {}, ''


def _read_case_helpfile(index: int, case_dir: Path) -> pd.DataFrame | None:
    """Return a case's runtime helpfile, or None if it is absent or unreadable.

    Read through the coupler's exact reader, which refuses a cut or malformed file
    (for example a run killed mid-write) instead of returning a partial last row.
    """
    runtime_file = Path(helpfile_path(case_dir))
    if not runtime_file.exists():
        log.warning('Case %06d has no runtime_helpfile.csv', index)
        return None
    try:
        return read_helpfile_table(runtime_file)
    except (OSError, HelpfileFormatError) as exc:
        log.warning('Case %06d: cannot read runtime_helpfile.csv: %s', index, exc)
        return None


def _load_case(index: int, case_dir: Path, base_solid: dict) -> dict:
    """Load one case's status and config, and reduce its helpfile to the summary values.

    The helpfile is read, reduced to its final row and solidification time, and
    dropped, so a grid of long runs never holds more than one helpfile in memory.
    """
    # Status code is read tolerantly (ReadStatus returns STATUS_MISSING if absent).
    code = ReadStatus(case_dir)
    comment = CommentFromStatus(code) if code >= 0 else 'Missing status file'
    if code < 0:
        log.warning('Case %06d has no readable status file', index)

    init_params, config_source = _load_case_config(index, case_dir)
    output = _read_case_helpfile(index, case_dir)
    solid = _case_solid_settings(init_params, base_solid)
    final_row = {} if output is None or output.empty else output.iloc[-1].to_dict()
    return {
        'index': index,
        'status_code': code,
        'status': comment,
        'category': status_category(code),
        'init_params': init_params,
        'config_source': config_source,
        'final_row': final_row,
        'solidification_time': extract_solidification_time(output, solid['phi_crit']),
        'solid_stop_enabled': solid['enabled'],
    }


def load_grid_cases(grid_dir: str | Path, base_solid: dict | None = None) -> list[dict]:
    """Load status, config and summary results for every case in a grid.

    Parameters
    ----------
    grid_dir : str or Path
        Grid output directory containing the ``case_*`` folders.
    base_solid : dict or None
        ``{'enabled': bool, 'phi_crit': float}`` for a case whose config omits
        ``params.stop.solid`` (normally the base config's values; default the
        schema defaults).

    Returns
    -------
    list of dict
        One dict per case (ascending case index), each with keys
        ``index`` (int), ``status_code`` (int), ``status`` (str comment),
        ``category`` (str), ``init_params`` (dict from init_coupler.toml, else
        from the grid's cfgs/case_<N>.toml), ``config_source`` (str, the file it
        came from, empty if neither is readable), ``final_row`` (dict of the last
        runtime-helpfile row, empty without a readable helpfile),
        ``solidification_time`` (float, see :func:`extract_solidification_time`)
        and ``solid_stop_enabled`` (bool).
    """
    if base_solid is None:
        base_solid = {'enabled': DEFAULT_SOLID_ENABLED, 'phi_crit': DEFAULT_PHI_CRIT}
    grid_dir = Path(grid_dir)
    cases = [_load_case(index, path, base_solid) for index, path in _case_folders(grid_dir)]
    log.info("Loaded %d cases from '%s'", len(cases), grid_dir)
    stop_off = [c['index'] for c in cases if not c['solid_stop_enabled']]
    if stop_off:
        log.info(
            '%d case(s) ran with the solidification stop disabled; their '
            'solidification_time is still the first time Phi_global <= phi_crit',
            len(stop_off),
        )
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
    It is found the same way when the solidification stop is disabled (a grid with
    a fixed end time), since the run then continues past the crossing. Returns NaN
    when the case never solidifies, has no results, or lacks the required columns.

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


def _case_solid_settings(init_params: dict, base_solid: dict) -> dict:
    """Return a case's solidification stop ``{'enabled', 'phi_crit'}``, base values filling gaps."""
    solid = _dig(init_params, 'params.stop.solid', {})
    return {
        'enabled': bool(solid.get('enabled', base_solid['enabled'])),
        'phi_crit': float(solid.get('phi_crit', base_solid['phi_crit'])),
    }


def build_summary_dataframe(cases: list[dict], param_paths: list[str]) -> pd.DataFrame:
    """Assemble one summary row per case: status, tested parameters, final outputs.

    Parameters
    ----------
    cases : list of dict
        Output of :func:`load_grid_cases`.
    param_paths : list of str
        Swept grid-parameter paths to record per case (from each case's config).

    Returns
    -------
    pandas.DataFrame
        One row per case. Columns: ``case_number``, ``status``,
        ``status_category``, ``config_source``, the tested parameters, every
        runtime-helpfile column at the final row, and ``solidification_time``
        (NaN when the case never reached ``Phi_global <= phi_crit``).
    """
    records = []
    for case in cases:
        record = {
            'case_number': case['index'],
            'status': case['status'],
            'status_category': case['category'],
            'config_source': case.get('config_source', ''),
        }
        # Tested parameter values actually used by this case.
        for path in param_paths:
            record[path] = _dig(case['init_params'], path, np.nan)
        record.update(case['final_row'])
        record['solidification_time'] = case['solidification_time']
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
        cases = load_grid_cases(grid_dir, _resolve_base_solid(ref))
        if not cases:
            raise FileNotFoundError(
                f"No case_<N> folders in '{grid_dir}': no grid case has started yet."
            )
        summary = build_summary_dataframe(cases, list(tested_params))
        return write_summary_csvs(summary, extracted_dir)

    csv_paths = {m: extracted_dir / f'final_extracted_data_{m}.csv' for m in CSV_MODES}
    missing = [p.name for p in csv_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f'update_csv is false but these CSVs are missing from {extracted_dir}: '
            f'{missing}. Set update_csv = true to generate them.'
        )
    return csv_paths


def _resolve_inputs(config_path: Path | None, grid_dir: Path | None) -> tuple[dict, Path, Path]:
    """Return (config, config path, grid folder) from a grid file, a grid folder or both.

    With only a grid folder, its copy.grid.toml (saved at launch) is the config.
    With both, the folder is analysed with the options of the given file.
    """
    if config_path is None and grid_dir is None:
        raise ValueError('Give a grid config (-c), a grid folder (-o) or both')
    if config_path is None:
        config_path = Path(grid_dir) / 'copy.grid.toml'
        if not config_path.exists():
            raise FileNotFoundError(
                f"No copy.grid.toml in '{grid_dir}'; pass the grid file with -c as well."
            )
    config_path = Path(config_path)
    with config_path.open('rb') as f:
        config = tomllib.load(f)
    grid_dir = resolve_grid_dir(config) if grid_dir is None else Path(grid_dir).resolve()
    if not grid_dir.is_dir():
        raise FileNotFoundError(f"Grid output directory not found: '{grid_dir}'")
    return config, config_path, grid_dir


def analyse_grid(
    config_path: str | Path | None = None, grid_dir: str | Path | None = None
) -> None:
    """Post-process a PROTEUS grid from its ``.grid.toml`` config or its output folder.

    Reads the optional ``[post_processing]`` table for its options, extracts
    per-case final-row results into summary CSVs, and (optionally) makes the
    status summary and ECDF grid plots.

    Parameters
    ----------
    config_path : str, Path or None
        Grid config file (the same ``.grid.toml`` used to run the grid). Without
        ``grid_dir`` its ``output`` names the grid folder, so it cannot be ``'auto'``.
    grid_dir : str, Path or None
        Grid output folder. Without ``config_path`` its ``copy.grid.toml`` gives
        the options.
    """
    config, config_path, grid_dir = _resolve_inputs(config_path, grid_dir)
    opts = read_post_processing_options(config)
    log.info("Analysing grid '%s' at '%s'", get_grid_name(grid_dir), grid_dir)

    tested_params = _run_tested_parameters(config, config_path, grid_dir)
    ref = _read_ref_config(grid_dir)
    csv_paths = _extract_csvs(grid_dir, tested_params, ref, opts['update_csv'])

    if opts['plot_status'] or opts['plot_ecdf']:
        # Import plotting lazily so the CSV-only path needs no matplotlib.
        from proteus.grid.plot_grid import make_grid_plots

        base_fmt = _dig(ref, 'params.out.plot_fmt')
        make_grid_plots(opts, csv_paths, tested_params, grid_dir, base_fmt)
