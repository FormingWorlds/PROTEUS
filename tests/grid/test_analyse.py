"""Unit tests for ``proteus.grid.analyse``.

Covers the grid post-processing extraction with small grids built under tmp_path:
- solidification time is the FIRST row with Phi_global <= phi_crit (the stop
  criterion), uses each case's own phi_crit, and is still found when the case's
  solidification stop is disabled
- each case keeps only its final helpfile row and solidification time, not the
  whole helpfile
- tested-parameter axes come from copy.grid.toml when present, and the
  [post_processing] table is never read as an axis
- the summary CSVs filter cases by status category (Unknown goes to running_error)
- the [post_processing] table is validated and filled with documented defaults
- a grid folder (-o) is analysed with its copy.grid.toml, even for output = "auto"
- an empty grid, stray case_* entries and missing files are handled explicitly

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from proteus.grid import analyse as ga
from proteus.grid import plot_grid
from proteus.grid.analyse import (
    DEFAULT_PHI_CRIT,
    _dig,
    _resolve_base_solid,
    analyse_grid,
    build_summary_dataframe,
    extract_solidification_time,
    get_grid_name,
    get_tested_parameters,
    load_grid_cases,
    resolve_grid_dir,
    write_summary_csvs,
)
from proteus.grid.analyse_options import DEFAULT_OUTPUT_VARIABLES
from proteus.grid.analyse_options import (
    read_post_processing_options as _read_post_processing_options,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_SETTINGS_TOML = """output = "g"
symlink = ""
ref_config = "base.toml"
use_slurm = false
max_jobs = 1
max_days = 1
max_mem = 1"""

_GRID_SETTINGS = {
    'output': 'g',
    'symlink': '',
    'ref_config': 'base.toml',
    'use_slurm': False,
    'max_jobs': 1,
    'max_days': 1,
    'max_mem': 1,
}


def test_get_grid_name_returns_final_path_component(tmp_path):
    """get_grid_name returns the directory's own name, independent of its parent."""
    target = tmp_path / 'my_grid'
    target.mkdir()
    assert get_grid_name(target) == 'my_grid'
    assert get_grid_name(tmp_path / 'other' / 'deep_grid') == 'deep_grid'


def test_resolve_grid_dir_strips_slashes_under_proteus_output(tmp_path, monkeypatch):
    """resolve_grid_dir places 'output' under <proteus>/output/; a leading slash
    must not turn it into an absolute path outside that folder.
    """
    monkeypatch.setattr(ga, 'get_proteus_dir', lambda: str(tmp_path))
    assert resolve_grid_dir({'output': 'run_a/'}) == (tmp_path / 'output' / 'run_a').resolve()
    assert resolve_grid_dir({'output': '/run_b/'}) == (tmp_path / 'output' / 'run_b').resolve()


def test_resolve_grid_dir_rejects_auto_output():
    """An 'auto' output names no folder (it is generated at launch), so the error
    points to the -o option instead.
    """
    with pytest.raises(ValueError, match='grid-analyse -o'):
        resolve_grid_dir({'output': 'auto'})
    with pytest.raises(ValueError, match='auto'):
        resolve_grid_dir({'output': 'AUTO'})  # case-insensitive


def test_dig_returns_nested_value_or_default():
    """_dig walks a dotted path and returns the default when any key is absent."""
    data = {'params': {'stop': {'solid': {'phi_crit': 0.005}}}}
    assert _dig(data, 'params.stop.solid.phi_crit') == pytest.approx(0.005)
    assert _dig(data, 'params.stop.missing', 'fallback') == 'fallback'
    assert _dig(data, 'params.stop.solid.phi_crit.too_deep', -1) == -1


# ---------------------------------------------------------------------------
# Solidification time
# ---------------------------------------------------------------------------


def test_extract_solidification_time_is_first_crossing_not_last():
    """Phi dips below phi_crit, recovers, then dips again: the first dip counts.

    A last-crossing implementation would return 1e4 yr instead of 100 yr.
    """
    df = pd.DataFrame(
        {
            'Time': [1.0, 10.0, 100.0, 1000.0, 10000.0],
            'Phi_global': [1.0, 0.5, 0.001, 0.5, 0.0001],
        }
    )
    t_solid = extract_solidification_time(df, phi_crit=0.005)
    assert t_solid == pytest.approx(100.0)
    assert t_solid != pytest.approx(10000.0)


def test_extract_solidification_time_counts_phi_equal_to_phi_crit():
    """Phi exactly equal to phi_crit has solidified ('<=', as in the stop criterion
    of utils.terminate); a strict '<' would return 100 yr instead of 10 yr, and a
    run stopped with status 10 on that row would get no solidification time.
    """
    df = pd.DataFrame({'Time': [1.0, 10.0, 100.0], 'Phi_global': [1.0, 0.005, 0.004]})
    t_solid = extract_solidification_time(df, phi_crit=0.005)
    assert t_solid == pytest.approx(10.0)
    assert t_solid != pytest.approx(100.0)


def test_extract_solidification_time_matches_the_solid_stop_criterion():
    """The analysis flags the same first row as terminate._check_solid, which
    stops the run, for rows just above, at and below phi_crit.
    """
    from types import SimpleNamespace

    from proteus.utils import terminate

    phi_crit = 0.01
    phis = [0.5, 0.010000001, 0.01, 0.002]
    stop_rows = []
    for phi in phis:
        handler = SimpleNamespace(
            hf_row={'Phi_global': phi},
            config=SimpleNamespace(
                params=SimpleNamespace(
                    stop=SimpleNamespace(solid=SimpleNamespace(phi_crit=phi_crit))
                )
            ),
            directories={},
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(terminate, 'UpdateStatusfile', lambda *a: None)
            stop_rows.append(terminate._check_solid(handler))
    df = pd.DataFrame({'Time': [1.0, 2.0, 3.0, 4.0], 'Phi_global': phis})
    assert stop_rows == [False, False, True, True]
    assert extract_solidification_time(df, phi_crit) == pytest.approx(3.0)


def test_extract_solidification_time_nan_without_crossing_or_columns():
    """No crossing, no helpfile, or missing columns give NaN, never a spurious time."""
    assert np.isnan(
        extract_solidification_time(
            pd.DataFrame({'Time': [1.0, 2.0], 'Phi_global': [1.0, 0.9]}), 0.005
        )
    )
    assert np.isnan(extract_solidification_time(None, phi_crit=0.005))
    assert np.isnan(extract_solidification_time(pd.DataFrame({'Time': [1.0]}), 0.005))
    assert np.isnan(extract_solidification_time(pd.DataFrame({'Phi_global': [0.0]}), 0.005))


def _case(index, category, status, init_params=None, final_row=None, t_solid=np.nan):
    """Return one case dict as load_grid_cases builds it."""
    return {
        'index': index,
        'status': status,
        'category': category,
        'init_params': init_params or {},
        'final_row': final_row or {},
        'solidification_time': t_solid,
        'solid_stop_enabled': True,
    }


def _phi_series():
    """Phi crosses 0.01 at 20 yr and 0.005 only at 50 yr, so phi_crit choice shows."""
    return pd.DataFrame(
        {
            'Time': [1.0, 20.0, 50.0],
            'Phi_global': [1.0, 0.007, 0.001],
            'P_surf': [10.0, 7.0, 5.0],
        }
    )


def _solid_toml(phi_crit=None, enabled=None):
    """Return init_coupler.toml text with a mass and an optional [params.stop.solid]."""
    lines = ['[planet]', 'mass_tot = 1.0']
    if phi_crit is not None or enabled is not None:
        lines.append('[params.stop.solid]')
    if phi_crit is not None:
        lines.append(f'phi_crit = {phi_crit}')
    if enabled is not None:
        lines.append(f'enabled = {str(enabled).lower()}')
    return '\n'.join(lines) + '\n'


def test_load_grid_cases_uses_each_case_phi_crit_and_ignores_the_stop_switch(tmp_path, caplog):
    """Case 0 sets phi_crit = 0.005 (solidifies at 50 yr); case 1 omits it and uses
    the base 0.01 (20 yr); case 2 ran with its solid stop disabled and phi_crit =
    0.005, so it still solidifies at 50 yr (a fixed-end-time grid), with one info
    line. Ignoring the per-case value would give 20 yr for cases 0 and 2.
    """
    for index, text in enumerate(
        [_solid_toml(0.005, True), _solid_toml(), _solid_toml(0.005, False)]
    ):
        case_dir = _write_case(tmp_path, index, status=10, helpfile=_phi_series())
        (case_dir / 'init_coupler.toml').write_text(text)
    with caplog.at_level('INFO'):
        cases = load_grid_cases(tmp_path, {'enabled': True, 'phi_crit': 0.01})
    df = build_summary_dataframe(cases, [])
    assert df['solidification_time'].tolist() == pytest.approx([50.0, 20.0, 50.0])
    assert [c['solid_stop_enabled'] for c in cases] == [True, True, False]
    assert any('1 case(s) ran with the solidification stop disabled' in r.message
               for r in caplog.records)  # fmt: skip


def test_base_disabled_solid_stop_still_gives_a_solidification_time(tmp_path):
    """A case without its own solid settings inherits a disabled base stop; its
    solidification time is still the first crossing of the base phi_crit (20 yr),
    not NaN, and its final row is recorded. A case that never reaches phi_crit
    stays NaN.
    """
    case_dir = _write_case(tmp_path, 0, status=13, helpfile=_phi_series())
    (case_dir / 'init_coupler.toml').write_text(_solid_toml())
    hot = _phi_series().assign(Phi_global=[1.0, 0.9, 0.8])
    (_write_case(tmp_path, 1, status=13, helpfile=hot) / 'init_coupler.toml').write_text(
        _solid_toml()
    )
    df = build_summary_dataframe(
        load_grid_cases(tmp_path, {'enabled': False, 'phi_crit': 0.01}), []
    )
    assert df.loc[0, 'solidification_time'] == pytest.approx(20.0)
    assert df.loc[0, 'P_surf'] == pytest.approx(5.0)  # final row still recorded
    assert np.isnan(df.loc[1, 'solidification_time'])


def test_load_grid_cases_keeps_only_the_summary_of_each_helpfile(tmp_path):
    """No case holds its helpfile after loading: only the final row (a dict, the
    last of 2000 rows) and the solidification time, so memory does not grow with
    the length of the runs. The summary matches the full helpfile.
    """
    n_rows = 2000
    long_run = pd.DataFrame(
        {
            'Time': np.linspace(1.0, 2.0e6, n_rows),
            'Phi_global': np.linspace(1.0, 0.0, n_rows),
            'T_surf': np.linspace(3000.0, 1500.0, n_rows),
        }
    )
    _write_case(tmp_path, 0, status=10, mass=1.0, helpfile=long_run)
    (case,) = load_grid_cases(tmp_path)
    assert not any(isinstance(v, (pd.DataFrame, pd.Series)) for v in case.values())
    assert case['final_row'] == pytest.approx(long_run.iloc[-1].to_dict())
    assert case['final_row']['T_surf'] == pytest.approx(1500.0)  # last row, not first
    first = long_run.loc[long_run['Phi_global'] <= 0.005, 'Time'].iloc[0]
    assert case['solidification_time'] == pytest.approx(first)


def test_build_summary_records_params_final_row_and_category():
    """Each row carries the case's tested parameter, the LAST helpfile row, and its
    status category; a case without a helpfile has NaN outputs, not copied values.
    """
    cases = [
        _case(0, 'Completed', 'Completed (solidified)', {'planet': {'mass_tot': 1.5}},
              _phi_series().iloc[-1].to_dict(), 50.0),
        _case(1, 'Error', 'Error (Interior model)', {'planet': {'mass_tot': 2.5}}),
    ]  # fmt: skip
    df = build_summary_dataframe(cases, ['planet.mass_tot'])
    assert list(df['case_number']) == [0, 1]
    assert list(df['status_category']) == ['Completed', 'Error']
    assert df['planet.mass_tot'].tolist() == pytest.approx([1.5, 2.5])
    assert df.loc[0, 'P_surf'] == pytest.approx(5.0)  # last row (first row has 10)
    assert pd.isna(df.loc[1, 'P_surf'])
    assert np.isnan(df.loc[1, 'solidification_time'])


def test_default_solid_settings_match_the_config_schema():
    """The fallback solid settings are the StopSolid schema defaults, not a copy."""
    assert DEFAULT_PHI_CRIT == pytest.approx(0.01)
    default = _resolve_base_solid({})
    assert default['enabled'] is True
    assert default['phi_crit'] == pytest.approx(0.01)
    base = _resolve_base_solid(
        {'params': {'stop': {'solid': {'phi_crit': 0.02, 'enabled': False}}}}
    )
    assert base['enabled'] is False
    assert base['phi_crit'] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# CSV filtering
# ---------------------------------------------------------------------------


def test_write_summary_csvs_filters_by_status_category(tmp_path):
    """completed keeps only Completed; running_error keeps Running, Error and
    Unknown (a case with no status did not finish); all keeps every case.
    """
    cases = [
        _case(0, 'Completed', 'Completed (solidified)'),
        _case(1, 'Running', 'Running'),
        _case(2, 'Error', 'Error (Atmosphere model)'),
        _case(3, 'Unknown', 'Missing status file'),
    ]
    paths = write_summary_csvs(build_summary_dataframe(cases, []), tmp_path)
    read = {mode: pd.read_csv(path, sep='\t') for mode, path in paths.items()}
    assert list(read['all']['case_number']) == [0, 1, 2, 3]
    assert list(read['completed']['case_number']) == [0]
    assert list(read['running_error']['case_number']) == [1, 2, 3]


# ---------------------------------------------------------------------------
# [post_processing] options
# ---------------------------------------------------------------------------


def test_post_processing_defaults_when_table_absent():
    """Without a table every step is on and the ECDF uses the documented outputs."""
    opts = _read_post_processing_options({})
    assert opts['update_csv'] and opts['plot_status'] and opts['plot_ecdf']
    assert opts['output_variables'] == list(DEFAULT_OUTPUT_VARIABLES)
    assert opts['output_variables'][0] == 'solidification_time'
    assert opts['colormap'] is None
    assert opts['plot_format'] is None
    assert opts['style'] is None


def test_post_processing_accepts_both_colourmap_spellings():
    """Either 'colormap' or 'colourmap' sets the colour map; different values in
    both raise rather than one silently winning.
    """
    assert (
        _read_post_processing_options({'post_processing': {'colormap': 'vik'}})['colormap']
        == 'vik'
    )
    assert (
        _read_post_processing_options({'post_processing': {'colourmap': 'batlow'}})['colormap']
        == 'batlow'
    )
    with pytest.raises(ValueError, match='both colormap and colourmap'):
        _read_post_processing_options(
            {'post_processing': {'colormap': 'vik', 'colourmap': 'batlow'}}
        )


@pytest.mark.parametrize(
    'table, match',
    [
        ({'plot_ecfd': True}, 'Unknown key'),
        ({'update_csv': 'false'}, 'update_csv must be a bool'),
        ({'output_variables': 'T_surf'}, 'output_variables must be a list'),
        ({'output_variables': ['T_surf', 3]}, 'list of strings'),
        (True, 'must be a table'),
        ({'style': 'grey'}, 'style'),
        ({'plot_format': 'svg'}, 'plot_format'),
        ({'colormap': 'batlowk_r'}, 'colour map'),
    ],
    ids=['typo_key', 'string_bool', 'scalar_outputs', 'non_string_output', 'not_a_table',
         'unknown_style', 'unsupported_format', 'misspelt_colour_map'],
)  # fmt: skip
def test_post_processing_rejects_invalid_tables(table, match):
    """A typo or wrong type in [post_processing] raises with the offending key,
    instead of being ignored ('false' as a string would otherwise be truthy).
    """
    with pytest.raises(ValueError, match=match):
        _read_post_processing_options({'post_processing': table})
    assert _read_post_processing_options({'post_processing': {}})['plot_ecdf'] is True
    valid = {'style': 'dark', 'plot_format': 'PDF', 'colormap': 'cmc.batlowK_r'}
    assert _read_post_processing_options({'post_processing': valid})['style'] == 'dark'


def test_get_tested_parameters_reads_axes_but_not_post_processing():
    """Axis values are the swept values (pinned here), and the [post_processing]
    table is a grid setting, never an axis.
    """
    config = {
        **_GRID_SETTINGS,
        'post_processing': {'plot_ecdf': False},
        'planet.mass_tot': {'method': 'direct', 'values': [1.0, 0.5, 1.0]},
        'orbit.eccentricity': {'method': 'logspace', 'start': 0.001, 'stop': 0.1, 'count': 3},
    }
    tested = get_tested_parameters(config, 'fake.grid.toml')
    assert set(tested) == {'planet.mass_tot', 'orbit.eccentricity'}
    assert tested['planet.mass_tot'] == pytest.approx([0.5, 1.0])
    assert tested['orbit.eccentricity'] == pytest.approx([1e-3, 1e-2, 1e-1])


# ---------------------------------------------------------------------------
# Loading cases and the end-to-end analysis on a tmp_path grid
# ---------------------------------------------------------------------------


def _write_case(grid_dir, index, status=None, mass=None, helpfile=None):
    """Write case_<index> with an optional status code, init_coupler and helpfile."""
    case_dir = grid_dir / f'case_{index:06d}'
    case_dir.mkdir(parents=True)
    if status is not None:
        (case_dir / 'status').write_text(f'{status}\n')
    if mass is not None:
        (case_dir / 'init_coupler.toml').write_text(
            f'[planet]\nmass_tot = {mass}\n[params.stop.solid]\nenabled = true\n'
            'phi_crit = 0.005\n'
        )
    if helpfile is not None:
        helpfile.to_csv(case_dir / 'runtime_helpfile.csv', sep='\t', index=False)
    return case_dir


def test_load_grid_cases_skips_stray_entries_and_tolerates_missing_files(tmp_path):
    """A case_* file or non-numeric folder is skipped (not a crash); a case with no
    status file is Unknown and one without a helpfile has no output.
    """
    _write_case(tmp_path, 0, status=10, mass=1.0, helpfile=_phi_series())
    _write_case(tmp_path, 2)  # started, but no status or helpfile written yet
    (tmp_path / 'case_notes').mkdir()
    (tmp_path / 'case_000001.tar').write_text('archive')
    cases = load_grid_cases(tmp_path)
    assert [c['index'] for c in cases] == [0, 2]
    assert [c['category'] for c in cases] == ['Completed', 'Unknown']
    assert cases[0]['status'] == 'Completed (solidified)'
    assert cases[0]['init_params']['planet']['mass_tot'] == pytest.approx(1.0)
    assert cases[0]['final_row']['P_surf'] == pytest.approx(5.0)
    assert cases[1]['final_row'] == {}
    assert np.isnan(cases[1]['solidification_time'])


def test_case_that_failed_before_init_coupler_reads_its_grid_config(tmp_path, caplog):
    """A case that stopped before writing init_coupler.toml (status 20, a config
    error) takes its swept value from cfgs/case_<N>.toml instead of NaN; a case
    with neither file has an empty config_source and a NaN value.
    """
    (tmp_path / 'cfgs').mkdir()
    (tmp_path / 'cfgs' / 'case_000000.toml').write_text('[planet]\nmass_tot = 2.5\n')
    _write_case(tmp_path, 0, status=20)
    _write_case(tmp_path, 1, status=20)
    _write_case(tmp_path, 2, status=10, mass=0.5, helpfile=_phi_series())
    with caplog.at_level('WARNING'):
        cases = load_grid_cases(tmp_path)
    df = build_summary_dataframe(cases, ['planet.mass_tot'])
    assert df.loc[0, 'planet.mass_tot'] == pytest.approx(2.5)
    assert df.loc[0, 'config_source'] == 'cfgs/case_000000.toml'
    assert np.isnan(df.loc[1, 'planet.mass_tot'])
    assert df.loc[1, 'config_source'] == ''
    assert df.loc[2, 'config_source'] == 'init_coupler.toml'
    assert any('reading cfgs/case_000000.toml' in r.message for r in caplog.records)


def test_load_grid_cases_refuses_a_cut_helpfile(tmp_path, caplog):
    """A helpfile whose last row was cut mid-write (no final newline, too few
    fields) is refused with a warning, so no partial final row reaches the CSV.
    """
    case_dir = _write_case(tmp_path, 0, status=25, mass=1.0)
    (case_dir / 'runtime_helpfile.csv').write_text('Time\tPhi_global\n1.0\t0.9\n2.0')
    with caplog.at_level('WARNING'):
        cases = load_grid_cases(tmp_path)
    assert cases[0]['final_row'] == {}
    assert cases[0]['category'] == 'Error'
    assert any('cannot read runtime_helpfile.csv' in rec.message for rec in caplog.records)


def _grid_project(tmp_path, monkeypatch, copy_axis=None, post=''):
    """Lay out <proteus>/output/g with a 3-case mass axis; return the -c config path."""
    monkeypatch.setattr(ga, 'get_proteus_dir', lambda: str(tmp_path))
    grid_dir = tmp_path / 'output' / 'g'
    grid_dir.mkdir(parents=True)
    settings = _SETTINGS_TOML
    config = tmp_path / 'g.grid.toml'
    config.write_text(
        f'{settings}\n{post}\n["planet.mass_tot"]\nmethod = "direct"\nvalues = [9.0]\n'
    )
    if copy_axis is not None:
        (grid_dir / 'copy.grid.toml').write_text(f'{settings}\n{copy_axis}\n')
    (grid_dir / 'ref_config.toml').write_text('[params.out]\nplot_fmt = "pdf"\n')
    return config, grid_dir


_COPY_AXIS = '["planet.mass_tot"]\nmethod = "direct"\nvalues = [0.5, 1.0, 2.0]\n'


def test_analyse_grid_prefers_copy_grid_and_counts_categories(tmp_path, monkeypatch):
    """Axes come from copy.grid.toml (3 values), not the edited -c file (9.0); the
    CSVs hold the two started cases; plotting gets the base plot format.
    """
    config, grid_dir = _grid_project(tmp_path, monkeypatch, copy_axis=_COPY_AXIS)
    _write_case(grid_dir, 0, status=10, mass=0.5, helpfile=_phi_series())
    _write_case(grid_dir, 1, status=22, mass=1.0)
    calls = []
    monkeypatch.setattr(plot_grid, 'make_grid_plots', lambda *a, **k: calls.append(a))
    analyse_grid(config)

    extracted = grid_dir / 'post_processing' / 'extracted_data'
    all_df = pd.read_csv(extracted / 'final_extracted_data_all.csv', sep='\t')
    assert list(all_df['status_category']) == ['Completed', 'Error']
    assert all_df.loc[0, 'solidification_time'] == pytest.approx(50.0)  # case phi_crit 0.005
    opts, _, tested, _, base_fmt = calls[0]
    assert list(tested) == ['planet.mass_tot']
    assert tested['planet.mass_tot'] == pytest.approx([0.5, 1.0, 2.0])
    assert base_fmt == 'pdf'
    assert opts['output_variables'] == list(DEFAULT_OUTPUT_VARIABLES)


def test_analyse_grid_raises_on_empty_grid_and_outdated_copy(tmp_path, monkeypatch):
    """No case folder yet raises FileNotFoundError (not a KeyError); a copy.grid.toml
    with a key the current schema refuses raises an error naming that file.
    """
    config, grid_dir = _grid_project(tmp_path, monkeypatch, post='[post_processing]\n')
    with pytest.raises(FileNotFoundError, match='No case_<N> folders'):
        analyse_grid(config)
    (grid_dir / 'copy.grid.toml').write_text(
        config.read_text() + '\n["delivery.elements.CH_ratio"]\nmethod = "direct"\n'
        'values = [1.0]\n'
    )
    with pytest.raises(ValueError, match='copy.grid.toml'):
        analyse_grid(config)
    assert not (grid_dir / 'post_processing').exists()  # nothing written on failure


def test_analyse_grid_without_update_requires_existing_csvs(tmp_path, monkeypatch):
    """update_csv = false plots from existing CSVs, so missing ones raise early."""
    config, grid_dir = _grid_project(
        tmp_path, monkeypatch, post='[post_processing]\nupdate_csv = false\n'
    )
    _write_case(grid_dir, 0, status=10, mass=0.5, helpfile=_phi_series())
    with pytest.raises(FileNotFoundError, match='update_csv is false'):
        analyse_grid(config)
    assert not (grid_dir / 'post_processing').exists()


def test_unreadable_init_coupler_warns_and_falls_back_to_grid_config(tmp_path, caplog):
    """A corrupt init_coupler.toml (a run killed while writing it) is reported, and
    the swept value comes from cfgs/case_<N>.toml instead of a TOMLDecodeError.
    """
    (tmp_path / 'cfgs').mkdir()
    (tmp_path / 'cfgs' / 'case_000000.toml').write_text('[planet]\nmass_tot = 2.5\n')
    case_dir = _write_case(tmp_path, 0, status=20)
    (case_dir / 'init_coupler.toml').write_text('[planet\nmass_tot = ')
    with caplog.at_level('WARNING'):
        cases = load_grid_cases(tmp_path)
    df = build_summary_dataframe(cases, ['planet.mass_tot'])
    assert df.loc[0, 'planet.mass_tot'] == pytest.approx(2.5)
    assert df.loc[0, 'config_source'] == 'cfgs/case_000000.toml'
    assert any('cannot read init_coupler.toml' in r.message for r in caplog.records)


def test_analyse_grid_without_ref_config_uses_schema_defaults(tmp_path, monkeypatch):
    """Without ref_config.toml the base solid stop is the schema default (enabled,
    phi_crit = 0.01): a case with no [params.stop.solid] table solidifies at the
    first row with Phi_global <= 0.01, which is 20 yr in _phi_series (Phi = 0.007),
    not 50 yr (the 0.005 crossing). Plotting gets no base format.
    """
    config, grid_dir = _grid_project(tmp_path, monkeypatch)
    (grid_dir / 'ref_config.toml').unlink()
    case_dir = _write_case(grid_dir, 0, status=10, helpfile=_phi_series())
    (case_dir / 'init_coupler.toml').write_text('[planet]\nmass_tot = 0.5\n')
    calls = []
    monkeypatch.setattr(plot_grid, 'make_grid_plots', lambda *a, **k: calls.append(a))
    analyse_grid(config)
    all_df = pd.read_csv(
        grid_dir / 'post_processing' / 'extracted_data' / 'final_extracted_data_all.csv',
        sep='\t',
    )
    assert all_df.loc[0, 'solidification_time'] == pytest.approx(20.0)
    assert calls[0][4] is None


def test_analyse_grid_without_update_plots_the_existing_csvs(tmp_path, monkeypatch):
    """update_csv = false hands the CSVs already on disk to the plots without
    reading the cases again: it works once the case folders are gone, and the CSV
    contents are left as they were.
    """
    config, grid_dir = _grid_project(tmp_path, monkeypatch)
    _write_case(grid_dir, 0, status=10, mass=0.5, helpfile=_phi_series())
    calls = []
    monkeypatch.setattr(plot_grid, 'make_grid_plots', lambda *a, **k: calls.append(a))
    analyse_grid(config)
    extracted = grid_dir / 'post_processing' / 'extracted_data'
    before = {p.name: p.read_text() for p in extracted.iterdir()}

    shutil.rmtree(grid_dir / 'case_000000')
    config.write_text(config.read_text() + '\n[post_processing]\nupdate_csv = false\n')
    analyse_grid(config)
    assert calls[1][1] == {
        m: extracted / f'final_extracted_data_{m}.csv'
        for m in ('all', 'completed', 'running_error')
    }
    assert {p.name: p.read_text() for p in extracted.iterdir()} == before


def test_analyse_grid_raises_when_grid_output_is_missing(tmp_path, monkeypatch):
    """A grid config whose output folder does not exist raises a clear error naming
    it, before anything is written.
    """
    config, grid_dir = _grid_project(tmp_path, monkeypatch)
    shutil.rmtree(grid_dir)
    with pytest.raises(FileNotFoundError, match='Grid output directory not found'):
        analyse_grid(config)
    assert not grid_dir.exists()


def test_analyse_grid_with_both_plots_off_only_writes_csvs(tmp_path, monkeypatch):
    """plot_status = false and plot_ecdf = false write the CSVs and never call the
    plotting code (so the CSV-only path needs no matplotlib).
    """
    config, grid_dir = _grid_project(
        tmp_path,
        monkeypatch,
        post='[post_processing]\nplot_status = false\nplot_ecdf = false\n',
    )
    _write_case(grid_dir, 0, status=10, mass=0.5, helpfile=_phi_series())
    calls = []
    monkeypatch.setattr(plot_grid, 'make_grid_plots', lambda *a, **k: calls.append(a))
    analyse_grid(config)
    assert calls == []
    extracted = grid_dir / 'post_processing' / 'extracted_data'
    assert sorted(p.name for p in extracted.iterdir()) == [
        'final_extracted_data_all.csv',
        'final_extracted_data_completed.csv',
        'final_extracted_data_running_error.csv',
    ]
    assert not (grid_dir / 'post_processing' / 'grid_plots').exists()


def test_analyse_grid_from_a_folder_reads_its_copy_grid_with_auto_output(tmp_path, monkeypatch):
    """The documented example has output = "auto": given the grid folder alone, the
    analysis takes the axes and the [post_processing] options from its
    copy.grid.toml and writes the CSVs, where -c alone would stop on 'auto'.
    """
    grid_dir = tmp_path / 'grid_20261016_120000_abcd'
    grid_dir.mkdir()
    copy = _SETTINGS_TOML.replace('output = "g"', 'output = "auto"')
    (grid_dir / 'copy.grid.toml').write_text(
        f'{copy}\n[post_processing]\nplot_format = "png"\n{_COPY_AXIS}'
    )
    _write_case(grid_dir, 0, status=10, mass=0.5, helpfile=_phi_series())
    calls = []
    monkeypatch.setattr(plot_grid, 'make_grid_plots', lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match='grid-analyse -o'):
        analyse_grid(grid_dir / 'copy.grid.toml')
    analyse_grid(grid_dir=grid_dir)
    all_df = pd.read_csv(
        grid_dir / 'post_processing' / 'extracted_data' / 'final_extracted_data_all.csv',
        sep='\t',
    )
    assert all_df['planet.mass_tot'].tolist() == pytest.approx([0.5])
    opts, _, tested, out_dir, _ = calls[0]
    assert opts['plot_format'] == 'png'
    assert tested['planet.mass_tot'] == pytest.approx([0.5, 1.0, 2.0])
    assert out_dir == grid_dir.resolve()


def test_analyse_grid_folder_with_config_uses_the_config_options(tmp_path, monkeypatch):
    """With both a grid folder and a grid file, the folder is analysed with the
    options of the file (here no plots), while the axes still come from the copy.
    A folder without copy.grid.toml and no file raises before writing anything.
    """
    config, grid_dir = _grid_project(
        tmp_path, monkeypatch, copy_axis=_COPY_AXIS,
        post='[post_processing]\nplot_status = false\nplot_ecdf = false\n',
    )  # fmt: skip
    _write_case(grid_dir, 0, status=10, mass=0.5, helpfile=_phi_series())
    calls = []
    monkeypatch.setattr(plot_grid, 'make_grid_plots', lambda *a, **k: calls.append(a))
    analyse_grid(config, grid_dir)
    assert calls == []
    assert (grid_dir / 'post_processing' / 'extracted_data').is_dir()
    bare = tmp_path / 'bare'
    bare.mkdir()
    with pytest.raises(FileNotFoundError, match='No copy.grid.toml'):
        analyse_grid(grid_dir=bare)
    assert not (bare / 'post_processing').exists()
