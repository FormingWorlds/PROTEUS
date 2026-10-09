"""Unit tests for ``proteus.grid.analyse``.

Covers the grid post-processing extraction helpers with no running grid:
- solidification time is the FIRST crossing of Phi_global < phi_crit, not the last
- tested-parameter axes match proteus.grid.manage.dimension_values exactly
- the summary CSVs filter cases by status category
- a missing output column becomes NaN rather than a silent substitute
- plot_format resolution and validation

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from proteus.grid import analyse as ga
from proteus.grid.analyse import (
    DEFAULT_PHI_CRIT,
    _dig,
    _read_post_processing_options,
    _resolve_base_phi_crit,
    _resolve_plot_format,
    build_summary_dataframe,
    extract_solidification_time,
    get_grid_name,
    get_tested_parameters,
    resolve_grid_dir,
    validate_output_variables,
    write_summary_csvs,
)
from proteus.grid.manage import dimension_values

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def test_get_grid_name_returns_final_path_component(tmp_path):
    """get_grid_name returns the directory's own name, independent of its parent."""
    target = tmp_path / 'my_grid'
    target.mkdir()
    assert get_grid_name(target) == 'my_grid'
    assert get_grid_name(tmp_path / 'other' / 'deep_grid') == 'deep_grid'


def test_resolve_grid_dir_joins_output_under_proteus_output(tmp_path, monkeypatch):
    """resolve_grid_dir places the config 'output' under <proteus>/output/."""
    monkeypatch.setattr(ga, 'get_proteus_dir', lambda: str(tmp_path))
    resolved = resolve_grid_dir({'output': 'run_a/'})
    assert resolved == (tmp_path / 'output' / 'run_a').resolve()
    # A stray leading/trailing slash must not escape the output directory.
    assert resolve_grid_dir({'output': 'run_b'}).parent == (tmp_path / 'output').resolve()


def test_dig_returns_nested_value_or_default():
    """_dig walks a dotted path and returns the default when any key is absent."""
    data = {'params': {'stop': {'solid': {'phi_crit': 0.005}}}}
    assert _dig(data, 'params.stop.solid.phi_crit') == pytest.approx(0.005)
    assert _dig(data, 'params.stop.missing', 'fallback') == 'fallback'
    assert _dig(data, 'params.stop.solid.phi_crit.too_deep', -1) == -1


def test_extract_solidification_time_is_first_crossing_not_last():
    """Solidification time is the FIRST time Phi_global < phi_crit.

    Phi dips below phi_crit, recovers above it, then dips again. The correct
    answer is the first dip; a last-crossing implementation would return the
    much later second dip, so the discrimination guard rejects it.
    """
    df = pd.DataFrame(
        {
            'Time': [1.0, 10.0, 100.0, 1000.0, 10000.0],
            'Phi_global': [1.0, 0.5, 0.001, 0.5, 0.0001],
        }
    )
    t_solid = extract_solidification_time(df, phi_crit=0.005)
    assert t_solid == pytest.approx(100.0)  # first crossing
    assert t_solid != pytest.approx(10000.0)  # not the last crossing


def test_extract_solidification_time_nan_when_never_solidifies():
    """A mantle that stays above phi_crit yields NaN, not a spurious time."""
    df = pd.DataFrame({'Time': [1.0, 2.0], 'Phi_global': [1.0, 0.9]})
    assert np.isnan(extract_solidification_time(df, phi_crit=0.005))
    assert np.isnan(extract_solidification_time(None, phi_crit=0.005))


def test_extract_solidification_time_nan_when_columns_missing():
    """Missing Phi_global or Time columns give NaN rather than raising."""
    df = pd.DataFrame({'Time': [1.0, 2.0]})
    assert np.isnan(extract_solidification_time(df, phi_crit=0.005))
    df2 = pd.DataFrame({'Phi_global': [1.0, 0.0]})
    assert np.isnan(extract_solidification_time(df2, phi_crit=0.005))


def test_get_tested_parameters_matches_manage_dimension_values():
    """The analysis axes are built by the same manage.dimension_values the grid
    uses, so a parameter's swept values match exactly for every method.
    """
    config = {
        'output': 'x',
        'symlink': '',
        'ref_config': 'y',
        'use_slurm': False,
        'max_jobs': 1,
        'max_days': 1,
        'max_mem': 1,
        'planet.mass_tot': {'method': 'direct', 'values': [1.0, 0.5, 1.0]},
        'orbit.eccentricity': {
            'method': 'logspace',
            'start': 0.001,
            'stop': 0.1,
            'count': 3,
        },
    }
    tested = get_tested_parameters(config, 'fake.grid.toml')
    assert tested['planet.mass_tot'] == dimension_values(config['planet.mass_tot'])
    assert tested['orbit.eccentricity'] == dimension_values(config['orbit.eccentricity'])


def _two_cases():
    """Two cases: one completed with results, one errored without a helpfile."""
    completed = {
        'index': 0,
        'status': 'Completed (solidified)',
        'category': 'Completed',
        'init_params': {
            'planet': {'mass_tot': 1.0},
            'params': {'stop': {'solid': {'phi_crit': 0.005}}},
        },
        'output': pd.DataFrame(
            {'Time': [1.0, 50.0], 'Phi_global': [1.0, 0.001], 'P_surf': [10.0, 5.0]}
        ),
    }
    errored = {
        'index': 1,
        'status': 'Error (died)',
        'category': 'Error',
        'init_params': {'planet': {'mass_tot': 2.0}},
        'output': None,
    }
    return [completed, errored]


def test_build_summary_dataframe_one_row_per_case_with_params():
    """Each case becomes one row carrying its tested parameter and final outputs."""
    df = build_summary_dataframe(_two_cases(), ['planet.mass_tot'])
    assert list(df['case_number']) == [0, 1]
    assert df.loc[0, 'planet.mass_tot'] == pytest.approx(1.0)
    # Final-row output taken from the last helpfile row.
    assert df.loc[0, 'P_surf'] == pytest.approx(5.0)
    assert df.loc[0, 'solidification_time'] == pytest.approx(50.0)


def test_build_summary_dataframe_missing_output_is_nan():
    """A case with no helpfile has NaN outputs and NaN solidification time,
    not a value copied from another column or case.
    """
    df = build_summary_dataframe(_two_cases(), ['planet.mass_tot'])
    assert np.isnan(df.loc[1, 'solidification_time'])
    assert pd.isna(df.loc[1, 'P_surf'])


def test_write_summary_csvs_filters_by_status_category(tmp_path):
    """The completed and running_error CSVs keep only their category's cases,
    while the all CSV keeps every case.
    """
    cases = _two_cases()
    summary = build_summary_dataframe(cases, ['planet.mass_tot'])
    paths = write_summary_csvs(summary, cases, tmp_path)

    all_df = pd.read_csv(paths['all'], sep='\t')
    completed_df = pd.read_csv(paths['completed'], sep='\t')
    running_error_df = pd.read_csv(paths['running_error'], sep='\t')

    assert len(all_df) == 2
    assert list(completed_df['case_number']) == [0]
    assert list(running_error_df['case_number']) == [1]


def test_validate_output_variables_drops_missing_without_substitute():
    """Requested outputs absent from the data are dropped (never replaced by
    another column such as Time), and present ones are returned in order.
    """
    df = pd.DataFrame({'Time': [1.0], 'P_surf': [2.0]})
    present = validate_output_variables(df, ['P_surf', 'not_a_column'])
    assert present == ['P_surf']
    assert 'Time' not in present


def test_resolve_grid_dir_rejects_auto_output():
    """An 'auto' output cannot be analysed (the run folder name is unknowable),
    so resolve_grid_dir raises a clear error instead of pointing at output/auto.
    """
    with pytest.raises(ValueError) as exc:
        resolve_grid_dir({'output': 'auto'})
    assert 'auto' in str(exc.value)
    with pytest.raises(ValueError) as exc_upper:
        resolve_grid_dir({'output': 'AUTO'})  # case-insensitive
    assert 'auto' in str(exc_upper.value).lower()


def test_read_post_processing_options_accepts_both_colourmap_spellings():
    """Either 'colormap' or 'colourmap' sets the colour map, so the UK spelling
    used elsewhere in the plotting code is not silently ignored.
    """
    us = _read_post_processing_options({'post_processing': {'colormap': 'vik'}})
    uk = _read_post_processing_options({'post_processing': {'colourmap': 'batlow'}})
    assert us['colormap'] == 'vik'
    assert uk['colormap'] == 'batlow'


def test_resolve_base_phi_crit_reads_ref_config_else_default(tmp_path):
    """The base phi_crit comes from ref_config.toml, falling back to the schema
    default when the file or key is absent.
    """
    (tmp_path / 'ref_config.toml').write_text('[params.stop.solid]\nphi_crit = 0.02\n')
    assert _resolve_base_phi_crit(tmp_path) == pytest.approx(0.02)
    assert _resolve_base_phi_crit(tmp_path / 'nowhere') == pytest.approx(DEFAULT_PHI_CRIT)


def test_resolve_plot_format_defaults_reads_base_and_validates(tmp_path):
    """plot_format falls back to the base config's params.out.plot_fmt, then
    'png', and an unsupported value raises instead of writing a '.None' file.
    """
    ref = tmp_path / 'ref_config.toml'
    ref.write_text('[params.out]\nplot_fmt = "pdf"\n')
    assert _resolve_plot_format('pdf', tmp_path) == 'pdf'  # explicit request wins
    assert _resolve_plot_format(None, tmp_path) == 'pdf'  # from base config
    assert _resolve_plot_format(None, tmp_path / 'empty') == 'png'  # final fallback
    with pytest.raises(ValueError):
        _resolve_plot_format('svg', tmp_path)  # svg is not a supported format
