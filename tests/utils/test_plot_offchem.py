"""Tests for proteus.utils.plot_offchem readers.

The VULCAN result of each chemistry snapshot is the NetCDF file
``data/<time>_chem.nc``, with the time in the sub-year form of
`format_subyear_time` (``12p320`` at 12.32 yr). The readers take mixing
ratios, layer pressures and temperatures from it, convert pressure from
dyn cm-2 to bar, clip the mixing ratios, and refuse the retired
``read_const`` path. Checked: unit conversion (against a wrong divisor),
species-column order, clipping bounds, and sorted float years per grid case.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from proteus.utils.plot_offchem import offchem_read_year
from tests.helpers._vulcan_netcdf import vulcan_layers, write_vulcan_netcdf

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Three layers from 1e8 to 1e4 dyn cm-2 (100 bar to 0.01 bar), so a /1e5 or
# /1e7 divisor lands a factor 10 away from the expected bar values.
_LAYERS = {
    'p': np.array([1.0e8, 1.0e6, 1.0e4]),
    'pl': np.array([3.0e8, 1.0e7, 1.0e5, 3.0e3]),
    'tmp': np.array([1850.0, 950.0, 300.0]),
    'zl': np.array([0.0, 1.0e7, 2.0e7, 3.0e7]),
    'Kzz': np.array([1.0e6, 1.0e7]),
}
# Asymmetric columns, so a swapped species axis is visible
_X_GAS = np.array([[0.40, 0.10], [0.30, 0.20], [0.20, 0.30]])


def _write_chem(out_dir, token, x_gas=_X_GAS, layers=_LAYERS):
    """Write a two-species VULCAN NetCDF as ``<out_dir>/data/<token>_chem.nc``."""
    data_dir = os.path.join(out_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    write_vulcan_netcdf(
        os.path.join(data_dir, f'{token}_chem.nc'), layers, ['H2O', 'CO2'], x_gas
    )


def test_offchem_read_year_reads_subyear_netcdf(tmp_path):
    """A snapshot at 12.32 yr is read from data/12p320_chem.nc, with each
    species column, pressure in bar and temperature in K."""
    _write_chem(str(tmp_path), '12p320')

    out = offchem_read_year(str(tmp_path), 12.32)

    assert out['year'] == pytest.approx(12.32)
    # Column 0 is H2O; a swapped axis would give 0.10 at the bottom layer
    np.testing.assert_allclose(out['mx_H2O'], [0.40, 0.30, 0.20])
    np.testing.assert_allclose(out['mx_CO2'], [0.10, 0.20, 0.30])
    np.testing.assert_allclose(out['temperature'], [1850.0, 950.0, 300.0])


def test_offchem_read_year_converts_pressure_to_bar(tmp_path):
    """Pressure is divided by 1e6 (dyn cm-2 to bar), not by 1e5 (Pa to bar)."""
    _write_chem(str(tmp_path), '1000p000')

    out = offchem_read_year(str(tmp_path), 1000.0)

    np.testing.assert_allclose(out['pressure'], [100.0, 1.0, 0.01], rtol=1e-12)
    # Discrimination: the Pa-to-bar divisor would give 1000 bar at the bottom
    assert abs(out['pressure'][0] - 1.0e8 / 1.0e5) > 100.0
    # Bottom-up order is kept: pressure falls with index
    assert np.all(np.diff(out['pressure']) < 0)


def test_offchem_read_year_reads_whole_year_name(tmp_path):
    """A result named in the whole-year form (500_chem.nc) is still found."""
    _write_chem(str(tmp_path), '500')

    out = offchem_read_year(str(tmp_path), 500.0)

    assert out['year'] == pytest.approx(500.0)
    np.testing.assert_allclose(out['mx_CO2'], [0.10, 0.20, 0.30])


def test_offchem_read_year_raises_when_no_file(tmp_path):
    """A missing snapshot raises FileNotFoundError naming the expected path."""
    _write_chem(str(tmp_path), '100p000')

    with pytest.raises(FileNotFoundError, match='200p000_chem.nc'):
        offchem_read_year(str(tmp_path), 200.0)
    # The file at another time is not taken as a substitute
    assert sorted(os.listdir(tmp_path / 'data')) == ['100p000_chem.nc']


def test_offchem_read_year_read_const_unsupported(tmp_path):
    """read_const=True is the retired vulcan_cfg.py path and raises, while the
    same call without it succeeds on the same file."""
    _write_chem(str(tmp_path), '500p000')

    with pytest.raises(NotImplementedError, match='read_const'):
        offchem_read_year(str(tmp_path), 500.0, read_const=True)
    out = offchem_read_year(str(tmp_path), 500.0, read_const=False)
    assert out['year'] == pytest.approx(500.0)
    assert set(out) >= {'mx_H2O', 'mx_CO2', 'pressure', 'temperature'}


def test_offchem_read_year_clip_applied_to_extremes(tmp_path):
    """Mixing ratios outside [mx_clip_min, mx_clip_max] are clipped to the bounds."""
    # 0 and 1e-40 lie below the 1e-30 floor; 1.5 and 2.0 above the ceiling of 1
    x_gas = np.array([[0.0, 1.5], [1.0e-40, 2.0], [0.5, 0.5]])
    _write_chem(str(tmp_path), '300p000', x_gas=x_gas)

    out = offchem_read_year(str(tmp_path), 300.0, mx_clip_min=1.0e-30, mx_clip_max=1.0)

    np.testing.assert_allclose(out['mx_H2O'][:2], [1.0e-30, 1.0e-30])
    np.testing.assert_allclose(out['mx_CO2'][:2], [1.0, 1.0])
    assert out['mx_H2O'][2] == pytest.approx(0.5)
    assert out['mx_CO2'][2] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# offchem_read_grid and offchem_slice_grid: grid-level aggregators.
# ---------------------------------------------------------------------------


def _build_grid_case(case_dir, case_options, snapshot_tokens):
    """Lay down a case folder with a TOML config and one data/<token>_chem.nc
    per snapshot."""
    os.makedirs(case_dir, exist_ok=True)
    for token in snapshot_tokens:
        _write_chem(case_dir, token)
    toml_text = (
        '\n'.join(['[params.case]'] + [f'{k} = {v!r}' for k, v in case_options.items()]) + '\n'
    )
    with open(os.path.join(case_dir, 'init_coupler.toml'), 'w') as fh:
        fh.write(toml_text)


def test_offchem_read_grid_raises_when_folder_is_empty(tmp_path):
    """offchem_read_grid raises when no case_* subfolders are present, rather
    than returning empty arrays."""
    from proteus.utils.plot_offchem import offchem_read_grid

    with pytest.raises(Exception, match='no grid points were found'):
        offchem_read_grid(str(tmp_path))
    assert tmp_path.is_dir()  # directory not deleted by the function


def test_offchem_read_grid_aggregates_year_data_across_cases(tmp_path, monkeypatch):
    """Two cases with two snapshots each give 2x2 float years, sorted per case,
    including a sub-year time, paired with the right case options."""
    from proteus.utils.plot_offchem import offchem_read_grid

    # Unsorted on disk; 0p500 is a sub-year time that an int parse would lose
    _build_grid_case(str(tmp_path / 'case_00'), {'name': 'alpha'}, ['100p000', '0p500'])
    _build_grid_case(str(tmp_path / 'case_01'), {'name': 'beta'}, ['200p250', '25p000'])

    def _fake_read_config(path):
        if 'case_00' in path:
            return ({'name': 'alpha'}, None)
        return ({'name': 'beta'}, None)

    monkeypatch.setattr('proteus.utils.plot_offchem.read_config', _fake_read_config)

    years, opts, data = offchem_read_grid(str(tmp_path))
    assert years.shape == (2, 2)
    assert data.shape == (2, 2)
    assert years.dtype == np.float64
    np.testing.assert_array_equal(np.sort(years, axis=1), years)
    expected = {'alpha': [0.5, 100.0], 'beta': [25.0, 200.25]}
    for i, row in enumerate(opts):
        np.testing.assert_allclose(years[i], expected[row['name']])
        assert data[i][0]['year'] == pytest.approx(expected[row['name']][0])


def test_vulcan_layers_helper_is_bottom_up():
    """The shared NetCDF fixture builds a column with pressure falling upward."""
    layers = vulcan_layers(P_b=1.0e8, P_t=1.0e2, nz=5)
    assert np.all(np.diff(layers['p']) < 0)
    assert layers['pl'][0] > layers['p'][0] > layers['pl'][1]


def test_offchem_slice_grid_keeps_only_matching_grid_points():
    """offchem_slice_grid returns a subset of the grid where every
    point's options dict matches every key/value in cvar_filter.

    Discriminating: a regression that flipped the match logic (kept
    non-matching points or excluded matching ones) would fail the
    cardinality assertion AND the per-row name check.
    """
    import numpy as np

    from proteus.utils.plot_offchem import offchem_slice_grid

    years = np.array([[100, 200], [150, 250], [300, 400]], dtype=int)
    opts = np.array([{'name': 'alpha'}, {'name': 'beta'}, {'name': 'alpha'}], dtype=dict)
    data = np.array(
        [
            [{'mx_H2O': 0.1}, {'mx_H2O': 0.2}],
            [{'mx_H2O': 0.5}, {'mx_H2O': 0.6}],
            [{'mx_H2O': 0.7}, {'mx_H2O': 0.8}],
        ],
        dtype=dict,
    )
    s_years, s_opts, s_data = offchem_slice_grid(
        years, opts, data, cvar_filter={'name': 'alpha'}
    )
    assert s_opts.shape == (2,)
    assert s_years.shape == (2, 2)
    assert s_data.shape == (2, 2)
    for row in s_opts:
        assert row['name'] == 'alpha'
    # Discrimination: a regression returning the inverse subset
    # (beta only, 1 row) would fail the shape check above.


def test_offchem_slice_grid_warns_when_filter_excludes_every_point(caplog):
    """An overly-strict filter that excludes every grid point must
    log a warning and return empty arrays.

    Edge: limit-input case. Discriminating: pin both the warning
    in the log and the zero-length result.
    """
    import logging

    import numpy as np

    from proteus.utils.plot_offchem import offchem_slice_grid

    years = np.array([[100], [200]], dtype=int)
    opts = np.array([{'name': 'alpha'}, {'name': 'beta'}], dtype=dict)
    data = np.array([[{'x': 1}], [{'x': 2}]], dtype=dict)
    with caplog.at_level(logging.WARNING, logger='fwl'):
        s_years, s_opts, s_data = offchem_slice_grid(
            years, opts, data, cvar_filter={'name': 'no-such-value'}
        )
    assert 'No grid points left after slicing' in caplog.text
    assert s_opts.shape == (0,)


def test_offchem_slice_grid_warns_on_filter_key_not_in_options(caplog):
    """A filter key absent from a grid point's options must trigger
    a warning. The point itself is then included (the source's
    `continue` skips the exclusion check for unknown keys).

    Discriminating: pin both the warning content (the missing key
    name) AND the inclusion behaviour. A regression that excluded on
    missing keys would land at zero matches.
    """
    import logging

    import numpy as np

    from proteus.utils.plot_offchem import offchem_slice_grid

    years = np.array([[100]], dtype=int)
    opts = np.array([{'name': 'alpha'}], dtype=dict)
    data = np.array([[{'x': 1}]], dtype=dict)
    with caplog.at_level(logging.WARNING, logger='fwl'):
        s_years, s_opts, s_data = offchem_slice_grid(
            years, opts, data, cvar_filter={'unknown_key': 'whatever'}
        )
    assert "'unknown_key'" in caplog.text
    assert 'is not present in OPTIONS' in caplog.text
    # The single grid point survives because the unknown filter key
    # is treated as a no-op exclusion check.
    assert s_opts.shape == (1,)
