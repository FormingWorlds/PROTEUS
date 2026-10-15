"""Unit tests for ``proteus.atmos_chem.common``.

Exercises ``read_result`` (chemistry CSV reader), ``chem_csv_name``,
``resolve_chemistry_times``, ``helpfile_row_at`` and ``read_vulcan_netcdf``.

Invariants tested:
  - Correct file path construction from module name and optional filename
  - Guard: returns None for 'none' or None module
  - Guard: returns None when the CSV file does not exist
  - Successful read of well-formed whitespace-delimited CSV
  - Snapshot names use the sub-year time token
  - Resolved chemistry times are existing snapshots, ascending, without duplicates
  - The helpfile row used for a snapshot is the nearest in time
  - VULCAN NetCDF values come back in CGS, bottom up, with filled Kzz ends as NaN

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from proteus.atmos_chem.common import (
    chem_csv_name,
    helpfile_row_at,
    read_result,
    read_vulcan_netcdf,
    resolve_chemistry_times,
)
from tests.helpers._vulcan_netcdf import vulcan_layers, write_vulcan_netcdf

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# -----------------------------------------------------------------------
# Guard: disabled module
# -----------------------------------------------------------------------


def test_read_result_none_module_returns_none():
    """When module is None, read_result returns None without touching
    the filesystem.

    Edge case: chemistry is disabled in the config.
    """
    result = read_result('/nonexistent/path', module=None)
    assert result is None
    # Adjacent check: the string 'none' is also a disabled sentinel
    result2 = read_result('/nonexistent/path', module='none')
    assert result2 is None


# -----------------------------------------------------------------------
# Guard: missing file
# -----------------------------------------------------------------------


def test_read_result_missing_file_returns_none(tmp_path):
    """When the CSV file does not exist on disk, read_result returns
    None and does not raise.

    Edge case: the output directory exists but the chemistry run
    did not produce output (solver failure, wrong module name).
    """
    # Create the offchem subdirectory but no CSV file
    offchem_dir = tmp_path / 'offchem'
    offchem_dir.mkdir()

    result = read_result(str(tmp_path), module='vulcan')
    assert result is None
    # The expected path would be <outdir>/offchem/vulcan.csv
    assert not os.path.exists(str(offchem_dir / 'vulcan.csv'))


# -----------------------------------------------------------------------
# Successful read
# -----------------------------------------------------------------------


def test_read_result_reads_whitespace_delimited_csv(tmp_path):
    """read_result parses a whitespace-delimited CSV into a DataFrame
    with correct column names and values.

    Uses a minimal 2-column, 3-row file to verify parsing. The
    delimiter is whitespace (consistent with VULCAN output format).
    """
    offchem_dir = tmp_path / 'offchem'
    offchem_dir.mkdir()

    # Write a whitespace-delimited CSV (VULCAN output format)
    csv_content = 'pressure   temperature\n1.0e5   300.0\n1.0e4   250.0\n1.0e3   200.0\n'
    csv_path = offchem_dir / 'vulcan.csv'
    csv_path.write_text(csv_content, encoding='utf-8')

    result = read_result(str(tmp_path), module='vulcan')
    assert result is not None
    assert len(result) == 3
    assert 'pressure' in result.columns
    assert 'temperature' in result.columns
    # Verify numeric values were parsed correctly
    assert result['pressure'].iloc[0] == pytest.approx(1.0e5, rel=1e-12)
    assert result['temperature'].iloc[2] == pytest.approx(200.0, rel=1e-12)


def test_read_result_custom_filename(tmp_path):
    """When filename is provided, read_result uses it instead of
    constructing '<module>.csv'.

    This path reads the file of one snapshot (e.g. 'vulcan_5000p000.csv').
    """
    offchem_dir = tmp_path / 'offchem'
    offchem_dir.mkdir()

    csv_content = 'species   vmr\nH2O   0.01\nCO2   0.001\n'
    csv_path = offchem_dir / 'vulcan_5000p000.csv'
    csv_path.write_text(csv_content, encoding='utf-8')

    result = read_result(str(tmp_path), module='vulcan', filename='vulcan_5000p000.csv')
    assert result is not None
    assert len(result) == 2
    assert result['species'].iloc[0] == 'H2O'
    # Default filename would have looked for 'vulcan.csv' and returned None
    result_default = read_result(str(tmp_path), module='vulcan')
    assert result_default is None


# -----------------------------------------------------------------------
# Snapshot names
# -----------------------------------------------------------------------


@pytest.mark.parametrize(
    'module,time,expected',
    [
        ('vulcan', 12.32, 'vulcan_12p320.csv'),
        ('dummy', 0.0, 'dummy_0p000.csv'),
        ('vulcan', 4.2e6, 'vulcan_4200000p000.csv'),
    ],
    ids=['subyear_time', 'initial_state', 'late_evolution'],
)
def test_chem_csv_name_uses_the_subyear_token(module, time, expected):
    """A chemistry CSV is named <module>_<time>.csv with the sub-year token,
    so two snapshots within one year get distinct names."""
    assert chem_csv_name(module, time) == expected
    # The whole-year form collides for 12.32 and 12.9 yr; the sub-year one does not
    assert chem_csv_name(module, time) != chem_csv_name(module, time + 0.5)


# -----------------------------------------------------------------------
# Time resolution
# -----------------------------------------------------------------------


def _write_atm_snapshots(outdir, times):
    """Create empty data/<time>_atm.nc files, as named by the climate writers."""
    data = outdir / 'data'
    data.mkdir(parents=True, exist_ok=True)
    for t in times:
        (data / ('%.3f_atm.nc' % t).replace('.', 'p', 1)).write_text('')


def _helpfile(times):
    """Minimal helpfile with the given Time column [yr] and a distinct T_surf per row."""
    return pd.DataFrame({'Time': times, 'T_surf': np.linspace(3000.0, 1500.0, len(times))})


def test_resolve_end_maps_to_the_nearest_atmosphere_snapshot(tmp_path):
    """'end' gives the snapshot nearest the final helpfile time, not the raw
    helpfile time, so the backend finds a snapshot file to read.

    The final helpfile time 1000.0004 yr rounds to the 1000p000 snapshot.
    """
    _write_atm_snapshots(tmp_path, [0.0, 10.0, 1000.0])
    hf_all = _helpfile([0.0, 10.0, 500.0, 1000.0004])

    times = resolve_chemistry_times(str(tmp_path), hf_all, 'end')

    assert times == [pytest.approx(1000.0, abs=1e-9)]
    assert len(times) == 1


def test_resolve_all_returns_every_snapshot_ascending(tmp_path):
    """'all' returns the time of each data/*_atm.nc file, ascending, and
    ignores files that are not atmosphere snapshots."""
    _write_atm_snapshots(tmp_path, [1000.0, 0.25, 10.0])
    (tmp_path / 'data' / '50p000_int.nc').write_text('')  # interior, not atmosphere
    (tmp_path / 'data' / '60p000_atm.nc.incomplete').write_text('')  # quarantined
    hf_all = _helpfile([0.0, 0.25, 10.0, 50.0, 60.0, 1000.0])

    times = resolve_chemistry_times(str(tmp_path), hf_all, 'all')

    np.testing.assert_allclose(times, [0.25, 10.0, 1000.0], rtol=0, atol=1e-12)
    assert times == sorted(times)


def test_resolve_list_maps_to_nearest_and_drops_duplicates(tmp_path):
    """A list of requested times maps each to the nearest snapshot; two that
    map to the same snapshot give it once, and the result is ascending
    whatever order the list was given in.

    Requests 900 and 1100 yr both map to 1000 yr; 3 yr maps to 0 yr and not
    to 10 yr, which a floor or ceiling instead of nearest would pick.
    """
    _write_atm_snapshots(tmp_path, [0.0, 10.0, 1000.0])
    hf_all = _helpfile([0.0, 10.0, 1000.0])

    times = resolve_chemistry_times(str(tmp_path), hf_all, [1100.0, 3.0, 900.0])

    np.testing.assert_allclose(times, [0.0, 1000.0], rtol=0, atol=1e-12)
    assert len(times) == len(set(times))


def test_resolve_falls_back_to_helpfile_times_without_snapshots(tmp_path):
    """With no atmosphere snapshot on disk (the dummy climate writes none),
    the helpfile times are the snapshots: 'all' gives each distinct helpfile
    time once, and a list maps to the nearest of them."""
    hf_all = _helpfile([0.0, 0.0, 0.0, 5.0, 25.0])  # init stage repeats t = 0

    all_times = resolve_chemistry_times(str(tmp_path), hf_all, 'all')
    listed = resolve_chemistry_times(str(tmp_path), hf_all, [6.0, 24.0])

    np.testing.assert_allclose(all_times, [0.0, 5.0, 25.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(listed, [5.0, 25.0], rtol=0, atol=1e-12)
    assert not (tmp_path / 'data').exists()  # nothing was created on disk


# -----------------------------------------------------------------------
# Helpfile row selection
# -----------------------------------------------------------------------


def test_helpfile_row_at_picks_the_nearest_row():
    """The row nearest the snapshot time is returned as a dict, so the
    chemistry of a snapshot uses the planet state at that time.

    7 yr is nearer 10 yr (T_surf 1500 K) than 0 yr (3000 K).
    """
    hf_all = pd.DataFrame({'Time': [0.0, 10.0, 100.0], 'T_surf': [3000.0, 1500.0, 900.0]})

    row = helpfile_row_at(hf_all, 7.0)

    assert row['Time'] == pytest.approx(10.0, abs=1e-12)
    assert row['T_surf'] == pytest.approx(1500.0, abs=1e-12)
    assert isinstance(row, dict)


def test_helpfile_row_at_takes_the_last_of_repeated_init_rows():
    """The init stage holds Time at 0 over several rows; the last of them is
    the state the t = 0 snapshot was written from, so it wins."""
    hf_all = pd.DataFrame(
        {'Time': [0.0, 0.0, 0.0, 50.0], 'T_surf': [3200.0, 3100.0, 3050.0, 2500.0]}
    )

    row = helpfile_row_at(hf_all, 0.0)

    assert row['T_surf'] == pytest.approx(3050.0, abs=1e-12)
    assert row['T_surf'] != pytest.approx(3200.0, abs=1.0)  # the first init row


# -----------------------------------------------------------------------
# VULCAN NetCDF reader
# -----------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_read_vulcan_netcdf_returns_cgs_columns_bottom_up(tmp_path):
    """The reader returns VULCAN's arrays unchanged in CGS, bottom up: layer
    pressures decrease with height and stay positive, Kzz is NaN on the
    bottom and top interfaces that VULCAN fills, and x_gas keeps one column
    per species in the order of the species names.

    An asymmetric composition (0.6, 0.3, 0.1) catches a transposed x_gas.
    """
    layers = vulcan_layers(P_b=1.0e8, P_t=1.0e2, nz=6)
    species = ['H2O', 'CO2', 'H2O_l_s']
    x_gas = np.tile([0.6, 0.3, 0.1], (6, 1))
    fpath = str(tmp_path / '12p320_chem.nc')
    write_vulcan_netcdf(fpath, layers, species, x_gas)

    out = read_vulcan_netcdf(fpath)

    assert out['species'] == species
    assert out['x_gas'].shape == (6, 3)
    np.testing.assert_allclose(out['x_gas'][:, 0], 0.6, rtol=1e-12)
    np.testing.assert_allclose(out['x_gas'].sum(axis=1), 1.0, rtol=1e-12)
    # dyn cm-2 as stored: a Pa conversion would give 1e7 at the bottom
    np.testing.assert_allclose(out['p'], layers['p'], rtol=1e-12)
    assert out['p'][0] == pytest.approx(1.0e8, rel=1e-12)
    assert np.all(out['p'] > 0.0) and np.all(np.diff(out['p']) < 0.0)
    np.testing.assert_allclose(out['tmp'], layers['tmp'], rtol=1e-12)
    np.testing.assert_allclose(out['zl'], layers['zl'], rtol=1e-12)
    assert np.isnan(out['Kzz'][0]) and np.isnan(out['Kzz'][-1])
    np.testing.assert_allclose(out['Kzz'][1:-1], layers['Kzz'], rtol=1e-12)


def test_read_vulcan_netcdf_missing_file_raises(tmp_path):
    """A missing NetCDF raises rather than returning an empty result, and the
    reader creates nothing in its place."""
    fpath = tmp_path / 'missing_chem.nc'
    with pytest.raises(FileNotFoundError):
        read_vulcan_netcdf(str(fpath))
    assert not fpath.exists()
