"""Unit tests for ``proteus.atmos_chem.common``.

Exercises ``read_result`` (chemistry CSV reader), ``chem_csv_name``,
``resolve_chemistry_times``, the target functions ``crossing_rows``,
``resolve_chemistry_targets`` and ``chem_targets_crossed``, ``snapshot_star_file``
and ``helpfile_row_at``.

Invariants tested:
  - Correct file path construction from module name and optional filename
  - Guard: returns None for 'none' or None module
  - Guard: returns None when the CSV file does not exist
  - Successful read of well-formed whitespace-delimited CSV
  - Snapshot names use the sub-year time token
  - Resolved chemistry times are existing snapshots, ascending, without duplicates
  - The helpfile row used for a snapshot is the nearest in time
  - A table target selects every row at which its column reaches or crosses the value
  - The main loop marks exactly the rows that the targets later resolve to
  - A snapshot uses the stellar spectrum it recorded, else the newest earlier one

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
    chem_targets_crossed,
    crossing_rows,
    helpfile_row_at,
    read_result,
    resolve_chemistry_targets,
    resolve_chemistry_times,
    snapshot_star_file,
)

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
# Targets on helpfile columns
# -----------------------------------------------------------------------


def test_crossing_rows_finds_each_crossing_in_either_direction():
    """A surface pressure that rises during outgassing and falls under escape crosses
    10 bar twice: on the way up at row 2 (8 -> 12 bar) and down at row 4 (15 -> 9 bar).

    A rule that only looks for the first row at or above the value would give row 2 alone.
    """
    rows = crossing_rows([2.0, 8.0, 12.0, 15.0, 9.0, 4.0], 10.0)

    np.testing.assert_array_equal(rows, [2, 4])
    assert crossing_rows([2.0, 8.0], 10.0).size == 0  # never reached


def test_crossing_rows_counts_reaching_a_value_once():
    """Reaching a value exactly counts at the first equal row; staying on it, leaving
    it, or touching it and turning back adds no second crossing. Row 0 counts when it
    equals the value; a NaN row never counts."""
    np.testing.assert_array_equal(crossing_rows([2000.0, 1500.0, 1500.0, 1400.0], 1500.0), [1])
    np.testing.assert_array_equal(crossing_rows([1600.0, 1500.0, 1600.0], 1500.0), [1])
    np.testing.assert_array_equal(crossing_rows([1500.0, np.nan, 1400.0], 1500.0), [0])


def _remelting_helpfile():
    """Helpfile [yr] with init rows at t = 0 and a melt fraction that re-melts after an impact."""
    return pd.DataFrame(
        {
            'Time': [0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            'Phi_global': [1.0, 1.0, 0.8, 0.4, 0.9, 0.3, 0.1],
            'T_surf': [3000.0, 2900.0, 2500.0, 1800.0, 2600.0, 1700.0, 1400.0],
        }
    )


def test_resolve_targets_from_a_table_takes_every_crossing():
    """Phi_global crosses 0.5 at 20 yr (cooling), 30 yr (re-melting after an impact) and
    40 yr (cooling again); Time = 15 gives the first row at or after 15 yr, 20 yr; and
    end = true adds the final row, 50 yr.

    Taking only the first crossing of each value would give 20 and 50 yr alone.
    """
    targets = resolve_chemistry_targets(
        _remelting_helpfile(), {'Phi_global': [0.5], 'Time': [15.0], 'end': True}
    )

    np.testing.assert_allclose(targets, [20.0, 30.0, 40.0, 50.0], rtol=0, atol=1e-12)
    assert targets == sorted(set(targets))


def test_resolve_targets_skips_values_never_reached_with_a_warning(caplog):
    """T_surf never reaches 500 K and the helpfile has no column 'R_star'; both are
    skipped with a warning, and the targets that are reached are kept."""
    hf_all = _remelting_helpfile()
    with caplog.at_level('WARNING'):
        targets = resolve_chemistry_targets(
            hf_all, {'T_surf': [500.0, 2000.0], 'R_star': [7e8], 'end': False}
        )

    # T_surf crosses 2000 K at 20 yr (2500 -> 1800), 30 yr (up) and 40 yr (down)
    np.testing.assert_allclose(targets, [20.0, 30.0, 40.0], rtol=0, atol=1e-12)
    assert 'T_surf = 500 is never reached' in caplog.text
    assert "'R_star' skipped" in caplog.text


def test_resolve_list_keeps_its_times_and_warns_for_those_outside_the_run(caplog):
    """A list keeps its own times: a time after the end of the run still maps to the
    final snapshot, as before, and is now reported."""
    hf_all = _remelting_helpfile()
    with caplog.at_level('WARNING'):
        targets = resolve_chemistry_targets(hf_all, [1e4, 25.0, 25.0])

    assert targets == [25.0, 1e4]
    assert 'outside the run' in caplog.text


def test_chem_targets_crossed_marks_exactly_the_rows_the_targets_resolve_to():
    """Walking the helpfile row by row, as the main loop does, the check is true at the
    last row exactly when that row is one of the resolved targets, so the run writes a
    snapshot at each row that offline chemistry later asks for."""
    hf_all = _remelting_helpfile()
    table = {'Phi_global': [0.5], 'T_surf': [2000.0], 'Time': [15.0]}

    marked = [
        float(hf_all['Time'].iloc[n - 1])
        for n in range(1, len(hf_all) + 1)
        if chem_targets_crossed(hf_all.iloc[:n], table)
    ]

    assert marked == resolve_chemistry_targets(hf_all, table)
    assert marked == [20.0, 30.0, 40.0]


def test_chem_targets_crossed_for_a_list_and_for_end_and_all():
    """A listed time marks the first row at or after it, from (t_prev, t_now]; 'end',
    'all' and a single row mark nothing."""
    hf_all = _remelting_helpfile()
    marked = [
        float(hf_all['Time'].iloc[n - 1])
        for n in range(1, len(hf_all) + 1)
        if chem_targets_crossed(hf_all.iloc[:n], [5.0, 30.0])
    ]

    assert marked == [10.0, 30.0]
    assert not chem_targets_crossed(hf_all, 'end')
    assert not chem_targets_crossed(hf_all, 'all')
    assert not chem_targets_crossed(hf_all.iloc[:1], [0.0])


def test_resolve_table_uses_own_snapshots_and_warns_when_one_is_missing(tmp_path, caplog):
    """A target with a snapshot of its own (20 yr) uses it; one without (30 yr, from a
    run made before such snapshots were written) uses the nearest and is reported."""
    _write_atm_snapshots(tmp_path, [0.0, 20.0, 50.0])
    hf_all = _remelting_helpfile()

    with caplog.at_level('WARNING'):
        times = resolve_chemistry_times(
            str(tmp_path), hf_all, {'Phi_global': [0.5], 'end': False}
        )

    # targets 20, 30, 40 yr map to the snapshots at 20, 20 and 50 yr
    np.testing.assert_allclose(times, [20.0, 50.0], rtol=0, atol=1e-12)
    assert 't=3.000e+01 yr has no snapshot of its own' in caplog.text
    assert 't=2.000e+01 yr has no snapshot' not in caplog.text


# -----------------------------------------------------------------------
# Stellar spectrum of a snapshot
# -----------------------------------------------------------------------


def _write_sflux(outdir, years):
    data = outdir / 'data'
    data.mkdir(parents=True, exist_ok=True)
    for y in years:
        (data / ('%d.sflux' % y)).write_text('# WL(nm) Flux\n')


def _write_starfile_snapshot(outdir, time, starfile):
    """An atmosphere snapshot holding only the `starfile` that AGNI records."""
    import netCDF4 as nc

    path = outdir / 'data' / ('%.3f_atm.nc' % time).replace('.', 'p', 1)
    with nc.Dataset(path, 'w') as ds:
        var = ds.createVariable('starfile', str, ())
        var[...] = starfile


def test_snapshot_star_file_uses_the_spectrum_the_snapshot_recorded(tmp_path):
    """The recorded spectrum is found by name in this run's data folder, so a run folder
    that was moved after the run still works; the newest earlier file is not taken."""
    _write_sflux(tmp_path, [0, 100, 1000])
    _write_starfile_snapshot(tmp_path, 900.0, '/old/place/run/data/0.sflux')

    path = snapshot_star_file(str(tmp_path), 900.0)

    assert path == os.path.join(str(tmp_path), 'data', '0.sflux')
    assert not os.path.exists('/old/place/run/data/0.sflux')  # only the name could be used


def test_snapshot_star_file_falls_back_to_the_newest_earlier_spectrum(tmp_path):
    """Without a recorded spectrum, the newest written at or before the snapshot is used,
    as the atmosphere step read it: at 900 yr that is 100 yr, not the nearer 1000 yr. A
    snapshot before every spectrum takes the earliest, and a run without any raises."""
    _write_sflux(tmp_path, [100, 1000])
    _write_starfile_snapshot(tmp_path, 900.0, '_ALREADY_IN_SPECTRAL_FILE')

    assert snapshot_star_file(str(tmp_path), 900.0).endswith(os.sep + '100.sflux')
    assert snapshot_star_file(str(tmp_path), 50.0).endswith(os.sep + '100.sflux')
    with pytest.raises(FileNotFoundError, match='No stellar spectrum'):
        snapshot_star_file(str(tmp_path / 'elsewhere'), 900.0)
