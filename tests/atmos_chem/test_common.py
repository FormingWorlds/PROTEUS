"""Unit tests for ``proteus.atmos_chem.common``.

Exercises ``read_result`` (chemistry CSV reader), ``chem_csv_name``,
``resolve_chemistry_times``, the target functions ``crossing_rows``,
``resolve_chemistry_targets`` and ``chem_targets_crossed``, ``snapshot_star_file``,
``helpfile_row_at`` and ``escape_atom_fluxes``.

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
  - Escape atom fluxes return the helpfile escape rates [kg s-1] through a sphere of R_xuv;
    a non-finite rate gives zero with a warning, a negative rate a negative flux

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
    escape_atom_fluxes,
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


def test_resolve_list_takes_the_first_row_at_or_after_each_time(tmp_path, caplog):
    """Each listed time selects the first helpfile row at or after it, as the run marks
    it, and its snapshot; times that select the same row give it once, ascending, and a
    time after the end of the run is skipped with a warning.

    3 yr selects the row at 10 yr, not the nearer one at 0 yr; 900 and 950 yr both select
    the row at 1000 yr; 1100 yr is after the last row.
    """
    _write_atm_snapshots(tmp_path, [0.0, 10.0, 1000.0])
    hf_all = _helpfile([0.0, 10.0, 1000.0])

    with caplog.at_level('WARNING'):
        times = resolve_chemistry_times(str(tmp_path), hf_all, [1100.0, 3.0, 950.0, 900.0])

    np.testing.assert_allclose(times, [10.0, 1000.0], rtol=0, atol=1e-12)
    assert 'Time = 1100 is never reached' in caplog.text


def test_resolve_falls_back_to_helpfile_times_without_snapshots(tmp_path):
    """With no atmosphere snapshot on disk (the dummy climate writes none),
    the helpfile times are the snapshots: 'all' gives each distinct helpfile
    time once, and a list maps to the nearest of them."""
    hf_all = _helpfile([0.0, 0.0, 0.0, 5.0, 25.0])  # init stage repeats t = 0

    all_times = resolve_chemistry_times(str(tmp_path), hf_all, 'all')
    listed = resolve_chemistry_times(str(tmp_path), hf_all, [4.0, 6.0, 24.0])

    np.testing.assert_allclose(all_times, [0.0, 5.0, 25.0], rtol=0, atol=1e-12)
    # 4 yr selects the row at 5 yr; 6 and 24 yr both select the row at 25 yr
    np.testing.assert_allclose(listed, [5.0, 25.0], rtol=0, atol=1e-12)
    assert not (tmp_path / 'data').exists()  # nothing was created on disk


# -----------------------------------------------------------------------
# Helpfile row selection
# -----------------------------------------------------------------------


def test_helpfile_row_at_picks_the_latest_row_at_or_before():
    """The latest row at or before the snapshot time is returned as a dict, the state the
    run had reached by then: at 7 yr the row at 0 yr (3000 K), not the nearer row at 10 yr.
    A snapshot named with the time rounded to 1e-3 yr still finds its own row, and a time
    before every row is an error."""
    hf_all = pd.DataFrame({'Time': [0.0, 10.0, 100.0], 'T_surf': [3000.0, 1500.0, 900.0]})

    row = helpfile_row_at(hf_all, 7.0)
    assert row['T_surf'] == pytest.approx(3000.0, abs=1e-12)
    assert isinstance(row, dict)
    # 9.9996 yr is 10 yr rounded down by less than half the file-name resolution
    assert helpfile_row_at(hf_all, 9.9996)['T_surf'] == pytest.approx(1500.0, abs=1e-12)
    with pytest.raises(ValueError, match='no row at or before'):
        helpfile_row_at(hf_all, -1.0)


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


def test_resolve_list_is_the_time_column_of_a_table(caplog):
    """A list of times resolves as the table {Time = [...]}: 25 yr selects the row at 30 yr,
    twice listed it gives it once, and 1e4 yr, after the run, is skipped with a warning."""
    hf_all = _remelting_helpfile()
    with caplog.at_level('WARNING'):
        targets = resolve_chemistry_targets(hf_all, [1e4, 25.0, 25.0])

    assert targets == resolve_chemistry_targets(hf_all, {'Time': [1e4, 25.0]})
    assert targets == [30.0]
    assert 'Time = 10000 is never reached' in caplog.text


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


def test_resolve_table_uses_own_snapshots_else_the_latest_before(tmp_path, caplog):
    """A target with a snapshot of its own (20 yr) uses it; one without (30 and 40 yr, in a
    run made before such snapshots were written) uses the latest snapshot before it, here
    20 yr, not the nearer later one at 50 yr, and is reported."""
    _write_atm_snapshots(tmp_path, [0.0, 20.0, 50.0])
    hf_all = _remelting_helpfile()

    with caplog.at_level('WARNING'):
        times = resolve_chemistry_times(
            str(tmp_path), hf_all, {'Phi_global': [0.5], 'end': False}
        )

    # targets 20, 30, 40 yr all use the snapshot at 20 yr
    np.testing.assert_allclose(times, [20.0], rtol=0, atol=1e-12)
    assert 't=4.000e+01 yr has no snapshot of its own' in caplog.text
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
    """Without a recorded spectrum, the latest written at or before the snapshot is used,
    as the atmosphere step read it: at 900 yr that is 100 yr, not the nearer 1000 yr. A
    snapshot before every spectrum of the evolution takes the earliest, never -1.sflux,
    the copy of the modern spectrum, and a run without any raises."""
    _write_sflux(tmp_path, [-1, 100, 1000])
    _write_starfile_snapshot(tmp_path, 900.0, '_ALREADY_IN_SPECTRAL_FILE')

    assert snapshot_star_file(str(tmp_path), 900.0).endswith(os.sep + '100.sflux')
    assert snapshot_star_file(str(tmp_path), 50.0).endswith(os.sep + '100.sflux')
    with pytest.raises(FileNotFoundError, match='No stellar spectrum'):
        snapshot_star_file(str(tmp_path / 'elsewhere'), 900.0)
    (tmp_path / 'modern_only').mkdir()
    _write_sflux(tmp_path / 'modern_only', [-1])
    with pytest.raises(FileNotFoundError, match='No stellar spectrum'):
        snapshot_star_file(str(tmp_path / 'modern_only'), 900.0)


# ---------------------------------------------------------------------------
# escape_atom_fluxes
# ---------------------------------------------------------------------------

_R_INT = 6.371e6  # m
_R_XUV = 1.5 * _R_INT  # m; far enough from R_int that using R_int fails the tolerance

# Asymmetric escape rates [kg s-1], so a swapped element changes the result
_ESC_RATES = {'H': 3.7e3, 'O': 2.9e4, 'C': 1.1e3, 'N': 0.0, 'S': 5.3e2, 'Si': 8.0e2, 'He': 4.0}


def _escape_row(rates=_ESC_RATES, r_xuv=_R_XUV):
    """Helpfile row with escape rates [kg s-1] for every PROTEUS element and R_xuv [m]."""
    from proteus.utils.constants import element_list

    row = {'Time': 2.5e5, 'R_int': _R_INT, 'R_xuv': r_xuv}
    for e in element_list:
        row['esc_rate_' + e] = rates.get(e, 0.0)
    return row


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_escape_atom_fluxes_pin_one_kg_per_second_of_hydrogen():
    """1 kg/s of H through a sphere of radius 1e7 m is 4.7542e7 atoms cm-2 s-1.

    By hand: 1 / 1.008e-3 kg mol-1 * 6.02214076e23 mol-1 = 5.97435e26 atoms s-1,
    over 4 pi (1e9 cm)^2 = 1.256637e19 cm2. Guards: the molar mass of H2, the area
    in m2 instead of cm2, and pi R^2 instead of 4 pi R^2 each fall far outside.
    """
    row = {'Time': 0.0, 'R_xuv': 1.0e7, 'R_int': 6.371e6, 'esc_rate_H': 1.0}
    flux = escape_atom_fluxes(row, ['H'])
    assert list(flux) == ['H']
    assert flux['H'] == pytest.approx(4.7542e7, rel=1e-4)

    atoms_per_s = 1.0 / 1.008e-3 * 6.02214076e23
    wrong = {
        'H2 molar mass': atoms_per_s / 2.0 / (4 * np.pi * 1.0e9**2),
        'area in m2': atoms_per_s / (4 * np.pi * 1.0e7**2),
        'disc area': atoms_per_s / (np.pi * 1.0e9**2),
    }
    for name, value in wrong.items():
        assert flux['H'] != pytest.approx(value, rel=1e-2), name


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_escape_atom_fluxes_return_the_escape_rates_through_the_r_xuv_sphere():
    """Each atom flux times the atomic mass and the area of the R_xuv sphere gives
    back the helpfile escape rate of that element [kg s-1], to round-off.

    The same flux spread over the R_int sphere is (R_xuv/R_int)^2 = 2.25 times
    larger, so using R_int fails this check. N escapes at zero rate and gets a
    zero flux, not a missing entry.
    """
    from proteus.utils.constants import const_Nav, element_mmw

    elements = ['H', 'O', 'C', 'N', 'S']
    flux = escape_atom_fluxes(_escape_row(), elements)
    assert set(flux) == set(elements)
    assert all(f >= 0 for f in flux.values())
    assert flux['N'] == 0

    area_xuv = 4 * np.pi * (_R_XUV * 100.0) ** 2  # cm2
    area_int = 4 * np.pi * (_R_INT * 100.0) ** 2  # cm2
    for e, f in flux.items():
        back = f * area_xuv * element_mmw[e] / const_Nav
        assert back == pytest.approx(_ESC_RATES[e], rel=1e-12, abs=0.0)
        if _ESC_RATES[e] > 0:
            assert f * area_int * element_mmw[e] / const_Nav != pytest.approx(
                _ESC_RATES[e], rel=1e-2
            )

    # Flux through a sphere falls as R^-2 at a fixed rate
    flux_2r = escape_atom_fluxes(_escape_row(r_xuv=2 * _R_XUV), elements)
    for e in flux:
        assert flux_2r[e] == pytest.approx(flux[e] / 4.0, rel=1e-12, abs=0.0)


@pytest.mark.unit
def test_escape_atom_fluxes_keys_are_the_network_elements_in_order():
    """The fluxes are keyed by exactly the given network elements, in their
    order, which is the row order of the VULCAN top boundary file. Si and He
    escape but are outside the network, so they get no entry; a CHO network
    leaves out N and S, and the O flux does not depend on which others are
    asked for."""
    flux = escape_atom_fluxes(_escape_row(), ['H', 'O', 'C', 'N', 'S'])
    assert list(flux) == ['H', 'O', 'C', 'N', 'S']
    assert 'Si' not in flux and 'He' not in flux

    flux_cho = escape_atom_fluxes(_escape_row(), ['C', 'O', 'H'])
    assert list(flux_cho) == ['C', 'O', 'H']
    assert flux_cho['O'] == pytest.approx(flux['O'], rel=1e-12)
    assert escape_atom_fluxes(_escape_row(), []) == {}


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_escape_atom_fluxes_without_escape_are_zero():
    """With every escape rate zero (escape off) each network element has a zero
    flux, so the top boundary removes nothing."""
    elements = ['H', 'O', 'C', 'N', 'S']
    flux = escape_atom_fluxes(_escape_row(rates={}), elements)
    assert list(flux) == elements
    assert all(f == 0 for f in flux.values())


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'r_xuv',
    [0.0, -1.0e7, float('nan'), 0.5 * _R_INT],
    ids=['unset', 'negative', 'nan', 'below_surface'],
)
def test_escape_atom_fluxes_clip_r_xuv_to_the_surface(r_xuv):
    """An R_xuv that is unset (0), negative, NaN or below the surface is replaced
    by R_int, so the fluxes are finite and equal those for R_xuv = R_int; with
    every rate zero they stay zero instead of 0/0."""
    elements = ['H', 'O', 'C', 'N', 'S']
    flux = escape_atom_fluxes(_escape_row(r_xuv=r_xuv), elements)
    at_surface = escape_atom_fluxes(_escape_row(r_xuv=_R_INT), elements)
    assert all(np.isfinite(f) for f in flux.values())
    for e in elements:
        assert flux[e] == pytest.approx(at_surface[e], rel=1e-12, abs=0.0)

    zero = escape_atom_fluxes(_escape_row(rates={}, r_xuv=r_xuv), elements)
    assert all(f == 0 for f in zero.values())


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_escape_atom_fluxes_use_an_r_xuv_above_the_surface():
    """An R_xuv just above R_int is used as it is: 1 % above the surface lowers
    every flux by 1/1.01^2, which the clip to R_int would miss."""
    elements = ['H', 'O', 'C', 'S']
    above = escape_atom_fluxes(_escape_row(r_xuv=1.01 * _R_INT), elements)
    at_surface = escape_atom_fluxes(_escape_row(r_xuv=_R_INT), elements)
    for e in elements:
        assert above[e] == pytest.approx(at_surface[e] / 1.01**2, rel=1e-12)
        assert above[e] != pytest.approx(at_surface[e], rel=1e-3)


@pytest.mark.unit
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
def test_escape_atom_fluxes_replace_a_non_finite_rate_by_zero(bad, caplog):
    """A non-finite escape rate of one element gives that element a zero flux and
    a warning naming it; the other elements keep their fluxes."""
    caplog.set_level('WARNING', logger='fwl.proteus.atmos_chem.common')
    elements = ['H', 'O', 'C']
    good = escape_atom_fluxes(_escape_row(), elements)
    flux = escape_atom_fluxes(_escape_row(rates={**_ESC_RATES, 'O': bad}), elements)

    assert flux['O'] == 0
    assert all(np.isfinite(f) for f in flux.values())
    for e in ('H', 'C'):
        assert flux[e] == pytest.approx(good[e], rel=1e-12)
    assert f'Escape rate of O is {bad} kg/s; using zero' in caplog.text


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_escape_atom_fluxes_keep_a_negative_rate_as_a_negative_flux(caplog):
    """A negative escape rate (gas delivered at the top) is kept: its flux has the
    same magnitude as for the positive rate and the opposite sign, with no warning."""
    caplog.set_level('WARNING', logger='fwl.proteus.atmos_chem.common')
    pos = escape_atom_fluxes(_escape_row(), ['O'])
    neg = escape_atom_fluxes(_escape_row(rates={'O': -_ESC_RATES['O']}), ['O'])
    assert neg['O'] < 0
    assert neg['O'] == pytest.approx(-pos['O'], rel=1e-12)
    assert caplog.text == ''


@pytest.mark.unit
def test_escape_atom_fluxes_report_a_helpfile_without_escape_columns():
    """A helpfile written before the per-element escape columns raises the
    schema-drift report naming the column, rather than a bare KeyError."""
    from proteus.utils.coupler import HelpfileRow, HelpfileSchemaDriftError

    row = HelpfileRow(
        {'Time': 1.0, 'R_xuv': _R_XUV, 'R_int': _R_INT, 'esc_rate_H': 1.0}, 'old_run.csv'
    )
    with pytest.raises(HelpfileSchemaDriftError, match="'esc_rate_O'"):
        escape_atom_fluxes(row, ['H', 'O'])
    assert escape_atom_fluxes(row, ['H'])['H'] > 0
