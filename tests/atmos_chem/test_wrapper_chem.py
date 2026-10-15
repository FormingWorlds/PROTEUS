"""Unit tests for ``proteus.atmos_chem.wrapper``.

Exercises ``run_chemistry``, which post-processes the snapshots set by
``atmos_chem.times`` with the configured chemistry backend after the
simulation, whatever ``atmos_chem.mode`` is.

Invariants tested:
  - Guard: no module returns None without touching the output folder
  - Error contract: unknown module raises ValueError and records status 20
  - Sequence: snapshots run in ascending time, and each run gets the
    helpfile row of its time; the VULCAN runs of one call share one
    network-compilation state
  - Output: the last snapshot is copied to '<module>.csv'
  - Error contract: a failed snapshot raises RuntimeError and stops the
    sequence; the earlier results stay, and no '<module>.csv' is written
  - Desiccation: snapshots without an atmosphere (P_surf <= 0 or NaN) are
    left out; with none left, nothing runs
  - The offchem folder is cleared once, at the start
  - The real dummy backend gives mixing ratios that sum to 1 on every level

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from proteus.atmos_chem.common import chem_csv_name
from proteus.atmos_chem.dummy import _ALL_SPECIES
from proteus.atmos_chem.wrapper import run_chemistry

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _make_config(module: str | None = 'dummy', times='end') -> MagicMock:
    """Build a minimal mock config for run_chemistry."""
    config = MagicMock()
    config.atmos_chem.module = module
    config.atmos_chem.times = times
    config.atmos_clim.p_top = 1e-6  # bar, read by the dummy backend
    return config


def _dirs(tmp_path) -> dict:
    """Output directories as PROTEUS sets them up."""
    return {'output': str(tmp_path), 'output/offchem': str(tmp_path / 'offchem')}


def _helpfile(times) -> pd.DataFrame:
    """Helpfile rows with a physically plausible surface state at each time [yr].

    T_surf falls with time, so each row is told apart by its temperature.
    """
    n = len(times)
    return pd.DataFrame(
        {
            'Time': times,
            'T_surf': np.linspace(3000.0, 1500.0, n),
            'P_surf': np.full(n, 100.0),  # bar
            'gravity': np.full(n, 9.81),
            'atm_kg_per_mol': np.full(n, 0.018),
            'H2O_vmr': np.full(n, 0.9),
            'CO2_vmr': np.full(n, 0.1),
        }
    )


def _fake_backend(tmp_path, fail_at=()):
    """A backend that writes one CSV per snapshot, holding T_surf of its row.

    Returns False, writing nothing, at the times in `fail_at`.
    """
    calls = []

    def run(dirs, config, hf_row):
        calls.append((float(hf_row['Time']), float(hf_row['T_surf'])))
        if hf_row['Time'] in fail_at:
            return False
        os.makedirs(dirs['output/offchem'], exist_ok=True)
        fpath = os.path.join(
            dirs['output/offchem'], chem_csv_name(config.atmos_chem.module, hf_row['Time'])
        )
        pd.DataFrame({'tmp': [hf_row['T_surf']], 'p': [1.0e5]}).to_csv(
            fpath, sep='\t', index=False
        )
        return True

    return run, calls


# -----------------------------------------------------------------------
# Guards and error contract
# -----------------------------------------------------------------------


@pytest.mark.parametrize('module', [None, ''], ids=['module_none', 'module_empty'])
def test_run_chemistry_without_module_returns_none(tmp_path, module):
    """With no chemistry module, run_chemistry returns None and leaves the
    output folder alone: no offchem folder is created or cleared."""
    stale = tmp_path / 'offchem' / 'keep.csv'
    stale.parent.mkdir()
    stale.write_text('x')

    result = run_chemistry(_dirs(tmp_path), _make_config(module=module), _helpfile([0.0]))

    assert result is None
    assert stale.read_text() == 'x'


def test_run_chemistry_unknown_module_raises_and_records_status(tmp_path):
    """An unrecognised module raises ValueError naming it, records error
    status 20 in the run's status file, and runs nothing."""
    with pytest.raises(ValueError, match='nonexistent_chem'):
        run_chemistry(
            _dirs(tmp_path), _make_config(module='nonexistent_chem'), _helpfile([0.0])
        )
    status = (tmp_path / 'status').read_text()
    assert status.splitlines()[0].strip() == '20'
    assert not (tmp_path / 'offchem').exists()


# -----------------------------------------------------------------------
# Sequence of snapshots
# -----------------------------------------------------------------------


def test_run_chemistry_end_runs_the_backend_once(tmp_path):
    """times = 'end' runs the backend once, on the final helpfile row, and
    returns that result, also copied to '<module>.csv'."""
    run, calls = _fake_backend(tmp_path)
    hf_all = _helpfile([0.0, 10.0, 250.0])

    with patch('proteus.atmos_chem.dummy.run_dummy_chem', side_effect=run):
        result = run_chemistry(_dirs(tmp_path), _make_config(times='end'), hf_all)

    assert calls == [(250.0, pytest.approx(1500.0))]
    assert result['tmp'].iloc[0] == pytest.approx(1500.0)
    assert (tmp_path / 'offchem' / 'dummy.csv').exists()


def test_run_chemistry_all_runs_in_ascending_time(tmp_path):
    """times = 'all' runs every snapshot in ascending time, and each call gets
    the helpfile row of its own time, not the final one."""
    run, calls = _fake_backend(tmp_path)
    hf_all = _helpfile([0.0, 10.0, 250.0])  # T_surf 3000, 2250, 1500 K

    with patch('proteus.atmos_chem.dummy.run_dummy_chem', side_effect=run):
        run_chemistry(_dirs(tmp_path), _make_config(times='all'), hf_all)

    assert [c[0] for c in calls] == [0.0, 10.0, 250.0]
    assert [c[1] for c in calls] == pytest.approx([3000.0, 2250.0, 1500.0])


def test_run_chemistry_vulcan_snapshots_share_one_network_state(tmp_path):
    """The VULCAN runs of one call get the same network_state dict, so a
    network compiled by a successful run is reused by the later ones, and a
    second call starts from a new, empty state."""
    states = []

    def run(dirs, config, hf_row, *, network_state=None):
        states.append(network_state)
        network_state['compiled'] = True  # as after a successful VULCAN run
        os.makedirs(dirs['output/offchem'], exist_ok=True)
        fpath = os.path.join(dirs['output/offchem'], chem_csv_name('vulcan', hf_row['Time']))
        pd.DataFrame({'tmp': [hf_row['T_surf']], 'p': [1.0e5]}).to_csv(
            fpath, sep='\t', index=False
        )
        return True

    config = _make_config(module='vulcan', times='all')
    with patch('proteus.atmos_chem.vulcan.run_vulcan', side_effect=run):
        run_chemistry(_dirs(tmp_path), config, _helpfile([0.0, 10.0]))
        run_chemistry(_dirs(tmp_path), config, _helpfile([0.0, 10.0]))

    assert len(states) == 4
    assert states[0] is states[1]
    assert states[2] is states[3]
    assert states[0] is not states[2]


def test_run_chemistry_failed_snapshot_stops_the_sequence(tmp_path):
    """A snapshot whose backend fails raises RuntimeError naming the module
    and time, and the later snapshots do not run. The earlier result stays
    on disk, and no '<module>.csv' copy is written.

    The middle snapshot (10 yr) fails, so 250 yr must not run.
    """
    run, calls = _fake_backend(tmp_path, fail_at=(10.0,))
    hf_all = _helpfile([0.0, 10.0, 250.0])

    with patch('proteus.atmos_chem.dummy.run_dummy_chem', side_effect=run):
        with pytest.raises(RuntimeError, match=r'dummy chemistry failed at t=1\.000e\+01 yr'):
            run_chemistry(_dirs(tmp_path), _make_config(times='all'), hf_all)

    assert [c[0] for c in calls] == [0.0, 10.0]
    assert (tmp_path / 'offchem' / 'dummy_0p000.csv').exists()
    assert not (tmp_path / 'offchem' / 'dummy_250p000.csv').exists()
    assert not (tmp_path / 'offchem' / 'dummy.csv').exists()


def test_run_chemistry_leaves_out_snapshots_without_an_atmosphere(tmp_path):
    """Snapshots whose helpfile row has no atmosphere (P_surf <= 0, as after
    desiccation, or NaN) are left out; the others run, and the last of them
    is copied to '<module>.csv'.

    The rows at 10 yr (P_surf 0) and 250 yr (NaN) are left out, so the copy
    must come from 100 yr (T_surf 2000 K) and not the final 250 yr row.
    """
    run, calls = _fake_backend(tmp_path)
    hf_all = _helpfile([0.0, 10.0, 100.0, 250.0])  # T_surf 3000, 2500, 2000, 1500 K
    hf_all['P_surf'] = [100.0, 0.0, 50.0, np.nan]

    with patch('proteus.atmos_chem.dummy.run_dummy_chem', side_effect=run):
        result = run_chemistry(_dirs(tmp_path), _make_config(times='all'), hf_all)

    assert [c[0] for c in calls] == [0.0, 100.0]
    assert result['tmp'].iloc[0] == pytest.approx(2000.0)
    assert not (tmp_path / 'offchem' / 'dummy_10p000.csv').exists()


def test_run_chemistry_all_snapshots_desiccated_runs_nothing(tmp_path):
    """When every snapshot is desiccated (P_surf = 0), run_chemistry returns
    None without calling the backend or writing '<module>.csv'."""
    run, calls = _fake_backend(tmp_path)
    hf_all = _helpfile([0.0, 10.0])
    hf_all['P_surf'] = 0.0

    with patch('proteus.atmos_chem.dummy.run_dummy_chem', side_effect=run):
        result = run_chemistry(_dirs(tmp_path), _make_config(times='all'), hf_all)

    assert result is None
    assert calls == []
    assert not (tmp_path / 'offchem' / 'dummy.csv').exists()


def test_run_chemistry_clears_offchem_once_at_the_start(tmp_path):
    """A file left in offchem by an earlier call is removed, while the files
    of this call's earlier snapshots survive its later ones."""
    stale = tmp_path / 'offchem' / 'dummy_999p000.csv'
    stale.parent.mkdir()
    stale.write_text('stale')
    run, _ = _fake_backend(tmp_path)

    with patch('proteus.atmos_chem.dummy.run_dummy_chem', side_effect=run):
        run_chemistry(_dirs(tmp_path), _make_config(times='all'), _helpfile([0.0, 10.0]))

    names = sorted(f.name for f in (tmp_path / 'offchem').iterdir())
    assert names == ['dummy.csv', 'dummy_0p000.csv', 'dummy_10p000.csv']
    assert not stale.exists()


# -----------------------------------------------------------------------
# Real dummy backend
# -----------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_run_chemistry_with_dummy_backend_at_listed_times(tmp_path):
    """With the real dummy backend and a list of times, each requested time
    maps to the nearest helpfile time (no atmosphere snapshots exist), the
    duplicates collapse, and the returned profile has positive pressures and
    mixing ratios summing to 1 on every level.

    Requests 4 and 6 yr both map to 5 yr; 90 yr maps to 100 yr.
    """
    hf_all = _helpfile([0.0, 5.0, 100.0])
    config = _make_config(times=[4.0, 6.0, 90.0])

    result = run_chemistry(_dirs(tmp_path), config, hf_all)

    names = sorted(f.name for f in (tmp_path / 'offchem').iterdir())
    assert names == ['dummy.csv', 'dummy_100p000.csv', 'dummy_5p000.csv']
    assert np.all(result['p'] > 0.0)
    vmr_sum = result[list(_ALL_SPECIES)].sum(axis=1).to_numpy()
    np.testing.assert_allclose(vmr_sum, 1.0, rtol=1e-6)  # file holds 9 significant digits
