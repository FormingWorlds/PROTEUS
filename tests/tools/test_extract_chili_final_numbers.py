"""Tests for ``tools/extract_chili_final_numbers.py``.

The tool reads the CHILI run folders. These tests build small run folders under
``tmp_path`` and replace ``netCDF4`` with a stand-in module, so no run data is read.
Checked: the last interior snapshot is the one with the latest model time, and every
missing input is reported as not available or raises, never printed as zero.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_COLUMNS = (
    'Time Phi_global Phi_global_vol T_magma T_surf P_surf F_atm F_net '
    'H2O_bar CO2_bar O2_bar H_kg_total C_kg_total'
).split()
_FIRST = [
    0.0,
    1.0,
    1.0,
    3000.0,
    2900.0,
    300.0,
    5.0e4,
    5.0e4,
    250.0,
    50.0,
    1.0e-6,
    2.0e20,
    1.0e20,
]
_LAST = [
    8.6793e7,
    0.05,
    0.06,
    1776.0,
    1700.0,
    320.0,
    1.32,
    1.32,
    280.0,
    40.0,
    1.0e-8,
    1.5e20,
    0.9e20,
]


def _load_tool():
    """Load the co-located ``extract_chili_final_numbers.py`` as a module."""
    script = Path(__file__).resolve().parents[2] / 'tools' / 'extract_chili_final_numbers.py'
    spec = importlib.util.spec_from_file_location('extract_chili_under_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_run(tmp_path, drop=(), logs=('Total runtime: 2.5 hours\n',), snapshots=()):
    """Write a run folder with a two-row helpfile, a status file, logs and snapshot names."""
    run = tmp_path / 'chili_venus'
    (run / 'data').mkdir(parents=True)
    keep = [i for i, c in enumerate(_COLUMNS) if c not in drop]
    rows = [[_COLUMNS[i] for i in keep]] + [[str(r[i]) for i in keep] for r in (_FIRST, _LAST)]
    (run / 'runtime_helpfile.csv').write_text(
        '\n'.join('\t'.join(r) for r in rows) + '\n', encoding='utf-8'
    )
    (run / 'status').write_text('10\nCompleted (solidified)\n', encoding='utf-8')
    for i, text in enumerate(logs):
        (run / f'proteus_{i:02d}.log').write_text(text, encoding='utf-8')
    for name in snapshots:
        (run / 'data' / name).write_bytes(b'')
    return run


def _fake_netcdf4(monkeypatch, variables):
    """Install a netCDF4 stand-in that records the opened path and serves ``variables``."""
    opened = []

    def dataset(path):
        opened.append(Path(path).name)
        return types.SimpleNamespace(variables=variables)

    monkeypatch.setitem(sys.modules, 'netCDF4', types.SimpleNamespace(Dataset=dataset))
    return opened


def test_last_snapshot_is_the_one_with_the_latest_model_time(tmp_path, monkeypatch):
    """Snapshot names hold the model time in years; '9997456p548' sorts after
    '86793456p548' as a string, so a string sort picks the 10 Myr snapshot."""
    tool = _load_tool()
    names = ('0p000_int.nc', '10000p614_int.nc', '9997456p548_int.nc', '86793456p548_int.nc')
    run = _make_run(tmp_path, snapshots=names)
    opened = _fake_netcdf4(monkeypatch, {'phi_s': np.array([0.0104, 0.3104])})

    res = tool.extract_run_data(run, 'Nominal Venus')

    assert opened == ['86793456p548_int.nc']
    assert res['mush_node_range_pct'] == pytest.approx((1.04, 31.04))


def test_missing_species_columns_and_log_print_not_available(tmp_path):
    """A helpfile without H2O_bar and CO2_bar and a run without a log give 'n/a' in the
    table, not 0.0 bar and 0.0 h."""
    tool = _load_tool()
    run = _make_run(tmp_path, drop=('H2O_bar', 'CO2_bar'), logs=())

    res = tool.extract_run_data(run, 'Nominal Venus')
    row = tool.build_markdown_summary([res]).splitlines()[4]

    assert res['wall_time_hours'] is None
    assert row.endswith('| n/a | n/a |')
    assert '0.0 h' not in row


def test_missing_totals_and_oxygen_give_not_available_notes(tmp_path):
    """Without H_kg_total, C_kg_total and O2_bar the notes say so for each quantity."""
    tool = _load_tool()
    run = _make_run(tmp_path, drop=('H_kg_total', 'C_kg_total', 'O2_bar'))

    notes = tool.build_markdown_summary([tool.extract_run_data(run, 'Nominal Venus')])

    assert 'H retention: not available' in notes
    assert 'C retention: not available' in notes
    assert 'Delta IW: not available' in notes


def test_snapshot_without_netcdf4_raises_and_without_phi_s_is_not_available(
    tmp_path, monkeypatch
):
    """A snapshot that cannot be read because netCDF4 is missing is an error; a snapshot
    without phi_s gives 'mush range: not available'."""
    tool = _load_tool()
    run = _make_run(tmp_path, snapshots=('86793456p548_int.nc',))
    monkeypatch.setitem(sys.modules, 'netCDF4', None)

    with pytest.raises(ImportError, match='netCDF4'):
        tool.extract_run_data(run, 'Nominal Venus')

    _fake_netcdf4(monkeypatch, {})
    notes = tool.build_markdown_summary([tool.extract_run_data(run, 'Nominal Venus')])
    assert 'mush range: not available' in notes
