"""Tests for ``tools/extract_chili_final_numbers.py``.

The tool reads the CHILI run folders. These tests build small run folders under
``tmp_path`` and replace ``netCDF4`` with a stand-in module, so no run data is read.
Checked: the last interior snapshot is the one with the latest model time, and every
missing input is reported as not available or raises, never printed as zero.
"""

from __future__ import annotations

import importlib.util
import json
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


def _make_run(
    tmp_path, drop=(), logs=('Total runtime: 2.5 hours\n',), snapshots=(), name='chili_venus'
):
    """Write a run folder with a two-row helpfile, a status file, logs and snapshot names."""
    run = tmp_path / name
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


def test_retention_and_delta_iw_follow_the_first_and_last_rows(tmp_path):
    """H and C retention are last over first row totals; Delta IW is log10(pO2) minus the
    O'Neill and Eggins (2002) IW buffer at T_magma, in log10 units, not natural log."""
    tool = _load_tool()
    res = tool.extract_run_data(_make_run(tmp_path), 'Nominal Venus')
    t = 1776.0
    iw = 2.0 * (-244118.0 + 115.559 * t - 8.474 * t * np.log(t)) / (np.log(10) * 8.31441 * t)

    assert res['h_retention_pct'] == pytest.approx(75.0, rel=1e-12)
    assert res['c_retention_pct'] == pytest.approx(90.0, rel=1e-12)
    assert res['delta_iw'] == pytest.approx(-8.0 - iw, rel=1e-12)
    assert abs(res['delta_iw'] - (np.log(1.0e-8) - iw)) > 5.0  # natural log is far off
    assert res['reached_5pct_melt'] is True


def test_wall_time_from_log_timestamps_and_its_limits(tmp_path):
    """A log without a runtime line counts from its first to its last timestamp; a log
    with a single timestamp makes the total not available instead of an undercount."""
    tool = _load_tool()
    stamps = '2026-10-14 10:00:00 start\n2026-10-14 16:00:00 end\n'
    run = _make_run(tmp_path, logs=('Total runtime: 2.5 hours\n', stamps))

    assert tool.extract_wall_time(run) == pytest.approx(8.5, rel=1e-12)

    (run / 'proteus_02.log').write_text('2026-10-15 10:00:00 only line\n', encoding='utf-8')
    assert tool.extract_wall_time(run) is None


def test_short_last_helpfile_row_raises(tmp_path):
    """A last row with fewer fields than the header is a partial write and raises,
    naming the file and both counts."""
    tool = _load_tool()
    run = _make_run(tmp_path)
    with (run / 'runtime_helpfile.csv').open('a', encoding='utf-8') as f:
        f.write('9.0e7\t0.04\n')

    with pytest.raises(ValueError, match='expected 13 columns, got 2') as excinfo:
        tool.extract_run_data(run, 'Nominal Venus')
    assert 'runtime_helpfile.csv' in str(excinfo.value)


def test_main_prints_every_case_and_refuses_a_missing_folder(tmp_path, monkeypatch, capsys):
    """main reads the five case folders, prints one table row per case and writes the
    JSON; a missing case folder raises FileNotFoundError."""
    tool = _load_tool()
    for name, _ in tool.CASES:
        _make_run(tmp_path, name=name)
    out_json = tmp_path / 'numbers.json'
    monkeypatch.setattr(
        sys, 'argv', ['extract', '--runs', str(tmp_path), '--json-out', str(out_json)]
    )

    tool.main()

    printed = capsys.readouterr().out
    assert all(f'| {disp} |' in printed for _, disp in tool.CASES)
    assert len(json.loads(out_json.read_text(encoding='utf-8'))) == len(tool.CASES)

    monkeypatch.setattr(sys, 'argv', ['extract', '--runs', str(tmp_path / 'absent')])
    with pytest.raises(FileNotFoundError, match='Run folder not found'):
        tool.main()
