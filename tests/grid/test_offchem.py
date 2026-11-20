"""Unit tests for ``proteus.grid.offchem`` (``proteus offchem --grid``).

Contract clauses checked:
  - The job limits come from the three keys of the grid file, with their types; a
    missing file or key is refused by name.
  - Only cases with a helpfile and a config in cfgs/ are selected, gaps in the
    numbering included.
  - The VULCAN network is compiled once when the cases share it, and not at all when
    every case reuses the compiled one.
  - Local runs never exceed the grid's `max_jobs` at once, do run that many together,
    and a failed case is reported without stopping the others; a grid without cases
    is refused.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from proteus.grid import offchem

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.fixture
def grid(tmp_path, monkeypatch):
    """Empty grid folder 'g1' in an output root under tmp_path, with a grid file."""
    monkeypatch.setenv('PROTEUS_OUTPUT_PATH', str(tmp_path / 'output'))
    folder = tmp_path / 'output' / 'g1'
    (folder / 'cfgs').mkdir(parents=True)
    _write_grid_file(folder, max_jobs=2, max_days=1.5, max_mem=4)
    return folder


def _write_grid_file(folder, **limits):
    # A dimension table as well, which grid_limits must leave alone
    lines = [f'{k} = {v}' for k, v in limits.items()]
    lines += ['', '["planet.mass_tot"]', 'method = "direct"', 'values = [0.7, 1.0]']
    (folder / 'copy.grid.toml').write_text('\n'.join(lines) + '\n')


def _fake_config(name, module='dummy', path=None, network='CHO', photo_on=True, make=True):
    """Config with the fields offchem reads; path defaults to the case in grid 'g1'."""
    return SimpleNamespace(
        params=SimpleNamespace(out=SimpleNamespace(path=path or f'g1/{name}/')),
        atmos_chem=SimpleNamespace(
            module=module,
            photo_on=photo_on,
            vulcan=SimpleNamespace(network=network, make_funs=make),
        ),
    )


def _add_cases(grid, monkeypatch, configs, helpfile=True, status=None):
    """Add case folders with a helpfile and a config each; read_config_object returns
    the given fake config for that config file."""
    by_path = {}
    for name, config in configs.items():
        (grid / name).mkdir()
        if helpfile:
            (grid / name / 'runtime_helpfile.csv').write_text('Time\n0.0\n')
        if status is not None:
            (grid / name / 'status').write_text(f'{status}\n')
        cfg = grid / 'cfgs' / f'{name}.toml'
        cfg.write_text('# case config\n')
        by_path[str(cfg)] = config
    monkeypatch.setattr(offchem, 'read_config_object', lambda p: by_path[str(p)])


def test_grid_limits_reads_the_three_keys_with_their_types(grid):
    """max_jobs is an int and the time [days] and memory [GB] are floats; the dimension
    tables of the grid file are not read as limits."""
    limits = offchem.grid_limits(grid)

    assert limits == {'max_jobs': 2, 'max_days': 1.5, 'max_mem': 4.0}
    assert isinstance(limits['max_jobs'], int)
    assert isinstance(limits['max_mem'], float)


@pytest.mark.parametrize('drop', ['file', 'max_mem'], ids=['no_grid_file', 'no_max_mem'])
def test_grid_limits_refuses_a_missing_grid_file_or_key(grid, drop):
    """A grid without copy.grid.toml, or one lacking a limit, is refused by name."""
    if drop == 'file':
        (grid / 'copy.grid.toml').unlink()
    else:
        _write_grid_file(grid, max_jobs=2, max_days=1.0)

    with pytest.raises(ValueError, match='copy.grid.toml') as err:
        offchem.grid_limits(grid)
    if drop == 'max_mem':
        assert 'lacks max_mem' in str(err.value)


def test_find_cases_keeps_cases_with_a_helpfile_and_a_config(grid, monkeypatch, caplog):
    """Of four case folders, the one without a helpfile and the one without a config are
    left out (the second with a warning); the gap in the numbering does not matter."""
    _add_cases(grid, monkeypatch, {'case_000000': None, 'case_000004': None})
    (grid / 'case_000001').mkdir()  # not run far enough to write a helpfile
    (grid / 'case_000003').mkdir()
    (grid / 'case_000003' / 'runtime_helpfile.csv').write_text('Time\n0.0\n')

    cases = offchem.find_cases(grid)

    assert [name for name, _ in cases] == ['case_000000', 'case_000004']
    assert cases[1][1] == grid / 'cfgs' / 'case_000004.toml'
    assert 'case_000003' in caplog.text
    assert 'case_000001' not in caplog.text


def test_check_cases_compiles_a_shared_vulcan_network_once(grid, monkeypatch):
    """Three VULCAN cases with the CHO photochemical network compile it once, and the
    function says so, so that the cases skip compiling it."""
    names = ['case_000000', 'case_000001', 'case_000005']
    _add_cases(grid, monkeypatch, {n: _fake_config(n, module='vulcan') for n in names})
    compiled = []
    monkeypatch.setattr(offchem, 'vulcan_problem', lambda config: None)
    monkeypatch.setattr(
        'proteus.atmos_chem.vulcan.compile_network', lambda c: compiled.append(c) or 'f'
    )

    assert offchem.check_cases(grid, offchem.find_cases(grid)) is True
    assert len(compiled) == 1
    assert compiled[0].atmos_chem.vulcan.network == 'CHO'


def test_check_cases_compiles_nothing_when_every_case_reuses_its_network(grid, monkeypatch):
    """With make_funs = false in every case, the cases use the network compiled before,
    as their configs ask, even with different networks."""
    configs = {
        'case_000000': _fake_config('case_000000', module='vulcan', make=False),
        'case_000001': _fake_config('case_000001', module='vulcan', network='NCHO', make=False),
    }
    _add_cases(grid, monkeypatch, configs)
    compiled = []
    monkeypatch.setattr(offchem, 'vulcan_problem', lambda config: None)
    monkeypatch.setattr('proteus.atmos_chem.vulcan.compile_network', compiled.append)

    assert offchem.check_cases(grid, offchem.find_cases(grid)) is False
    assert compiled == []


def test_offchem_grid_runs_max_jobs_cases_at_once_and_reports_failures(grid, monkeypatch):
    """Four cases with max_jobs = 2: two run together (a two-party barrier would time out
    otherwise), never more than two, and the failure of case 1 leaves the others running."""
    names = ['case_000000', 'case_000001', 'case_000002', 'case_000003']
    _add_cases(grid, monkeypatch, {n: _fake_config(n) for n in names})
    monkeypatch.setattr(offchem.os, 'cpu_count', lambda: 8)
    barrier, lock = threading.Barrier(2, timeout=5), threading.Lock()
    state = {'now': 0, 'peak': 0, 'cmds': []}

    def fake_run(cmd, stdout, stderr, check):
        with lock:
            state['now'] += 1
            state['peak'] = max(state['peak'], state['now'])
            state['cmds'].append(cmd)
        barrier.wait()
        time.sleep(0.01)
        with lock:
            state['now'] -= 1
        stdout.write('ran\n')
        return SimpleNamespace(returncode=1 if 'case_000001' in cmd[3] else 0)

    monkeypatch.setattr(offchem.subprocess, 'run', fake_run)

    failed = offchem.offchem_grid(grid)

    assert failed == ['case_000001']
    assert state['peak'] == 2
    assert sorted(c[3] for c in state['cmds']) == [
        str(grid / 'cfgs' / f'{n}.toml') for n in names
    ]
    assert all(
        c[:3] == ['proteus', 'offchem', '--config'] and len(c) == 4 for c in state['cmds']
    )
    assert (grid / 'logs' / 'offchem_case_000003.log').read_text() == 'ran\n'


def test_offchem_grid_refuses_a_grid_without_cases(grid):
    """A grid in which no case has a helpfile is refused before the logs folder is made."""
    (grid / 'case_000000').mkdir()

    with pytest.raises(ValueError, match='No case'):
        offchem.offchem_grid(grid)
    assert not (grid / 'logs').exists()
