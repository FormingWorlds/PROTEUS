"""Unit tests for ``tools/whole_run_ledger.py``: the command reads the run settings from the run folder,
prints the closure of a closed run and refuses a run it cannot ledger with exit code 2."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tests.utils.test_energy_ledger import _table

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_TOOL = Path(__file__).resolve().parents[2] / 'tools' / 'whole_run_ledger.py'


def _tool():
    spec = importlib.util.spec_from_file_location('whole_run_ledger_tool', _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(tmp_path, core_bc='core_module', toml=True):
    _table(core_bc).to_csv(tmp_path / 'runtime_helpfile.csv', sep='\t', index=False)
    if toml:
        (tmp_path / 'init_coupler.toml').write_text(
            f'[interior_energetics]\ncore_tfac_avg = 1.2\n[interior_energetics.aragog]\ncore_bc = "{core_bc}"\n'
        )
    return tmp_path


def test_the_command_prints_the_closure_of_a_closed_run(tmp_path, capsys):
    """A closed synthetic core_module run: exit 0, the settings of the run folder in the header, and a
    planet residual of 0 to round-off relative to L_int."""
    rc = _tool().main([str(_run(tmp_path))])
    out = capsys.readouterr().out
    assert rc == 0
    assert 'core_bc core_module, core_tfac 1.2' in out
    line = next(x for x in out.splitlines() if x.startswith('R_p'))
    assert abs(float(line.split('(')[1].split()[0])) < 1e-12


def test_the_core_bc_option_replaces_a_missing_config(tmp_path, capsys):
    """Without init_coupler.toml the command needs --core-bc: exit 2 without it, 0 with it, and the
    default core_tfac 1.147."""
    tool = _tool()
    run = _run(tmp_path, 'energy_balance', toml=False)
    assert tool.main([str(run)]) == 2
    assert 'pass --core-bc' in capsys.readouterr().err
    assert tool.main([str(run), '--core-bc', 'energy_balance']) == 0
    assert 'core_tfac 1.147' in capsys.readouterr().out


def test_a_run_the_ledger_refuses_exits_with_2(tmp_path, capsys):
    """A window outside the table and a missing helpfile give exit code 2 and the reason on stderr."""
    tool = _tool()
    run = _run(tmp_path)
    assert tool.main([str(run), '--start', '50']) == 2
    assert 'start_row' in capsys.readouterr().err
    assert tool.main([str(tmp_path / 'absent'), '--core-bc', 'core_module']) == 2
    assert 'runtime_helpfile.csv' in capsys.readouterr().err
