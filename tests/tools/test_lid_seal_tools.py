"""Unit tests for the lid seal tools in ``tools/verification/lid_seal/``.

The data tools run on small synthetic run folders: the closing row is the first active row, the
data extractor keeps the closing row and the 13 after it and every 5th row before closure, the
ensemble tool writes the superheat offset and the end values of each start, and the number check
fills its templates and runs the code blocks of a page. The replay tools need the full tutorial
runs and have no stored input small enough for a unit test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_DIR = Path(__file__).resolve().parents[2] / 'tools' / 'verification' / 'lid_seal'


def _load(name: str):
    """Load a script of the folder; its sibling imports resolve from the folder."""
    sys.path.insert(0, str(_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f'{name}_uut', _DIR / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_DIR))
    return module


def _run(folder: Path, n=30, close=12, h2o=1.0, superheat=None):
    """A run folder with ``n`` rows whose seal is active from row ``close`` on."""
    folder.mkdir()
    t = np.arange(n) * 100.0
    data = {
        'Time': t, 'Phi_global': np.linspace(0.5, 0.1, n), 'M_mantle': np.full(n, 4e24),
        'R_int': np.full(n, 6.4e6), 'gravity': np.full(n, 9.8), 'T_magma': np.linspace(3000, 1500, n),
        'fO2_shift_IW_derived': np.full(n, 2.0), 'M_atm': np.full(n, 1e19),
        'seal_active': (np.arange(n) >= close).astype(float), 'H2O_kg_atm': h2o * (1 + t / 1e4),
        'H2O_mol_atm': np.full(n, 3.0), 'CO2_kg_atm': np.zeros(n),
        'H2O_kg_sealed': np.where(np.arange(n) >= close, 5e18, 0.0),
        'H2O_kg_eq': np.where(np.arange(n) >= close, 4e18, 0.0),
        'esc_step_kg': np.full(n, 1e10), 'esc_rate_total': np.full(n, 1e3),
    }  # fmt: skip
    pd.DataFrame(data).to_csv(folder / 'runtime_helpfile.csv', sep=' ', index=False)
    planet = '' if superheat is None else f'delta_T_super = {superheat!r}\n'
    (folder / 'init_coupler.toml').write_text(f'[planet]\n{planet}')
    return folder


def test_the_closing_row_is_the_first_active_row_and_a_run_without_one_is_refused(tmp_path):
    runs = _load('_runs')
    seal = runs.read_run(_run(tmp_path / 'a', close=7))
    assert runs.closing_row(seal) == 7
    with pytest.raises(ValueError, match='never closes'):
        runs.closing_row(runs.read_run(_run(tmp_path / 'b', close=99)))


def test_the_data_extractor_keeps_the_closing_rows_and_thins_the_span_before(tmp_path):
    tool = _load('extract_data')
    seal, off, hold = (_run(tmp_path / name) for name in ('seal', 'off', 'hold'))
    tool.main(
        [
            str(seal),
            str(off),
            str(hold),
            str(tmp_path / 'out'),
            '--commit',
            'abc',
            '--aragog',
            'def',
            '--calliope',
            '1',
        ]
    )
    rows = pd.read_csv(tmp_path / 'out' / 'tutorial_closing_rows.csv', comment='#')
    assert list(rows['row']) == list(range(12, 26))
    assert 'H2O_kg_sealed' in rows and 'CO2_kg_atm' not in rows  # zero on the closing row
    water = pd.read_csv(tmp_path / 'out' / 'tutorial_water.csv', comment='#')
    assert list(water['Time'] / 100) == [0, 5, 10, *range(12, 30)]
    head = (tmp_path / 'out' / 'tutorial_water.csv').read_text().splitlines()[0]
    assert 'PROTEUS commit abc, aragog commit def, CALLIOPE 1.' in head


def test_the_ensemble_writes_the_superheat_offset_and_refuses_unpaired_starts(tmp_path):
    tool = _load('ensemble')
    v = 500.0 + 5e-10
    s, o = _run(tmp_path / 's', superheat=v, h2o=2.0), _run(tmp_path / 'o', superheat=v)
    out = tmp_path / 'e.csv'
    tool.main(
        [str(out), '--commit', 'abc', '--aragog', 'def', '--calliope', '1', f'k1={s},{o}']
    )
    d = pd.read_csv(out, comment='#', float_precision='round_trip')
    assert d['dT'].item() == v - 500.0 and d['t_close'].item() == 1200.0
    assert d['seal_atm'].item() == 2.0 * d['open_atm'].item()
    other = _run(tmp_path / 'p', superheat=500.0)
    with pytest.raises(AssertionError, match='other superheats'):
        tool.main(
            [
                str(out),
                '--commit',
                'abc',
                '--aragog',
                'def',
                '--calliope',
                '1',
                f'k1={s},{other}',
            ]
        )


def test_the_number_check_formats_its_values_and_runs_the_page_blocks(tmp_path):
    tool = _load('check_numbers')
    assert tool._sci(1.3391e15) == '1.339e15' and tool._sci(6.7994e-4) == '6.799e-4'
    assert tool._tex(8.678e-4) == '$8.7 \\times 10^{-4}$'
    seal = _run(tmp_path / 'seal')
    page = (
        '```python\nimport pandas as pd\n'
        "on = pd.read_csv('output/tutorial_lid_seal/runtime_helpfile.csv', sep=r'\\s+')\n"
        'print(len(on))\n```\n\n```text\n30\n```\n'
    )
    assert tool.tutorial_blocks(page, seal, seal) == [('30', '30')]
    names = {n for _, t in tool.TEMPLATES for n in __import__('re').findall(r'{(\w+)', t)}
    assert names == {
        'crust', 'phi_k', 'phi_k1', 'melt', 'total', 'spec', 'leak', 'leak_abs', 'rest', 'plow',
        'ptot', 'co', 'co2', 'h2o', 't_spread', 'seal_spread_tex', 'open_spread_tex',
        'band_printed', 'n_below', 'span_below', 'below_max', 't_close_kyr',
        'trap_h2o', 'trap_all', 'trap_ratio', 'seq', 'closed_gas',
    }  # fmt: skip
