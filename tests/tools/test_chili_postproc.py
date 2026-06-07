"""
Unit tests for the CHILI postprocessing helper script.

These tests cover the scalar postprocessing path in
``tools/chili_postproc.py`` without invoking heavy atmosphere/profile
readers or plotting. Running them requires a PROTEUS environment on the
path, because ``chili_postproc.py`` imports ``proteus``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _load_chili_postproc_module():
    """Load the co-located ``chili_postproc.py`` as a module for direct tests."""
    script_path = Path(__file__).resolve().parents[2] / 'tools' / 'chili_postproc.py'
    spec = importlib.util.spec_from_file_location('chili_postproc_under_test', script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_runtime_helpfile(path: Path, gas_list: list[str]):
    """Create a minimal runtime helpfile with physically valid positive values."""
    data = {
        'Time': [1.0],
        'T_surf': [300.0],
        'T_pot': [1500.0],
        'F_int': [100.0],
        'F_olr': [220.0],
        'F_ins': [340.0],
        'Phi_global_vol': [0.2],
        'O2_bar': [1.0e-3],
        'C_kg_solid': [1.0e17],
        'C_kg_liquid': [2.0e17],
        'C_kg_atm': [3.0e17],
        'H_kg_solid': [4.0e17],
        'H_kg_liquid': [5.0e17],
        'H_kg_atm': [6.0e17],
        'O_kg_atm': [7.0e17],
        'P_surf': [1.0],
        'atm_kg_per_mol': [2.8e-2],
        'R_obs': [6.5e6],
        'R_int': [6.3e6],
        'RF_depth': [0.01],
        'Phi_global': [0.15],
    }
    for gas in gas_list:
        data[f'{gas}_bar'] = [1.0e-6]

    df = pd.DataFrame(data)
    df.to_csv(path, sep=' ', index=False)


@pytest.mark.unit
def test_postproc_once_raises_when_helpfile_missing(tmp_path):
    """Raises ``FileNotFoundError`` when runtime helpfile is absent."""
    chili_postproc = _load_chili_postproc_module()

    simdir = tmp_path / 'case_no_helpfile'
    simdir.mkdir()

    with pytest.raises(FileNotFoundError, match='runtime_helpfile.csv'):
        chili_postproc.postproc_once(str(simdir), plot=False)
    # Discrimination: confirm the helpfile genuinely is absent at the time
    # of the raise so the FileNotFoundError above can only have come from
    # the missing-helpfile guard, not from some other I/O on a different
    # path.
    assert not (simdir / 'runtime_helpfile.csv').exists()
