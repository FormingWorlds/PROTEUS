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
    # The guard runs before any output is touched: a failed call must
    # leave no chili/ folder behind.
    assert not (simdir / 'chili').exists()


@pytest.mark.unit
def test_postproc_once_writes_protocol_scalars(tmp_path, monkeypatch):
    """The scalar path writes the CHILI-MIP evolution CSV from a helpfile.

    Heavy pieces (the Proteus handler, config read, profile sampling,
    plotting) are stubbed; the contract under test is the column
    mapping from the runtime helpfile to the protocol schema, including
    the derived absorbed-flux and solid-radius columns.
    """
    import types

    import numpy as np

    chili_postproc = _load_chili_postproc_module()

    simdir = tmp_path / 'chili_earth'
    (simdir / 'data').mkdir(parents=True)
    _make_runtime_helpfile(simdir / 'runtime_helpfile.csv', chili_postproc.vol_list)
    (simdir / 'init_coupler.toml').write_text('unused')

    fake_config = types.SimpleNamespace(
        orbit=types.SimpleNamespace(s0_factor=0.375),
        atmos_clim=types.SimpleNamespace(albedo_pl=0.1, surface_d=0.01),
    )
    handler = types.SimpleNamespace(
        extract_archives=lambda: None,
        create_archives=lambda: None,
    )
    monkeypatch.setattr(chili_postproc, 'read_config_object', lambda path: fake_config)
    monkeypatch.setattr(chili_postproc, 'Proteus', lambda config_path: handler)

    chili_postproc.postproc_once(str(simdir), plot=False)

    csvs = list((simdir / 'chili').glob('evolution-proteus-earth-data.csv'))
    assert len(csvs) == 1
    out = pd.read_csv(csvs[0])
    # Column mapping: helpfile values land under the protocol names.
    assert out['T_surf(K)'].iloc[0] == pytest.approx(300.0, rel=1e-12)
    assert out['p_surf(bar)'].iloc[0] == pytest.approx(1.0, rel=1e-12)
    # Derived absorbed flux: F_ins * s0_factor * (1 - albedo). The
    # asymmetric factors discriminate dropped terms: forgetting the
    # albedo factor lands at 127.5, forgetting s0 at 306.
    assert out['flux_ASR(W/m2)'].iloc[0] == pytest.approx(340.0 * 0.375 * 0.9, rel=1e-9)
    # Derived solid radius: R_int * (1 - RF_depth) = 6.3e6 * 0.99.
    assert out['R_solid(m)'].iloc[0] == pytest.approx(6.237e6, rel=1e-9)
    # Missing interior snapshots produce a NaN viscosity, not a crash.
    assert np.isnan(out['viscosity(Pa.s)'].iloc[0])
