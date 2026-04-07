"""
Unit tests for CHILI postprocessing helper script.

These tests cover the scalar postprocessing path in ``tools/chili_postproc.py``
without invoking heavy atmosphere/profile readers or plotting.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _load_chili_postproc_module():
    """Load ``tools/chili_postproc.py`` as a module for direct function tests."""
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / 'tools' / 'chili_postproc.py'
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
def test_postproc_once_writes_scalar_csv_and_orders_log_message(tmp_path):
    """Writes scalar CSV and logs scalar-write message before config parsing."""
    chili_postproc = _load_chili_postproc_module()

    simdir = tmp_path / 'run_chili_tr1b'
    simdir.mkdir()

    init_path = simdir / 'init_coupler.toml'
    init_path.write_text('placeholder = true\n', encoding='utf-8')

    all_gases = sorted(set(chili_postproc.vol_list))
    _make_runtime_helpfile(simdir / 'runtime_helpfile.csv', all_gases)

    class _FakeInteriorData:
        def get_dict_values(self, keys):
            if keys == ['data', 'temp_b']:
                return np.array([1000.0, 1500.0, 2000.0])
            if keys == ['data', 'visc_b']:
                return np.array([1.0e20, 2.0e20, 3.0e20])
            raise KeyError(keys)

    recorded_messages = []
    original_print = builtins.print

    def fake_print(*args, **kwargs):
        msg = ' '.join(str(a) for a in args)
        recorded_messages.append(msg)

    def fake_read_config(_config_path):
        recorded_messages.append('READ_CONFIG_CALLED')
        return SimpleNamespace(
            orbit=SimpleNamespace(s0_factor=1.0),
            atmos_clim=SimpleNamespace(albedo_pl=0.3, surface_d=1000.0),
        )

    chili_postproc.read_config_object = fake_read_config
    chili_postproc.read_jsons = lambda _simdir, _times: [_FakeInteriorData()]
    chili_postproc.read_ncdf_profile = lambda *_args, **_kwargs: None

    try:
        builtins.print = fake_print
        name = chili_postproc.postproc_once(str(simdir), plot=False)
    finally:
        builtins.print = original_print

    assert name == 'trappist1b'
    assert '    write CSV file for scalars' in recorded_messages
    assert 'READ_CONFIG_CALLED' in recorded_messages
    assert recorded_messages.index('    write CSV file for scalars') < recorded_messages.index(
        'READ_CONFIG_CALLED'
    )

    out_csv = simdir / 'chili' / 'evolution-proteus-trappist1b-data.csv'
    assert out_csv.is_file()

    out_df = pd.read_csv(out_csv)
    assert 'flux_ASR(W/m2)' in out_df.columns
    assert 'viscosity(Pa.s)' in out_df.columns
    assert out_df.loc[0, 'viscosity(Pa.s)'] == pytest.approx(2.0e20)


@pytest.mark.unit
def test_postproc_once_raises_when_helpfile_missing(tmp_path):
    """Raises ``FileNotFoundError`` when runtime helpfile is absent."""
    chili_postproc = _load_chili_postproc_module()

    simdir = tmp_path / 'case_no_helpfile'
    simdir.mkdir()

    with pytest.raises(FileNotFoundError, match='runtime_helpfile.csv'):
        chili_postproc.postproc_once(str(simdir), plot=False)
