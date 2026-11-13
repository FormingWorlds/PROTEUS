"""Unit tests for ``proteus.atmos_chem.vulcan_agni``: the climate hook for coupled chemistry.

`SnapshotClimate` is tested with AGNI replaced by a stand-in atmosphere carrying the fields
the hook reads, with `init_agni_atmos`, `run_agni` and the snapshot loader mocked, and with
a stand-in `vulcan.agni` (VULCAN is not installed for the unit tests; its own tests cover
`set_composition` and `climate_update`, and the integration test runs the real ones).

Invariants tested:
  - The AGNI atmosphere is set up from the run's own config (the gases of the snapshot), with
    the stellar spectrum the snapshot used, then loaded from the snapshot file
  - The solve has AGNI's equilibrium chemistry off, so the composition stays VULCAN's
  - VULCAN's composition is handed to AGNI before the solve, and the profile handed back to
    VULCAN is no colder than the VULCAN floor
  - An unconverged AGNI solve or an unreadable snapshot is an error, never a silent climate
"""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from proteus.atmos_chem.vulcan_agni import VULCAN_TMP_FLOOR, SnapshotClimate

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.fixture(autouse=True)
def fake_vulcan_agni():
    """Stand-in `vulcan.agni`: records the composition hand-off and returns AGNI's profiles."""
    agni = types.ModuleType('vulcan.agni')
    agni.set_composition = MagicMock()
    agni.climate_update = MagicMock(
        side_effect=lambda atmos: SimpleNamespace(
            p=np.asarray(atmos.p, dtype=float) * 10.0, tmp=np.array(atmos.tmp, dtype=float)
        )
    )
    package = types.ModuleType('vulcan')
    package.agni = agni
    with patch.dict(sys.modules, {'vulcan': package, 'vulcan.agni': agni}):
        yield agni


def _fake_atmos():
    """AGNI-like atmosphere [SI]: 3 layers from 1e4 to 1e2 Pa, top down, Kzz on 4 edges."""
    return SimpleNamespace(
        p=np.array([1e2, 1e3, 1e4]),
        pl=np.array([3e1, 3e2, 3e3, 3e4]),
        tmp=np.array([150.0, 600.0, 1500.0]),  # the top layer is below the VULCAN floor
        Kzz=np.array([1.0, 10.0, 100.0, 100.0]),  # m2 s-1
        gas_names=['H2O', 'H2'],
        gas_vmr={'H2O': np.full(3, 0.5), 'H2': np.full(3, 0.5)},
        gas_ovmr={'H2O': np.full(3, 0.5), 'H2': np.full(3, 0.5)},
        flux_tot=np.array([12.0, 30.0, 50.0]),
    )


def _config():
    """Config shaped for the hook: equilibrium chemistry on during the evolution."""
    return SimpleNamespace(atmos_clim=SimpleNamespace(agni=SimpleNamespace(chemistry='eq')))


def _setup(tmp_path, *, loaded=True):
    """A SnapshotClimate for the snapshot at 700 yr, with AGNI mocked; returns it and mocks."""
    (tmp_path / 'data').mkdir(exist_ok=True)
    (tmp_path / 'data' / '100.sflux').write_text('# WL(nm) Flux\n')
    dirs = {'output': str(tmp_path), 'output/offchem': str(tmp_path / 'offchem')}
    atmos = _fake_atmos()
    jl = MagicMock()
    jl.AGNI.load.load_ncdf_b.return_value = loaded
    config = _config()
    hf_row = {'Time': 700.0, 'F_atm': 25.0}
    with (
        patch('proteus.atmos_chem.vulcan_agni._ensure_agni'),
        patch('juliacall.Main', jl),
        patch('proteus.atmos_clim.agni.init_agni_atmos', return_value=atmos) as mock_init,
        patch('proteus.atmos_clim.agni.deallocate_atmos') as mock_dealloc,
    ):
        climate = SnapshotClimate(dirs, config, hf_row) if loaded else None
        if not loaded:
            with pytest.raises(RuntimeError, match='could not load the atmosphere snapshot'):
                SnapshotClimate(dirs, config, hf_row)
    return climate, dict(
        atmos=atmos,
        jl=jl,
        init=mock_init,
        dealloc=mock_dealloc,
        dirs=dirs,
        config=config,
        hf_row=hf_row,
    )


def test_setup_uses_the_snapshot_gases_spectrum_and_file(tmp_path):
    """The setup keeps the run's config, so the gases match the snapshot file, and uses the
    snapshot's stellar spectrum and a working folder in offchem/; the solves switch AGNI's
    equilibrium chemistry off on a copy, leaving the run's config as it was."""
    climate, m = _setup(tmp_path)

    args, kwargs = m['init'].call_args
    assert args == (m['dirs'], m['config'], m['hf_row'])
    assert kwargs == dict(
        use_cache=False,
        star_file=os.path.join(str(tmp_path), 'data', '100.sflux'),
        io_dir=os.path.join(str(tmp_path), 'offchem', 'agni'),
    )
    snapshot = os.path.join(str(tmp_path), 'data', '700p000_atm.nc')
    m['jl'].AGNI.load.load_ncdf_b.assert_called_once_with(m['atmos'], snapshot)
    assert climate.config_solve.atmos_clim.agni.chemistry is None
    assert m['config'].atmos_clim.agni.chemistry == 'eq'


def test_an_unreadable_snapshot_is_an_error_and_frees_the_atmosphere(tmp_path):
    """A snapshot AGNI cannot load (another level or gas count) raises, after freeing AGNI."""
    _, m = _setup(tmp_path, loaded=False)

    m['dealloc'].assert_called_once_with(m['atmos'])
    assert m['init'].call_count == 1


@pytest.mark.physics_invariant
def test_climate_update_solves_with_vulcan_gases_and_returns_a_floored_profile(
    tmp_path, fake_vulcan_agni
):
    """One climate update gives AGNI VULCAN's composition, then solves through run_agni with
    the chemistry-free config and the snapshot left unwritten, and hands back AGNI's
    temperature with the 150 K top layer raised to VULCAN_TMP_FLOOR and the rest unchanged."""
    climate, m = _setup(tmp_path)
    pco = np.array([1e5, 1e4, 1e3])  # dyn cm-2, bottom up: AGNI's 1e4, 1e3, 1e2 Pa
    ymix = np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])

    order = []
    fake_vulcan_agni.set_composition.side_effect = lambda *a: order.append('composition')

    def _solve(*args):
        order.append('solve')
        return m['atmos'], {'agni_converged': True}

    with patch('proteus.atmos_clim.agni.run_agni', side_effect=_solve) as mock_run:
        update = climate(pco, np.full(3, 900.0), ['H2O', 'H2'], ymix)

    assert order == ['composition', 'solve']
    atmos, p_given, species, ymix_given = fake_vulcan_agni.set_composition.call_args.args
    assert atmos is m['atmos'] and species == ['H2O', 'H2']
    np.testing.assert_array_equal(p_given, pco)
    np.testing.assert_array_equal(ymix_given, ymix)
    args = mock_run.call_args.args
    assert args[3] is climate.config_solve
    assert args[5] is False  # the snapshot file is left as it is
    np.testing.assert_allclose(update.tmp, [VULCAN_TMP_FLOOR, 600.0, 1500.0], rtol=1e-12)
    assert update.tmp.min() >= VULCAN_TMP_FLOOR
    assert climate.calls == 1


def test_an_unconverged_climate_never_reaches_vulcan(tmp_path):
    """When AGNI does not converge, the update raises instead of returning that climate."""
    climate, m = _setup(tmp_path)
    with patch(
        'proteus.atmos_clim.agni.run_agni', return_value=(m['atmos'], {'agni_converged': False})
    ):
        with pytest.raises(RuntimeError, match='did not converge at climate update 1'):
            climate(np.array([1e5, 1e3]), np.full(2, 900.0), ['H2O'], np.ones((2, 1)))
    assert climate.calls == 1


def test_write_keeps_the_snapshot_and_reports_the_flux_change(tmp_path, caplog):
    """The coupled state goes to <time>_chematm.nc, not over the snapshot, and the log gives
    the top-of-atmosphere net flux of the evolution (25 W m-2) and of the coupled state (12)."""
    climate, m = _setup(tmp_path)
    with (
        patch('proteus.atmos_clim.agni.write_atmos_ncdf') as mock_write,
        caplog.at_level('INFO'),
    ):
        climate.write()

    mock_write.assert_called_once_with(m['atmos'], m['dirs'], 700.0, suffix='_chematm.nc')
    assert '2.5000e+01 W m-2 in the evolution' in caplog.text
    assert '1.2000e+01 W m-2 with the chemistry' in caplog.text
