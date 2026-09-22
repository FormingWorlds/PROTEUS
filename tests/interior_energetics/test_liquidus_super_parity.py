"""Cross-path parity for the ``liquidus_super`` initial-condition solvers.

``solve_superliquidus_entropy_from_tables`` (the table path, used when
``interior_struct.module != 'zalmoxis'``) and ``solve_superliquidus_adiabat``
(the zalmoxis-adiabat path) must make the same raise-vs-clamp decision for
the same config: both raise when even the hottest reachable adiabat cannot
reach the liquidus, and both clamp (rather than raise) when the hottest
reachable adiabat is molten but falls short of the requested superheat
margin ``delta``. Each path is driven here through a linear analytic
stand-in physics model so the comparison is exact and independent of any
real EOS or melting-curve table.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from proteus.interior_energetics import common
from proteus.interior_energetics.aragog import AragogRunner
from proteus.interior_energetics.common import solve_superliquidus_entropy_from_tables
from proteus.interior_energetics.spider import _compute_spider_initial_entropy
from proteus.interior_struct import zalmoxis as zmod
from proteus.interior_struct.zalmoxis import solve_superliquidus_adiabat

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Shared analytic stand-in, identical constants to test_liquidus_super_tables.py:
# T(P, S) = T0 + A*S + B*P_GPa, liquidus T_liq(P) = L0 + L1*P_GPa.
T0, A, B = 500.0, 0.8, 8.0
L0, L1 = 1800.0, 10.0
S_MAX = 3000.0


def _table_liquidus(P):
    return L0 + L1 * np.asarray(P, dtype=float) / 1e9


class _FakeEOS:
    """Linear stand-in entropy EOS for the table-path solver."""

    P_min = 1e4
    P_max = 1.0e13
    S_min = 0.0
    S_max = S_MAX

    def temperature(self, P, S):
        return T0 + A * np.asarray(S, dtype=float) + B * np.asarray(P, dtype=float) / 1e9


def _shared_config(delta):
    from types import SimpleNamespace

    return SimpleNamespace(
        planet=SimpleNamespace(delta_T_super=delta, mass_tot=5.972e24),
        interior_struct=SimpleNamespace(
            core_frac=0.5,
            core_frac_mode='mass',
            melting_dir='unused',
            zalmoxis=SimpleNamespace(mantle_eos='PALEOS:MgSiO3'),
        ),
    )


@pytest.fixture(autouse=True)
def _clear_cache():
    zmod._clear_superliquidus_cache()
    yield
    zmod._clear_superliquidus_cache()


@pytest.fixture
def _table_env(monkeypatch):
    monkeypatch.setattr(common, '_load_entropy_eos', lambda eos_dir: _FakeEOS())
    monkeypatch.setattr(
        'proteus.utils.data.get_zalmoxis_melting_curves',
        lambda config: (None, _table_liquidus),
    )


@pytest.fixture
def _zalmoxis_env(monkeypatch):
    import zalmoxis.eos_export as eos_export

    def _fake_adiabat(
        eos_file, T_surface, P_surface, P_cmb, n_points, solidus_func, liquidus_func,
        solid_eos_file, liquid_eos_file,
    ):
        S = (float(T_surface) - T0 - B * (P_surface / 1e9)) / A
        P = np.linspace(P_surface, P_cmb, n_points)
        T = T0 + A * S + B * (P / 1e9)
        return {
            'P': P,
            'T': T,
            'S_target': float(S),
            'S_profile': np.full(n_points, float(S)),
        }

    monkeypatch.setattr(zmod, 'load_zalmoxis_material_dictionaries', lambda: {})
    monkeypatch.setattr(
        zmod, 'resolve_2phase_mgsio3_paths', lambda mantle_eos, mat_dicts: ('solid.eos', 'liquid.eos')
    )
    monkeypatch.setattr(
        zmod,
        'load_zalmoxis_solidus_liquidus_functions',
        lambda mantle_eos, config: (lambda P: 0.0, _table_liquidus),
    )
    monkeypatch.setattr(eos_export, 'compute_entropy_adiabat', _fake_adiabat)


def test_sub_liquidus_raises_identically_on_both_paths(_table_env, _zalmoxis_env):
    """Neither solver can reach the liquidus anywhere: both must raise, with
    the same error phrasing, rather than silently clamping to a cold result.
    """
    config = _shared_config(delta=200.0)
    hf_row = {'P_cmb': 3000e9}
    match = 'no fully-molten initial condition is reachable within'

    with pytest.raises(RuntimeError, match=match):
        solve_superliquidus_entropy_from_tables(config, hf_row, eos_dir='unused')

    zmod._clear_superliquidus_cache()
    with pytest.raises(RuntimeError, match=match):
        solve_superliquidus_adiabat(config, hf_row)


def test_table_path_clamps_at_its_own_ceiling(_table_env):
    """The table path clamps to its own entropy ceiling (S_max) when the
    hottest tabulated adiabat is molten but short of the requested delta.
    """
    config = _shared_config(delta=1000.0)
    hf_row = {'P_cmb': 100e9}

    res = solve_superliquidus_entropy_from_tables(config, hf_row, eos_dir='unused')

    assert res['clamped'] is True
    assert 0.0 <= res['achieved_superheat'] < 1000.0
    assert res['achieved_superheat'] == pytest.approx(900.0, abs=1.0)


def _shared_ic_config(delta, ic_module):
    """Same table-path config as ``_shared_config``, extended with the
    fields ``AragogRunner._set_entropy_ic`` and ``_compute_spider_initial_entropy``
    read on their way to ``compute_initial_entropy``: ``interior_energetics.module``
    records which module is under test, ``interior_struct.module`` stays off
    ``'zalmoxis'`` so both modules resolve through the same table path.
    """
    config = _shared_config(delta)
    config.interior_energetics = SimpleNamespace(module=ic_module)
    config.planet.temperature_mode = 'liquidus_super'
    config.planet.ini_dsdr = 0.0
    config.interior_struct.module = 'non_zalmoxis_struct'
    return config


def _aragog_S_target(config, hf_row):
    """Drive Aragog's own IC entry point and read back its uniform S_target."""
    solver = MagicMock()
    solver._P_stag_flat = np.array([1e10, 5e10])
    solver._r_basic_flat = np.array([6e6, 5e6, 4e6])
    interior_o = SimpleNamespace(aragog_solver=solver, _spider_eos_dir='unused')

    AragogRunner._set_entropy_ic(config, interior_o, 'unused_outdir', hf_row)

    S_init = solver.set_initial_entropy.call_args[0][0]
    assert np.allclose(S_init, S_init[0]), 'ini_dsdr=0 must give a uniform profile'
    return float(S_init[0])


def test_aragog_and_spider_raise_identically_on_sub_liquidus_config(_table_env):
    """Same config, same sub-liquidus case, driven through each module's own
    IC entry point: both must raise, with the same error phrase.
    """
    hf_row = {'P_cmb': 3000e9}
    match = 'no fully-molten initial condition is reachable within'

    aragog_config = _shared_ic_config(delta=200.0, ic_module='aragog')
    with pytest.raises(RuntimeError, match=match):
        _aragog_S_target(aragog_config, hf_row)

    spider_config = _shared_ic_config(delta=200.0, ic_module='spider')
    with pytest.raises(RuntimeError, match=match):
        _compute_spider_initial_entropy(spider_config, hf_row, spider_eos_dir='unused')


def test_aragog_and_spider_clamp_to_same_S_target(_table_env):
    """Same config, same reduced-superheat case, driven through each
    module's own IC entry point: both must clamp to the same S_target.
    """
    hf_row = {'P_cmb': 100e9}

    aragog_config = _shared_ic_config(delta=1000.0, ic_module='aragog')
    aragog_S = _aragog_S_target(aragog_config, hf_row)

    spider_config = _shared_ic_config(delta=1000.0, ic_module='spider')
    spider_S = _compute_spider_initial_entropy(spider_config, hf_row, spider_eos_dir='unused')

    assert aragog_S == pytest.approx(spider_S, abs=1e-6)


def test_zalmoxis_path_clamps_at_its_own_ceiling(_zalmoxis_env):
    """The zalmoxis path clamps to its own scan-span ceiling
    (``_SUPERLIQ_SCAN_SPAN_K``) when the hottest scanned adiabat is molten
    but short of the requested delta. This ceiling is a different quantity
    from the table path's entropy cap (a temperature span, not an entropy
    bound), so the two paths' achieved superheats are not expected to match
    numerically; only the clamp-not-raise decision is asserted to match.
    """
    config = _shared_config(delta=4000.0)
    hf_row = {'P_cmb': 100e9}

    res = solve_superliquidus_adiabat(config, hf_row)

    assert res['clamped'] is True
    assert 0.0 <= res['achieved_superheat'] < 4000.0
    assert res['achieved_superheat'] == pytest.approx(3800.0, abs=5.0)
