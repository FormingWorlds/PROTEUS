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

import re
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

    _liquidus = {'P': np.array([1e4, 1.0e13])}

    def temperature(self, P, S):
        return T0 + A * np.asarray(S, dtype=float) + B * np.asarray(P, dtype=float) / 1e9

    def liquidus_entropy(self, P):
        # The entropy whose temperature is the shared liquidus _table_liquidus(P).
        return (_table_liquidus(P) - T0 - B * np.asarray(P, dtype=float) / 1e9) / A


def _shared_config(delta):
    from types import SimpleNamespace

    return SimpleNamespace(
        planet=SimpleNamespace(delta_T_super=delta, ini_dsdr=0.0, mass_tot=5.972e24),
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


def _install_zalmoxis_deps(monkeypatch, s_ceiling=S_MAX, invalid_bands=()):
    """Drive ``solve_superliquidus_adiabat`` through the shared linear model.

    Adiabats with ``S > s_ceiling`` plateau at ``s_ceiling`` at the deepest
    point, as a real EOS table does when the deep adiabat runs out of range;
    ``s_ceiling=None`` removes the ceiling. Adiabats with ``S`` inside any
    ``(S_lo, S_hi)`` of ``invalid_bands`` get a 5 K cooling-with-depth kink
    near the surface, the shape of the real near-liquidus monotonicity band,
    while their superheat is left unchanged.
    """
    import zalmoxis.eos_export as eos_export

    def _fake_adiabat(
        eos_file,
        T_surface,
        P_surface,
        P_cmb,
        n_points,
        solidus_func,
        liquidus_func,
        solid_eos_file,
        liquid_eos_file,
    ):
        S = (float(T_surface) - T0 - B * (P_surface / 1e9)) / A
        P = np.linspace(P_surface, P_cmb, n_points)
        S_profile = np.full(n_points, float(S))
        if s_ceiling is not None and S > s_ceiling:
            S_profile[-1] = s_ceiling
        T = T0 + A * S_profile + B * (P / 1e9)
        if any(lo <= S < hi for lo, hi in invalid_bands):
            T[1] = T[0] - 5.0
        return {
            'P': P,
            'T': T,
            'S_target': float(S),
            'S_profile': S_profile,
        }

    monkeypatch.setattr(zmod, 'load_zalmoxis_material_dictionaries', lambda: {})
    monkeypatch.setattr(
        zmod,
        'resolve_2phase_mgsio3_paths',
        lambda mantle_eos, mat_dicts: ('solid.eos', 'liquid.eos'),
    )
    monkeypatch.setattr(
        zmod,
        'load_zalmoxis_solidus_liquidus_functions',
        lambda mantle_eos, config: (lambda P: 0.0, _table_liquidus),
    )
    monkeypatch.setattr(eos_export, 'compute_entropy_adiabat', _fake_adiabat)


@pytest.fixture
def _zalmoxis_env(monkeypatch):
    _install_zalmoxis_deps(monkeypatch)


def _surface_T_for(delta, P_cmb_GPa=100.0):
    """Exact surface T whose linear-model adiabat is ``delta`` above the liquidus.

    Superheat is ``A*S - (L0 - T0) - (L1 - B)*P_GPa``, smallest at the CMB.
    """
    S = (delta + (L0 - T0) + (L1 - B) * P_cmb_GPa) / A
    return T0 + A * S + B * 1e-4


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
    with pytest.raises(RuntimeError, match=match) as exc:
        solve_superliquidus_adiabat(config, hf_row)
    # The hottest valid adiabat sits at the S_MAX ceiling (surface T = T0 + A*S_MAX
    # = 2900 K, plus up to 3 K of the solver's 1e-3 entropy-drift tolerance), so
    # the raise comes from the table ceiling, not the scan span (~5800 K).
    assert re.search(r'surface T=290[0-3] K', str(exc.value)), str(exc.value)


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
    """The zalmoxis path clamps to the fake adiabat's own entropy ceiling
    (``S_MAX``, the same value the table path is capped at) when the hottest
    reachable adiabat is molten but short of the requested delta. Both paths
    are bound by the same physical ceiling here, so the achieved superheats
    match the table path's own ceiling test (``test_table_path_clamps_at_its_own_ceiling``).
    """
    config = _shared_config(delta=4000.0)
    hf_row = {'P_cmb': 100e9}

    res = solve_superliquidus_adiabat(config, hf_row)

    assert res['clamped'] is True
    assert 0.0 <= res['achieved_superheat'] < 4000.0
    assert res['achieved_superheat'] == pytest.approx(900.0, abs=2.0)


def test_zalmoxis_path_reaches_delta_via_ceiling_refinement(_zalmoxis_env):
    """A delta just below the ceiling found in ``test_zalmoxis_path_clamps_at_its_own_ceiling``
    (~900 K) sits inside the 200 K coarse-scan spacing (4000 K span / 20 steps),
    so the coarse/extension grid alone never samples a point at or above it: only
    the validity-edge bisection reaches delta. This pins the crossing to
    sub-Kelvin accuracy, which a bracket drawn from the coarse grid alone
    could not achieve.
    """
    config = _shared_config(delta=899.9)
    hf_row = {'P_cmb': 100e9}

    res = solve_superliquidus_adiabat(config, hf_row)

    assert res['clamped'] is False
    assert res['achieved_superheat'] == pytest.approx(899.9, abs=0.5)


_LEADING_BAND = ((0.0, 1900.0),)  # invalid below S=1900, i.e. surface T < 2020 K


@pytest.mark.parametrize(
    ('delta', 'expected_T', 'expected_superheat'),
    [
        # delta=0 is met inside the invalid band (surface T 2000 to 2020 K has
        # superheat 0 to 20 K), so the answer is the band's upper edge.
        (0.0, 2020.0, 20.0),
        (100.0, _surface_T_for(100.0), 100.0),
    ],
)
def test_zalmoxis_path_tolerates_leading_invalid_band(
    monkeypatch, delta, expected_T, expected_superheat
):
    """One invalid band at the cold end of the scan does not raise, and an
    invalid midpoint during the bisection counts as not satisfied, so the
    solve returns the coolest adiabat that is both valid and at least delta
    above the liquidus, refined past the coarse scan point (2221 K here).
    """
    _install_zalmoxis_deps(monkeypatch, invalid_bands=_LEADING_BAND)

    res = solve_superliquidus_adiabat(_shared_config(delta=delta), {'P_cmb': 100e9})

    assert res['clamped'] is False
    assert res['surface_T'] == pytest.approx(expected_T, abs=0.5)
    assert res['achieved_superheat'] == pytest.approx(expected_superheat, abs=0.5)


@pytest.mark.parametrize('bands', [((2400.0, 2500.0),), _LEADING_BAND + ((2400.0, 2500.0),)])
def test_zalmoxis_path_raises_on_second_validity_edge(monkeypatch, bands):
    """A valid -> invalid -> valid pattern after the first valid scan point
    breaks the single validity edge the bisection needs, so the solve raises
    (with or without a tolerated leading band before it).
    """
    _install_zalmoxis_deps(monkeypatch, invalid_bands=bands)

    with pytest.raises(RuntimeError, match='valid, then invalid, then valid again'):
        solve_superliquidus_adiabat(_shared_config(delta=100.0), {'P_cmb': 100e9})


def test_zalmoxis_path_extends_scan_past_window_without_ceiling(monkeypatch):
    """With no table ceiling, a delta just past the 4000 K scan window is
    reached by the scan extension and bisected to the exact surface T.
    """
    _install_zalmoxis_deps(monkeypatch, s_ceiling=None)

    res = solve_superliquidus_adiabat(_shared_config(delta=4000.0), {'P_cmb': 100e9})

    assert res['clamped'] is False
    assert res['surface_T'] == pytest.approx(_surface_T_for(4000.0), abs=0.5)
    assert res['achieved_superheat'] == pytest.approx(4000.0, abs=0.5)


def test_zalmoxis_path_reports_search_window_limit(monkeypatch, caplog):
    """With no table ceiling and a delta beyond the scan plus its three
    doubling extensions, the clamp names the search window as the limit,
    not the EOS table.
    """
    _install_zalmoxis_deps(monkeypatch, s_ceiling=None)
    step = zmod._SUPERLIQ_SCAN_SPAN_K / (zmod._SUPERLIQ_SCAN_STEPS - 1)
    T_top = _table_liquidus(1e5) + zmod._SUPERLIQ_SCAN_SPAN_K + step * (1 + 2 + 4)

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_struct.zalmoxis'):
        res = solve_superliquidus_adiabat(_shared_config(delta=6000.0), {'P_cmb': 100e9})

    assert res['clamped'] is True
    assert res['surface_T'] == pytest.approx(T_top, abs=1e-6)
    assert res['achieved_superheat'] == pytest.approx(T_top - _surface_T_for(0.0), abs=0.01)
    msgs = [r.getMessage() for r in caplog.records]
    assert any('search window' in m and 'EOS table was not exhausted' in m for m in msgs), msgs


# A ceiling 8 K above the liquidus sits below the second scan point
# (surface T 2010.5 K, superheat +10.5 K), so only the edge bisection finds it.
_S_CEIL_8K = (8.0 + (L0 - T0) + (L1 - B) * 100.0) / A


@pytest.mark.parametrize(
    ('delta', 'clamped', 'expected_superheat'),
    [(0.0, False, 0.0), (5.0, False, 5.0), (10.0, True, 8.0)],
)
def test_zalmoxis_path_refines_ceiling_near_liquidus(
    monkeypatch, delta, clamped, expected_superheat
):
    """A table ceiling between the first two scan points still gives a
    molten solution: small deltas are reached and a larger one clamps to the
    exact +8 K ceiling instead of raising on the coarse -200 K point.
    """
    _install_zalmoxis_deps(monkeypatch, s_ceiling=_S_CEIL_8K)

    res = solve_superliquidus_adiabat(_shared_config(delta=delta), {'P_cmb': 100e9})

    assert res['clamped'] is clamped
    assert res['achieved_superheat'] == pytest.approx(expected_superheat, abs=0.5)


def test_aragog_ic_with_ini_dsdr_stays_inside_table(_table_env):
    """A clamped table-path solve with the default ini_dsdr keeps every Aragog
    initial node at or below S_max, and SPIDER's deepest node (at
    core_frac * R_int) at S_max exactly.
    """
    config = _shared_ic_config(delta=1000.0, ic_module='aragog')
    config.planet.ini_dsdr = -4.698e-6
    R_int, R_core = 6.0e6, 4.0e6
    hf_row = {'P_cmb': 100e9, 'R_int': R_int, 'R_core': R_core}

    solver = MagicMock()
    solver._P_stag_flat = np.array([5e10, 1e10])
    solver._r_basic_flat = np.array([R_core, 5.0e6, R_int])
    interior_o = SimpleNamespace(aragog_solver=solver, _spider_eos_dir='unused')
    AragogRunner._set_entropy_ic(config, interior_o, 'unused_outdir', hf_row)
    S_init = solver.set_initial_entropy.call_args[0][0]

    assert S_init.max() <= S_MAX
    assert S_init.max() > S_init.min()  # the perturbation was applied

    S_top = _compute_spider_initial_entropy(config, hf_row, spider_eos_dir='unused')
    R_cmb_spider = config.interior_struct.core_frac * R_int
    assert S_top + 4.698e-6 * (R_int - R_cmb_spider) == pytest.approx(S_MAX, abs=1e-9)
