"""
Unit tests for the table-based ``liquidus_super`` initial entropy.

With ``interior_struct.module`` other than ``zalmoxis``, the ``liquidus_super``
initial entropy is solved on the interior P-S tables against their own P-S
liquidus, and no Zalmoxis or PALEOS data are read. The tests use an analytic
stand-in for the entropy EOS so that the expected entropy is known in closed
form.

Testing standards and documentation:
- docs/How-to/testing.md: Running, writing, and marking tests; coverage and CI
- docs/Explanations/test_framework.md: Test tiers, physics invariants, and quality rules

Functions tested:
- solve_superliquidus_entropy_from_tables(): reachable target, clamp and warning,
  table liquidus as the reference, undefined liquidus, non-finite temperatures,
  ini_dsdr entropy ceiling
- compute_initial_entropy(): dispatch on interior_struct.module
- _load_entropy_eos(): missing directory, cache reuse and invalidation
- planet_liquidus_super_needs_tables(): rejection of a configuration with no
  table route
"""

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from proteus.interior_energetics import common
from proteus.interior_energetics.common import (
    compute_initial_entropy,
    solve_superliquidus_entropy_from_tables,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Analytic stand-in: T(P, S) = T0 + A*S + B*P_GPa, table liquidus T_liq(P) = L0 + L1*P_GPa.
# The superheat margin is then A*S + T0 - L0 + (B - L1)*P_GPa, which is
# smallest at the largest pressure whenever B < L1, so the solution is exact.
T0, A, B = 500.0, 0.8, 8.0
L0, L1 = 1800.0, 10.0
P_CMB = 1.0e11
S_MAX = 3000.0


def _T(P, S):
    """Return the stand-in adiabat temperature [K] at pressure P [Pa], entropy S."""
    return T0 + A * np.asarray(S, dtype=float) + B * np.asarray(P, dtype=float) / 1e9


def _T_liq(P):
    """Return the stand-in liquidus temperature [K] at pressure P [Pa]."""
    return L0 + L1 * np.asarray(P, dtype=float) / 1e9


def _S_expected(delta, P=P_CMB):
    """Return the exact entropy [J/kg/K] whose superheat at P equals delta."""
    return (delta + L0 + L1 * P / 1e9 - T0 - B * P / 1e9) / A


class _FakeEOS:
    """Analytic entropy EOS exposing the attributes the solver uses."""

    P_min = 1e4
    P_max = 1.0e12
    S_min = 0.0
    S_max = S_MAX

    def temperature(self, P, S):
        return _T(P, S)

    # Pressure range of the liquidus_P-S.dat file, as aragog's EntropyEOS holds it.
    _liquidus = {'P': np.array([1e4, 1e13])}

    def liquidus_entropy(self, P):
        # The entropy whose temperature is the table liquidus T_liq(P).
        return (_T_liq(P) - T0 - B * np.asarray(P, dtype=float) / 1e9) / A


def _config(delta=200.0, module='spider', melting_dir='Monteux-600'):
    """Minimal config namespace carrying only the fields the solver reads."""
    return SimpleNamespace(
        planet=SimpleNamespace(
            temperature_mode='liquidus_super',
            delta_T_super=delta,
            ini_dsdr=0.0,
            mass_tot=1.0,
        ),
        interior_energetics=SimpleNamespace(module='spider'),
        interior_struct=SimpleNamespace(
            module=module,
            melting_dir=melting_dir,
            core_frac=0.55,
            core_frac_mode='radius',
            zalmoxis=SimpleNamespace(mantle_eos='PALEOS-2phase:MgSiO3'),
        ),
    )


def _write_table_set(d, text='1 2 3\n'):
    """Write every file of a SPIDER-format table set into ``d`` with ``text``."""
    d.mkdir(parents=True, exist_ok=True)
    for name in common._SPIDER_EOS_PHASE_FILES + common._SPIDER_EOS_MELTING_CURVES:
        (d / name).write_text(text)


@pytest.fixture
def fake_tables(monkeypatch, tmp_path):
    """Patch the table loader with the analytic stand-in."""
    monkeypatch.setattr(common, '_load_entropy_eos', lambda d: _FakeEOS())
    return str(tmp_path)


def test_reachable_target_is_honoured(fake_tables):
    """A reachable superheat is met to numerical tolerance without clamping."""
    res = solve_superliquidus_entropy_from_tables(_config(200.0), {'P_cmb': P_CMB}, fake_tables)
    S_exp = _S_expected(200.0)

    assert res['S_target'] == pytest.approx(S_exp, rel=1e-6)
    assert res['achieved_superheat'] == pytest.approx(200.0, abs=1e-2)
    assert res['clamped'] is False
    assert res['binding_P'] == pytest.approx(P_CMB, rel=1e-9)
    # Discrimination: the answer is neither the table maximum nor the value
    # that would put the surface, rather than the CMB, at the target.
    assert res['S_target'] < S_MAX - 100.0
    S_surface_bound = (200.0 + L0 - T0) / A
    assert abs(res['S_target'] - S_surface_bound) > 100.0


def test_unreachable_target_clamps_and_warns(fake_tables, caplog):
    """A target above the EOS ceiling clamps to the ceiling and logs a warning."""
    delta = 1000.0
    achievable = A * S_MAX + T0 - L0 - (L1 - B) * P_CMB / 1e9
    assert achievable < delta  # the requested superheat is out of reach

    with caplog.at_level(logging.WARNING):
        res = solve_superliquidus_entropy_from_tables(
            _config(delta), {'P_cmb': P_CMB}, fake_tables
        )

    assert res['clamped'] is True
    assert res['S_target'] == pytest.approx(S_MAX, rel=1e-12)
    assert res['achieved_superheat'] == pytest.approx(achievable, abs=1e-6)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    clamp_msgs = [r.getMessage() for r in warnings if 'not reachable' in r.getMessage()]
    assert len(clamp_msgs) == 1
    # The message names both the requested and the achieved superheat.
    assert '1000 K' in clamp_msgs[0]
    assert f'{achievable:.0f} K' in clamp_msgs[0]


def test_reachable_target_does_not_warn(fake_tables, caplog):
    """The clamp warning must not fire when the target is reachable."""
    with caplog.at_level(logging.WARNING):
        solve_superliquidus_entropy_from_tables(_config(200.0), {'P_cmb': P_CMB}, fake_tables)

    assert not [r for r in caplog.records if 'not reachable' in r.getMessage()]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_sub_liquidus_at_table_maximum_raises(monkeypatch, fake_tables):
    """When the adiabat stays below the liquidus even at the table's maximum
    entropy, no fully-molten initial condition exists and the solve must
    raise, not clamp. Lowering S_max to 50 keeps T(P, S_max) below T_liq(P)
    across the whole pressure range: at P_cmb, T=1340 K vs T_liq=2800 K.
    """
    monkeypatch.setattr(_FakeEOS, 'S_max', 50.0)

    with pytest.raises(
        common.InitialConditionError,
        match='no fully-molten initial condition is reachable within',
    ) as exc:
        solve_superliquidus_entropy_from_tables(_config(200.0), {'P_cmb': P_CMB}, fake_tables)
    assert 'exceeds the table maximum (50.0 J/kg/K) at 200 of 200 pressures' in str(exc.value)


@pytest.mark.physics_invariant
def test_entropy_is_monotonic_in_requested_superheat(fake_tables):
    """A larger requested superheat needs a strictly larger initial entropy."""
    deltas = [0.0, 100.0, 300.0, 600.0]
    S = [
        solve_superliquidus_entropy_from_tables(_config(d), {'P_cmb': P_CMB}, fake_tables)[
            'S_target'
        ]
        for d in deltas
    ]

    assert all(b > a for a, b in zip(S, S[1:]))
    # Each step of 100 K raises the entropy by 100/A, the analytic slope.
    assert S[1] - S[0] == pytest.approx(100.0 / A, rel=1e-6)


def test_table_floor_when_target_below_lowest_entropy(fake_tables, monkeypatch):
    """A target already met at the lowest table entropy returns that entropy."""
    calls = []
    real = _FakeEOS.temperature

    def counting(self, P, S):
        calls.append(1)
        return real(self, P, S)

    monkeypatch.setattr(_FakeEOS, 'temperature', counting)
    res = solve_superliquidus_entropy_from_tables(
        _config(-2000.0), {'P_cmb': P_CMB}, fake_tables
    )

    assert res['S_target'] == pytest.approx(_FakeEOS.S_min, abs=1e-12)
    assert res['clamped'] is False
    # Bisection would need 60 probes; the floor short-circuit needs a handful.
    assert len(calls) < 10


def test_melting_dir_curve_is_not_the_reference(fake_tables, monkeypatch):
    """The superheat is measured against the tables' own P-S liquidus, which
    sets the solver's melt fraction; a melting_dir P-T curve 500 K hotter must
    not move the answer, and it is never read.
    """

    def _boom(cfg):
        raise AssertionError('melting_dir curve read on the table path')

    monkeypatch.setattr('proteus.utils.data.get_zalmoxis_melting_curves', _boom)

    res = solve_superliquidus_entropy_from_tables(_config(200.0), {'P_cmb': P_CMB}, fake_tables)

    assert res['S_target'] == pytest.approx(_S_expected(200.0), rel=1e-6)
    assert res['achieved_superheat'] == pytest.approx(200.0, abs=1e-2)


def test_table_liquidus_undefined_at_depth_raises(fake_tables, monkeypatch):
    """A liquidus file that stops short of the CMB leaves the deep mantle
    unchecked, so the solve raises and names the covered and missing ranges.
    """
    monkeypatch.setattr(_FakeEOS, '_liquidus', {'P': np.array([1e4, 5.0e10])})

    with pytest.raises(RuntimeError, match='table liquidus is undefined') as exc:
        solve_superliquidus_entropy_from_tables(_config(200.0), {'P_cmb': P_CMB}, fake_tables)

    msg = str(exc.value)
    assert 'covers 1e-05 to 50 GPa' in msg
    assert 'and 100 GPa' in msg


def test_liquidus_file_ending_one_ulp_below_p_max_is_covered(fake_tables, monkeypatch, caplog):
    """A liquidus file whose top pressure rounds one ULP below the table edge
    P_max still covers a CMB at P_max, with no warning.
    """
    P_max = 5.0e11
    monkeypatch.setattr(_FakeEOS, 'P_max', P_max)
    monkeypatch.setattr(_FakeEOS, '_liquidus', {'P': np.array([1e4, np.nextafter(P_max, 0.0)])})

    with caplog.at_level(logging.WARNING):
        res = solve_superliquidus_entropy_from_tables(
            _config(50.0), {'P_cmb': P_max}, fake_tables
        )

    # At 500 GPa the model reaches at most 100 K of superheat, so 50 K is met.
    assert res['P_cmb'] == pytest.approx(P_max, rel=1e-12)
    assert res['clamped'] is False
    assert res['S_target'] == pytest.approx(_S_expected(50.0, P_max), rel=1e-6)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_liquidus_entropy_above_table_range_raises(fake_tables, monkeypatch):
    """Where the liquidus entropy exceeds the table's S_max, no adiabat in the
    table is molten there (and T(P, S_liq) would be a clipped table-edge
    value), so the solve raises and names the excess.
    """
    real = _FakeEOS.liquidus_entropy

    def _s_liq_leaves_table(self, P):
        P = np.asarray(P, dtype=float)
        return np.where(P > 6.0e10, S_MAX + 100.0, real(self, P))

    monkeypatch.setattr(_FakeEOS, 'liquidus_entropy', _s_liq_leaves_table)

    with pytest.raises(RuntimeError, match='no fully-molten initial condition') as exc:
        solve_superliquidus_entropy_from_tables(_config(200.0), {'P_cmb': P_CMB}, fake_tables)
    assert 'exceeds the table maximum (3000.0 J/kg/K)' in str(exc.value)
    assert 'by up to 100.0 J/kg/K' in str(exc.value)


def test_liquidus_undefined_everywhere_raises(monkeypatch, tmp_path):
    """A table liquidus that is NaN everywhere raises a clear error."""

    class _NaNLiqEOS(_FakeEOS):
        def liquidus_entropy(self, P):
            return np.full_like(np.asarray(P, dtype=float), np.nan)

    monkeypatch.setattr(common, '_load_entropy_eos', lambda d: _NaNLiqEOS())

    with pytest.raises(common.InitialConditionError, match='undefined at 200 of 200') as exc:
        solve_superliquidus_entropy_from_tables(_config(200.0), {'P_cmb': P_CMB}, str(tmp_path))
    assert 'between 0.0001 and 100 GPa' in str(exc.value)


def test_surface_anchor_is_raised_to_table_minimum_pressure(fake_tables, monkeypatch):
    """A table whose minimum pressure exceeds 1 bar is never evaluated below it."""
    seen = []
    real = _FakeEOS.temperature

    def guarded(self, P, S):
        P = np.asarray(P, dtype=float)
        seen.append(float(P.min()))
        return np.where(P < self.P_min, np.nan, real(self, P, S))

    monkeypatch.setattr(_FakeEOS, 'P_min', 1.0e7)
    monkeypatch.setattr(_FakeEOS, 'temperature', guarded)
    res = solve_superliquidus_entropy_from_tables(_config(200.0), {'P_cmb': P_CMB}, fake_tables)

    assert min(seen) >= 1.0e7
    assert np.isfinite(res['S_target'])
    assert res['S_target'] == pytest.approx(_S_expected(200.0), rel=1e-6)


def test_missing_p_cmb_uses_mass_aware_estimate(fake_tables, monkeypatch):
    """Without a helpfile P_cmb the Noack and Lasbleis estimate sets the CMB pressure."""
    monkeypatch.setattr(
        'proteus.utils.structure_estimate.estimate_P_cmb_NL20', lambda m, f, mode: 7.5e10
    )
    res = solve_superliquidus_entropy_from_tables(_config(200.0), {}, fake_tables)

    assert res['P_cmb'] == pytest.approx(7.5e10, rel=1e-12)
    assert res['S_target'] == pytest.approx(_S_expected(200.0, 7.5e10), rel=1e-6)


def test_nan_p_cmb_uses_mass_aware_estimate(fake_tables, monkeypatch):
    """A NaN P_cmb also falls back to the Noack and Lasbleis estimate:
    ``resolve_P_cmb``'s finite check catches it where a sign check alone
    would not.
    """
    monkeypatch.setattr(
        'proteus.utils.structure_estimate.estimate_P_cmb_NL20', lambda m, f, mode: 7.5e10
    )
    res = solve_superliquidus_entropy_from_tables(
        _config(200.0), {'P_cmb': float('nan')}, fake_tables
    )

    assert res['P_cmb'] == pytest.approx(7.5e10, rel=1e-12)
    assert res['S_target'] == pytest.approx(_S_expected(200.0, 7.5e10), rel=1e-6)


def test_cache_keeps_other_directories_when_one_reloads(monkeypatch, tmp_path):
    """Loading a second table set does not evict the first; the cache stays bounded."""
    aragog_entropy = pytest.importorskip('aragog.eos.entropy')
    monkeypatch.setattr(aragog_entropy, 'EntropyEOS', lambda d: object())
    monkeypatch.setattr(common, '_EOS_CACHE', {})
    dirs = []
    for i in range(common._EOS_CACHE_MAX + 2):
        d = tmp_path / f'set{i}'
        _write_table_set(d)
        dirs.append(str(d))

    first = common._load_entropy_eos(dirs[0])
    second = common._load_entropy_eos(dirs[1])
    assert common._load_entropy_eos(dirs[0]) is first
    assert common._load_entropy_eos(dirs[1]) is second
    for d in dirs[2:]:
        common._load_entropy_eos(d)
    assert len(common._EOS_CACHE) == common._EOS_CACHE_MAX


def test_nan_temperatures_at_depth_raise(fake_tables, monkeypatch):
    """A table that has no finite temperature at some mantle pressure, for
    every entropy, cannot certify a molten adiabat there: the solve raises
    instead of skipping those pressures.
    """
    real = _FakeEOS.temperature

    def holey(self, P, S):
        P = np.asarray(P, dtype=float)
        T = np.asarray(real(self, P, S), dtype=float)
        # The liquidus itself stays defined; only adiabat lookups are NaN.
        if np.ndim(S) and np.allclose(S, self.liquidus_entropy(P)):
            return T
        return np.where((P > 2.0e10) & (P < 3.0e10), np.nan, T)

    monkeypatch.setattr(_FakeEOS, 'temperature', holey)

    with pytest.raises(
        common.InitialConditionError, match='non-finite temperature at P=2'
    ) as exc:
        solve_superliquidus_entropy_from_tables(_config(200.0), {'P_cmb': P_CMB}, fake_tables)
    assert 'highest usable entropy (3000.0 J/kg/K)' in str(exc.value)


def test_all_nan_table_raises(fake_tables, monkeypatch):
    """An all-NaN temperature table raises instead of returning S_min with an
    infinite superheat. Its table liquidus T(P, S_liq) is NaN too, so the
    liquidus check is what raises.
    """
    monkeypatch.setattr(
        _FakeEOS,
        'temperature',
        lambda self, P, S: np.full(np.broadcast(np.asarray(P), np.asarray(S)).shape, np.nan),
    )

    with pytest.raises(common.InitialConditionError, match='undefined at 200 of 200') as exc:
        solve_superliquidus_entropy_from_tables(_config(200.0), {'P_cmb': P_CMB}, fake_tables)
    assert 'between 0.0001 and 100 GPa' in str(exc.value)


def test_entropy_dependent_nan_at_depth_is_not_satisfied(fake_tables, monkeypatch):
    """Where the deep table is NaN below some entropy, those entropies do not
    count as molten: the solve moves up to the first entropy with a finite
    deep adiabat instead of dropping the deep constraint.
    """
    S_edge = _S_expected(400.0)
    real = _FakeEOS.temperature

    def holey(self, P, S):
        P = np.asarray(P, dtype=float)
        S_arr = np.broadcast_to(np.asarray(S, dtype=float), P.shape)
        T = np.asarray(real(self, P, S), dtype=float)
        if np.allclose(S_arr, self.liquidus_entropy(P)):
            return T
        return np.where((P > 5.0e10) & (S_arr < S_edge), np.nan, T)

    monkeypatch.setattr(_FakeEOS, 'temperature', holey)

    res = solve_superliquidus_entropy_from_tables(_config(200.0), {'P_cmb': P_CMB}, fake_tables)

    assert res['S_target'] == pytest.approx(S_edge, rel=1e-9)
    assert res['achieved_superheat'] == pytest.approx(400.0, abs=1e-3)


def test_ini_dsdr_lowers_the_entropy_ceiling(fake_tables):
    """With ini_dsdr < 0 the clamp lands at S_max minus the entropy the
    perturbation adds between the surface and hf_row R_core, where both
    solvers' meshes end, so every initial node stays inside the table.
    """
    cfg = _config(1000.0)
    cfg.planet.ini_dsdr = -4.698e-6
    R_int, R_core = 6.371e6, 3.6e6
    hf_row = {'P_cmb': P_CMB, 'R_int': R_int, 'R_core': R_core}

    res = solve_superliquidus_entropy_from_tables(cfg, hf_row, fake_tables)

    span = R_int - R_core
    assert res['clamped'] is True
    assert res['S_target'] == pytest.approx(S_MAX - 4.698e-6 * span, rel=1e-12)
    assert res['S_target'] + 4.698e-6 * span == pytest.approx(S_MAX, rel=1e-12)


def test_ini_dsdr_ceiling_ignores_a_mass_core_fraction(fake_tables):
    """In core_frac_mode 'mass', core_frac * R_int is not a radius. The
    ceiling uses R_core, so a molten state that exists within the table
    clamps (+3.1 K here) instead of raising: reading 0.325 * R_int as the
    CMB would give a 1.6 times longer span and a ceiling 2.2 K below the
    liquidus at 543 GPa.
    """
    cfg = _config(50.0)
    cfg.planet.ini_dsdr = -4.698e-6
    cfg.interior_struct.core_frac = 0.325
    cfg.interior_struct.core_frac_mode = 'mass'
    R_int, R_core, P_cmb = 6.371e6, 3.48e6, 543e9
    hf_row = {'P_cmb': P_cmb, 'R_int': R_int, 'R_core': R_core}

    res = solve_superliquidus_entropy_from_tables(cfg, hf_row, fake_tables)

    S_ceiling = S_MAX - 4.698e-6 * (R_int - R_core)
    expected = A * S_ceiling + T0 - L0 - (L1 - B) * P_cmb / 1e9
    assert res['clamped'] is True
    assert res['S_target'] == pytest.approx(S_ceiling, rel=1e-12)
    assert res['achieved_superheat'] == pytest.approx(expected, abs=1e-6)
    assert 0.0 < res['achieved_superheat'] < 50.0
    # The mass-fraction reading lands below the liquidus at this pressure.
    S_wrong = S_MAX - 4.698e-6 * (R_int - 0.325 * R_int)
    assert A * S_wrong + T0 - L0 - (L1 - B) * P_cmb / 1e9 < 0.0


def test_ini_dsdr_ceiling_below_the_liquidus_raises(fake_tables):
    """The liquidus entropy stays inside the table (2987.5 J/kg/K at 545 GPa),
    but the ini_dsdr ceiling S_max - 4.698e-6 * 2.891e6 = 2986.4 J/kg/K puts
    the adiabat 0.9 K below the liquidus there, so no molten IC exists.
    """
    cfg = _config(50.0)
    cfg.planet.ini_dsdr = -4.698e-6
    hf_row = {'P_cmb': 545e9, 'R_int': 6.371e6, 'R_core': 3.48e6}

    with pytest.raises(common.InitialConditionError, match='1 K below the liquidus') as exc:
        solve_superliquidus_entropy_from_tables(cfg, hf_row, fake_tables)
    assert 'highest usable entropy (2986.4 J/kg/K)' in str(exc.value)
    assert 'P=545 GPa' in str(exc.value)


def test_liquidus_entropy_below_table_minimum_raises(fake_tables, monkeypatch):
    """Below S_min, T(P, S_liq) is a clipped table-edge value, not the
    liquidus. With S_min = 1700 J/kg/K the stand-in liquidus entropy
    (1625 + 2.5 P_GPa) is below it for P < 30 GPa, so the solve raises.
    """
    monkeypatch.setattr(_FakeEOS, 'S_min', 1700.0)

    with pytest.raises(
        common.InitialConditionError, match='table liquidus is undefined'
    ) as exc:
        solve_superliquidus_entropy_from_tables(_config(200.0), {'P_cmb': P_CMB}, fake_tables)
    msg = str(exc.value)
    assert 'table entropy from 1700 J/kg/K' in msg
    assert 'between 0.0001 and 28.7 GPa' in msg


@pytest.mark.parametrize('node_source', ['melt_table', 'liquidus_file'])
def test_liquidus_bump_between_grid_points_is_checked(fake_tables, monkeypatch, node_source):
    """A 50 K liquidus bump that rises and falls between two points of the
    200-point pressure grid, as a step in the melt temperature table does,
    is still checked because the margin is also evaluated at the melt-table
    or liquidus-file pressure nodes. It binds at the deep end of its top.
    """
    grid = np.geomspace(1e5, P_CMB, 200)
    i = int(np.searchsorted(grid, 90e9))
    lo, hi = grid[i - 1], grid[i]
    nodes = lo + (hi - lo) * np.array([0.2, 0.4, 0.6, 0.8])
    assert ((grid > nodes[0]) & (grid < nodes[-1])).sum() == 0

    class _BumpEOS(_FakeEOS):
        if node_source == 'melt_table':
            _tables = {'temperature_melt': {'P': nodes}}
        else:
            _liquidus = {'P': np.concatenate([[1e4], nodes, [1e13]])}

        def liquidus_entropy(self, P):
            bump = np.interp(np.asarray(P, dtype=float), nodes, [0.0, 50.0, 50.0, 0.0])
            return super().liquidus_entropy(P) + bump / A

    monkeypatch.setattr(common, '_load_entropy_eos', lambda d: _BumpEOS())

    res = solve_superliquidus_entropy_from_tables(_config(200.0), {'P_cmb': P_CMB}, fake_tables)

    P_bump_GPa = nodes[2] / 1e9
    expected = (200.0 + L0 - T0 + (L1 - B) * P_bump_GPa + 50.0) / A
    assert res['clamped'] is False
    assert res['S_target'] == pytest.approx(expected, abs=0.05)
    assert res['S_target'] > _S_expected(200.0) + 5.0


def test_liquidus_crossing_a_table_entropy_node_is_checked(fake_tables, monkeypatch):
    """With bilinear tables, the liquidus temperature T(P, S_liq(P)) can
    peak where S_liq(P) crosses a melt-table entropy node, which is neither
    a table nor a liquidus-file pressure node. A 50 K bump in T around that
    entropy node, 0.75 J/kg/K wide (0.3 GPa of S_liq), binds at the crossing.
    """
    grid = np.geomspace(1e5, P_CMB, 200)
    i = int(np.searchsorted(grid, 90e9))
    P_x = 0.5 * (grid[i - 1] + grid[i])
    S_x = float(_FakeEOS().liquidus_entropy(P_x))
    P_file = np.array([1e4, 1e13])

    class _CrossEOS(_FakeEOS):
        _liquidus = {'P': P_file, 'S': _FakeEOS().liquidus_entropy(P_file)}
        _tables = {'temperature_melt': {'P': P_file, 'S': np.array([0.0, S_x, S_MAX])}}

        def temperature(self, P, S):
            hat = np.clip(1.0 - np.abs(np.asarray(S, dtype=float) - S_x) / 0.75, 0.0, None)
            return super().temperature(P, S) + 50.0 * hat

    monkeypatch.setattr(common, '_load_entropy_eos', lambda d: _CrossEOS())

    res = solve_superliquidus_entropy_from_tables(_config(200.0), {'P_cmb': P_CMB}, fake_tables)

    expected = (200.0 + L0 - T0 + (L1 - B) * P_x / 1e9 + 50.0) / A
    assert res['S_target'] == pytest.approx(expected, abs=0.05)
    assert res['binding_P'] == pytest.approx(P_x, rel=1e-6)


@pytest.mark.parametrize('P_cmb', [5e4, 1e5])
def test_p_cmb_at_the_surface_raises(fake_tables, P_cmb):
    """A core-mantle boundary pressure at or below the 1 bar surface leaves no
    mantle to check, so the solve raises instead of reversing the grid or
    evaluating a single pressure.
    """
    with pytest.raises(common.InitialConditionError, match='not above') as exc:
        solve_superliquidus_entropy_from_tables(_config(200.0), {'P_cmb': P_cmb}, fake_tables)
    assert 'surface pressure 0.0001 GPa' in str(exc.value)


def test_margin_kink_pressures_find_every_crossing():
    """Crossings of a non-monotone, partly flat S_liq(P) with several table
    entropy nodes, on a liquidus file with more segments than nodes so a
    segment index cannot stand in for a node index. A flat segment lying on
    a node adds no crossing (its ends are file nodes already).
    """
    P_file = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    S_file = np.array([10.0, 30.0, 20.0, 20.0, 40.0, 45.0])
    S_nodes = np.array([15.0, 20.0, 25.0, 35.0])
    eos = SimpleNamespace(
        _liquidus={'P': P_file, 'S': S_file},
        _tables={'temperature_melt': {'P': np.array([3.5]), 'S': S_nodes}},
    )

    with np.errstate(all='raise'):
        kinks = common._margin_kink_pressures(eos)

    crossings = np.sort(kinks[~np.isin(kinks, np.concatenate([P_file, [3.5]]))])
    # Roots of S_liq(P) = S_node inside each segment, found by hand.
    expected = [1.25, 1.5, 1.75, 2.5, 4.25, 4.75]
    np.testing.assert_allclose(crossings, expected, rtol=0, atol=1e-12)
    assert kinks.size == P_file.size + 1 + len(expected)


class _NarrowMeltEOS(_FakeEOS):
    """Solid table reaches 0-3000 J/kg/K, melt table only 1700-2600."""

    _tables = {
        'temperature_melt': {'P': np.array([1e4, 1e13]), 'S': np.array([1700.0, 2600.0])}
    }


def test_liquidus_below_the_melt_table_raises(fake_tables, monkeypatch):
    """T(P, S_liq) reads the melt table, so a liquidus entropy below the melt
    table's lowest entropy (1625 + 2.5 P_GPa < 1700 for P < 30 GPa) is
    undefined even though the solid table covers it.
    """
    monkeypatch.setattr(common, '_load_entropy_eos', lambda d: _NarrowMeltEOS())

    with pytest.raises(
        common.InitialConditionError, match='table liquidus is undefined'
    ) as exc:
        solve_superliquidus_entropy_from_tables(_config(200.0), {'P_cmb': P_CMB}, fake_tables)
    assert 'table entropy from 1700 J/kg/K' in str(exc.value)


def test_melt_table_maximum_is_the_entropy_ceiling(fake_tables, monkeypatch):
    """The molten adiabat reads the melt table, so the clamp ceiling is the
    melt table's highest entropy (2600), not the 3000 J/kg/K overall maximum.
    """

    class _Eos(_NarrowMeltEOS):
        pass

    monkeypatch.setattr(common, '_load_entropy_eos', lambda d: _Eos())
    monkeypatch.setattr(
        _Eos,
        '_tables',
        {'temperature_melt': {'P': np.array([1e4, 1e13]), 'S': np.array([1600.0, 2600.0])}},
    )

    res = solve_superliquidus_entropy_from_tables(
        _config(1500.0), {'P_cmb': P_CMB}, fake_tables
    )

    assert res['clamped'] is True
    assert res['S_target'] == pytest.approx(2600.0, rel=1e-12)
    assert res['achieved_superheat'] == pytest.approx(A * 2600.0 - (L0 - T0) - (L1 - B) * 100.0)


def test_melt_table_floor_tolerates_one_ulp_at_the_surface_liquidus(fake_tables, monkeypatch):
    """A melt table whose lowest entropy is the surface liquidus entropy
    rounded 1 ULP up still covers the liquidus; 1 J/kg/K up does not.
    """
    S_surf = float(_FakeEOS().liquidus_entropy(1e5))

    class _Eos(_NarrowMeltEOS):
        pass

    monkeypatch.setattr(common, '_load_entropy_eos', lambda d: _Eos())
    for S_floor, ok in ((np.nextafter(S_surf, np.inf), True), (S_surf + 1.0, False)):
        monkeypatch.setattr(
            _Eos,
            '_tables',
            {
                'temperature_melt': {
                    'P': np.array([1e4, 1e13]),
                    'S': np.array([S_floor, 2600.0]),
                }
            },
        )
        if ok:
            res = solve_superliquidus_entropy_from_tables(
                _config(200.0), {'P_cmb': P_CMB}, fake_tables
            )
            assert res['S_target'] == pytest.approx(_S_expected(200.0), rel=1e-6)
        else:
            with pytest.raises(common.InitialConditionError, match='undefined'):
                solve_superliquidus_entropy_from_tables(
                    _config(200.0), {'P_cmb': P_CMB}, fake_tables
                )


def test_ini_dsdr_without_radii_warns(fake_tables, caplog):
    """Without mantle radii the ceiling cannot be lowered, and a warning says so."""
    cfg = _config(1000.0)
    cfg.planet.ini_dsdr = -4.698e-6

    with caplog.at_level(logging.WARNING):
        res = solve_superliquidus_entropy_from_tables(cfg, {'P_cmb': P_CMB}, fake_tables)

    assert res['S_target'] == pytest.approx(S_MAX, rel=1e-12)
    assert any('no mantle radii' in r.getMessage() for r in caplog.records)


def test_surface_binding_pressure_is_printed_in_gpa(fake_tables, monkeypatch, caplog):
    """A binding at the 1 bar surface prints as 0.0001 GPa, not 0 GPa."""
    # An adiabat steeper than the liquidus (12 K/GPa against 10 K/GPa), with
    # the table liquidus kept at L0 + L1*P_GPa, binds at the surface.
    monkeypatch.setattr(
        _FakeEOS,
        'temperature',
        lambda self, P, S: T0 + A * np.asarray(S, dtype=float) + 12.0 * np.asarray(P) / 1e9,
    )
    monkeypatch.setattr(
        _FakeEOS,
        'liquidus_entropy',
        lambda self, P: (_T_liq(P) - T0 - 12.0 * np.asarray(P, dtype=float) / 1e9) / A,
    )

    with caplog.at_level(logging.INFO, logger='fwl.proteus.interior_energetics.common'):
        res = solve_superliquidus_entropy_from_tables(
            _config(200.0), {'P_cmb': P_CMB}, fake_tables
        )

    assert res['binding_P'] == pytest.approx(1e5)
    assert any('at P=0.0001 GPa' in r.getMessage() for r in caplog.records)


def test_p_cmb_above_table_maximum_raises(fake_tables, monkeypatch):
    """A CMB pressure beyond the table maximum leaves the deepest mantle
    unchecked, so the solve raises and names both pressures instead of
    clipping to the table edge.
    """
    monkeypatch.setattr(_FakeEOS, 'P_max', 2.0e11)

    with pytest.raises(
        common.InitialConditionError, match='above the EOS table maximum'
    ) as exc:
        solve_superliquidus_entropy_from_tables(_config(200.0), {'P_cmb': 5e12}, fake_tables)

    msg = str(exc.value)
    assert '5e+03 GPa' in msg
    assert 'table covers 1e-05 to 200 GPa' in msg


def test_ic_entropy_reads_zalmoxis_only_on_the_zalmoxis_route(fake_tables, monkeypatch):
    """The initial entropy comes from the P-S tables for every structure
    module. The spider and dummy routes never touch Zalmoxis; the Zalmoxis
    route calls the P-T anchor once, and an anchor that reaches delta leaves
    the entropy equal to the other routes.
    """
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')
    calls = []

    def _anchor(config, hf_row):
        calls.append(config.interior_struct.module)
        return {'clamped': False, 'achieved_superheat': 200.0, 'P_cmb': P_CMB}

    monkeypatch.setattr(zal, 'solve_superliquidus_adiabat', _anchor)
    monkeypatch.setitem(sys.modules, 'zalmoxis.melting_curves', None)

    S = {
        module: compute_initial_entropy(
            _config(200.0, module=module), {'P_cmb': P_CMB}, 3300.0, fake_tables
        )
        for module in ('spider', 'dummy', 'zalmoxis')
    }

    assert calls == ['zalmoxis']
    for module, value in S.items():
        assert value == pytest.approx(_S_expected(200.0), rel=1e-6), module
    assert S['zalmoxis'] == pytest.approx(S['dummy'], rel=1e-12)


def _anchor_result(S_target, clamped=True, window_limited=False, achieved=150.0):
    """Return a P-T anchor result as solve_superliquidus_adiabat builds it."""
    return {
        'S_target': S_target,
        'clamped': clamped,
        'window_limited': window_limited,
        'achieved_superheat': achieved,
        'P_cmb': P_CMB,
    }


@pytest.mark.parametrize(
    ('ini_dsdr', 'S_exp'),
    [(0.0, 2300.0), (-1.0e-5, 2300.0 - 1.0e-5 * 3.0e6)],
)
def test_zalmoxis_anchor_clamp_caps_the_ic_entropy(
    fake_tables, monkeypatch, caplog, ini_dsdr, S_exp
):
    """When the PALEOS P-T anchor clamps at the table, the P-S IC entropy is
    capped at the anchor entropy; with ini_dsdr < 0 the deepest entropy stays
    at the cap. Repeated IC calls log one warning in total, naming the anchor
    superheat, delta, P_cmb, the capped and anchor entropies and the P-S
    superheat.
    """
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')
    monkeypatch.setattr(common, '_ANCHOR_CAP_WARNED', set())
    monkeypatch.setattr(
        zal, 'solve_superliquidus_adiabat', lambda config, hf_row: _anchor_result(2300.0)
    )
    cfg = _config(500.0, module='zalmoxis')
    cfg.planet.ini_dsdr = ini_dsdr
    hf_row = {'P_cmb': P_CMB, 'R_int': 6.0e6, 'R_core': 3.0e6}
    assert _S_expected(500.0) > 2300.0  # the cap binds

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.common'):
        S = [compute_initial_entropy(cfg, hf_row, 3300.0, fake_tables) for _ in range(3)]

    assert S[0] == pytest.approx(S_exp, rel=1e-12)
    assert S[0] == S[1] == S[2]
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(msgs) == 1
    assert 'reaches only 150 K of the requested 500 K' in msgs[0]
    assert 'P_cmb=100 GPa' in msgs[0]
    assert f'capped at {S_exp:.1f} J/kg/K (anchor entropy 2300.0 J/kg/K)' in msgs[0]
    superheat = A * S_exp + T0 - L0 - (L1 - B) * P_CMB / 1e9
    assert f'{superheat:.0f} K above the P-S table liquidus' in msgs[0]


@pytest.mark.parametrize(
    'anchor',
    [
        _anchor_result(2300.0, clamped=False, achieved=500.0),
        _anchor_result(2300.0, window_limited=True),
        _anchor_result(2600.0),
    ],
    ids=['reached', 'window-limited', 'cap-above-target'],
)
def test_zalmoxis_anchor_leaves_the_ic_uncapped(fake_tables, monkeypatch, caplog, anchor):
    """An anchor that reaches delta, a clamp set by the search window rather
    than the table, and a cap above the P-S target all leave the IC at delta,
    with no cap warning.
    """
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')
    monkeypatch.setattr(common, '_ANCHOR_CAP_WARNED', set())
    monkeypatch.setattr(zal, 'solve_superliquidus_adiabat', lambda config, hf_row: anchor)

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.common'):
        S = compute_initial_entropy(
            _config(500.0, module='zalmoxis'), {'P_cmb': P_CMB}, 3300.0, fake_tables
        )

    assert S == pytest.approx(_S_expected(500.0), rel=1e-6)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_entropy_ceiling_above_the_table_maximum_is_ignored(fake_tables, caplog):
    """An S_ceiling above the melt-table maximum leaves the table maximum as the
    clamp, reported as a table clamp at WARNING; a lower one binds instead.
    """
    with caplog.at_level(logging.INFO, logger='fwl.proteus.interior_energetics.common'):
        high = solve_superliquidus_entropy_from_tables(
            _config(1000.0), {'P_cmb': P_CMB}, fake_tables, S_ceiling=S_MAX + 500.0
        )
        low = solve_superliquidus_entropy_from_tables(
            _config(1000.0), {'P_cmb': P_CMB}, fake_tables, S_ceiling=S_MAX - 200.0
        )

    assert high['S_target'] == pytest.approx(S_MAX, rel=1e-12)
    assert high['capped_by_ceiling'] is False
    assert low['S_target'] == pytest.approx(S_MAX - 200.0, rel=1e-12)
    assert low['capped_by_ceiling'] is True
    clamp = [r for r in caplog.records if 'is not reachable below' in r.getMessage()]
    assert [r.levelno for r in clamp] == [logging.WARNING, logging.INFO]
    assert 'below the EOS table' in clamp[0].getMessage()
    assert 'below the P-T anchor entropy' in clamp[1].getMessage()


def test_zalmoxis_route_with_a_non_paleos_mantle_skips_the_anchor(fake_tables, monkeypatch):
    """The PALEOS anchor bounds only PALEOS-generated tables: a non-PALEOS
    mantle on the Zalmoxis route solves the IC on its tables alone.
    """
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')
    calls = []
    monkeypatch.setattr(
        zal,
        'solve_superliquidus_adiabat',
        lambda config, hf_row: calls.append(1) or _anchor_result(2300.0),
    )
    cfg = _config(500.0, module='zalmoxis')
    cfg.interior_struct.zalmoxis.mantle_eos = 'WolfBower2018:MgSiO3'

    S = compute_initial_entropy(cfg, {'P_cmb': P_CMB}, 3300.0, fake_tables)

    assert calls == []
    assert S == pytest.approx(_S_expected(500.0), rel=1e-6)


def test_zalmoxis_anchor_raise_is_the_ic_raise(fake_tables, monkeypatch):
    """No molten P-T anchor at this P_cmb means no initial condition on the
    Zalmoxis route: the anchor's InitialConditionError reaches the caller.
    """
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')

    def _raise(config, hf_row):
        raise common.InitialConditionError('liquidus_super: no valid molten adiabat found')

    monkeypatch.setattr(zal, 'solve_superliquidus_adiabat', _raise)

    with pytest.raises(common.InitialConditionError, match='no valid molten adiabat') as exc:
        compute_initial_entropy(
            _config(500.0, module='zalmoxis'), {'P_cmb': P_CMB}, 3300.0, fake_tables
        )
    assert 'P-S' not in str(exc.value)


def test_zalmoxis_structure_without_table_dir_raises_initial_condition_error(tmp_path):
    """With the Zalmoxis structure there is no fallback to a P-T adiabat: a
    missing or non-directory table path raises and names the field."""
    missing = str(tmp_path / 'absent')
    for table_dir in (None, missing):
        with pytest.raises(common.InitialConditionError) as exc:
            compute_initial_entropy(
                _config(200.0, module='zalmoxis'), {'P_cmb': P_CMB}, 3300.0, table_dir
            )
        assert "dirs['spider_eos_dir']" in str(exc.value)
        assert repr(table_dir) in str(exc.value)


def test_non_zalmoxis_path_without_table_dir_raises_naming_both_fields():
    """Without a table directory the error names temperature_mode and the module."""
    with pytest.raises(FileNotFoundError) as exc:
        compute_initial_entropy(_config(200.0, module='spider'), {'P_cmb': P_CMB}, 3300.0, None)

    msg = str(exc.value)
    assert 'temperature_mode' in msg
    assert 'interior_struct.module' in msg


def test_load_entropy_eos_missing_directory_raises(tmp_path):
    """A non-existent table directory raises FileNotFoundError naming the path."""
    missing = tmp_path / 'absent'

    with pytest.raises(FileNotFoundError, match='absent') as exc:
        common._load_entropy_eos(str(missing))
    assert str(missing) in str(exc.value)


def test_load_entropy_eos_caches_and_invalidates_on_file_change(monkeypatch, tmp_path):
    """The table set is loaded once per unchanged directory and reloaded on change."""
    made = []

    class _Counting:
        def __init__(self, d):
            made.append(d)

    aragog_entropy = pytest.importorskip('aragog.eos.entropy')
    monkeypatch.setattr(aragog_entropy, 'EntropyEOS', _Counting)
    monkeypatch.setattr(common, '_EOS_CACHE', {})
    _write_table_set(tmp_path)

    a = common._load_entropy_eos(str(tmp_path))
    b = common._load_entropy_eos(str(tmp_path))
    assert a is b
    assert len(made) == 1

    (tmp_path / 'density_melt.dat').write_text('1 2 3 4 5 6\n')
    c = common._load_entropy_eos(str(tmp_path))
    assert c is not a
    assert len(made) == 2
    assert len(common._EOS_CACHE) == 1


def test_rewrites_of_one_directory_keep_one_entry_and_spare_others(monkeypatch, tmp_path):
    """Each rewrite of a table set replaces that directory's entry, so 4
    rewrites of one directory neither fill the cache nor evict another set.
    """
    made = []
    aragog_entropy = pytest.importorskip('aragog.eos.entropy')
    monkeypatch.setattr(aragog_entropy, 'EntropyEOS', lambda d: made.append(d) or object())
    monkeypatch.setattr(common, '_EOS_CACHE', {})
    a, b = tmp_path / 'a', tmp_path / 'b'
    _write_table_set(a)
    _write_table_set(b)
    eos_b = common._load_entropy_eos(str(b))

    for i in range(4):
        (a / 'temperature_melt.dat').write_text('1 2 3' + ' 4' * (i + 1) + '\n')
        common._load_entropy_eos(str(a))

    assert len(common._EOS_CACHE) == 2
    assert common._load_entropy_eos(str(b)) is eos_b
    assert made.count(str(b)) == 1


def test_extra_files_do_not_change_the_cache_key(monkeypatch, tmp_path):
    """Files other than the table set, such as the cache marker or a
    short-lived temporary file, do not force a reload when they appear or
    vanish. An absent optional table file is keyed as absent; a table file
    that cannot be stat'ed loads without caching.
    """
    made = []
    aragog_entropy = pytest.importorskip('aragog.eos.entropy')
    monkeypatch.setattr(aragog_entropy, 'EntropyEOS', lambda d: made.append(d) or object())
    monkeypatch.setattr(common, '_EOS_CACHE', {})
    _write_table_set(tmp_path)

    first = common._load_entropy_eos(str(tmp_path))
    (tmp_path / '.cache_info.txt').write_text('marker\n')
    tmp_file = tmp_path / '.tmp-123'
    tmp_file.write_text('partial')
    assert common._load_entropy_eos(str(tmp_path)) is first
    tmp_file.unlink()
    assert common._load_entropy_eos(str(tmp_path)) is first
    assert len(made) == 1

    # An absent table file is part of the key: one reload, then cached.
    (tmp_path / 'thermal_exp_melt.dat').unlink()
    second = common._load_entropy_eos(str(tmp_path))
    assert common._load_entropy_eos(str(tmp_path)) is second
    assert len(made) == 2

    # A table file that exists but cannot be stat'ed loads without caching.
    real_stat = os.stat

    def _stat(path, *args, **kwargs):
        if str(path).endswith('liquidus_P-S.dat'):
            raise PermissionError(path)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(common.os, 'stat', _stat)
    common._load_entropy_eos(str(tmp_path))
    common._load_entropy_eos(str(tmp_path))
    assert len(made) == 4
    assert len(common._EOS_CACHE) == 1


def test_jax_eos_cache_keys_on_path_and_modification(monkeypatch, tmp_path):
    """Two table sets with the same file sizes but different content load
    two JAX EOS instances, and an in-place rewrite reloads it.
    """
    jax_eos = pytest.importorskip('aragog.jax.eos')
    from proteus.interior_energetics import aragog as aragog_mod

    made = []
    monkeypatch.setattr(jax_eos, 'EntropyEOS_JAX', lambda d: made.append(d) or object())
    monkeypatch.setattr(aragog_mod, '_entropy_eos_jax_cache', {})
    a, b = tmp_path / 'a', tmp_path / 'b'
    _write_table_set(a, '1 2 3\n')
    _write_table_set(b, '4 5 6\n')

    eos_a = aragog_mod._cached_entropy_eos_jax(str(a))
    eos_b = aragog_mod._cached_entropy_eos_jax(str(b))
    assert eos_a is not eos_b
    assert aragog_mod._cached_entropy_eos_jax(str(a)) is eos_a

    (a / 'temperature_melt.dat').write_text('7 8 9\n')
    os.utime(a / 'temperature_melt.dat', ns=(1, 1))
    assert aragog_mod._cached_entropy_eos_jax(str(a)) is not eos_a
    assert made == [str(a), str(b), str(a)]


def _planet_and_struct(temperature_mode, struct_module):
    """Return validated Planet and Struct objects for the route check."""
    import pathlib

    import attrs

    from proteus.config import read_config_object

    root = pathlib.Path(__file__).resolve().parents[2]
    cfg = read_config_object(root / 'input' / 'dummy.toml')
    planet = attrs.evolve(cfg.planet, temperature_mode=temperature_mode)
    struct = attrs.evolve(
        cfg.interior_struct,
        module=struct_module,
        core_density=10738.33,
        core_heatcap=880.0,
        melting_dir='Monteux600',
        eos_dir='WolfBower2018_MgSiO3',
    )
    return planet, struct


def _instance(energetics, struct):
    """Return the Config-like instance the planet validator receives."""
    return SimpleNamespace(
        interior_energetics=SimpleNamespace(module=energetics), interior_struct=struct
    )


@pytest.mark.parametrize('energetics', ['spider', 'aragog'])
def test_no_route_configuration_is_rejected_naming_both_fields(energetics):
    """liquidus_super with no structure module fails validation for spider and aragog."""
    from proteus.config._config import planet_liquidus_super_needs_tables

    planet, struct = _planet_and_struct('liquidus_super', None)

    with pytest.raises(ValueError) as exc:
        planet_liquidus_super_needs_tables(_instance(energetics, struct), None, planet)

    msg = str(exc.value)
    assert 'temperature_mode' in msg
    assert 'interior_struct.module' in msg
    assert energetics in msg


@pytest.mark.parametrize('energetics', ['spider', 'aragog'])
def test_full_config_with_no_structure_module_is_rejected(energetics):
    """The check is attached to Config, so building a no-route Config fails."""
    import pathlib

    import attrs

    from proteus.config import read_config_object

    root = pathlib.Path(__file__).resolve().parents[2]
    cfg = read_config_object(root / 'input' / 'dummy.toml')
    planet = attrs.evolve(cfg.planet, temperature_mode='liquidus_super')
    struct = attrs.evolve(cfg.interior_struct, module=None)
    energ = attrs.evolve(cfg.interior_energetics, module=energetics)

    with pytest.raises(ValueError) as exc:
        attrs.evolve(cfg, planet=planet, interior_struct=struct, interior_energetics=energ)
    assert 'temperature_mode' in str(exc.value)
    assert 'interior_struct.module' in str(exc.value)
    # Discrimination: the same Config builds once a structure module is set.
    ok = attrs.evolve(cfg, planet=planet, interior_energetics=energ)
    assert ok.interior_struct.module == cfg.interior_struct.module


@pytest.mark.parametrize('struct_module', ['spider', 'dummy', 'zalmoxis'])
def test_routes_with_a_structure_module_are_accepted(struct_module):
    """Every structure module that provides tables passes the route check."""
    from proteus.config._config import planet_liquidus_super_needs_tables

    planet, struct = _planet_and_struct('liquidus_super', struct_module)

    assert planet_liquidus_super_needs_tables(_instance('spider', struct), None, planet) is None
    assert struct.module == struct_module


def test_route_check_ignores_other_temperature_modes_and_solvers():
    """The route check only applies to liquidus_super under spider or aragog."""
    from proteus.config._config import planet_liquidus_super_needs_tables

    planet_iso, struct = _planet_and_struct('isentropic', None)
    planet_ls, _ = _planet_and_struct('liquidus_super', None)

    assert (
        planet_liquidus_super_needs_tables(_instance('spider', struct), None, planet_iso)
        is None
    )
    assert (
        planet_liquidus_super_needs_tables(_instance('dummy', struct), None, planet_ls) is None
    )
    # Discrimination: the same structure module is rejected once the mode and
    # solver both apply.
    with pytest.raises(ValueError):
        planet_liquidus_super_needs_tables(_instance('spider', struct), None, planet_ls)


def test_aragog_setup_and_ic_share_one_table_load(monkeypatch, tmp_path):
    """The Aragog solver setup and the liquidus_super IC read the same table
    set through one cache, so a fresh aragog run builds EntropyEOS once; a
    copy with the same names, sizes and modification times is another table
    set and loads its own instance.
    """
    import shutil

    from proteus.interior_energetics.aragog import _cached_entropy_eos

    made = []

    class _Counting:
        def __init__(self, d):
            made.append(d)

    aragog_entropy = pytest.importorskip('aragog.eos.entropy')
    monkeypatch.setattr(aragog_entropy, 'EntropyEOS', _Counting)
    monkeypatch.setattr(common, '_EOS_CACHE', {})
    src = tmp_path / 'spider_eos'
    _write_table_set(src)
    copy = tmp_path / 'copy'
    shutil.copytree(src, copy, copy_function=shutil.copy2)

    setup_eos = _cached_entropy_eos(str(src))
    ic_eos = common._load_entropy_eos(str(src))
    copy_eos = common._load_entropy_eos(str(copy))

    assert setup_eos is ic_eos
    assert copy_eos is not ic_eos
    assert made == [str(src), str(copy)]


def test_clearing_the_anchor_cache_rearms_the_cap_warning():
    """The cap warning is deduplicated with the same lifetime as the anchor
    memo, so clearing the memo also clears the warned keys.
    """
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')
    common._ANCHOR_CAP_WARNED.add((100000, 500.0, 'PALEOS-2phase:MgSiO3'))
    zal._SUPERLIQ_FAILED[(100000, 500.0, 'PALEOS-2phase:MgSiO3')] = 'failed'

    zal._clear_superliquidus_cache()

    assert common._ANCHOR_CAP_WARNED == set()
    assert zal._SUPERLIQ_FAILED == {}


def test_structure_anchor_raise_falls_back_and_the_ic_at_the_final_p_cmb_decides(
    fake_tables, monkeypatch, caplog
):
    """The first structure solve runs the anchor at the Noack & Lasbleis (2020)
    P_cmb estimate (about 138 GPa here); a raise there falls back to tcmb_init
    with a warning. The initial entropy then runs the anchor at the converged
    P_cmb: below the 120 GPa limit of this fake it gives the P-S entropy, above
    it the run stops with the anchor's error.
    """
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')
    from proteus.utils.structure_estimate import resolve_P_cmb

    P_limit = 1.2e11

    def _anchor(config, hf_row):
        P = resolve_P_cmb(hf_row, config)[0]
        if P > P_limit:
            raise common.InitialConditionError(
                f'liquidus_super: no valid molten adiabat found (P_cmb={P / 1e9:.0f} GPa)'
            )
        return _anchor_result(2600.0, clamped=False, achieved=500.0)

    monkeypatch.setattr(zal, 'solve_superliquidus_adiabat', _anchor)
    monkeypatch.setattr(zal, '_SUPERLIQ_LAST_ANCHOR', None)
    cfg = _config(500.0, module='zalmoxis')
    cfg.planet.tcmb_init = 6000.0
    P_estimate = resolve_P_cmb({}, cfg)[0]
    assert P_estimate > P_limit  # the first structure solve hits the raise

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_struct.zalmoxis'):
        T_cmb = zal._resolve_zalmoxis_cmb_temperature(cfg, {}, 'liquidus_super')

    assert T_cmb == pytest.approx(6000.0)
    msgs = [r.getMessage() for r in caplog.records if 'no P-T anchor' in r.getMessage()]
    assert len(msgs) == 1
    assert f'the estimated P_cmb={P_estimate / 1e9:.0f} GPa' in msgs[0]
    assert 'tcmb_init' in msgs[0]

    S = compute_initial_entropy(cfg, {'P_cmb': P_CMB}, 3300.0, fake_tables)
    assert S == pytest.approx(_S_expected(500.0), rel=1e-6)
    with pytest.raises(common.InitialConditionError, match='P_cmb=130 GPa'):
        compute_initial_entropy(cfg, {'P_cmb': 1.3e11}, 3300.0, fake_tables)


@pytest.mark.parametrize('energetics', ['spider', 'aragog'])
def test_structure_anchor_raise_reuses_the_last_solved_anchor(monkeypatch, caplog, energetics):
    """After a successful anchor solve, a later raise in a structure solve
    falls back to that anchor's CMB temperature, not to tcmb_init, under
    either energetics module that re-solves the anchor in the IC.
    """
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')

    def _anchor(config, hf_row):
        raise common.InitialConditionError('liquidus_super: no valid molten adiabat found')

    monkeypatch.setattr(zal, 'solve_superliquidus_adiabat', _anchor)
    monkeypatch.setattr(zal, '_SUPERLIQ_LAST_ANCHOR', 8765.0)
    cfg = _config(500.0, module='zalmoxis')
    cfg.planet.tcmb_init = 6000.0
    cfg.interior_energetics.module = energetics

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_struct.zalmoxis'):
        T_cmb = zal._resolve_zalmoxis_cmb_temperature(cfg, {'P_cmb': P_CMB}, 'liquidus_super')

    assert T_cmb == pytest.approx(8765.0)
    msgs = [r.getMessage() for r in caplog.records if 'no P-T anchor' in r.getMessage()]
    assert len(msgs) == 1
    assert 'at P_cmb=100 GPa' in msgs[0]
    assert 'estimated' not in msgs[0]


def test_zalmoxis_route_without_a_zalmoxis_section_raises(fake_tables):
    """The Zalmoxis route reads interior_struct.zalmoxis.mantle_eos; without
    that section the IC stops with an error that names it.
    """
    cfg = _config(500.0, module='zalmoxis')
    cfg.interior_struct.zalmoxis = None

    with pytest.raises(
        common.InitialConditionError, match='interior_struct.zalmoxis section'
    ) as exc:
        compute_initial_entropy(cfg, {'P_cmb': P_CMB}, 3300.0, fake_tables)
    assert 'mantle_eos' in str(exc.value)


def test_anchor_failures_other_than_the_ic_error_are_chained(fake_tables, monkeypatch):
    """A table-load or integration failure inside the anchor stops the IC as
    InitialConditionError with the cause chained, so SPIDER does not retry it.
    """
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')

    def _anchor(config, hf_row):
        raise ValueError('f(a) and f(b) must have different signs')

    monkeypatch.setattr(zal, 'solve_superliquidus_adiabat', _anchor)

    with pytest.raises(common.InitialConditionError, match='ValueError: f\\(a\\)') as exc:
        compute_initial_entropy(
            _config(500.0, module='zalmoxis'), {'P_cmb': P_CMB}, 3300.0, fake_tables
        )
    assert isinstance(exc.value.__cause__, ValueError)


@pytest.mark.parametrize(
    ('S_anchor', 'ini_dsdr', 'expect'),
    [
        (1800.0, 0.0, 'below the P-S table liquidus'),
        (1900.0, -1.0e-5, 'comes from the ini_dsdr allowance'),
    ],
    ids=['anchor-entropy-not-molten', 'ini_dsdr-allowance'],
)
def test_cap_raise_names_its_cause(fake_tables, monkeypatch, S_anchor, ini_dsdr, expect):
    """A capped IC that raises names the P-S margin at the anchor entropy
    itself. When that margin is negative the message gives both superheats;
    when it is positive, as with S_anchor = 1900 and a 30 J/kg/K ini_dsdr
    allowance, the message names the allowance as the cause.
    """
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')
    monkeypatch.setattr(
        zal,
        'solve_superliquidus_adiabat',
        lambda config, hf_row: _anchor_result(S_anchor, achieved=20.0),
    )
    cfg = _config(500.0, module='zalmoxis')
    cfg.planet.ini_dsdr = ini_dsdr
    hf_row = {'P_cmb': P_CMB, 'R_int': 6.0e6, 'R_core': 3.0e6}

    def superheat(S):
        return A * S + T0 - L0 - (L1 - B) * P_CMB / 1e9

    dS = -ini_dsdr * 3.0e6
    assert superheat(S_anchor - dS) < 0  # the uniform-entropy check fails

    with pytest.raises(common.InitialConditionError) as exc:
        compute_initial_entropy(cfg, hf_row, 3300.0, fake_tables)

    msg = str(exc.value)
    assert 'below the P-T anchor entropy' in msg
    assert expect in msg
    m = superheat(S_anchor)
    if m < 0:
        assert f'P-S adiabat is {-m:.0f} K below the P-S table liquidus' in msg
        assert 'is 20 K above the P-T liquidus' in msg
    else:
        assert f'P-S adiabat is {m:.0f} K above the P-S liquidus' in msg
        assert 'P-T liquidus' not in msg


def test_clamp_warnings_give_the_deepest_node_superheat(fake_tables, monkeypatch, caplog):
    """With ini_dsdr < 0 the deepest node holds S + |ini_dsdr| (R_int - R_core),
    so both clamp warnings give its superheat next to the uniform-entropy one.
    """
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')
    monkeypatch.setattr(common, '_ANCHOR_CAP_WARNED', set())
    monkeypatch.setattr(
        zal, 'solve_superliquidus_adiabat', lambda config, hf_row: _anchor_result(2300.0)
    )
    hf_row = {'P_cmb': P_CMB, 'R_int': 6.0e6, 'R_core': 3.0e6}
    dS = 1.0e-5 * 3.0e6
    S_cap = 2300.0 - dS

    def superheat(S):
        return A * S + T0 - L0 - (L1 - B) * P_CMB / 1e9

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.common'):
        cfg = _config(500.0, module='zalmoxis')
        cfg.planet.ini_dsdr = -1.0e-5
        compute_initial_entropy(cfg, hf_row, 3300.0, fake_tables)
        cfg_table = _config(1000.0)
        cfg_table.planet.ini_dsdr = -1.0e-5
        res = solve_superliquidus_entropy_from_tables(cfg_table, hf_row, fake_tables)

    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(msgs) == 2
    assert f'{superheat(S_cap):.0f} K above the P-S table liquidus' in msgs[0]
    assert f'({superheat(2300.0):.0f} K at the deepest node with ini_dsdr)' in msgs[0]
    assert res['cmb_node_superheat'] == pytest.approx(superheat(S_MAX), rel=1e-9)
    assert f'({superheat(S_MAX):.0f} K at the deepest node with ini_dsdr)' in msgs[1]
    # Discrimination: the uniform-entropy superheat is lower by A * dS.
    assert superheat(S_MAX) - res['achieved_superheat'] == pytest.approx(A * dS)


def test_cap_warning_key_separates_delta_and_p_cmb(fake_tables, monkeypatch, caplog):
    """The cap warning is logged once per (P_cmb, delta): a new delta or a
    new P_cmb logs again, a repeat does not.
    """
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')
    monkeypatch.setattr(common, '_ANCHOR_CAP_WARNED', set())

    def _anchor(config, hf_row):
        return {**_anchor_result(2300.0), 'P_cmb': hf_row['P_cmb']}

    monkeypatch.setattr(zal, 'solve_superliquidus_adiabat', _anchor)
    calls = [(500.0, P_CMB), (500.0, P_CMB), (600.0, P_CMB), (500.0, 0.99 * P_CMB)]
    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.common'):
        for delta, P in calls:
            compute_initial_entropy(
                _config(delta, module='zalmoxis'), {'P_cmb': P}, 3300.0, fake_tables
            )

    msgs = [r.getMessage() for r in caplog.records if 'PALEOS P-T anchor' in r.getMessage()]
    assert len(msgs) == 3
    assert 'requested 600 K' in msgs[1]
    assert 'P_cmb=99 GPa' in msgs[2]


def test_entropy_ceiling_equal_to_the_table_maximum_is_a_table_clamp(fake_tables):
    """An S_ceiling equal to the melt-table maximum is not lower than it, so
    the clamp is reported as a table clamp.
    """
    res = solve_superliquidus_entropy_from_tables(
        _config(1000.0), {'P_cmb': P_CMB}, fake_tables, S_ceiling=S_MAX
    )

    assert res['S_target'] == pytest.approx(S_MAX, rel=1e-12)
    assert res['clamped'] is True
    assert res['capped_by_ceiling'] is False


def test_structure_anchor_raise_propagates_for_a_non_paleos_mantle(monkeypatch):
    """The initial entropy re-solves the anchor only for a PALEOS mantle, so
    with another mantle the structure solve keeps the anchor's raise.
    """
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')

    def _anchor(config, hf_row):
        raise common.InitialConditionError('liquidus_super: no valid molten adiabat found')

    monkeypatch.setattr(zal, 'solve_superliquidus_adiabat', _anchor)
    cfg = _config(500.0, module='zalmoxis')
    cfg.planet.tcmb_init = 6000.0
    cfg.interior_struct.zalmoxis.mantle_eos = 'WolfBower2018:MgSiO3'

    with pytest.raises(common.InitialConditionError, match='no valid molten adiabat'):
        zal._resolve_zalmoxis_cmb_temperature(cfg, {'P_cmb': P_CMB}, 'liquidus_super')
    # Discrimination: a PALEOS mantle with the same input falls back.
    cfg.interior_struct.zalmoxis.mantle_eos = 'PALEOS-2phase:MgSiO3'
    monkeypatch.setattr(zal, '_SUPERLIQ_LAST_ANCHOR', None)
    T_cmb = zal._resolve_zalmoxis_cmb_temperature(cfg, {'P_cmb': P_CMB}, 'liquidus_super')
    assert T_cmb == pytest.approx(6000.0)


def test_structure_anchor_integration_error_falls_back(monkeypatch, caplog):
    """A ValueError from the anchor integration is raised by the anchor as a
    chained InitialConditionError, so a structure solve falls back on it.
    """
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')

    def _solve(config, hf_row):
        raise ValueError('f(a) and f(b) must have different signs')

    monkeypatch.setattr(zal, '_solve_superliquidus_adiabat', _solve)
    monkeypatch.setattr(zal, '_SUPERLIQ_LAST_ANCHOR', None)
    cfg = _config(500.0, module='zalmoxis')
    cfg.planet.tcmb_init = 6000.0

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_struct.zalmoxis'):
        T_cmb = zal._resolve_zalmoxis_cmb_temperature(cfg, {'P_cmb': P_CMB}, 'liquidus_super')

    assert T_cmb == pytest.approx(6000.0)
    assert any('ValueError: f(a)' in r.getMessage() for r in caplog.records)
    # A first raise at a new key chains the ValueError itself.
    with pytest.raises(common.InitialConditionError, match='ValueError: f') as exc:
        zal.solve_superliquidus_adiabat(cfg, {'P_cmb': 1.1 * P_CMB})
    assert isinstance(exc.value.__cause__, ValueError)


def test_failed_anchor_solve_is_memoised(monkeypatch):
    """A failing anchor solve at one key runs once; later calls raise the
    same message without solving, and clearing the memo re-arms the solve.
    """
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')
    zal._clear_superliquidus_cache()
    calls = []

    def _solve(config, hf_row):
        calls.append(hf_row['P_cmb'])
        raise common.InitialConditionError('liquidus_super: no valid molten adiabat found')

    monkeypatch.setattr(zal, '_solve_superliquidus_adiabat', _solve)
    cfg = _config(500.0, module='zalmoxis')

    raised = []
    for _ in range(3):
        with pytest.raises(
            common.InitialConditionError, match='no valid molten adiabat'
        ) as exc:
            zal.solve_superliquidus_adiabat(cfg, {'P_cmb': P_CMB})
        raised.append(exc.value)
    assert calls == [P_CMB]
    # Later raises are chained to one stored copy that holds no traceback.
    stored = raised[1].__cause__
    assert stored is raised[2].__cause__
    assert stored is not raised[0]
    assert stored.__traceback__ is None
    assert str(stored) == str(raised[0])
    # Another P_cmb is another key.
    with pytest.raises(common.InitialConditionError):
        zal.solve_superliquidus_adiabat(cfg, {'P_cmb': 1.1 * P_CMB})
    assert len(calls) == 2

    zal._clear_superliquidus_cache()
    with pytest.raises(common.InitialConditionError):
        zal.solve_superliquidus_adiabat(cfg, {'P_cmb': P_CMB})
    assert len(calls) == 3
    zal._clear_superliquidus_cache()


def test_deepest_node_superheat_is_nan_without_mantle_radii(fake_tables):
    """With ini_dsdr < 0 and no mantle radii the deepest-node entropy is not
    known, so its superheat is NaN rather than the uniform value.
    """
    cfg = _config(1000.0)
    cfg.planet.ini_dsdr = -1.0e-5

    res = solve_superliquidus_entropy_from_tables(cfg, {'P_cmb': P_CMB}, fake_tables)

    assert np.isnan(res['cmb_node_superheat'])
    assert np.isfinite(res['achieved_superheat'])


def test_structure_anchor_raise_propagates_without_an_ic_re_solve(monkeypatch, caplog):
    """With dummy energetics nothing re-solves the anchor at the converged
    P_cmb, so a structure solve keeps the anchor's raise even for a PALEOS
    mantle.
    """
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')

    def _anchor(config, hf_row):
        raise common.InitialConditionError('liquidus_super: no valid molten adiabat found')

    monkeypatch.setattr(zal, 'solve_superliquidus_adiabat', _anchor)
    cfg = _config(500.0, module='zalmoxis')
    cfg.planet.tcmb_init = 6000.0
    cfg.interior_energetics.module = 'dummy'

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_struct.zalmoxis'):
        with pytest.raises(common.InitialConditionError, match='no valid molten adiabat'):
            zal._resolve_zalmoxis_cmb_temperature(cfg, {'P_cmb': P_CMB}, 'liquidus_super')
    assert not [r for r in caplog.records if 'no P-T anchor' in r.getMessage()]

    # Discrimination: the same input under spider energetics falls back.
    cfg.interior_energetics.module = 'spider'
    monkeypatch.setattr(zal, '_SUPERLIQ_LAST_ANCHOR', None)
    T_cmb = zal._resolve_zalmoxis_cmb_temperature(cfg, {'P_cmb': P_CMB}, 'liquidus_super')
    assert T_cmb == pytest.approx(6000.0)


def test_clamp_warning_says_unknown_without_mantle_radii(fake_tables, caplog):
    """Without mantle radii the deepest-node superheat is unknown, and the
    clamp warning says so instead of printing nan.
    """
    cfg = _config(1000.0)
    cfg.planet.ini_dsdr = -1.0e-5

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_energetics.common'):
        solve_superliquidus_entropy_from_tables(cfg, {'P_cmb': P_CMB}, fake_tables)

    clamp = [r.getMessage() for r in caplog.records if 'not reachable below' in r.getMessage()]
    assert len(clamp) == 1
    assert '(unknown at the deepest node with ini_dsdr)' in clamp[0]
    assert 'nan' not in clamp[0]


def test_cap_raise_without_a_table_temperature_at_the_anchor_entropy(fake_tables, monkeypatch):
    """When the P-S tables give no finite temperature at the anchor entropy,
    the capped raise says so instead of printing an infinite margin.
    """
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')
    monkeypatch.setattr(
        zal,
        'solve_superliquidus_adiabat',
        lambda config, hf_row: _anchor_result(1900.0, achieved=20.0),
    )
    monkeypatch.setattr(
        _FakeEOS,
        'temperature',
        lambda self, P, S: np.where(np.asarray(S, dtype=float) > 1890.0, np.nan, _T(P, S)),
    )
    cfg = _config(500.0, module='zalmoxis')
    cfg.planet.ini_dsdr = -1.0e-5

    with pytest.raises(common.InitialConditionError) as exc:
        compute_initial_entropy(
            cfg, {'P_cmb': P_CMB, 'R_int': 6.0e6, 'R_core': 3.0e6}, 3300.0, fake_tables
        )

    msg = str(exc.value)
    # Every pressure is non-finite at 1900, so the first (1 bar) is named.
    assert 'no finite temperature at the anchor entropy at P=0.0001 GPa' in msg
    assert 'inf' not in msg


@pytest.mark.parametrize(
    'error',
    [
        ValueError('f(a) and f(b) must have different signs'),
        KeyError('S_profile'),
        IndexError('index 200 is out of bounds'),
        ZeroDivisionError('float division by zero'),
        FloatingPointError('overflow encountered'),
        RuntimeError('integration did not converge'),
    ],
    ids=lambda e: type(e).__name__,
)
def test_numerical_anchor_failures_are_wrapped_and_memoised(monkeypatch, error):
    """Each numerical failure of the anchor integration is raised as a
    chained InitialConditionError that names its type, and is memoised.
    """
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')
    calls = []

    def _solve(config, hf_row):
        calls.append(1)
        raise error

    monkeypatch.setattr(zal, '_solve_superliquidus_adiabat', _solve)
    cfg = _config(500.0, module='zalmoxis')

    with pytest.raises(common.InitialConditionError, match=type(error).__name__) as exc:
        zal.solve_superliquidus_adiabat(cfg, {'P_cmb': P_CMB})
    assert exc.value.__cause__ is error
    with pytest.raises(common.InitialConditionError, match=type(error).__name__) as again:
        zal.solve_superliquidus_adiabat(cfg, {'P_cmb': P_CMB})
    assert len(calls) == 1
    # The memo holds a copy without traceback or cause, not the raised error.
    stored = again.value.__cause__
    assert stored is not exc.value
    assert stored.__traceback__ is None
    assert stored.__cause__ is None


@pytest.mark.parametrize(
    'error',
    [
        TypeError('unexpected keyword argument'),
        AttributeError('module has no attribute'),
        FileNotFoundError('paleos_mgsio3_tables_pt_proteus_liquid.dat'),
        PermissionError('lock.pid'),
    ],
    ids=lambda e: type(e).__name__,
)
def test_programming_and_io_errors_propagate_and_are_not_memoised(monkeypatch, error):
    """Programming and I/O errors from the anchor propagate unchanged,
    through the structure solve as well, and a later call solves again.
    """
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')
    calls = []

    def _solve(config, hf_row):
        calls.append(1)
        raise error

    monkeypatch.setattr(zal, '_solve_superliquidus_adiabat', _solve)
    cfg = _config(500.0, module='zalmoxis')
    cfg.planet.tcmb_init = 6000.0

    with pytest.raises(type(error)) as exc:
        zal.solve_superliquidus_adiabat(cfg, {'P_cmb': P_CMB})
    assert exc.value is error
    with pytest.raises(type(error)):
        zal._resolve_zalmoxis_cmb_temperature(cfg, {'P_cmb': P_CMB}, 'liquidus_super')
    assert len(calls) == 2
    assert not zal._SUPERLIQ_FAILED


def test_ic_passes_programming_errors_from_the_anchor_through(fake_tables, monkeypatch):
    """compute_initial_entropy wraps only numerical anchor failures; a
    TypeError reaches the caller unchanged.
    """
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')

    def _anchor(config, hf_row):
        raise TypeError('unexpected keyword argument')

    monkeypatch.setattr(zal, 'solve_superliquidus_adiabat', _anchor)

    with pytest.raises(TypeError, match='unexpected keyword') as exc:
        compute_initial_entropy(
            _config(500.0, module='zalmoxis'), {'P_cmb': P_CMB}, 3300.0, fake_tables
        )
    assert not isinstance(exc.value, common.InitialConditionError)
