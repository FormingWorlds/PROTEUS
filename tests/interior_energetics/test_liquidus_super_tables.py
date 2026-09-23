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
        interior_struct=SimpleNamespace(
            module=module,
            melting_dir=melting_dir,
            core_frac=0.55,
            core_frac_mode='radius',
            zalmoxis=None,
        ),
    )


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
        d.mkdir()
        (d / 'f.dat').write_text('x')
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
        S_min = 1500.0  # liquidus stays inside the melt range at every P

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


@pytest.mark.parametrize('achieved', [150.0, 0.0])
def test_zalmoxis_anchor_clamp_caps_the_ic_superheat(
    fake_tables, monkeypatch, caplog, achieved
):
    """When the PALEOS P-T anchor clamps below delta, the P-S tables (whose
    filled cells hide the validity ceiling) are solved for the superheat the
    anchor reached, with one warning naming both numbers and P_cmb.
    """
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')
    monkeypatch.setattr(
        zal,
        'solve_superliquidus_adiabat',
        lambda config, hf_row: {
            'clamped': True,
            'achieved_superheat': achieved,
            'P_cmb': P_CMB,
        },
    )

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.common'):
        S = compute_initial_entropy(
            _config(500.0, module='zalmoxis'), {'P_cmb': P_CMB}, 3300.0, fake_tables
        )

    assert S == pytest.approx(_S_expected(achieved), rel=1e-6)
    msgs = [r.getMessage() for r in caplog.records if 'PALEOS P-T anchor' in r.getMessage()]
    assert len(msgs) == 1
    assert f'reaches only {achieved:.0f} K of the requested 500 K' in msgs[0]
    assert 'P_cmb=100 GPa' in msgs[0]


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
    table = tmp_path / 'density_melt.dat'
    table.write_text('1 2 3\n')

    a = common._load_entropy_eos(str(tmp_path))
    b = common._load_entropy_eos(str(tmp_path))
    assert a is b
    assert len(made) == 1

    table.write_text('1 2 3 4 5 6\n')
    c = common._load_entropy_eos(str(tmp_path))
    assert c is not a
    assert len(made) == 2


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
    copy with the same names, sizes and modification times reuses it.
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
    src.mkdir()
    (src / 'density_melt.dat').write_text('1 2 3\n')
    copy = tmp_path / 'copy'
    shutil.copytree(src, copy, copy_function=shutil.copy2)

    setup_eos = _cached_entropy_eos(str(src))
    ic_eos = common._load_entropy_eos(str(src))
    copy_eos = common._load_entropy_eos(str(copy))

    assert len(made) == 1
    assert setup_eos is ic_eos is copy_eos
