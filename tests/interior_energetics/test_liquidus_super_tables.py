"""
Unit tests for the table-based ``liquidus_super`` initial entropy.

With ``interior_struct.module`` other than ``zalmoxis``, the ``liquidus_super``
initial entropy is solved on the interior P-S tables and the selected melting
curve, and no Zalmoxis or PALEOS data are read. The tests use an analytic
stand-in for the entropy EOS so that the expected entropy is known in closed
form.

Testing standards and documentation:
- docs/How-to/testing.md: Running, writing, and marking tests; coverage and CI
- docs/Explanations/test_framework.md: Test tiers, physics invariants, and quality rules

Functions tested:
- solve_superliquidus_entropy_from_tables(): reachable target, clamp and warning,
  table-liquidus fallback, partial melting-curve coverage
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

# Analytic stand-in: T(P, S) = T0 + A*S + B*P_GPa, liquidus T_liq(P) = L0 + L1*P_GPa.
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

    def liquidus_entropy(self, P):
        # Entropy of the table liquidus: a curve 50 K below the melting curve.
        return (_T_liq(P) - 50.0 - T0 - B * np.asarray(P, dtype=float) / 1e9) / A


def _config(delta=200.0, module='spider', melting_dir='Monteux-600'):
    """Minimal config namespace carrying only the fields the solver reads."""
    return SimpleNamespace(
        planet=SimpleNamespace(
            temperature_mode='liquidus_super',
            delta_T_super=delta,
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
    """Patch the table loader and melting-curve reader with analytic stand-ins."""
    monkeypatch.setattr(common, '_load_entropy_eos', lambda d: _FakeEOS())
    monkeypatch.setattr(
        'proteus.utils.data.get_zalmoxis_melting_curves',
        lambda cfg: (None, _T_liq),
    )
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


def test_table_floor_when_target_below_lowest_entropy(fake_tables):
    """A target already met at the lowest table entropy returns that entropy."""
    res = solve_superliquidus_entropy_from_tables(
        _config(-2000.0), {'P_cmb': P_CMB}, fake_tables
    )

    assert res['S_target'] == pytest.approx(_FakeEOS.S_min, abs=1e-12)
    assert res['clamped'] is False


def test_missing_melting_curve_falls_back_to_table_liquidus(monkeypatch, tmp_path, caplog):
    """Without a melting-curve file the table liquidus is used, with a warning."""
    monkeypatch.setattr(common, '_load_entropy_eos', lambda d: _FakeEOS())
    monkeypatch.setattr('proteus.utils.data.get_zalmoxis_melting_curves', lambda cfg: None)

    with caplog.at_level(logging.WARNING):
        res = solve_superliquidus_entropy_from_tables(
            _config(200.0), {'P_cmb': P_CMB}, str(tmp_path)
        )

    # The table liquidus lies 50 K below the melting curve, so 200 K above it
    # needs 50/A less entropy than 200 K above the melting curve.
    assert res['S_target'] == pytest.approx(_S_expected(200.0) - 50.0 / A, rel=1e-6)
    assert any('unavailable' in r.getMessage() for r in caplog.records)


def test_melting_curve_undefined_at_depth_uses_covered_range(monkeypatch, tmp_path, caplog):
    """A melting curve that is NaN above some pressure binds only where defined."""
    P_edge = 5.0e10

    def liq_partial(P):
        out = _T_liq(P)
        return np.where(np.asarray(P) > P_edge, np.nan, out)

    monkeypatch.setattr(common, '_load_entropy_eos', lambda d: _FakeEOS())
    monkeypatch.setattr(
        'proteus.utils.data.get_zalmoxis_melting_curves', lambda cfg: (None, liq_partial)
    )

    with caplog.at_level(logging.WARNING):
        res = solve_superliquidus_entropy_from_tables(
            _config(200.0), {'P_cmb': P_CMB}, str(tmp_path)
        )

    # The binding pressure is the highest covered grid pressure, not the CMB.
    assert res['binding_P'] <= P_edge
    assert res['binding_P'] > 0.9 * P_edge
    assert res['S_target'] < _S_expected(200.0)
    assert any('undefined above' in r.getMessage() for r in caplog.records)


def test_p_cmb_above_table_range_is_clipped(fake_tables, monkeypatch):
    """A CMB pressure beyond the table maximum is clipped to the maximum."""
    monkeypatch.setattr(_FakeEOS, 'P_max', 2.0e11)
    res = solve_superliquidus_entropy_from_tables(_config(200.0), {'P_cmb': 5e12}, fake_tables)

    assert res['P_cmb'] == pytest.approx(2.0e11, rel=1e-12)
    assert res['S_target'] == pytest.approx(_S_expected(200.0, 2.0e11), rel=1e-6)
    # Discrimination: differs from the entropy solved at the unclipped 5 TPa.
    assert res['S_target'] != pytest.approx(_S_expected(200.0, 5.0e12), rel=1e-3)


def test_no_zalmoxis_or_paleos_read_on_non_zalmoxis_path(fake_tables, monkeypatch):
    """The non-Zalmoxis path completes with every Zalmoxis entry point raising."""
    zal = pytest.importorskip('proteus.interior_struct.zalmoxis')

    def _boom(*args, **kwargs):
        raise AssertionError('Zalmoxis was called on the table-based path')

    monkeypatch.setattr(zal, 'solve_superliquidus_adiabat', _boom)
    monkeypatch.setitem(sys.modules, 'zalmoxis.melting_curves', None)

    for module in ('spider', 'dummy'):
        S = compute_initial_entropy(
            _config(200.0, module=module), {'P_cmb': P_CMB}, 3300.0, fake_tables
        )
        assert S == pytest.approx(_S_expected(200.0), rel=1e-6)

    # Control: the same patch is live, so the Zalmoxis structure module reaches it.
    with pytest.raises(AssertionError, match='table-based path'):
        compute_initial_entropy(
            _config(200.0, module='zalmoxis'), {'P_cmb': P_CMB}, 3300.0, fake_tables
        )


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

    with pytest.raises(FileNotFoundError, match='absent'):
        common._load_entropy_eos(str(missing))


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
