"""Unit tests for the pure-Python helpers in ``proteus.atmos_clim.wrapper``.

Covers ``update_wtg_surf`` (weak-temperature-gradient surface parameter),
``update_bolometry`` (transit + eclipse depth closed-form relations),
``ShallowMixedOceanLayer`` (thin-ocean thermal evolution via scipy.solve_ivp),
``write_atmosphere_snapshot`` (per-module snapshot dispatch), and
``carry_converged_levels`` (photospheric and XUV levels are not taken from a
structure the solver rejected). The ``run_atmosphere`` paths reached here are
those a mocked atmosphere module can drive; the full coupled dispatch is
exercised by integration tests in the nightly tier.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

import proteus.atmos_clim.wrapper as atmos_wrapper
from proteus.atmos_clim.common import Atmos_t, LevelsSource
from proteus.utils.constants import const_R

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# update_wtg_surf: weak-temperature-gradient surface scaling
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_update_wtg_surf_closed_form_at_unit_inputs():
    """``update_wtg_surf`` computes wtg_surf = sqrt(R_mix * T_surf) /
    (omega * R_int) where omega = 2*pi / axial_period. With chosen
    inputs the closed-form value can be pinned to high precision.

    Discrimination: a regression that dropped the sqrt would land at
    R_mix * T_surf / (omega * R_int) which is much larger; a regression
    that flipped the period to omega would land at axial_period /
    (2*pi * R_int) * sqrt(R_mix * T_surf), much smaller.
    """
    hf_row = {
        'axial_period': 86400.0,  # 1 day in seconds
        'atm_kg_per_mol': 0.029,  # Earth-like mean molecular weight
        'T_surf': 300.0,  # K
        'R_int': 6.371e6,  # Earth radius in m
    }
    atmos_wrapper.update_wtg_surf(hf_row)

    omega = 2.0 * math.pi / 86400.0
    R_mix = const_R / 0.029
    expected = math.sqrt(R_mix * 300.0) / (omega * 6.371e6)

    assert hf_row['wtg_surf'] == pytest.approx(expected, rel=1e-12)
    # Discrimination: positivity guard rules out a sign flip
    assert hf_row['wtg_surf'] > 0
    # Discrimination: scale guard. For Earth-like conditions wtg_surf
    # is order ~0.05 (dimensionless WTG parameter). A regression that
    # dropped the sqrt would yield ~3e7; one that kept R_int^2 would
    # be ~1e-14.
    assert 1e-3 < hf_row['wtg_surf'] < 10.0


@pytest.mark.physics_invariant
def test_update_wtg_surf_scales_inversely_with_planet_rotation_rate():
    """A slower-rotating planet (larger axial_period -> smaller omega)
    has a LARGER wtg_surf, because the WTG approximation is more valid
    when rotation is slow.

    Discrimination: a regression that put omega in the numerator would
    invert this scaling.
    """
    hf_fast = {
        'axial_period': 86400.0,
        'atm_kg_per_mol': 0.029,
        'T_surf': 300.0,
        'R_int': 6.371e6,
    }
    hf_slow = {
        'axial_period': 86400.0 * 10.0,
        'atm_kg_per_mol': 0.029,
        'T_surf': 300.0,
        'R_int': 6.371e6,
    }
    atmos_wrapper.update_wtg_surf(hf_fast)
    atmos_wrapper.update_wtg_surf(hf_slow)

    # Slower rotation -> larger wtg
    assert hf_slow['wtg_surf'] > hf_fast['wtg_surf']
    # Discrimination: 10x slower rotation means 10x larger wtg_surf
    # (linear in axial_period via omega in denominator)
    ratio = hf_slow['wtg_surf'] / hf_fast['wtg_surf']
    assert ratio == pytest.approx(10.0, rel=1e-12)


# ---------------------------------------------------------------------------
# update_bolometry: transit + eclipse depth
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_update_bolometry_transit_depth_is_ratio_of_radii_squared():
    """Transit depth = (R_obs / R_star)^2. With Earth-like geometry
    (R_obs ~ 6e6 m, R_star ~ 7e8 m), the depth is ~(6e6/7e8)^2 ~ 7e-5
    (about 73 ppm).

    Discrimination: a regression that used (R_obs/R_star) instead of
    squaring would yield ~8e-3 (1000x larger); a regression that
    flipped the ratio would yield ~1.3e4 (impossibly large).
    """
    hf_row = {
        'R_obs': 6.371e6,
        'R_star': 6.96e8,  # Solar radius in m
        'F_olr': 200.0,
        'F_sct': 100.0,
        'F_ins': 1361.0,
        'separation': 1.5e11,
    }
    atmos_wrapper.update_bolometry(hf_row)

    expected_transit = (6.371e6 / 6.96e8) ** 2
    assert hf_row['transit_depth'] == pytest.approx(expected_transit, rel=1e-12)
    # Discrimination: positivity + scale guards
    assert hf_row['transit_depth'] > 0
    # Earth-like transit depth is ~8e-5; a regression that dropped the
    # square would give ~9e-3, two orders of magnitude bigger.
    assert 1e-5 < hf_row['transit_depth'] < 1e-3


@pytest.mark.physics_invariant
def test_update_bolometry_eclipse_depth_is_flux_ratio_times_radius_ratio_squared():
    """Eclipse depth = ((F_olr + F_sct) / F_ins) * (R_obs / separation)^2.
    The F_olr + F_sct is the planet's thermal+scattered flux at TOA;
    F_ins is the incoming stellar flux at TOA; the (R_obs/separation)^2
    factor accounts for the inverse-square attenuation from the planet
    to the star.

    Discrimination: a regression that dropped the (R_obs/separation)^2
    would yield a much larger depth (~order unity).
    """
    hf_row = {
        'R_obs': 6.371e6,
        'R_star': 6.96e8,
        'F_olr': 200.0,
        'F_sct': 100.0,
        'F_ins': 1361.0,
        'separation': 1.5e11,  # 1 AU
    }
    atmos_wrapper.update_bolometry(hf_row)

    expected_eclipse = (300.0 / 1361.0) * (6.371e6 / 1.5e11) ** 2
    assert hf_row['eclipse_depth'] == pytest.approx(expected_eclipse, rel=1e-12)
    # Discrimination: positivity + scale guards
    assert hf_row['eclipse_depth'] > 0
    # Earth-like eclipse depth is ~4e-10 (1 AU separation, Earth radius);
    # a regression that dropped the (R_obs/separation)^2 geometric factor
    # would land near 0.22 (the flux ratio alone), 9 orders larger.
    assert 1e-12 < hf_row['eclipse_depth'] < 1e-7


@pytest.mark.physics_invariant
def test_update_bolometry_eclipse_depth_scales_with_flux_excess():
    """At fixed geometry, doubling (F_olr + F_sct) doubles the eclipse
    depth (linear in planet's emission flux). Discrimination: a
    regression that squared or exponentiated the flux ratio would not
    show this 2x scaling.
    """
    base_geometry = {
        'R_obs': 6.371e6,
        'R_star': 6.96e8,
        'F_ins': 1361.0,
        'separation': 1.5e11,
    }
    hf_base = {**base_geometry, 'F_olr': 200.0, 'F_sct': 100.0}
    hf_hot = {**base_geometry, 'F_olr': 400.0, 'F_sct': 200.0}

    atmos_wrapper.update_bolometry(hf_base)
    atmos_wrapper.update_bolometry(hf_hot)

    ratio = hf_hot['eclipse_depth'] / hf_base['eclipse_depth']
    assert ratio == pytest.approx(2.0, rel=1e-12)
    # Discrimination: both eclipse depths are positive (catches a sign
    # flip on F_sct or F_olr).
    assert hf_base['eclipse_depth'] > 0
    assert hf_hot['eclipse_depth'] > hf_base['eclipse_depth']


# ---------------------------------------------------------------------------
# ShallowMixedOceanLayer: thin-ocean thermal evolution
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_shallow_mixed_ocean_layer_cools_under_positive_net_flux():
    """A positive ``F_net`` (upward) draws heat out of the ocean layer.

    The function solves dT/dt = -F_net / mu where mu = c_p * rho * d =
    1000 * 3000 * 1000 = 3e9 J K-1 m-2. With F_net = 100 W m-2 over
    1 year (3.154e7 s) the analytical drop is 100 * 3.154e7 / 3e9 =
    1.0513 K. The implementation feeds this ODE to scipy.solve_ivp
    at default tolerances (rtol=1e-3, atol=1e-6); on a constant-RHS
    linear problem the integrator reaches its rtol bound, so rel=5e-3
    leaves only 5x headroom and still rejects a factor-of-2 mu bug.
    """
    hf_pre = {'Time': 0.0, 'T_surf': 300.0}
    hf_cur = {'Time': 1.0, 'F_net': 100.0}

    T_cur = atmos_wrapper.ShallowMixedOceanLayer(hf_cur, hf_pre)

    expected_drop = 100.0 * 3.154e7 / 3e9
    assert T_cur == pytest.approx(300.0 - expected_drop, rel=5e-3)
    # Sign guard: positive F_net cools the layer; T_cur < T_pre.
    assert T_cur < 300.0
    # Scale guard: the drop is roughly 1 K, not 0.001 K (forgotten
    # year-to-second conversion) and not 1000 K (missing mu).
    drop = 300.0 - T_cur
    assert 0.5 < drop < 5.0


@pytest.mark.physics_invariant
def test_shallow_mixed_ocean_layer_warms_under_negative_net_flux():
    """Negative ``F_net`` (downward heating) raises the layer's
    temperature. The sign behaviour pins that F_net is treated as an
    outgoing flux convention; a regression that flipped the sign of
    the RHS would cool the layer here instead of warming it.
    """
    hf_pre = {'Time': 0.0, 'T_surf': 250.0}
    hf_cur = {'Time': 1.0, 'F_net': -50.0}

    T_cur = atmos_wrapper.ShallowMixedOceanLayer(hf_cur, hf_pre)

    expected_rise = 50.0 * 3.154e7 / 3e9
    assert T_cur == pytest.approx(250.0 + expected_rise, rel=5e-3)
    # Sign guard: negative F_net warms the layer; T_cur > T_pre.
    assert T_cur > 250.0
    # Scale guard: same magnitude as the cooling test by symmetry.
    rise = T_cur - 250.0
    assert 0.1 < rise < 2.0


@pytest.mark.physics_invariant
def test_shallow_mixed_ocean_layer_zero_flux_keeps_temperature_constant():
    """F_net = 0 is the conservative-isolated-layer limit: dT/dt = 0,
    so T_cur must equal T_pre to machine precision. This pins the
    structural correctness of the ODE setup independent of mu.
    """
    hf_pre = {'Time': 0.0, 'T_surf': 1500.0}
    hf_cur = {'Time': 1.0, 'F_net': 0.0}

    T_cur = atmos_wrapper.ShallowMixedOceanLayer(hf_cur, hf_pre)

    assert T_cur == pytest.approx(1500.0, abs=1e-6)
    # Sign guard: with zero forcing the temperature must not drift in
    # either direction. A regression that introduced a constant offset
    # would surface here.
    assert abs(T_cur - 1500.0) < 1e-3


# ---------------------------------------------------------------------------
# write_atmosphere_snapshot: end-of-simulation force-write dispatch
# ---------------------------------------------------------------------------


def _fake_atmos_o(atm, solved=None):
    """Minimal Atmos_t stand-in exposing the two atmosphere attributes.

    `_atm` is the module's own struct; `_atm_janus_last` is the column JANUS
    returns from a solve, which is a different object from the one it was
    given and is the only one whose profiles and fluxes share a grid.
    """
    return SimpleNamespace(_atm=atm, _atm_janus_last=solved)


def _fake_config(module):
    return SimpleNamespace(atmos_clim=SimpleNamespace(module=module))


def test_write_atmosphere_snapshot_noop_when_atm_is_none():
    """No atmosphere struct means nothing can be written; the dispatch must
    return without touching either backend writer.

    Edge case: an aborted run may reach the end-of-sim block with
    ``atmos_o._atm is None`` (atmosphere never solved).
    """
    atmos_o = _fake_atmos_o(None)
    config = _fake_config('agni')
    with (
        patch('proteus.atmos_clim.agni.write_atmos_ncdf') as agni_w,
        patch('proteus.atmos_clim.janus.write_atmos_ncdf') as janus_w,
    ):
        atmos_wrapper.write_atmosphere_snapshot(
            atmos_o, config, {'output': '/tmp/x'}, {'Time': 100.0}
        )
    agni_w.assert_not_called()
    janus_w.assert_not_called()


def test_write_atmosphere_snapshot_dummy_module_writes_nothing():
    """The dummy atmosphere has no NetCDF representation, so the dispatch is a
    no-op even when a non-None struct is present."""
    atmos_o = _fake_atmos_o(object())
    config = _fake_config('dummy')
    with (
        patch('proteus.atmos_clim.agni.write_atmos_ncdf') as agni_w,
        patch('proteus.atmos_clim.janus.write_atmos_ncdf') as janus_w,
    ):
        atmos_wrapper.write_atmosphere_snapshot(
            atmos_o, config, {'output': '/tmp/x'}, {'Time': 100.0}
        )
    agni_w.assert_not_called()
    janus_w.assert_not_called()


def test_write_atmosphere_snapshot_agni_writes_when_allocated():
    """For AGNI with an allocated struct, the dispatch calls the AGNI writer
    once with the current time, and NOT the JANUS writer.

    Discrimination: asserting the exact `time` argument (and that janus is not
    called) rules out a regression that dispatched to the wrong backend or
    dropped the time.
    """
    atm = SimpleNamespace(is_alloc=True)
    atmos_o = _fake_atmos_o(atm)
    config = _fake_config('agni')
    dirs = {'output': '/tmp/x'}
    with (
        patch('proteus.atmos_clim.agni.write_atmos_ncdf') as agni_w,
        patch('proteus.atmos_clim.janus.write_atmos_ncdf') as janus_w,
    ):
        atmos_wrapper.write_atmosphere_snapshot(atmos_o, config, dirs, {'Time': 384913130.0})
    agni_w.assert_called_once_with(atm, dirs, 384913130.0)
    janus_w.assert_not_called()


def test_write_atmosphere_snapshot_janus_writes():
    """For JANUS the dispatch writes the solved column, not the seed.

    JANUS integrates its adiabat on a high-resolution grid and resamples onto
    the radiative grid when it solves, so the object it was handed keeps the
    integration profiles while the fluxes written back onto it sit on the
    radiative grid. Writing that object mixes two grids and the NetCDF write
    fails on the shape mismatch, so the snapshot has to come from the solved
    column the solver returned.

    Edge case: a run that has not solved a column yet has nothing coherent to
    write, and the dispatch must return rather than fall back to the seed.
    """
    seed, solved = object(), object()
    atmos_o = _fake_atmos_o(seed, solved=solved)
    config = _fake_config('janus')
    dirs = {'output': '/tmp/x'}
    with (
        patch('proteus.atmos_clim.agni.write_atmos_ncdf') as agni_w,
        patch('proteus.atmos_clim.janus.write_atmos_ncdf') as janus_w,
    ):
        atmos_wrapper.write_atmosphere_snapshot(atmos_o, config, dirs, {'Time': 4675466.0})
    janus_w.assert_called_once_with(solved, dirs, 4675466.0)
    # Discrimination: the seed is a different object, so a dispatch that still
    # reached for `_atm` would have been called with it instead.
    assert janus_w.call_args.args[0] is not seed
    agni_w.assert_not_called()

    # Seed present but never solved: nothing is written.
    unsolved = _fake_atmos_o(seed, solved=None)
    with (
        patch('proteus.atmos_clim.agni.write_atmos_ncdf') as agni_w2,
        patch('proteus.atmos_clim.janus.write_atmos_ncdf') as janus_w2,
    ):
        atmos_wrapper.write_atmosphere_snapshot(unsolved, config, dirs, {'Time': 4675466.0})
    janus_w2.assert_not_called()
    agni_w2.assert_not_called()


def test_run_atmosphere_keeps_the_column_janus_solved():
    """The JANUS dispatch stores the solved column where the writer reads it.

    JANUS returns a column that is not the object it was given, and the
    end-of-run snapshot is written from the returned one. That wiring is the
    single point where a regression would silently put the run back to writing
    the seed, which cannot be written at all: its profile arrays are sized for
    the integration grid while the fluxes on it are on the radiative grid.

    Verifies:
    - What the solver returned is what the struct carries afterwards.
    - The seed is left in place, so the next iteration still integrates from
      the object JANUS expects rather than from a resampled column.
    - The struct starts out carrying nothing, so the attribute is genuinely
      written by the call rather than pre-set by the fixture.
    """
    from proteus.atmos_clim.common import Atmos_t

    seed, solved = object(), object()
    atmos_o = Atmos_t()
    atmos_o._atm = seed
    assert atmos_o._atm_janus_last is None

    config = SimpleNamespace(
        atmos_clim=SimpleNamespace(
            module='janus',
            albedo_pl=0.0,
            rayleigh=False,
            cloud_enabled=False,
            surf_state='fixed',
        ),
        interior_energetics=SimpleNamespace(module='aragog'),
    )
    hf_row = {
        'T_magma': 1800.0,
        'M_planet': 6.0e24,
        'R_obs': 6.4e6,
        'F_int': 120.0,
        'F_atm': 100.0,
        'axial_period': 86400.0,
        'atm_kg_per_mol': 0.029,
        'T_surf': 0.0,
        'R_int': 6.371e6,
        'R_star': 6.96e8,
        'F_olr': 200.0,
        'F_sct': 100.0,
        'F_ins': 1361.0,
        'separation': 1.5e11,
    }
    atm_output = {'albedo': 0.2, 'F_atm': 100.0}

    with patch(
        'proteus.atmos_clim.janus.RunJANUS', return_value=(solved, atm_output)
    ) as run_janus:
        atmos_wrapper.run_atmosphere(
            atmos_o,
            config,
            {'output': '/tmp/x'},
            {'total': 5},
            [1.0],
            [1.0],
            False,
            None,
            hf_row,
        )

    run_janus.assert_called_once()
    # The seed is what JANUS was handed, and it is what the struct keeps as
    # its seed; the solved column is stored separately.
    assert run_janus.call_args.args[0] is seed
    assert atmos_o._atm is seed
    assert atmos_o._atm_janus_last is solved


# ---------------------------------------------------------------------------
# carry_converged_levels: levels of a rejected structure are not used
# ---------------------------------------------------------------------------


def _levels(r_obs: float, r_xuv: float, h2o_vmr: float = 0.4) -> dict:
    """Build a helpfile row carrying one consistent set of levels.

    Pressures, temperature and gravity are scaled off the radii, so a
    substitution that touched only the radii would leave a level whose pressure
    no longer matches it, and the test can see that. The XUV composition is
    carried too, since BOREAS reads it alongside the radius.
    """
    return {
        'R_obs': r_obs,
        'p_obs': 1.0e-3 * (2.0e7 / r_obs) ** 2,  # bar
        'T_obs': 900.0 * (2.0e7 / r_obs),  # K
        'g_obs': 9.8 * (2.0e7 / r_obs) ** 2,  # m s-2
        'R_xuv': r_xuv,
        'p_xuv': 1.0e-6 * (2.4e7 / r_xuv) ** 2,  # bar
        'T_xuv': 400.0 * (2.4e7 / r_xuv),  # K
        'g_xuv': 6.8 * (2.4e7 / r_xuv) ** 2,  # m s-2
        'H2O_vmr_xuv': h2o_vmr,
    }


@pytest.mark.physics_invariant
def test_carry_levels_substitutes_last_converged_after_failed_solve():
    """A converged solve records its levels; the next solve that fails has its
    photospheric and XUV levels replaced by the recorded ones.

    The energy-limited escape rate goes as R_xuv**3, so the discriminating
    quantity is the cube of the radius ratio, not the ratio itself. The
    rejected structure is inflated 12-fold in radius, which is 1728-fold in
    escape rate; a substitution that held a squared quantity, or held nothing,
    lands at 144 or 1728 and is far outside the tolerance on 1.
    """
    atmos_o = Atmos_t()
    assert atmos_o.levels_converged == {}

    good = _levels(r_obs=2.0e7, r_xuv=2.4e7, h2o_vmr=0.4)
    atmos_o.converged = True
    atmos_wrapper.carry_converged_levels(atmos_o, dict(good))
    assert atmos_o.levels_converged['R_xuv'] == pytest.approx(2.4e7, rel=1e-12)

    # A rejected structure that has ballooned well past any bound atmosphere.
    bad = _levels(r_obs=2.4e8, r_xuv=2.88e8, h2o_vmr=0.9)
    rejected_r_xuv = bad['R_xuv']
    atmos_o.converged = False
    atmos_wrapper.carry_converged_levels(atmos_o, bad)

    # Every level came back to the converged structure, pressures and the
    # composition BOREAS reads included.
    for key, val in good.items():
        assert bad[key] == pytest.approx(val, rel=1e-12)

    # Escape rate ratio the substitution removes, and the guard that the
    # rejected value was discriminably different in the first place.
    assert (rejected_r_xuv / good['R_xuv']) ** 3 == pytest.approx(1728.0, rel=1e-9)
    assert (bad['R_xuv'] / good['R_xuv']) ** 3 == pytest.approx(1.0, rel=1e-9)
    assert rejected_r_xuv > 10.0 * good['R_xuv']


def test_carry_levels_keeps_fluxes_and_surface_state_moving():
    """Only level properties are carried. Fluxes and surface temperature must
    pass through untouched, because the coupling advances on them and the
    deadlock detector fires on an interior state that has stopped moving.
    """
    atmos_o = Atmos_t()
    atmos_o.converged = True
    atmos_wrapper.carry_converged_levels(atmos_o, _levels(2.0e7, 2.4e7))

    failed = _levels(2.4e8, 2.88e8)
    failed.update({'F_atm': 4.21e3, 'T_surf': 2870.0, 'albedo': 0.31})
    atmos_o.converged = False
    atmos_wrapper.carry_converged_levels(atmos_o, failed)

    assert failed['F_atm'] == pytest.approx(4.21e3, rel=1e-12)
    assert failed['T_surf'] == pytest.approx(2870.0, rel=1e-12)
    assert failed['albedo'] == pytest.approx(0.31, rel=1e-12)
    # ... while the levels did move back.
    assert failed['R_obs'] == pytest.approx(2.0e7, rel=1e-12)


def test_carry_levels_falls_back_on_the_last_committed_row_after_a_resume(caplog):
    """A resumed run starts with an empty record, since the record is not
    written to the output files. Until it converges a solve of its own it uses
    the last committed row, which is the state escape would have used anyway.

    Without this a resumed run reopens the defect for its first iterations,
    which is when escape is already active: resume sets the iteration counter
    past the point where escape switches on.
    """
    atmos_o = Atmos_t()
    atmos_o.converged = False
    committed = _levels(2.0e7, 2.4e7)
    rejected = _levels(2.4e8, 2.88e8)

    with caplog.at_level('WARNING', logger='fwl.proteus.atmos_clim.wrapper'):
        atmos_wrapper.carry_converged_levels(atmos_o, rejected, previous_row=committed)

    assert rejected['R_xuv'] == pytest.approx(2.4e7, rel=1e-12)
    assert rejected['R_obs'] == pytest.approx(2.0e7, rel=1e-12)
    # The warning names the source it actually used, since a committed row is
    # not necessarily one that converged.
    assert 'last committed row' in caplog.text

    # A second failure before anything converges must not start calling the
    # same never-converged levels a converged solve.
    caplog.clear()
    again = _levels(2.4e8, 2.88e8)
    with caplog.at_level('WARNING', logger='fwl.proteus.atmos_clim.wrapper'):
        atmos_wrapper.carry_converged_levels(atmos_o, again, previous_row=committed)
    assert again['R_obs'] == pytest.approx(2.0e7, rel=1e-12)
    assert 'last committed row' in caplog.text
    assert 'last converged solve' not in caplog.text
    # A converged solve of this run then supersedes the committed row.
    atmos_o.converged = True
    atmos_wrapper.carry_converged_levels(atmos_o, _levels(3.1e7, 3.7e7))
    later = _levels(2.4e8, 2.88e8)
    atmos_o.converged = False
    atmos_wrapper.carry_converged_levels(atmos_o, later, previous_row=committed)
    assert later['R_obs'] == pytest.approx(3.1e7, rel=1e-12)


def test_carry_levels_without_any_earlier_levels_keeps_the_rejected_ones(caplog):
    """With no converged solve and no committed row there is nothing to fall
    back on. The rejected levels are used unchanged and the run says so, rather
    than silently substituting a level nobody computed.
    """
    atmos_o = Atmos_t()
    atmos_o.converged = False
    rejected = _levels(2.4e8, 2.88e8)

    with caplog.at_level('WARNING', logger='fwl.proteus.atmos_clim.wrapper'):
        atmos_wrapper.carry_converged_levels(atmos_o, rejected, previous_row=None)

    assert rejected['R_xuv'] == pytest.approx(2.88e8, rel=1e-12)
    assert rejected['R_obs'] == pytest.approx(2.4e8, rel=1e-12)
    assert atmos_o.levels_converged == {}
    assert 'no earlier levels' in caplog.text


def test_carry_levels_holds_across_consecutive_failures(caplog):
    """A run of failed solves keeps falling back on the same converged levels.
    A rejected solve must never write to the record, or the second failure
    would hold the first failure's inflated radius and the fix would decay
    over a streak of failures.
    """
    atmos_o = Atmos_t()
    atmos_o.converged = True
    atmos_wrapper.carry_converged_levels(atmos_o, _levels(2.0e7, 2.4e7))

    atmos_o.converged = False
    first = _levels(2.4e8, 2.88e8)
    second = _levels(9.6e8, 1.152e9)
    with caplog.at_level('WARNING', logger='fwl.proteus.atmos_clim.wrapper'):
        atmos_wrapper.carry_converged_levels(atmos_o, first)
        atmos_wrapper.carry_converged_levels(atmos_o, second)

    assert 'last converged solve' in caplog.text
    assert 'last committed row' not in caplog.text

    assert first['R_xuv'] == pytest.approx(2.4e7, rel=1e-12)
    assert second['R_xuv'] == pytest.approx(2.4e7, rel=1e-12)
    assert atmos_o.levels_converged['R_xuv'] == pytest.approx(2.4e7, rel=1e-12)
    # Both failures were reported, not just the first.
    assert caplog.text.count('did not converge') == 2


def test_carry_levels_ignores_a_non_finite_level():
    """A solve that returns NaN for a level has not measured it. Recording that
    value would leave a record that cannot be fallen back on, so non-finite
    values are skipped and the previous finite value stands.
    """
    atmos_o = Atmos_t()
    atmos_o.converged = True
    atmos_wrapper.carry_converged_levels(atmos_o, _levels(2.0e7, 2.4e7))

    poisoned = _levels(2.0e7, 2.4e7)
    poisoned['R_xuv'] = float('nan')
    atmos_o.converged = True
    atmos_wrapper.carry_converged_levels(atmos_o, poisoned)
    assert atmos_o.levels_converged['R_xuv'] == pytest.approx(2.4e7, rel=1e-12)

    failed = _levels(2.4e8, 2.88e8)
    atmos_o.converged = False
    atmos_wrapper.carry_converged_levels(atmos_o, failed)
    assert failed['R_xuv'] == pytest.approx(2.4e7, rel=1e-12)


def test_carry_levels_record_survives_a_module_that_omits_a_level():
    """The record is merged, not replaced, so a converged solve that omits one
    level leaves the earlier value for that level in place instead of dropping
    it. A dropped key would silently return the rejected value to escape.
    """
    atmos_o = Atmos_t()
    atmos_o.converged = True
    atmos_wrapper.carry_converged_levels(atmos_o, _levels(2.0e7, 2.4e7))

    partial = {'R_obs': 3.1e7, 'p_obs': 4.2e-4}
    atmos_o.converged = True
    atmos_wrapper.carry_converged_levels(atmos_o, partial)

    failed = _levels(2.4e8, 2.88e8)
    atmos_o.converged = False
    atmos_wrapper.carry_converged_levels(atmos_o, failed)

    assert failed['R_obs'] == pytest.approx(3.1e7, rel=1e-12)  # the newer value
    assert failed['R_xuv'] == pytest.approx(2.4e7, rel=1e-12)  # the retained one


def test_carry_levels_leaves_a_row_that_lacks_the_key_alone():
    """Keys absent from the row are neither recorded nor written back, so a
    module that reports fewer levels does not gain invented ones.
    """
    atmos_o = Atmos_t()
    atmos_o.converged = True
    atmos_wrapper.carry_converged_levels(atmos_o, {'R_obs': 2.0e7, 'R_xuv': 2.4e7})
    assert set(atmos_o.levels_converged) == {'R_obs', 'R_xuv'}

    failed = {'R_obs': 2.4e8, 'p_obs': 5.0e-5}
    atmos_o.converged = False
    atmos_wrapper.carry_converged_levels(atmos_o, failed)

    assert failed['R_obs'] == pytest.approx(2.0e7, rel=1e-12)
    assert failed['p_obs'] == pytest.approx(5.0e-5, rel=1e-12)  # nothing recorded for it
    assert 'R_xuv' not in failed  # not invented out of the record


@pytest.mark.physics_invariant
def test_run_atmosphere_carries_levels_into_the_row_escape_reads():
    """The wiring test: `run_atmosphere` puts the carried levels into `hf_row`,
    which is where escape reads R_xuv on the following iteration, and the
    observables derived from R_obs follow the carried radius.

    Escape runs before the atmosphere within an iteration, so the radius it
    uses is the one the previous atmosphere call left in the row. A fix that
    only corrected the module output, without it reaching the row, would leave
    escape reading the rejected radius exactly as before.
    """
    atmos_o = Atmos_t()
    atmos_o._atm = object()

    config = SimpleNamespace(
        atmos_clim=SimpleNamespace(
            module='agni',
            albedo_pl=0.0,
            rayleigh=False,
            cloud_enabled=False,
            surf_state='fixed',
        ),
        interior_energetics=SimpleNamespace(module='aragog'),
    )
    hf_row = {
        'T_magma': 1800.0,
        'M_planet': 6.0e24,
        'F_int': 120.0,
        'F_atm': 100.0,
        'axial_period': 86400.0,
        'atm_kg_per_mol': 0.029,
        'T_surf': 0.0,
        'R_int': 6.371e6,
        'R_star': 6.96e8,
        'F_olr': 200.0,
        'F_sct': 100.0,
        'F_ins': 1361.0,
        'separation': 1.5e11,
        **{key: 0.0 for key in _levels(1.0, 1.0)},
    }

    def _call(levels, converged):
        out = dict(levels)
        out.update({'albedo': 0.2, 'F_atm': 100.0, 'agni_converged': converged})
        vmr = out.pop('H2O_vmr_xuv')

        def _run_agni(atm, *args, **kwargs):
            # AGNI writes the XUV composition straight into the row, so the
            # mock has to as well or the test would not exercise that path.
            hf_row['H2O_vmr_xuv'] = vmr
            return atm, out

        with (
            patch(
                'proteus.atmos_clim.agni.update_agni_atmos', side_effect=lambda a, *_, **__: a
            ),
            patch('proteus.atmos_clim.agni.run_agni', side_effect=_run_agni),
        ):
            atmos_wrapper.run_atmosphere(
                atmos_o,
                config,
                {'output': '/tmp/x'},
                {'total': 5},
                [1.0],
                [1.0],
                False,
                None,
                hf_row,
            )

    _call(_levels(r_obs=2.0e7, r_xuv=2.4e7, h2o_vmr=0.4), converged=True)
    assert hf_row['R_xuv'] == pytest.approx(2.4e7, rel=1e-12)
    # A converged row is marked as such, with no stale streak.
    assert hf_row['atm_converged'] == pytest.approx(1.0, rel=1e-12)
    assert hf_row['atm_levels_stale'] == pytest.approx(0.0, abs=1e-12)

    _call(_levels(r_obs=2.4e8, r_xuv=2.88e8, h2o_vmr=0.9), converged=False)

    # The row escape will read carries the converged radii and the converged
    # composition, not the rejected ones.
    assert hf_row['R_xuv'] == pytest.approx(2.4e7, rel=1e-12)
    assert hf_row['R_obs'] == pytest.approx(2.0e7, rel=1e-12)
    assert hf_row['H2O_vmr_xuv'] == pytest.approx(0.4, rel=1e-12)
    # Quantities derived from R_obs follow the carried radius, so the row stays
    # internally consistent rather than mixing two structures.
    assert hf_row['transit_depth'] == pytest.approx((2.0e7 / 6.96e8) ** 2, rel=1e-9)
    assert hf_row['rho_obs'] == pytest.approx(
        3 * 6.0e24 / (4 * math.pi * (2.0e7) ** 3), rel=1e-9
    )
    # The flag the deadlock detector reads still reports the failure.
    assert atmos_o.converged is False
    # And the transient flag never reaches the row.
    assert 'agni_converged' not in hf_row
    # The persisted outcome columns record the rejection and the streak, so a
    # carried row is identifiable from the output alone.
    assert hf_row['atm_converged'] == pytest.approx(-1.0, rel=1e-12)
    assert hf_row['atm_levels_stale'] == pytest.approx(1.0, rel=1e-12)
    # The carried temperature at the XUV level came back with the radius.
    assert hf_row['T_xuv'] == pytest.approx(400.0, rel=1e-12)


@pytest.mark.physics_invariant
def test_run_atmosphere_takes_the_resume_fallback_from_the_last_committed_row():
    """On the first iteration after a resume the record is empty, so
    `run_atmosphere` has to reach into the persisted history for it, and it has
    to take the LAST committed row.

    Escape is already active at that point, because resume sets the iteration
    counter past the point where escape switches on. Reaching one row too far
    back would hand escape a radius from an earlier state of the planet, which
    for an evolving atmosphere is a different radius, so the two rows here
    differ by a factor of four.
    """
    atmos_o = Atmos_t()
    atmos_o._atm = object()

    config = SimpleNamespace(
        atmos_clim=SimpleNamespace(
            module='agni',
            albedo_pl=0.0,
            rayleigh=False,
            cloud_enabled=False,
            surf_state='fixed',
        ),
        interior_energetics=SimpleNamespace(module='aragog'),
    )
    hf_row = {
        'T_magma': 1800.0,
        'M_planet': 6.0e24,
        'F_int': 120.0,
        'F_atm': 100.0,
        'axial_period': 86400.0,
        'atm_kg_per_mol': 0.029,
        'T_surf': 0.0,
        'R_int': 6.371e6,
        'R_star': 6.96e8,
        'F_olr': 200.0,
        'F_sct': 100.0,
        'F_ins': 1361.0,
        'separation': 1.5e11,
        **{key: 0.0 for key in _levels(1.0, 1.0)},
    }

    # Two committed rows: an older one and the one resume must actually use.
    hf_all = pd.DataFrame([_levels(5.0e6, 6.0e6, 0.1), _levels(2.0e7, 2.4e7, 0.4)])

    rejected = _levels(2.4e8, 2.88e8, 0.9)
    out = dict(rejected)
    out.update({'albedo': 0.2, 'F_atm': 100.0, 'agni_converged': False})
    vmr = out.pop('H2O_vmr_xuv')

    def _run_agni(atm, *args, **kwargs):
        hf_row['H2O_vmr_xuv'] = vmr
        return atm, out

    with (
        patch('proteus.atmos_clim.agni.update_agni_atmos', side_effect=lambda a, *_, **__: a),
        patch('proteus.atmos_clim.agni.run_agni', side_effect=_run_agni),
    ):
        atmos_wrapper.run_atmosphere(
            atmos_o,
            config,
            {'output': '/tmp/x'},
            {'total': 5},
            [1.0],
            [1.0],
            False,
            hf_all,
            hf_row,
        )

    assert hf_row['R_xuv'] == pytest.approx(2.4e7, rel=1e-12)
    assert hf_row['R_obs'] == pytest.approx(2.0e7, rel=1e-12)
    assert hf_row['H2O_vmr_xuv'] == pytest.approx(0.4, rel=1e-12)
    # Discrimination: the older row is a factor of four away, far outside the
    # tolerance, so reading the wrong row cannot pass as rounding.
    assert hf_row['R_obs'] != pytest.approx(5.0e6, rel=1e-1)
    # And the rejected structure did not survive anywhere in the row.
    assert hf_row['R_xuv'] < 0.1 * rejected['R_xuv']


def test_carry_levels_escalates_on_a_long_streak(caplog):
    """A solver that fails once is a stumble; one that fails for many
    iterations leaves escape running on a planet the run has not resolved,
    while the interior keeps evolving underneath it.

    The deadlock detector cannot catch that case, since it only fires when the
    interior is frozen as well, so the streak is reported here instead. The
    count is carried on the struct, so it survives across iterations and resets
    on the next solve that converges.
    """
    atmos_o = Atmos_t()
    atmos_o.converged = True
    atmos_wrapper.carry_converged_levels(atmos_o, _levels(2.0e7, 2.4e7))

    atmos_o.converged = False
    with caplog.at_level('WARNING', logger='fwl.proteus.atmos_clim.wrapper'):
        for _ in range(atmos_wrapper.CARRIED_LEVELS_ALERT - 1):
            atmos_wrapper.carry_converged_levels(atmos_o, _levels(2.4e8, 2.88e8))

    assert atmos_o.levels_stale_iters == atmos_wrapper.CARRIED_LEVELS_ALERT - 1
    errors = [r for r in caplog.records if r.levelname == 'ERROR']
    assert not errors  # one short of the threshold, so nothing has escalated

    with caplog.at_level('WARNING', logger='fwl.proteus.atmos_clim.wrapper'):
        atmos_wrapper.carry_converged_levels(atmos_o, _levels(2.4e8, 2.88e8))
    errors = [r for r in caplog.records if r.levelname == 'ERROR']
    assert len(errors) == 1
    assert str(atmos_wrapper.CARRIED_LEVELS_ALERT) in errors[0].getMessage()

    # A solve that converges clears the streak, so the next failure starts over.
    atmos_o.converged = True
    atmos_wrapper.carry_converged_levels(atmos_o, _levels(2.0e7, 2.4e7))
    assert atmos_o.levels_stale_iters == 0


def test_carry_levels_does_not_trust_rows_this_run_rejected(caplog):
    """Only the first solve of a run may fall back on the committed rows.

    A fresh run whose first solve fails commits the rejected levels, since
    there is nothing better to write. Treating that row as a fallback on the
    next failure would launder a rejected structure into a trusted one and keep
    re-propagating it, so a run that has never converged says it has nothing to
    fall back on instead.
    """
    atmos_o = Atmos_t()
    atmos_o.converged = False

    # First solve of a fresh run: no history at all, so the rejected levels stand.
    first = _levels(2.4e8, 2.88e8)
    atmos_wrapper.carry_converged_levels(atmos_o, first, previous_row=None)
    assert first['R_obs'] == pytest.approx(2.4e8, rel=1e-12)

    # That row is now the committed one. The second failure must not adopt it.
    second = _levels(4.8e8, 5.76e8)
    with caplog.at_level('WARNING', logger='fwl.proteus.atmos_clim.wrapper'):
        atmos_wrapper.carry_converged_levels(atmos_o, second, previous_row=first)

    assert second['R_obs'] == pytest.approx(4.8e8, rel=1e-12)
    assert atmos_o.levels_converged == {}
    assert 'no earlier levels' in caplog.text


def test_carry_levels_escalates_when_there_is_nothing_to_fall_back_on(caplog):
    """A run that has never converged and has no committed history is the worst
    case, not an exempt one: escape runs on the rejected structure itself. The
    streak is counted and escalated there too, so that run is not the one case
    that reports a bare warning forever.
    """
    atmos_o = Atmos_t()
    atmos_o.converged = False

    with caplog.at_level('WARNING', logger='fwl.proteus.atmos_clim.wrapper'):
        for _ in range(atmos_wrapper.CARRIED_LEVELS_ALERT):
            atmos_wrapper.carry_converged_levels(atmos_o, _levels(2.4e8, 2.88e8))

    assert atmos_o.levels_stale_iters == atmos_wrapper.CARRIED_LEVELS_ALERT
    assert atmos_o.levels_converged == {}  # still nothing recorded
    errors = [r for r in caplog.records if r.levelname == 'ERROR']
    assert len(errors) == 1
    assert 'no earlier levels' in caplog.text


def test_carry_levels_counts_converged_solves_towards_the_first_solve_rule():
    """The first-solve rule counts every solve, not only the failed ones. A run
    that converges once and then fails is past its first solve, so it must hold
    its own converged levels and never reach back to the committed rows.

    Counting only failures would let a committed row overwrite the record on
    the first failure, which is how a level this run never converged gets
    laundered into the record it trusts.
    """
    atmos_o = Atmos_t()
    atmos_o.converged = True
    atmos_wrapper.carry_converged_levels(atmos_o, _levels(2.0e7, 2.4e7))
    assert atmos_o.solves_seen == 1

    committed = _levels(5.0e6, 6.0e6)
    failed = _levels(2.4e8, 2.88e8)
    atmos_o.converged = False
    atmos_wrapper.carry_converged_levels(atmos_o, failed, previous_row=committed)

    # The converged levels of this run, not the committed row.
    assert failed['R_obs'] == pytest.approx(2.0e7, rel=1e-12)
    assert failed['R_xuv'] == pytest.approx(2.4e7, rel=1e-12)
    assert failed['R_obs'] != pytest.approx(5.0e6, rel=1e-1)
    assert atmos_o.levels_source is LevelsSource.CONVERGED_SOLVE
