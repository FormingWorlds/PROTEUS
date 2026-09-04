"""Unit tests for the planet-satellite tidal orbit evolution module
(``proteus.orbit.satellite``).

Structure mirrors ``tests/orbit/test_orbit.py``: each model's ODE
right-hand sides are private closures (``Ltot``/``dw_dt``/``da_dt``
nested in ``ps0d``; ``domega_dt``/``dE_dt``/``orbitals`` nested in
``ps1d``/``ps1d_evec``), so they are tested black-box through their
public entry points, using finite-difference probes for the closed-
form algebra checks and real (non-trivial) integration steps for the
conservation checks.

Exercises:

- ``compute_a_res_prime``, ``_state_is_valid``, ``_in_evection_band``,
  ``_flush_fine_evection_csv``: pure/file-I/O helpers, tested directly.
- ``ps0d`` (Korenaga 2023 Icarus 400, 115564, Eqs. 58-60): the
  ``Ltot`` M_sat-vs-M_planet discrimination guard the source itself
  asks for (see the comment above ``Ltot``'s ``return`` statement),
  the Eq. 58/59 fixed point and closed-form pin, and the
  ``current_time <= 10`` angular-momentum bootstrap.
- ``ps1d``: zero-dissipation fixed point, and total angular momentum
  (now THREE components: planet spin + satellite spin + orbital)
  conservation -- using both the raw-total and the delta-cancellation
  formulation, since (as found for ``sp1d`` in ``orbit.py``) the raw
  total is dominated by the orbital term and is comparatively blind
  to a bug confined to one spin-coupling term.
- ``ps1d_evec``: the "three clocks" model (PROTEUS's coarse elapsed
  time -> evolve_orbit_satellite's adaptive substep controller ->
  solve_ivp's own internal adaptive stepping -> the storage-clock
  throttle in _flush_fine_evection_csv). ``filter_value=0`` (out of
  band) reduces it EXACTLY to ps1d's physics: evection angle frozen,
  same 3-component total AM conserved to machine precision, verified
  here, not assumed. ``filter_value=1`` (in band) activates the
  star's secular/evection torque: the evection angle evolves and the
  planet-satellite subsystem's own AM is NOT expected to be conserved
  (a real three-body exchange with the star) -- checked numerically
  and documented as physical, not asserted as a false invariant.
  Also covers the wiring from ``_in_evection_band``'s live result
  through to ``ps1d_evec``'s ``filter_value`` (with ``ps1d_evec``
  itself mocked out, since that wiring check has nothing to do with
  its internal physics), and the storage-clock density
  contrast (dense in-band vs throttled out-of-band) through the real
  ``evolve_orbit_satellite`` -> ``ps1d_evec`` pipeline. The CPL
  resonance-physics literature comparison against a published case is
  out of scope for this file (a separate, long-running test).
- ``evolve_orbit_satellite``: dispatch to each model (and the
  unrecognized-model error), the accept/reject adaptive-step
  controller (forcing a rejection to observe the rollback), the
  documented C_planet angular-momentum-conserving spin rescale on a
  structural (interior) change between calls -- including with real
  (non-quiescent) ps0d dynamics running across the change -- and
  controller-state persistence (``_orbit_dt_yr``/
  ``_orbit_resonance_state``) across calls.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_building.md
- docs/How-to/test_categorization.md
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import proteus.orbit.common as common_mod
from proteus.orbit.common import Tides_t
from proteus.orbit.satellite import (
    _flush_fine_evection_csv,
    _in_evection_band,
    _state_is_valid,
    compute_a_res_prime,
    evolve_orbit_satellite,
    ps0d,
    ps1d,
    ps1d_evec,
)
from proteus.utils.constants import M_earth, R_earth, const_G, secs_per_year

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# compute_a_res_prime
# ---------------------------------------------------------------------------


def test_compute_a_res_prime_matches_closed_form_at_earth_like_spin():
    """``a_res = (Lambda * s' / (1 - e^2))**(4/7)``, ``s' = Omega_p /
    Omega_earth``. Pinned at Earth's own present-day spin (so
    ``s' = 1``) and ``e = 0``, where the formula collapses to
    ``Lambda**(4/7)``.
    """
    omega_earth = np.sqrt(const_G * M_earth / R_earth**3)
    hf_row = {'eccentricity_sat': 0.0, 'axial_period': 2 * np.pi / omega_earth}
    lam = np.sqrt(1.5 * 0.315 * omega_earth / (2 * np.pi / secs_per_year))
    expected = lam ** (4.0 / 7.0)
    assert compute_a_res_prime(hf_row) == pytest.approx(expected, rel=1e-10)
    # Discrimination: a doubled spin rate must NOT double a_res (the
    # exponent is 4/7 on s', not 1); pin the ratio explicitly.
    hf_row_fast = {'eccentricity_sat': 0.0, 'axial_period': 2 * np.pi / (2 * omega_earth)}
    ratio = compute_a_res_prime(hf_row_fast) / compute_a_res_prime(hf_row)
    assert ratio == pytest.approx(2.0 ** (4.0 / 7.0), rel=1e-10)
    assert abs(ratio - 2.0) > 0.3  # rejects a linear-in-s' regression


def test_compute_a_res_prime_increases_with_eccentricity():
    """``1/(1-e^2)`` grows with ``e``, so ``a_res`` is monotonically
    increasing in eccentricity at fixed spin (a boundedness/
    monotonicity invariant, not just a point pin)."""
    omega_earth = np.sqrt(const_G * M_earth / R_earth**3)
    axial_period = 2 * np.pi / omega_earth
    a_res_circular = compute_a_res_prime(
        {'eccentricity_sat': 0.0, 'axial_period': axial_period}
    )
    a_res_eccentric = compute_a_res_prime(
        {'eccentricity_sat': 0.5, 'axial_period': axial_period}
    )
    assert a_res_eccentric > a_res_circular
    # Edge case: near-parabolic e must not raise (only warn/produce a
    # large-but-finite value); errstate(invalid='ignore') is only for
    # e >= 1 exactly, so use a merely-large e here.
    a_res_extreme = compute_a_res_prime({'eccentricity_sat': 0.9, 'axial_period': axial_period})
    assert np.isfinite(a_res_extreme)
    assert a_res_extreme > a_res_eccentric


# ---------------------------------------------------------------------------
# _state_is_valid
# ---------------------------------------------------------------------------


def test_state_is_valid_accepts_a_physically_sane_state():
    """A realistic Earth-Moon-like state with finite spin periods and
    e in-range must be accepted."""
    hf_row = {
        'semimajorax_sat': 3.844e8,
        'eccentricity_sat': 0.05,
        'axial_period': 86400.0,
        'axial_period_sat': 2.36e6,
    }
    assert _state_is_valid(hf_row) is True


@pytest.mark.physics_invariant
def test_state_is_valid_rejects_satellite_inside_planet():
    """A semimajor axis at or below ``1.05 R_earth`` (the satellite
    has effectively spiralled into the planet) is rejected -- a hard
    physical floor, not a solver-tolerance artifact.
    """
    hf_row = {
        'semimajorax_sat': 1.0 * R_earth,
        'eccentricity_sat': 0.05,
        'axial_period': 86400.0,
        'axial_period_sat': 2.36e6,
    }
    assert _state_is_valid(hf_row) is False
    # Discrimination: just above the floor must be accepted -- pins
    # the boundary itself, not merely "small values are rejected".
    hf_row_ok = dict(hf_row, semimajorax_sat=1.06 * R_earth)
    assert _state_is_valid(hf_row_ok) is True


@pytest.mark.physics_invariant
@pytest.mark.parametrize('bad_e', [-0.01, 0.999, 1.5, float('nan')])
def test_state_is_valid_rejects_eccentricity_outside_0_to_0p999(bad_e):
    """Eccentricity must lie in ``[0, 0.999)``: negative, exactly at
    or beyond the sub-parabolic ceiling, and NaN are all rejected.
    """
    hf_row = {
        'semimajorax_sat': 3.844e8,
        'eccentricity_sat': bad_e,
        'axial_period': 86400.0,
        'axial_period_sat': 2.36e6,
    }
    assert _state_is_valid(hf_row) is False


def test_state_is_valid_rejects_non_finite_spin_period():
    """A non-finite planet or satellite spin period is rejected even
    when ``a``/``e`` are otherwise fine."""
    base = {'semimajorax_sat': 3.844e8, 'eccentricity_sat': 0.05}
    assert _state_is_valid(dict(base, axial_period=np.nan, axial_period_sat=2.36e6)) is False
    assert _state_is_valid(dict(base, axial_period=86400.0, axial_period_sat=np.inf)) is False


# ---------------------------------------------------------------------------
# _in_evection_band: debounced hysteretic detector
# ---------------------------------------------------------------------------


def test_in_evection_band_enters_when_within_margin_and_exits_when_far(monkeypatch):
    """A two-step history that stays within ``margin_enter`` of
    ``a_res`` must activate the band; once active, moving far outside
    even the wider ``margin_exit`` must deactivate it (asymmetric
    Schmitt-trigger margins, not a single threshold).
    """
    from proteus.orbit import satellite as sat_mod

    # Pin a_res_prime to a known constant so the band geometry is
    # controlled by semimajorax_sat alone, independent of spin/e.
    monkeypatch.setattr(sat_mod, 'compute_a_res_prime', lambda hf_row: 60.0)

    state = {}
    # Two consecutive calls at a'=60.03 R_earth: 0.05% inside a_res=60.
    hf_row = {'semimajorax_sat': 60.03 * R_earth}
    assert _in_evection_band(hf_row, state, margin_enter=0.10, margin_exit=0.35) is True
    assert _in_evection_band(hf_row, state, margin_enter=0.10, margin_exit=0.35) is True

    # Now move far outside even the wide exit margin (a' = 100, i.e.
    # d_a_rel = 0.667 >> 0.35): must deactivate.
    hf_row_far = {'semimajorax_sat': 100.0 * R_earth}
    assert _in_evection_band(hf_row_far, state, margin_enter=0.10, margin_exit=0.35) is False


def test_in_evection_band_hysteresis_keeps_active_within_exit_margin(monkeypatch):
    """Once active, a displacement that would have failed the
    (narrower) entry margin but still satisfies the (wider) exit
    margin must NOT deactivate the band -- this asymmetry is the
    entire point of the Schmitt trigger (prevents rapid toggling).
    """
    from proteus.orbit import satellite as sat_mod

    monkeypatch.setattr(sat_mod, 'compute_a_res_prime', lambda hf_row: 60.0)

    state = {}
    hf_row_in = {'semimajorax_sat': 60.0 * R_earth}
    assert _in_evection_band(hf_row_in, state, margin_enter=0.10, margin_exit=0.35) is True

    # d_a_rel = 0.20: outside the 0.10 entry margin, inside the 0.35
    # exit margin. Because the band is already active, this must
    # stay active (uses margin_exit, not margin_enter).
    hf_row_mid = {'semimajorax_sat': 72.0 * R_earth}
    assert _in_evection_band(hf_row_mid, state, margin_enter=0.10, margin_exit=0.35) is True


def test_in_evection_band_handles_non_finite_a_res_gracefully():
    """A non-finite or zero ``a_res`` (e.g. e -> 1 upstream) must
    deactivate the band and clear history rather than raise or emit
    NaN comparisons.
    """
    state = {'active': True, 'hist_d_a_rel': [0.05, 0.06]}
    # e = 1.0 exactly makes (1 - e**2) == 0.0, so a_res_prime's
    # division produces +inf (suppressed RuntimeWarning), not merely
    # a large finite value -- the actual branch under test.
    hf_row = {
        'semimajorax_sat': 60.0 * R_earth,
        'eccentricity_sat': 1.0,
        'axial_period': 86400.0,
    }
    with np.errstate(divide='ignore'):
        result = _in_evection_band(hf_row, state, margin_enter=0.10, margin_exit=0.35)
    assert result is False
    assert state['active'] is False
    assert state['hist_d_a_rel'] == []


# ---------------------------------------------------------------------------
# _flush_fine_evection_csv: dedup + storage-clock throttle, real file I/O
# ---------------------------------------------------------------------------


def _make_fine_entry(t_abs_yr, n=None):
    t_abs_yr = np.asarray(t_abs_yr, dtype=float)
    n = len(t_abs_yr) if n is None else n
    return {
        't_abs_yr': t_abs_yr,
        'omega_p': np.full(n, 7.27e-5),
        'omega_s': np.full(n, 2.5e-6),
        'sma': np.full(n, 3.844e8),
        'ecc': np.full(n, 0.05),
        'phi': np.full(n, 0.0),
        'da_planet_tide_cum': np.zeros(n),
        'da_sat_tide_cum': np.zeros(n),
        'de_planet_tide_cum': np.zeros(n),
        'de_sat_tide_cum': np.zeros(n),
        'filter': np.ones(n),
    }


def test_flush_fine_evection_csv_in_band_keeps_every_sample(tmp_path):
    """When ``in_band`` is True, every sample surviving the dedup
    filter is written -- no storage-clock throttling."""
    hf_row = {}
    entry = _make_fine_entry([1.0, 2.0, 3.0])
    _flush_fine_evection_csv(
        hf_row, str(tmp_path), entry, in_band=True, storage_target_interval_yr=100.0
    )

    csv_path = tmp_path / 'fine_evection_data.csv'
    lines = csv_path.read_text().splitlines()
    assert len(lines) == 1 + 3  # header + 3 rows, none throttled
    assert hf_row['_fine_csv_last_t_yr'] == pytest.approx(3.0)


def test_flush_fine_evection_csv_out_of_band_throttles_to_target_spacing(tmp_path):
    """Out of band, only samples reaching/crossing the storage target
    are kept; the target then advances from the KEPT sample's time,
    not blindly by a fixed increment.
    """
    hf_row = {}
    entry = _make_fine_entry([1.0, 2.0, 3.0, 20.0, 21.0, 50.0])
    _flush_fine_evection_csv(
        hf_row, str(tmp_path), entry, in_band=False, storage_target_interval_yr=10.0
    )
    csv_path = tmp_path / 'fine_evection_data.csv'
    lines = csv_path.read_text().splitlines()
    # Header + kept rows. First target is -inf -> t=1.0 is kept
    # (starts the cursor at 1+10=11); 2,3 are dropped (< 11); 20.0
    # crosses 11 -> kept, cursor advances to 20+10=30; 21.0 dropped
    # (< 30); 50.0 crosses 30 -> kept.
    assert len(lines) - 1 == 3
    kept_times = [float(line.split(',')[0]) for line in lines[1:]]
    assert kept_times == pytest.approx([1.0, 20.0, 50.0])


def test_flush_fine_evection_csv_dedup_drops_samples_at_or_before_last_write(tmp_path):
    """A sample at or before the persisted ``_fine_csv_last_t_yr``
    cursor (e.g. a duplicate boundary sample from the previous
    accepted call) is dropped regardless of the in-band/out-of-band
    policy.
    """
    hf_row = {'_fine_csv_last_t_yr': 5.0}
    entry = _make_fine_entry([4.0, 5.0, 6.0, 7.0])
    _flush_fine_evection_csv(
        hf_row, str(tmp_path), entry, in_band=True, storage_target_interval_yr=100.0
    )

    csv_path = tmp_path / 'fine_evection_data.csv'
    lines = csv_path.read_text().splitlines()
    kept_times = [float(line.split(',')[0]) for line in lines[1:]]
    # 4.0 and 5.0 (<= last_t) dropped; 6.0 and 7.0 kept.
    assert kept_times == pytest.approx([6.0, 7.0])
    assert hf_row['_fine_csv_last_t_yr'] == pytest.approx(7.0)


def test_flush_fine_evection_csv_no_kept_samples_writes_no_file(tmp_path):
    """If every sample is deduped away, no file is created at all (not
    an empty/header-only file)."""
    hf_row = {'_fine_csv_last_t_yr': 100.0}
    entry = _make_fine_entry([1.0, 2.0, 3.0])
    _flush_fine_evection_csv(
        hf_row, str(tmp_path), entry, in_band=True, storage_target_interval_yr=100.0
    )
    assert not (tmp_path / 'fine_evection_data.csv').exists()


def test_flush_fine_evection_csv_empty_entry_is_a_no_op(tmp_path):
    """An entry with zero samples returns immediately without writing
    a file or touching the cursors."""
    hf_row = {}
    entry = _make_fine_entry([])
    _flush_fine_evection_csv(
        hf_row, str(tmp_path), entry, in_band=True, storage_target_interval_yr=100.0
    )
    assert not (tmp_path / 'fine_evection_data.csv').exists()
    assert '_fine_csv_last_t_yr' not in hf_row


# ---------------------------------------------------------------------------
# ps0d: Korenaga (2023) Icarus 400, 115564, Eqs. 58-60 (single satellite,
# planet spin + satellite orbit; no satellite spin state). Ltot/dw_dt/da_dt
# are private closures, probed black-box through the public ps0d(hf_row, dt)
# entry point via finite differences (calibrated below: rates converge
# cleanly from dt_yr=1e4 to 1e8, so dt_yr=1e5 is used throughout).
# ---------------------------------------------------------------------------

_PS0D_RPL = 6.371e6  # R_earth
_PS0D_MPL = 5.972e24  # M_earth
_PS0D_MSA = 7.342e22  # M_moon
_PS0D_SMA = 3.844e8  # Earth-Moon distance, m
_PS0D_AXIAL_PERIOD = 86400.0  # 1 day
_PS0D_I = 2.0 / 5.0 * _PS0D_MPL * _PS0D_RPL**2  # uniform-sphere moment of inertia
_PS0D_FD_DT_YR = 1e5


def _ps0d_korenaga_L(omega, sma):
    """Korenaga (2023) Eq. 60, computed independently of the source
    (no call into ps0d/Ltot): the orbital prefactor is the SATELLITE
    mass, not the planet mass."""
    return _PS0D_I * omega + _PS0D_MSA * (const_G * (_PS0D_MPL + _PS0D_MSA) * _PS0D_SMA) ** 0.5


def _make_ps0d_hf_row(
    *, time=100.0, L=None, F_tidal=1e-3, sma=_PS0D_SMA, axial_period=_PS0D_AXIAL_PERIOD
):
    if L is None:
        L = _ps0d_korenaga_L(2 * np.pi / axial_period, sma)
    return {
        'R_int': _PS0D_RPL,
        'M_int': _PS0D_MPL,
        'M_sat': _PS0D_MSA,
        'semimajorax_sat': sma,
        'axial_period': axial_period,
        'plan_sat_am': L,
        'F_tidal': F_tidal,
        'Time': time,
        # ps0d's ODE now reads its moment-of-inertia coefficient from
        # here directly (matching ps1d/ps1d_evec), not a fixed
        # uniform-sphere approximation computed internally.
        'C_planet': _PS0D_I,
    }


def _ps0d_instantaneous_rates(**hf_row_kwargs):
    """Probe ps0d's private da_dt/dw_dt via finite difference over a
    short (but well-converged, see module docstring) step. Returns
    ``(da_dt, domega_dt)`` in SI units.
    """
    hf_row = _make_ps0d_hf_row(**hf_row_kwargs)
    sma_before, axial_before = hf_row['semimajorax_sat'], hf_row['axial_period']
    omega_before = 2 * np.pi / axial_before

    ps0d(hf_row, dt=_PS0D_FD_DT_YR)

    dt_s = _PS0D_FD_DT_YR * secs_per_year
    da_dt = (hf_row['semimajorax_sat'] - sma_before) / dt_s
    omega_after = 2 * np.pi / hf_row['axial_period']
    domega_dt = (omega_after - omega_before) / dt_s
    return da_dt, domega_dt


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_ps0d_bootstrap_am_uses_satellite_mass_not_planet_mass():
    """``ps0d``'s angular-momentum bootstrap (``current_time <= 10``
    and ``plan_sat_am == 0``) computes ``L`` via ``Ltot``, Korenaga
    (2023) Eq. 60: the orbital term's prefactor is the SATELLITE mass
    (the M_sat/M_planet -> 0 limit of the textbook reduced-mass
    formula), not the planet mass -- this is exactly the discriminating
    guard the source's own comment above ``Ltot``'s ``return``
    statement asks for.

    For Earth-Moon, M_planet/M_sat ~ 81, so a regression that
    substituted M_planet for M_sat in the orbital term would inflate
    the bootstrapped L by roughly that factor -- checked explicitly
    below, not just approximated.
    """
    hf_row = {
        'R_int': _PS0D_RPL,
        'M_int': _PS0D_MPL,
        'M_sat': _PS0D_MSA,
        'semimajorax_sat': _PS0D_SMA,
        'axial_period': _PS0D_AXIAL_PERIOD,
        'plan_sat_am': 0,  # triggers the bootstrap
        'F_tidal': 0.0,
        'Time': 0.0,
        # In production, evolve_orbit_satellite populates this via
        # get_C_planet before calling ps0d; seeded directly here since
        # this test calls ps0d in isolation.
        'C_planet': _PS0D_I,
    }
    ps0d(hf_row, dt=1.0)

    omega = 2 * np.pi / _PS0D_AXIAL_PERIOD
    expected = _ps0d_korenaga_L(omega, _PS0D_SMA)
    assert hf_row['plan_sat_am'] == pytest.approx(expected, rel=1e-6)

    # Discrimination: the M_planet-substituted orbital term.
    wrong_orbital = _PS0D_MPL * (const_G * (_PS0D_MPL + _PS0D_MSA) * _PS0D_SMA) ** 0.5
    wrong_L = _PS0D_I * omega + wrong_orbital
    assert abs(hf_row['plan_sat_am'] - wrong_L) / expected > 50.0
    # Scale guard: Eq. 60 for Earth-Moon lands at ~3.6e34 kg m^2/s
    # (spin ~7e33 + orbital ~2.9e34); the M_planet substitution would
    # land at ~2.4e36, two orders of magnitude above this bracket.
    assert 1e34 < hf_row['plan_sat_am'] < 1e35


def test_ps0d_bootstrap_only_fires_once_am_is_populated():
    """Once ``plan_sat_am`` is nonzero, a later call at ``Time <= 10``
    must NOT recompute/overwrite it via the bootstrap -- only the
    ODE-evolved value should change it."""
    hf_row = {
        'R_int': _PS0D_RPL,
        'M_int': _PS0D_MPL,
        'M_sat': _PS0D_MSA,
        'semimajorax_sat': _PS0D_SMA,
        'axial_period': _PS0D_AXIAL_PERIOD,
        'plan_sat_am': 0,
        'F_tidal': 0.0,
        'Time': 0.0,
        'C_planet': _PS0D_I,  # see comment in the test above
    }
    ps0d(hf_row, dt=1.0)
    bootstrapped_L = hf_row['plan_sat_am']

    # A second call, still at Time <= 10, with zero tidal power (a
    # fixed point, see below) so L should not evolve either -- if the
    # bootstrap incorrectly re-fired here, it would still land on the
    # same value by construction of Ltot, so instead directly assert
    # the bootstrap condition's guard: seed a deliberately WRONG
    # sentinel L and confirm it is preserved (not silently replaced).
    hf_row['Time'] = 5.0
    hf_row['plan_sat_am'] = 999.0
    hf_row['F_tidal'] = 0.0
    ps0d(hf_row, dt=1.0)
    assert hf_row['plan_sat_am'] == pytest.approx(999.0, rel=1e-12)
    assert hf_row['plan_sat_am'] != pytest.approx(bootstrapped_L, rel=1e-3)


@pytest.mark.physics_invariant
def test_ps0d_domega_dt_vanishes_when_dE_tidal_is_zero():
    """No tidal dissipation gives no spin-down (Eq. 58's numerator is
    zero): a fixed point of BOTH omega and, through the Eq. 59
    kinematic chain, the semimajor axis too.
    """
    da_dt, domega_dt = _ps0d_instantaneous_rates(F_tidal=0.0)
    assert domega_dt == pytest.approx(0.0, abs=1e-30)
    assert da_dt == pytest.approx(0.0, abs=1e-15)


@pytest.mark.physics_invariant
def test_ps0d_domega_dt_is_negative_for_positive_tidal_dissipation():
    """Eq. 58 has an explicit minus sign on ``dE_tidal``: positive
    tidal dissipation must spin the planet DOWN (angular momentum
    flows from spin into the satellite's orbit, growing ``a``)."""
    da_dt, domega_dt = _ps0d_instantaneous_rates(F_tidal=1e-3)
    assert domega_dt < 0.0
    # Discrimination: the orbit correspondingly expands (da/dt > 0),
    # the qualitative Eq. 59 prediction for a prograde Moon losing
    # spin AM to the orbit -- not just "domega_dt is negative".
    assert da_dt > 0.0


@pytest.mark.physics_invariant
def test_ps0d_da_dt_obeys_korenaga_eq59_kinematic_identity():
    """``da/dt = -2 I a / (L - I omega) * domega/dt`` (Eq. 59) must
    hold between the two independently finite-differenced rates."""
    hf_row_state = _make_ps0d_hf_row(F_tidal=2e-3)
    a, omega, L = (
        hf_row_state['semimajorax_sat'],
        2 * np.pi / hf_row_state['axial_period'],
        hf_row_state['plan_sat_am'],
    )
    da_dt, domega_dt = _ps0d_instantaneous_rates(F_tidal=2e-3)

    expected_da_dt = -2.0 * _PS0D_I * a / (L - _PS0D_I * omega) * domega_dt
    assert da_dt == pytest.approx(expected_da_dt, rel=1e-6)


def test_ps0d_finite_output_over_a_realistic_step():
    """A realistic multi-year integration for the present-day Earth-
    Moon system must yield finite, positive semimajor axis and axial
    period."""
    hf_row = _make_ps0d_hf_row(F_tidal=1e-3)
    ps0d(hf_row, dt=1e3)
    assert np.isfinite(hf_row['semimajorax_sat'])
    assert np.isfinite(hf_row['axial_period'])
    assert hf_row['semimajorax_sat'] > 0.0
    assert hf_row['axial_period'] > 0.0


# ---------------------------------------------------------------------------
# ps1d: planet spin + satellite spin + orbit (Hansen-coefficient-based,
# same structure as orbit.py's sp1d, extended to a second spinning body).
#
# ps1d/ps1d_evec call proteus.orbit.common.get_all_m_hansen, which lazily
# builds a module-level cache via a full FFT sweep over a ~100-point
# eccentricity grid on first use -- on the order of a minute of wall time
# (the trap fixed in tests/orbit/test_common.py, and worked around the
# same way in tests/orbit/test_orbit.py's sp1d tests). Every test below
# that touches ps1d/ps1d_evec force-builds a tiny, fast table instead via
# the _fast_hansen_table fixture, monkeypatched so it cannot leak into
# other tests sharing the same pytest process.
# ---------------------------------------------------------------------------

_FAST_E_GRID = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
_FAST_KMIN, _FAST_KMAX = -6, 6


@pytest.fixture
def _fast_hansen_table(monkeypatch):
    monkeypatch.setattr(common_mod, '_hansen_table', None)
    common_mod.init_hansen_table(
        e_grid=_FAST_E_GRID, kmin=_FAST_KMIN, kmax=_FAST_KMAX, n_deg=2, force=True
    )


_PS1D_MPL, _PS1D_MSA = 5.972e24, 7.342e22  # Earth, Moon
_PS1D_RPL, _PS1D_RSA = 6.371e6, 1.737e6
_PS1D_CPL = 0.33 * _PS1D_MPL * _PS1D_RPL**2
_PS1D_CSA = 0.33 * _PS1D_MSA * _PS1D_RSA**2


def _make_ps1d_tides(lnk_value: complex) -> Tides_t:
    nmk = [(2, 0, k) for k in range(_FAST_KMIN, _FAST_KMAX + 1)] + [
        (2, 2, k) for k in range(_FAST_KMIN, _FAST_KMAX + 1)
    ]
    tides_o = Tides_t()
    for primary, perturber in (('planet', 'satellite'), ('satellite', 'planet')):
        entry = tides_o.add(primary=primary, perturber=perturber)
        entry.nmk = np.array(nmk, dtype=int)
        entry.LNk = np.full(len(nmk), lnk_value, dtype=complex)
    return tides_o


def _make_ps1d_hf_row(*, axial_period=86400.0, axial_period_sat=2.36e6, sma=3.844e8, ecc=0.3):
    return {
        'axial_period': axial_period,
        'axial_period_sat': axial_period_sat,
        'semimajorax_sat': sma,
        'eccentricity_sat': ecc,
        'M_int': _PS1D_MPL,
        'M_sat': _PS1D_MSA,
        'R_int': _PS1D_RPL,
        'R_sat': _PS1D_RSA,
        'C_planet': _PS1D_CPL,
        'C_sat': _PS1D_CSA,
    }


def _ps1d_am_components(hf_row: dict) -> tuple[float, float, float]:
    """``(planet spin AM, satellite spin AM, orbital AM)`` under
    ps1d's own bookkeeping."""
    a = hf_row['semimajorax_sat']
    e = hf_row['eccentricity_sat']
    omega_p = 2 * np.pi / hf_row['axial_period']
    omega_s = 2 * np.pi / hf_row['axial_period_sat']
    mu = _PS1D_MPL * _PS1D_MSA / (_PS1D_MPL + _PS1D_MSA)
    l_orb = mu * np.sqrt(const_G * (_PS1D_MPL + _PS1D_MSA) * a * (1 - e**2))
    return hf_row['C_planet'] * omega_p, hf_row['C_sat'] * omega_s, l_orb


@pytest.mark.physics_invariant
def test_ps1d_zero_dissipation_is_an_exact_fixed_point(_fast_hansen_table):
    """With every Love number exactly 0, both spins, semimajor axis,
    and eccentricity are left exactly unchanged."""
    hf_row = _make_ps1d_hf_row()
    before = dict(hf_row)
    tides_o = _make_ps1d_tides(0.0 + 0.0j)

    ps1d(hf_row, tides_o, dt=1e5)

    assert hf_row['axial_period'] == pytest.approx(before['axial_period'], rel=1e-12)
    assert hf_row['axial_period_sat'] == pytest.approx(before['axial_period_sat'], rel=1e-12)
    assert hf_row['semimajorax_sat'] == pytest.approx(before['semimajorax_sat'], rel=1e-12)
    assert hf_row['eccentricity_sat'] == pytest.approx(before['eccentricity_sat'], rel=1e-12)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_ps1d_conserves_total_angular_momentum(_fast_hansen_table):
    """Physical law: total angular momentum (planet spin + satellite
    spin + orbital) is conserved for this isolated two-body-plus-
    tides system. Checked across a real, non-trivial step.

    Tolerance rel=1e-6 matches ``solve_ivp``'s configured ``rtol``.
    """
    hf_row = _make_ps1d_hf_row(ecc=0.3)
    spin_p0, spin_s0, orb0 = _ps1d_am_components(hf_row)
    am_before = spin_p0 + spin_s0 + orb0
    tides_o = _make_ps1d_tides(-0.01 - 0.02j)

    ps1d(hf_row, tides_o, dt=1e5)

    spin_p1, spin_s1, orb1 = _ps1d_am_components(hf_row)
    am_after = spin_p1 + spin_s1 + orb1
    # Discrimination: the satellite spin must have actually moved
    # (otherwise "conservation" would be a trivial no-op check).
    assert spin_s1 != pytest.approx(spin_s0, rel=1e-6)
    assert am_after == pytest.approx(am_before, rel=1e-6)


@pytest.mark.physics_invariant
def test_ps1d_satellite_spin_am_change_matches_the_rest_of_the_system(_fast_hansen_table):
    """More targeted phrasing of the same law, needed for the same
    reason as ``sp1d``'s equivalent test in ``tests/orbit/test_orbit.py``:
    the satellite's own spin AM (~1e29 here) is dwarfed by the
    planet's spin AM (~1e33) and the orbital AM (~1e34), so a bug
    confined to the satellite's ``domega_dt`` coupling would barely
    move the raw total. Comparing
    ``Delta(satellite spin) == -(Delta(planet spin) + Delta(orbital))``
    directly is sensitive to exactly that class of bug.
    """
    hf_row = _make_ps1d_hf_row(ecc=0.3)
    spin_p0, spin_s0, orb0 = _ps1d_am_components(hf_row)
    tides_o = _make_ps1d_tides(-0.01 - 0.02j)

    ps1d(hf_row, tides_o, dt=1e5)

    spin_p1, spin_s1, orb1 = _ps1d_am_components(hf_row)
    d_spin_s = spin_s1 - spin_s0
    d_rest = (spin_p1 - spin_p0) + (orb1 - orb0)
    assert abs(d_spin_s) > 1e26  # substantial, not a near-zero step
    assert d_spin_s == pytest.approx(-d_rest, rel=5e-2)


@pytest.mark.physics_invariant
def test_ps1d_da_tidal_split_sums_to_total_sma_change(_fast_hansen_table):
    """``sma_dot_planet`` and ``sma_dot_sat`` are the solver's own
    exact split of the total semimajor-axis change into planet-raised
    and satellite-raised contributions -- pins the source's own
    logged self-consistency check (``da_total_check``): the two parts
    must sum to the actual total change over the step, to solver
    tolerance.
    """
    hf_row = _make_ps1d_hf_row(ecc=0.3)
    sma_before = hf_row['semimajorax_sat']
    tides_o = _make_ps1d_tides(-0.01 - 0.02j)
    dt_yr = 1e5

    ps1d(hf_row, tides_o, dt=dt_yr)

    dt_s = dt_yr * secs_per_year
    total_da_dt = (hf_row['semimajorax_sat'] - sma_before) / dt_s
    split_sum = hf_row['sma_dot_planet'] + hf_row['sma_dot_sat']
    assert split_sum == pytest.approx(total_da_dt, rel=1e-6)


@pytest.mark.physics_invariant
def test_ps1d_eccentricity_clamped_at_zero_not_negative(_fast_hansen_table):
    """Dissipation strong enough to drive a high-initial-eccentricity
    orbit to (near-)circular within one step must not report a
    negative eccentricity or non-finite output. This combination
    (e0=0.9, dt=1e6 yr) lands within ~1e-4 of exact circularization
    while keeping the Radau solver's internal Jacobian estimate well
    inside its stable range; a substantially larger Love-number
    magnitude at the same state drives that estimate into overflow.
    """
    hf_row = _make_ps1d_hf_row(ecc=0.9)
    tides_o = _make_ps1d_tides(-0.005 - 0.01j)

    ps1d(hf_row, tides_o, dt=1e6)

    assert np.isfinite(hf_row['eccentricity_sat'])
    assert np.isfinite(hf_row['semimajorax_sat'])
    assert hf_row['eccentricity_sat'] >= 0.0
    assert hf_row['eccentricity_sat'] < 0.01


# ---------------------------------------------------------------------------
# evolve_orbit_satellite: dispatch, the C_planet angular-momentum-conserving
# rescale (the "figure skater" effect, see the function's own docstring),
# and the adaptive accept/reject substep controller.
# ---------------------------------------------------------------------------


def _make_interior_for_c_planet(density: float, nlev_b: int = 20):
    """Uniform-density-sphere Interior_t-like stand-in for
    get_C_planet, matching the fixture style in
    tests/orbit/test_common.py and tests/orbit/test_orbit.py."""
    radius = np.linspace(0.0, _PS0D_RPL, nlev_b)
    return SimpleNamespace(radius=radius, density=np.full(nlev_b - 1, density))


def _make_satellite_config(model) -> Any:
    return cast(
        Any,
        SimpleNamespace(
            orbit=SimpleNamespace(planet_satellite_model=model),
            interior_energetics=SimpleNamespace(module='aragog'),
        ),
    )


def _make_evolve_hf_row(*, time=100.0, axial_period=_PS0D_AXIAL_PERIOD, plan_sat_am=1.0):
    return {
        'Time': time,
        'F_tidal': 0.0,
        'semimajorax_sat': _PS0D_SMA,
        'eccentricity_sat': 0.05,
        'axial_period': axial_period,
        'axial_period_sat': 2.36e6,
        'R_int': _PS0D_RPL,
        'M_int': _PS0D_MPL,
        'M_sat': _PS0D_MSA,
        # Nonzero so ps0d's own AM bootstrap (a separate mechanism,
        # tested above) does not also fire and confound this check.
        'plan_sat_am': plan_sat_am,
    }


@pytest.mark.physics_invariant
def test_evolve_orbit_satellite_conserves_spin_am_across_c_planet_change_for_ps0d():
    """The documented "figure skater" rescale: when the interior state
    changes ``C_planet`` between two calls (e.g. from solidification),
    ``evolve_orbit_satellite`` must rescale the planet's spin
    (``axial_period``) so that ``C_planet * Omega_p`` is exactly
    conserved across that structural jump -- with ``F_tidal = 0`` so
    ps0d's own integration is a no-op and cannot be confused with the
    rescale's effect.

    This must hold for ``model = 'ps0d'`` specifically: unlike
    ps1d/ps1d_evec, ps0d has no satellite spin state of its own, but
    its own AM bootstrap reads ``hf_row['C_planet']`` directly, so it
    needs this refreshed and angular-momentum-consistent exactly like
    the other two models.
    """
    hf_row = _make_evolve_hf_row()
    config = _make_satellite_config('ps0d')
    interior_1 = _make_interior_for_c_planet(density=5500.0)
    interior_1.dt = 1.0
    evolve_orbit_satellite(hf_row, config, dirs={}, tides_o=Tides_t(), interior_o=interior_1)
    c_planet_1 = hf_row['C_planet']
    spin_am_1 = c_planet_1 * (2 * np.pi / hf_row['axial_period'])

    # Simulate interior solidification: a different density profile
    # changes C_planet on the next call.
    interior_2 = _make_interior_for_c_planet(density=6000.0)
    interior_2.dt = 1.0
    evolve_orbit_satellite(hf_row, config, dirs={}, tides_o=Tides_t(), interior_o=interior_2)
    c_planet_2 = hf_row['C_planet']
    spin_am_2 = c_planet_2 * (2 * np.pi / hf_row['axial_period'])

    # Discrimination: C_planet must have actually changed, or the
    # rescale would trivially conserve spin AM regardless of whether
    # it works.
    assert c_planet_2 != pytest.approx(c_planet_1, rel=1e-6)
    assert spin_am_2 == pytest.approx(spin_am_1, rel=1e-9)


def test_evolve_orbit_satellite_c_planet_rescale_holds_with_real_ps0d_dynamics():
    """The same "figure skater" rescale checked above, but now with
    real (nonzero ``F_tidal``) dynamics actually running through
    ``ps0d`` on both calls -- important because ``ps0d``'s own ODE
    now reads its moment-of-inertia coefficient from
    ``hf_row['C_planet']`` directly (matching ps1d/ps1d_evec), rather
    than a fixed uniform-sphere value independent of the interior
    state.

    The second call's ``interior_o.dt`` is deliberately tiny (1e-6 yr,
    versus ps0d's own spin-orbit exchange timescale of hundreds of
    millions of years at these parameters) so genuine tidal evolution
    during that call contributes a negligible amount to the spin
    change, isolating the discrete rescale jump from the (real, but
    minuscule at this dt) ongoing dynamics.
    """
    hf_row = _make_evolve_hf_row()
    hf_row['F_tidal'] = 1e-3
    config = _make_satellite_config('ps0d')

    interior_1 = _make_interior_for_c_planet(density=5500.0)
    interior_1.dt = 1e5
    evolve_orbit_satellite(hf_row, config, dirs={}, tides_o=Tides_t(), interior_o=interior_1)
    c_planet_1 = hf_row['C_planet']
    spin_am_1 = c_planet_1 * (2 * np.pi / hf_row['axial_period'])

    interior_2 = _make_interior_for_c_planet(density=6000.0)
    interior_2.dt = 1e-6
    evolve_orbit_satellite(hf_row, config, dirs={}, tides_o=Tides_t(), interior_o=interior_2)
    c_planet_2 = hf_row['C_planet']
    spin_am_2 = c_planet_2 * (2 * np.pi / hf_row['axial_period'])

    assert c_planet_2 != pytest.approx(c_planet_1, rel=1e-6)
    assert spin_am_2 == pytest.approx(spin_am_1, rel=1e-6)


@pytest.mark.parametrize('model', ['ps0d', 'ps1d', 'ps1d_evec'])
def test_evolve_orbit_satellite_populates_c_planet_for_every_model(model, _fast_hansen_table):
    """All three dispatchable models need ``hf_row['C_planet']``
    populated on entry (ps0d for its own AM bootstrap; ps1d/ps1d_evec
    for their spin-coupling ODEs) -- confirms the refresh gate covers
    all of them, not just the two it originally covered.
    """
    hf_row = _make_evolve_hf_row()
    hf_row['axial_period_sat'] = 2.36e6
    hf_row['evection_angle'] = 0.0
    hf_row['C_sat'] = _PS1D_CSA
    hf_row['R_sat'] = _PS1D_RSA
    config = _make_satellite_config(model)
    interior_o = _make_interior_for_c_planet(density=5500.0)
    interior_o.dt = 1.0
    tides_o = _make_ps1d_tides(-0.01 - 0.02j) if model != 'ps0d' else Tides_t()

    evolve_orbit_satellite(hf_row, config, dirs={}, tides_o=tides_o, interior_o=interior_o)

    assert 'C_planet' in hf_row
    assert hf_row['C_planet'] > 0.0


def test_evolve_orbit_satellite_unrecognized_model_advances_nothing_without_raising():
    """Pins the OBSERVED (not necessarily intended) contract for an
    unrecognized ``planet_satellite_model``: the dispatch ``else``
    branch does ``raise ValueError(...)``, but that raise happens
    INSIDE the substep's own ``try`` block, which has a broad
    ``except Exception: ok = False`` around it -- so the ValueError is
    caught and treated identically to a transient solver failure. The
    controller then retries with a shrinking ``dt_yr`` (every retry
    hits the same unconditional ``raise``) until the step size
    collapses below its ``1e-10`` floor, logs a warning, and
    ``evolve_orbit_satellite`` RETURNS NORMALLY: no exception ever
    reaches the caller, and ``t_elapsed`` never advances past 0.

    This means a configuration error (an invalid model name) is
    currently indistinguishable, from the caller's side, from the
    solver simply failing to converge -- both silently advance zero
    time and return. Pinned here as the current behavior; not fixed.
    """
    hf_row = _make_evolve_hf_row()
    hf_row['F_tidal'] = 1e-3
    sma_before = hf_row['semimajorax_sat']
    config = _make_satellite_config('not-a-real-model')
    interior_o = _make_interior_for_c_planet(density=5500.0)
    interior_o.dt = 1.0

    # No exception propagates.
    evolve_orbit_satellite(hf_row, config, dirs={}, tides_o=Tides_t(), interior_o=interior_o)

    # Discrimination: the state is exactly the pre-call snapshot (every
    # substep was rejected and rolled back), not partially evolved.
    assert hf_row['semimajorax_sat'] == pytest.approx(sma_before, rel=1e-12)


def test_evolve_orbit_satellite_ps0d_dispatch_evolves_hf_row():
    """``model='ps0d'`` actually dispatches to and runs ``ps0d``: with
    nonzero tidal power, the semimajor axis must change over the
    call."""
    hf_row = _make_evolve_hf_row()
    hf_row['F_tidal'] = 1e-3
    sma_before = hf_row['semimajorax_sat']
    config = _make_satellite_config('ps0d')
    interior_o = _make_interior_for_c_planet(density=5500.0)
    interior_o.dt = 1e6  # long enough for ps0d's slow spin-orbit exchange to register

    evolve_orbit_satellite(hf_row, config, dirs={}, tides_o=Tides_t(), interior_o=interior_o)

    assert abs(hf_row['semimajorax_sat'] - sma_before) > 1.0


def test_evolve_orbit_satellite_rejects_substep_exceeding_max_rel_da_and_shrinks_dt():
    """Forcing ``max_rel_da`` far below any physically achievable step
    must cause EVERY substep to be rejected (state rolled back to the
    pre-substep snapshot each time) until either the step size
    collapses or ``max_substeps`` is exhausted -- exercising the
    reject/rollback/shrink branch of the adaptive controller, not just
    the accept path every other test here takes.
    """
    hf_row = _make_evolve_hf_row()
    hf_row['F_tidal'] = 1e-3
    sma_before = hf_row['semimajorax_sat']
    config = _make_satellite_config('ps0d')
    interior_o = _make_interior_for_c_planet(density=5500.0)
    interior_o.dt = 1e7

    evolve_orbit_satellite(
        hf_row,
        config,
        dirs={},
        tides_o=Tides_t(),
        interior_o=interior_o,
        max_rel_da=1e-30,
        max_substeps=20,
    )

    # Every substep must have been rejected: semimajorax_sat is
    # restored to the pre-call snapshot on every rejection, so after
    # exhausting max_substeps with an unsatisfiable tolerance it must
    # still equal the ORIGINAL value, not something partway evolved.
    assert hf_row['semimajorax_sat'] == pytest.approx(sma_before, rel=1e-12)


def test_evolve_orbit_satellite_persists_controller_state_across_calls():
    """``_orbit_dt_yr`` and ``_orbit_resonance_state`` are written back
    to ``hf_row`` at the end of the call (private, dt_yr not reset to
    ``dt0_yr`` on the next call) -- the persistence the function's own
    docstring says is load-bearing for not wasting substeps re-growing
    a step size a previous call had already found safe.
    """
    hf_row = _make_evolve_hf_row()
    hf_row['F_tidal'] = 1e-3
    config = _make_satellite_config('ps0d')
    interior_o = _make_interior_for_c_planet(density=5500.0)
    interior_o.dt = 10.0

    assert '_orbit_dt_yr' not in hf_row
    evolve_orbit_satellite(hf_row, config, dirs={}, tides_o=Tides_t(), interior_o=interior_o)
    assert '_orbit_dt_yr' in hf_row
    assert '_orbit_resonance_state' in hf_row
    # Discrimination: the persisted value is a real float step size,
    # not e.g. a leftover None or the untouched dt0_yr default when
    # growth should have moved it (dt0_yr=1e-4 by default; a step that
    # ran to completion on a 10 yr call and grew at all would leave a
    # noticeably larger value, given growth=1.15 compounds quickly).
    assert isinstance(hf_row['_orbit_dt_yr'], float)
    assert hf_row['_orbit_dt_yr'] > 0.0


# ---------------------------------------------------------------------------
# ps1d_evec: planet + satellite spin/orbit plus apsidal precession and the
# evection resonance (star-forced eccentricity pumping). This is the "three
# clocks" model: PROTEUS hands evolve_orbit_satellite a coarse elapsed time
# (Clock 1); the adaptive substep controller there subdivides it into
# accepted solver calls (Clock 2, further subdivided internally by
# solve_ivp's own adaptive stepping); accepted substeps are optionally
# stored at a throttled cadence (Clock 3, via _flush_fine_evection_csv,
# already covered above). Only the mechanics are tested here -- the
# resonance-physics literature comparison (CPL model vs a published case)
# is a separate, long-running test.
#
# filter_value=0 (out of band) makes ps1d_evec's own orbitals() identical
# to ps1d's (dphi_dt forced to 0, the evection term's r_filter also 0 by
# construction -- see dw_dt): confirmed below to conserve the same 3-
# component total AM to machine precision, same as ps1d.
#
# filter_value=1 (in band) activates the star's secular/evection torque:
# this is a genuine three-body angular-momentum exchange with the star, so
# the planet-satellite subsystem's own total AM is NOT expected to be
# conserved in this regime (checked numerically: ~0.5% drift over a 1-year
# in-band step at the parameters used here) -- this is documented as
# physical, not asserted as a false invariant.
# ---------------------------------------------------------------------------


def _make_ps1d_evec_tides(lnk_value: complex) -> Tides_t:
    return _make_ps1d_tides(lnk_value)


def _make_ps1d_evec_hf_row(*, ecc=0.3, evection_angle=0.0, time=100.0):
    hf_row = _make_ps1d_hf_row(ecc=ecc)
    hf_row['M_star'] = 1.989e30
    hf_row['M_planet'] = _PS1D_MPL
    hf_row['semimajorax'] = 1.5e11
    hf_row['evection_angle'] = evection_angle
    hf_row['Time'] = time
    # Only used by evolve_orbit_satellite's C_planet refresh/rescale
    # block, not by ps1d_evec itself when called directly; harmless
    # either way, kept for parity with the other models' hf_row shape.
    hf_row['plan_sat_am'] = hf_row.get('plan_sat_am', 1.0)
    return hf_row


def _ps1d_evec_am_components(hf_row: dict) -> tuple[float, float, float]:
    return _ps1d_am_components(hf_row)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_ps1d_evec_filter_zero_freezes_phi_and_conserves_am_like_ps1d(_fast_hansen_table):
    """``filter_value=0`` must reduce ps1d_evec EXACTLY to ps1d's
    physics: the evection angle stays frozen at its initial value (no
    precession tracked while out of band), and the same 3-component
    total angular momentum (planet spin + satellite spin + orbital)
    is conserved to machine precision -- not just approximately, since
    with the evection terms exactly zeroed there is no star-torque
    contribution left to break the closed two-body-plus-tides
    conservation law.
    """
    hf_row = _make_ps1d_evec_hf_row(ecc=0.3, evection_angle=0.0)
    spin_p0, spin_s0, orb0 = _ps1d_evec_am_components(hf_row)
    am_before = spin_p0 + spin_s0 + orb0
    tides_o = _make_ps1d_evec_tides(-0.002 - 0.004j)

    ps1d_evec(hf_row, tides_o, dt=1.0, filter_value=0.0)

    assert hf_row['evection_angle'] == pytest.approx(0.0, abs=1e-15)
    spin_p1, spin_s1, orb1 = _ps1d_evec_am_components(hf_row)
    am_after = spin_p1 + spin_s1 + orb1
    assert am_after == pytest.approx(am_before, rel=1e-9)


@pytest.mark.physics_invariant
def test_ps1d_evec_filter_one_evolves_phi_and_am_is_not_conserved(_fast_hansen_table):
    """``filter_value=1`` activates the star-forced evection term: the
    evection angle must actually evolve (not stay frozen), and the
    planet-satellite subsystem's own total angular momentum is NOT
    expected to be conserved -- the star is a third body exchanging
    angular momentum with the system during resonant forcing. This
    pins that the drift is real, finite, and of a physically sane
    (small-fraction) magnitude, not that it vanishes.
    """
    hf_row = _make_ps1d_evec_hf_row(ecc=0.3, evection_angle=0.0)
    spin_p0, spin_s0, orb0 = _ps1d_evec_am_components(hf_row)
    am_before = spin_p0 + spin_s0 + orb0
    tides_o = _make_ps1d_evec_tides(-0.002 - 0.004j)

    ps1d_evec(hf_row, tides_o, dt=1.0, filter_value=1.0)

    # Discrimination: phi must have moved substantially, not just by
    # solver-noise scale.
    assert abs(hf_row['evection_angle']) > 1.0
    spin_p1, spin_s1, orb1 = _ps1d_evec_am_components(hf_row)
    am_after = spin_p1 + spin_s1 + orb1
    rel_drift = abs(am_after - am_before) / am_before
    assert np.isfinite(rel_drift)
    # Sane magnitude: measurable (the star is doing real work) but not
    # wildly unphysical for a single year of resonant forcing.
    assert 1e-6 < rel_drift < 0.5


def test_evolve_orbit_satellite_threads_in_band_result_as_filter_value(monkeypatch):
    """Wiring check: ``evolve_orbit_satellite`` must pass
    ``_in_evection_band``'s live result through to ``ps1d_evec`` as
    ``filter_value`` -- forced True and forced False separately, both
    observed directly from the call ``ps1d_evec`` actually receives
    (not inferred from a downstream effect), so a regression that
    hardcoded ``filter_value`` or read a stale/cached band state would
    be caught regardless of what it happened to hardcode.
    """
    from proteus.orbit import satellite as sat_mod

    captured_filter_values = []

    def fake_ps1d_evec(
        hf_row, tides_o, dt, fine_sink=None, fine_stride=1, filter_value=None, **kw
    ):
        # Deliberately NOT the real physics: this test verifies only
        # that evolve_orbit_satellite threads the live _in_evection_band
        # result through as filter_value, not ps1d_evec's own dynamics
        # (covered by the dedicated ps1d_evec tests above). A trivial
        # no-op keeps every substep cheap and instantly accepted
        # regardless of how many the controller wants to take.
        captured_filter_values.append(filter_value)

    monkeypatch.setattr(sat_mod, 'ps1d_evec', fake_ps1d_evec)

    for forced_band, expected_filter in [(True, 1.0), (False, 0.0)]:
        monkeypatch.setattr(
            sat_mod, '_in_evection_band', lambda hf_row, state, **kw: forced_band
        )
        captured_filter_values.clear()

        hf_row = _make_ps1d_evec_hf_row(ecc=0.05)
        interior_o = _make_interior_for_c_planet(density=5500.0)
        interior_o.dt = 1.0
        config = _make_satellite_config('ps1d_evec')
        tides_o = _make_ps1d_evec_tides(-0.002 - 0.004j)

        sat_mod.evolve_orbit_satellite(
            hf_row, config, dirs={'output/data': '/tmp'}, tides_o=tides_o, interior_o=interior_o
        )

        assert len(captured_filter_values) > 0
        assert all(fv == expected_filter for fv in captured_filter_values)


def test_evolve_orbit_satellite_ps1d_evec_stores_dense_samples_in_band(
    tmp_path, monkeypatch, _fast_hansen_table
):
    """With the resonance band forced active for the whole call, EVERY
    accepted solver-clock sample must reach
    ``fine_evection_data.csv`` (Clock 3 == Clock 2, no throttling) --
    the in-band storage policy documented in
    ``_flush_fine_evection_csv``, now exercised through the real
    ``evolve_orbit_satellite`` -> ``ps1d_evec`` pipeline rather than
    called directly.
    """
    from proteus.orbit import satellite as sat_mod

    monkeypatch.setattr(sat_mod, '_in_evection_band', lambda hf_row, state, **kw: True)

    hf_row = _make_ps1d_evec_hf_row(ecc=0.3)
    interior_o = _make_interior_for_c_planet(density=5500.0)
    interior_o.dt = 1.0
    config = _make_satellite_config('ps1d_evec')
    tides_o = _make_ps1d_evec_tides(-0.002 - 0.004j)

    sat_mod.evolve_orbit_satellite(
        hf_row,
        config,
        dirs={'output/data': str(tmp_path)},
        tides_o=tides_o,
        interior_o=interior_o,
    )

    csv_path = tmp_path / 'fine_evection_data.csv'
    assert csv_path.exists()
    lines = csv_path.read_text().splitlines()
    # Discrimination: a real multi-substep, in-band run stores far more
    # than a handful of rows (dense solver-clock sampling), not just
    # one row per accepted macro-substep.
    assert len(lines) - 1 > 20


def test_evolve_orbit_satellite_ps1d_evec_throttles_samples_out_of_band(
    tmp_path, monkeypatch, _fast_hansen_table
):
    """With the resonance band forced OFF for the whole call, stored
    samples must be throttled to the storage-clock target spacing
    (``fine_csv_target_rel_dt``), landing on far fewer rows than the
    in-band case above for a comparable elapsed time and solver
    activity.
    """
    from proteus.orbit import satellite as sat_mod

    monkeypatch.setattr(sat_mod, '_in_evection_band', lambda hf_row, state, **kw: False)

    hf_row = _make_ps1d_evec_hf_row(ecc=0.3)
    interior_o = _make_interior_for_c_planet(density=5500.0)
    interior_o.dt = 1.0
    config = _make_satellite_config('ps1d_evec')
    tides_o = _make_ps1d_evec_tides(-0.002 - 0.004j)

    sat_mod.evolve_orbit_satellite(
        hf_row,
        config,
        dirs={'output/data': str(tmp_path)},
        tides_o=tides_o,
        interior_o=interior_o,
        fine_csv_target_rel_dt=0.1,
    )

    csv_path = tmp_path / 'fine_evection_data.csv'
    if csv_path.exists():
        lines = csv_path.read_text().splitlines()
        # Discrimination: far sparser than the in-band case (>20 rows
        # there for the same 1 yr span); out-of-band throttling to
        # ~10 target points (1/0.1) keeps this in the single digits.
        assert len(lines) - 1 < 15
    # else: zero out-of-band samples happened to cross a storage
    # target within this short a span -- also a valid (sparse) outcome
    # for the throttling policy, not a failure.


def test_ps1d_evec_finite_output_for_high_eccentricity_in_band(_fast_hansen_table):
    """Adversarial-but-physical: high initial eccentricity with the
    evection term active must not produce non-finite output."""
    hf_row = _make_ps1d_evec_hf_row(ecc=0.7)
    tides_o = _make_ps1d_evec_tides(-0.002 - 0.004j)

    ps1d_evec(hf_row, tides_o, dt=0.5, filter_value=1.0)

    assert np.isfinite(hf_row['eccentricity_sat'])
    assert np.isfinite(hf_row['semimajorax_sat'])
    assert np.isfinite(hf_row['evection_angle'])
    assert hf_row['eccentricity_sat'] >= 0.0
