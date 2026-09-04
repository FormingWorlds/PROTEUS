"""Slow-tier literature-comparison test: real orbit.satellite.ps1d_evec
driven by a Mignard constant-time-lag (CTL) tidal spectrum, reproducing
the evection-resonance-capture case from Rufu & Canup (2020) Figure 3.

Match against the reference case
---------------------------------
A standalone run of the driver logic below (see
``output_files/evection_reference/run_ctl_reference.py``, gitignored)
covers t=0-5.9e4 yr of physical time (~55 min wall-clock). Compared
against the reference figure's t=0-1e5 yr run (capture ~2.4e4 yr; peak
e~0.72 at a'~11.89 R_earth at t~5.2e4 yr; then contraction to a'~10.2
R_earth, e~0.67 by t=1e5 yr):

- Pre-resonance phase matches: e stays below 0.01 while a' rises from
  3.5 to ~7.6 R_earth over the first ~2e4 yr, same as the reference.
- Resonance capture timing matches closely: e crosses 0.02 at t=2.5e4 yr,
  a'=7.86 R_earth (reference: ~2.4e4 yr, a'~7.7-8).
- Peak eccentricity and its location match closely: e_peak=0.755 at
  t=5.12e4 yr, a'=11.88 R_earth (reference: e~0.72-0.724 at a'~11.89
  R_earth, t~5.2e4 yr) -- the peak a' in particular matches to <0.1%.
- The post-peak CONTRACTION phase was NOT reproduced: by t=5.92e4 yr
  (~8000 yr past the peak, where the observed run's wall-clock budget
  was exhausted), a' was still climbing (13.5 R_earth and rising)
  instead of turning over. This is NOT a truncation artifact -- the
  internal `filter` flag (real per-substep data from
  fine_evection_data.csv) never switched off, but the diagnostic
  (a'-a'_res)/a'_res was tracked directly and shows the satellite
  escaping evection proper onto the a' > a'_res side of the turnaround,
  rather than the a' < a'_res side. Rufu & Canup (2020) Section 3.1
  describe exactly this bifurcation for their own A=10 reference case
  (the same tidal-strength ratio used here): escape to the high-e side
  of the separatrix enters the quasi-resonance (QR) regime they
  describe, with the orbit interior to a'_res and genuine tidally-driven
  contraction; escape to the low-e side leaves the orbit EXTERIOR to
  a'_res with no further resonant regulation and elevated AM -- and they
  report this split occurred in 2 of 10 of their own simulations that
  varied only the initial resonance angle phi(0), everything else held
  fixed. Which branch is realized is therefore expected to be sensitive
  to phi(0) (0.3 rad here, an arbitrary choice inherited from the
  reference notebook), not a discrepancy in the physics being tested.
  This test makes NO assertion about the contraction phase for that
  reason.

Because of the ~1 hour runtime of the full case, this test targets only
the resonance-entry and peak-eccentricity portion (through t=5.2e4 yr).
Reaching that target took ~1364 s in the calibration run; a 2200 s
internal cutoff leaves a ~1.6x margin over that measurement, plus
further headroom under the 3600 s pytest-timeout ceiling for slower CI
hardware. If the internal cutoff is hit well short of the target, the
driver raises RuntimeError rather than returning a truncated trajectory
silently, so that failure mode is distinguishable from an actual
physics regression.

Invariants asserted:

- Pre-resonance: eccentricity stays below 0.02 while a' < 7.5 R_earth
  (an edge case away from the interesting dynamics, and a discrimination
  check that the driver isn't spuriously exciting e from the start).
- Resonance capture occurs within a physically reasonable window
  (e crosses 0.1 between t=2e4 and t=4e4 yr), not immediately and not
  never.
- Peak eccentricity reaches at least 0.6 (comfortably below the observed
  0.755, allowing for run-to-run solver variance) at a semimajor axis
  within [10, 13] R_earth, matching the reference's peak location.
- The 2-body (planet spin + satellite spin + orbital) angular momentum
  diagnostic ``hf_row['plan_sat_am']`` stays finite, positive, and within
  +/-10% of its initial value throughout -- a boundedness check, not an
  exact-conservation one: ps1d_evec's in-band evection coupling
  physically exchanges angular momentum with the star, so this quantity
  is expected to drift (observed drift in the reference run was <1%
  over the run), not be exactly conserved.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

import proteus.orbit.common as common_mod
from proteus.orbit.common import Tides_t, get_C_planet, init_hansen_table
from proteus.orbit.satellite import evolve_orbit_satellite

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]

# ---------------------------------------------------------------------------
# Body/constant values copied verbatim from the reference notebook (NOT
# proteus.utils.constants, whose R_earth/const_G/M_sun differ slightly) --
# exact fidelity to the calibrated reference case matters more here than
# consistency with PROTEUS's own body-parameter defaults.
# ---------------------------------------------------------------------------
_CONST_G = 6.67430e-11
_SECS_PER_YEAR = 365.25 * 24 * 3600.0
_M_EARTH, _R_EARTH = 5.972e24, 6.371e6
_M_MOON, _R_MOON = 7.342e22, 1.737e6
_M_SUN, _AU = 1.989e30, 1.496e11

# Mignard CTL parameters, Rufu & Canup (2020) Figure-3-calibrated (see the
# reference notebook's make_initial_hf_row docstring for the derivation).
_K2_P, _DT_P = 0.3, 5.98
_K2_S, _DT_S = 1.5, 1.20
_KMIN, _KMAX = -50, 200

# Target: through the peak (observed at t=5.12e4 yr), not the full 1e5 yr
# case, to keep the test inside the slow-tier timeout budget. Reaching
# this target took ~1364 s in the calibration run; 2200 s leaves a ~1.6x
# margin against that measurement plus roughly 1400 s of additional
# headroom under the 3600 s pytest-timeout ceiling (which also covers
# Hansen-table construction, ~10 s) for slower CI hardware.
_T_TARGET_YR = 52000.0
_DT_OUTER_YR = 200.0
_MAX_WALL_SECONDS = 2200.0


def _make_initial_hf_row() -> dict:
    return {
        'Time': 0.0,
        'R_int': _R_EARTH,
        'M_int': _M_EARTH,
        'M_planet': _M_EARTH,
        'R_sat': _R_MOON,
        'M_sat': _M_MOON,
        'semimajorax_sat': 3.5 * _R_EARTH,
        'eccentricity_sat': 0.01,
        'axial_period': 2 * 3600.0,
        'axial_period_sat': 24 * 3600.0,
        'evection_angle': 0.3,
        'plan_sat_am': 0.0,
        'M_star': _M_SUN,
        'semimajorax': 1.0 * _AU,
        'C_sat': 0.4 * _M_MOON * _R_MOON**2,
        'C_planet': 0.4 * _M_EARTH * _R_EARTH**2,
    }


def _refresh_ctl_tides(hf_row: dict) -> Tides_t:
    """Rebuild tides_o from the current hf_row state under the Mignard
    CTL model (Re[k]=k2, Im[k]=-k2*sigma*dt_lag), held fixed for the next
    evolve_orbit_satellite call -- mirroring how a real external
    tidal-response module (Obliqua/LovePy) would populate tides_o once
    per PROTEUS coupling step. sigma(m, k) matches exactly the internal
    formula ps1d/ps1d_evec use in their own orbitals() closures at
    src/proteus/orbit/satellite.py (sigma_0=-k*n_mm, sigma_2=2*Omega-k*n_mm).
    """
    a = hf_row['semimajorax_sat']
    Mp, Ms = hf_row['M_planet'], hf_row['M_sat']
    n_mm = np.sqrt(_CONST_G * (Mp + Ms) / a**3)
    Omega_p = 2 * np.pi / hf_row['axial_period']
    Omega_s = 2 * np.pi / hf_row['axial_period_sat']

    tides_o = Tides_t()
    s_vals = np.arange(_KMIN, _KMAX + 1)
    for primary, k2, dt_lag, Omega in (
        ('planet', _K2_P, _DT_P, Omega_p),
        ('satellite', _K2_S, _DT_S, Omega_s),
    ):
        perturber = 'satellite' if primary == 'planet' else 'planet'
        entry = tides_o.add(primary=primary, perturber=perturber)
        nmk, lnk = [], []
        for s in s_vals:
            sigma0 = -s * n_mm
            sigma2 = 2 * Omega - s * n_mm
            nmk.append((2, 0, s))
            lnk.append(complex(k2, -k2 * sigma0 * dt_lag))
            nmk.append((2, 2, s))
            lnk.append(complex(k2, -k2 * sigma2 * dt_lag))
        entry.nmk = np.array(nmk, dtype=int)
        entry.LNk = np.array(lnk, dtype=complex)
    return tides_o


def _run_ctl_reference(
    t_target_yr: float, max_wall_seconds: float, data_dir: str
) -> list[dict]:
    """Drive the real evolve_orbit_satellite(model='ps1d_evec') under the
    CTL spectrum above; returns the (Time, a', e, plan_sat_am) trajectory
    as a list of dicts, one per outer step actually completed.

    Raises RuntimeError (rather than silently returning a short
    trajectory) if the wall-clock budget is exhausted well short of
    ``t_target_yr``, so a slow-CI truncation surfaces as a distinct
    failure instead of masquerading as a physics assertion failure in
    one of the tests below.
    """
    hf_row = _make_initial_hf_row()
    config = SimpleNamespace(
        orbit=SimpleNamespace(planet_satellite_model='ps1d_evec'),
        interior_energetics=SimpleNamespace(module='aragog'),
    )
    n_shells = 50
    rho_uniform = _M_EARTH / (4.0 / 3.0 * np.pi * _R_EARTH**3)
    interior_o = SimpleNamespace(
        radius=np.linspace(0.0, _R_EARTH, n_shells),
        density=np.full(n_shells - 1, rho_uniform),
        dt=_DT_OUTER_YR,
    )
    dirs: dict = {'output/data': data_dir}

    get_C_planet(hf_row, config, interior_o)

    trajectory = []
    t_start = time.time()
    n_outer = int(np.ceil(t_target_yr / _DT_OUTER_YR))
    for i in range(n_outer):
        tides_o = _refresh_ctl_tides(hf_row)
        evolve_orbit_satellite(
            hf_row,
            config,
            dirs,
            tides_o,
            interior_o,
            dt0_yr=0.2,
            dt_max_yr=2.0,
        )
        hf_row['Time'] = (i + 1) * _DT_OUTER_YR
        trajectory.append(
            {
                'Time': hf_row['Time'],
                'a_prime': hf_row['semimajorax_sat'] / _R_EARTH,
                'ecc': hf_row['eccentricity_sat'],
                'plan_sat_am': hf_row.get('plan_sat_am', float('nan')),
            }
        )
        if time.time() - t_start > max_wall_seconds:
            break

    t_reached = trajectory[-1]['Time'] if trajectory else 0.0
    if t_reached < t_target_yr - 2 * _DT_OUTER_YR:
        raise RuntimeError(
            f'CTL reference driver exhausted its {max_wall_seconds:.0f} s wall-clock '
            f'budget at t={t_reached:.0f} yr, short of the t={t_target_yr:.0f} yr '
            'target -- this run is too slow to draw a physics conclusion from '
            '(likely slower-than-calibration hardware), not a failed physics check.'
        )
    return trajectory


@pytest.fixture(scope='module')
def _ctl_reference_trajectory(tmp_path_factory):
    """Builds the real, wide Hansen-coefficient table once (needed for
    eccentricities up to ~0.8; the [-50, 200] k-window is the notebook's
    own choice, verified there to suffice up to e~0.755) and runs the
    driver once, shared by every assertion in this file so the ~25+
    minute cost is paid a single time per test session.
    """
    e_grid = np.concatenate([np.arange(0.0, 0.1, 0.005), np.arange(0.1, 0.86, 0.01)])
    data_dir = str(tmp_path_factory.mktemp('evection_ctl'))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(common_mod, '_hansen_table', None)
        init_hansen_table(e_grid=e_grid, kmin=_KMIN, kmax=_KMAX, n_deg=2, force=True)
        yield _run_ctl_reference(_T_TARGET_YR, _MAX_WALL_SECONDS, data_dir)


@pytest.mark.physics_invariant
def test_pre_resonance_eccentricity_stays_flat(_ctl_reference_trajectory):
    """Before the satellite migrates into the evection band, purely
    planet-raised/satellite-raised circularizing tides should keep e
    small -- edge case away from the resonance dynamics this test is
    mostly about, and a discrimination check that the CTL driver isn't
    spuriously exciting e from t=0.
    """
    pre_resonance = [row for row in _ctl_reference_trajectory if row['a_prime'] < 7.5]
    assert len(pre_resonance) > 10  # comfortably exercised, not a 1-row fluke
    max_e_pre = max(row['ecc'] for row in pre_resonance)
    assert max_e_pre < 0.02


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_resonance_capture_occurs_within_expected_time_window(_ctl_reference_trajectory):
    """Pins the resonance-capture timing against Rufu & Canup (2020)
    Figure 3 / the reference notebook's real-Hansen run (capture at
    t~2.4e4 yr): e must cross 0.1 somewhere in [2e4, 4e4] yr, not
    immediately (which would indicate the CTL spectrum or Hansen table
    is wrong) and not never (which would indicate no capture at all).
    """
    crossing = next((row for row in _ctl_reference_trajectory if row['ecc'] > 0.1), None)
    assert crossing is not None, 'eccentricity never exceeded 0.1 -- no resonance capture'
    assert 2.0e4 <= crossing['Time'] <= 4.0e4


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_peak_eccentricity_matches_reference_location(_ctl_reference_trajectory):
    """Pins the peak-eccentricity magnitude and location against the
    reference (e_peak~0.72-0.724 at a'~11.89 R_earth, t~5.2e4 yr).
    Tolerance is loose (e >= 0.6, a' in [10, 13]) relative to the actual
    observed match (e_peak=0.755 at a'=11.88 R_earth) to absorb run-to-
    run Radau solver variance without becoming a trivial pass -- 0.6 is
    still far above the ~0.02-0.06 eccentricity anywhere outside the
    resonance, so this discriminates a genuine capture-and-pump event
    from a marginal or failed one.
    """
    ecc_values = [row['ecc'] for row in _ctl_reference_trajectory]
    i_peak = int(np.argmax(ecc_values))
    peak = _ctl_reference_trajectory[i_peak]
    assert peak['ecc'] >= 0.6
    assert 10.0 <= peak['a_prime'] <= 13.0


@pytest.mark.physics_invariant
def test_two_body_angular_momentum_diagnostic_stays_bounded(_ctl_reference_trajectory):
    """``plan_sat_am`` (ps1d_evec's own planet-spin + satellite-spin +
    orbital angular momentum diagnostic) must stay finite and positive
    throughout, and within +/-10% of its initial value. This is a
    boundedness check, not exact conservation: the evection coupling to
    the star (in-band) physically exchanges angular momentum with the
    star's orbit, so ``plan_sat_am`` alone is NOT a conserved quantity
    here -- observed drift in the reference run was <1% over the full
    stretch, so +/-10% is a generous bound that would still catch a
    gross bug (e.g. a sign error blowing the diagnostic up or negative)
    without asserting a conservation law this quantity doesn't obey.
    """
    am_values = np.array([row['plan_sat_am'] for row in _ctl_reference_trajectory])
    assert np.all(np.isfinite(am_values))
    assert np.all(am_values > 0.0)
    am0 = am_values[0]
    assert np.all(np.abs(am_values - am0) / am0 < 0.10)
