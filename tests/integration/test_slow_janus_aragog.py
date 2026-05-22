"""Slow-tier integration test: real JANUS atmosphere coupled to real
Aragog interior.

JANUS (1D convective atmosphere, Python wrapper around SOCRATES
radiative transfer) is the only atmosphere wrapper in PROTEUS that
is currently 100% uncovered at the slow tier. This test closes the
gap by booting JANUS end-to-end alongside the production Aragog
interior. Outgas stays on calliope (the production default outgas
backend); star, orbit, escape, atmos_chem stay on dummy backends
so the test isolates the atmosphere + interior coupling boundary.

Differs from ``test_slow_aragog_calliope.py`` in atmosphere only:
that test uses ``atmos_clim='dummy'`` so the F_atm = sigma * T_surf
* (1 - gamma) grey-opacity stub bypasses radiative transfer. This
file boots SOCRATES via JANUS and computes F_atm from a real
spectral solution.

Invariants asserted:

- Helpfile has at least 2 rows.
- Discrimination guard: atmos_clim.module is 'janus' (not
  dummy / agni fallback).
- F_atm finite and positive at every row (the planet radiates).
- T_surf in [200, 4000] K (the JANUS solver's accepted range under
  the dummy.toml IC).
- TOA optical depth < surface optical depth (radiation thins
  upward; same monotonicity invariant the AGNI pair tests assert
  synthetically).
- Per-element mass closure for H, C, N, S, O at the final row
  within rel=1e-2 (the conservation invariant from the aragog +
  calliope pair).
- ``Phi_global`` in [0, 1].
- Cross-step continuity on T_magma (|dT_magma| < 1000 K) and
  Phi_global (|dPhi| < 0.5).
- Cross-cutting mass + stability helpers.

Runtime budget: ~3-5 min macOS GHA (SOCRATES init + JANUS Python
per-step solve is light), ~5-10 min Linux GHA (CVode setup tax on
Aragog dominates JANUS-side cost). The 3600 s timeout sits well
inside the slow-tier 120 min step cap.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.integration.conftest import (
    validate_mass_conservation,
    validate_stability,
)

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


@pytest.mark.slow
@pytest.mark.physics_invariant
def test_janus_aragog_two_timesteps(proteus_multi_timestep_run):
    """Two-step PROTEUS run with real JANUS atmosphere + real Aragog
    interior on the Earth-IC fiducial.

    Physical scenario: 1 M_Earth, 0.5 AU, IW+2 fO2 shift, 3000 ppmw
    H budget (from ``input/dummy.toml``). JANUS solves the 1D
    convective atmosphere via SOCRATES radiative transfer; Aragog
    steps the entropy ODE on the mantle (backend='jax'); calliope
    partitions volatiles. The atmosphere boundary couples to the
    interior through hf_row['F_atm'] (radiative-output flux) and
    hf_row['T_surf'] (boundary T at the bottom of the atmosphere).

    Verifies:

    - At least 2 helpfile rows.
    - JANUS module is on (discrimination guard against fallback).
    - F_atm finite and positive at every row.
    - T_surf in [200, 4000] K at every row.
    - TOA tau < surface tau (radiation thins outward).
    - Per-element mass closure within rel=1e-2.
    - Phi_global in [0, 1].
    - Cross-step T_magma and Phi continuity.
    - Cross-cutting mass + stability helpers.
    """
    # interior_struct stays on the dummy module; melting_dir must be
    # set explicitly because Aragog's legacy fallback EOS path reads
    # the Monteux-600 melting curve folder name from there. Mirrors
    # the override in ``test_slow_aragog_calliope.py``.
    runner = proteus_multi_timestep_run(
        config_path='input/dummy.toml',
        num_timesteps=2,
        max_time=1e3,
        min_time=1e2,
        atmos_clim__module='janus',
        interior_energetics__module='aragog',
        interior_struct__melting_dir='Monteux-600',
        outgas__module='calliope',
    )

    hf = runner.hf_all
    assert hf is not None, 'helpfile should be created'
    assert len(hf) >= 2, f'expected >= 2 rows, got {len(hf)}'

    # Discrimination guard: JANUS module ran, not dummy.
    assert runner.config.atmos_clim.module == 'janus', (
        'atmos_clim silently swapped away from janus'
    )
    assert runner.config.interior_energetics.module == 'aragog', (
        'interior_energetics silently swapped away from aragog'
    )

    # F_atm positive and finite. JANUS solves radiative transfer
    # via SOCRATES; the planet is radiating energy out, so
    # F_atm > 0 at every iteration over a 1e3 yr horizon.
    f_atm = hf['F_atm'].to_numpy()
    assert np.all(np.isfinite(f_atm)), 'F_atm contains NaN or Inf'
    assert np.all(f_atm > 0), f'F_atm non-positive: min={f_atm.min():.3e} W/m^2'

    # T_surf in the JANUS-supported range.
    t_surf = hf['T_surf'].to_numpy()
    assert np.all(np.isfinite(t_surf)), 'T_surf contains NaN or Inf'
    assert np.all(t_surf > 200.0), f'T_surf too low: min={t_surf.min():.1f} K'
    assert np.all(t_surf < 4000.0), f'T_surf too high: max={t_surf.max():.1f} K'

    # Optical-depth monotonicity: radiation must thin upward.
    # JANUS populates tau_atm_TOA and tau_atm_surface in hf_row at
    # every iteration (the same columns the AGNI pair tests pin
    # synthetically). A regression that swapped the two would fail
    # this row-by-row check.
    if {'tau_atm_TOA', 'tau_atm_surface'} <= set(hf.columns):
        tau_toa = hf['tau_atm_TOA'].to_numpy()
        tau_srf = hf['tau_atm_surface'].to_numpy()
        # Only check rows where both are non-zero (some JANUS init
        # configs leave the columns at 0.0 on the first row).
        valid = (tau_toa > 0) & (tau_srf > 0)
        if valid.any():
            assert np.all(tau_toa[valid] < tau_srf[valid]), (
                f'tau monotonicity broken: max(tau_TOA / tau_srf) = '
                f'{(tau_toa[valid] / tau_srf[valid]).max():.3f}'
            )

    final = hf.iloc[-1]

    # Per-element mass closure.
    for elt in ('H', 'C', 'N', 'S', 'O'):
        atm_key = f'{elt}_kg_atm'
        liq_key = f'{elt}_kg_liquid'
        sol_key = f'{elt}_kg_solid'
        tot_key = f'{elt}_kg_total'
        if not all(k in final for k in (atm_key, liq_key, sol_key, tot_key)):
            continue
        atm = float(final[atm_key])
        liq = float(final[liq_key])
        sol = float(final[sol_key])
        tot = float(final[tot_key])
        assert atm >= 0, f'{atm_key} negative: {atm:.3e}'
        assert liq >= 0, f'{liq_key} negative: {liq:.3e}'
        assert sol >= 0, f'{sol_key} negative: {sol:.3e}'
        if tot > 0:
            assert atm + liq + sol == pytest.approx(tot, rel=1e-2), (
                f'{elt} closure: atm+liq+sol={atm + liq + sol:.3e}, total={tot:.3e}'
            )

    # Phi_global bounded.
    if 'Phi_global' in hf.columns:
        phi = hf['Phi_global'].to_numpy()
        assert np.all((0 <= phi) & (phi <= 1)), (
            f'Phi_global out of [0,1], observed [{phi.min():.3e}, {phi.max():.3e}]'
        )

    # Cross-step continuity of T_magma and Phi_global.
    if 'T_magma' in hf.columns and len(hf) >= 2:
        dT = np.diff(hf['T_magma'].to_numpy())
        assert np.all(np.abs(dT) < 1000.0), (
            f'T_magma jump too large: max(|dT|)={np.max(np.abs(dT)):.1f} K'
        )
    if 'Phi_global' in hf.columns and len(hf) >= 2:
        dphi = np.diff(hf['Phi_global'].to_numpy())
        assert np.all(np.abs(dphi) < 0.5), (
            f'Phi_global jump too large: max(|dPhi|)={np.max(np.abs(dphi)):.3f}'
        )

    # Cross-cutting helpers.
    mass_results = validate_mass_conservation(hf, tolerance=0.2)
    assert mass_results.get('masses_positive', True), 'mass reservoirs negative'

    stability = validate_stability(hf, max_temp=1e6, max_pressure=1e10)
    assert stability['temps_stable'], 'temperature instability detected'
    assert stability['pressures_stable'], 'pressure instability detected'
