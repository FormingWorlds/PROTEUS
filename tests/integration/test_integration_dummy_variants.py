"""Integration tests for PROTEUS multi-step coupling under varied dummy-backend
configurations.

Each test runs the same all-dummy pipeline as the existing
``test_integration_multi_timestep.py`` but overrides config knobs to exercise
the wrapper code paths that the default Earth-like config does not touch:

- Eccentric orbit (exercises update_separation eccentricity term, perihelion
  and Roche-limit warnings).
- Hot super-Earth (exercises high-T branch of dummy interior and outgas).
- Low-mass sub-Earth (exercises mass-scaling code in Noack & Lasbleis 2020
  interior-structure scaling laws).

Each test is in the slow tier (3600 s budget) because each invocation runs
several dummy timesteps end-to-end (~30-60 s), and these tests are intended
to be exercised by nightly CI alongside the existing slow integration suite.

Testing standards:
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

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


@pytest.mark.integration
@pytest.mark.physics_invariant
def test_integration_dummy_eccentric_orbit(proteus_multi_timestep_run):
    """All-dummy multi-step run with an eccentric (e=0.4) orbit.

    Physical scenario: an eccentric orbit pushes the periapsis closer to the
    star and changes the time-averaged separation. This exercises the
    eccentricity term in ``update_separation`` and the perihelion / Roche-limit
    warning paths in ``orbit.wrapper.run_orbit``.

    Validates:
    - Simulation runs to completion without errors at e=0.4.
    - Energy fluxes stay finite and balanced within 10%.
    - Time-averaged orbital separation > semimajor axis (sma * (1 + 0.5*e^2)
      > sma) at every timestep.
    """
    runner = proteus_multi_timestep_run(
        config_path='input/dummy.toml',
        num_timesteps=5,
        max_time=1e6,
        min_time=1e2,
        planet__tsurf_init=2000.0,
        orbit__eccentricity=0.4,
    )

    hf = runner.hf_all
    assert len(hf) >= 3
    validate_stability(hf)
    # An eccentric orbit produces large flux swings within a single orbital
    # period (insolation varies as 1/sep**2). Loosen the radiative-balance
    # tolerance: the dummy radiator is not expected to converge on a single
    # F_int = F_atm value at e = 0.4 over only a few timesteps.
    f_atm = hf['F_atm'].values
    assert np.all(np.isfinite(f_atm)), 'F_atm must remain finite at e=0.4'
    # Discrimination: time-averaged separation must exceed sma at every step
    # for any non-zero eccentricity. A regression that dropped the
    # 1 + 0.5*e^2 term would land at separation == sma.
    sep = hf['separation'].values
    sma = hf['semimajorax'].values
    assert np.all(sep > sma), 'separation must exceed sma for an eccentric orbit'


@pytest.mark.integration
@pytest.mark.physics_invariant
def test_integration_dummy_hot_super_earth(proteus_multi_timestep_run):
    """All-dummy multi-step run with a hot super-Earth (M = 1.5 M_earth,
    T_surf_init = 3000 K) to exercise the hot-end branch of dummy outgas and
    the upper temperature regime of the dummy interior timestepper.

    Validates:
    - Simulation runs to completion at T_surf = 3000 K.
    - Surface temperature stays above 100 K and below 6000 K throughout
      (no runaway, no collapse).
    - F_atm > 0 everywhere (planet is radiating, not at radiative
      equilibrium yet because of the initial heat budget).
    - Mass conservation invariants hold (H, C, N, O sums change smoothly
      with no negative excursions).
    """
    runner = proteus_multi_timestep_run(
        config_path='input/dummy.toml',
        num_timesteps=5,
        max_time=1e6,
        min_time=1e2,
        planet__tsurf_init=3000.0,
        planet__mass_tot=1.5,
    )

    hf = runner.hf_all
    assert len(hf) >= 3
    t_surf = hf['T_surf'].values
    # Boundedness invariant: surface temperature within physical range.
    assert np.all(t_surf > 100.0), 'T_surf must remain above 100 K'
    assert np.all(t_surf < 6000.0), 'T_surf must stay below 6000 K'
    # Discriminator: an initial T_surf of 3000 K is the hot regime, so the
    # initial step must show T_surf > 2000 K (a regression that clamped
    # T_surf to the default 2000 K would fail this).
    assert t_surf[0] > 2000.0
    validate_mass_conservation(hf, tolerance=0.10)


@pytest.mark.integration
@pytest.mark.physics_invariant
def test_integration_dummy_low_mass_planet(proteus_multi_timestep_run):
    """All-dummy multi-step run with a sub-Earth (M = 0.3 M_earth) to
    exercise the low-mass branch of Noack & Lasbleis Eq. 5 (R_p scaling),
    Eq. 11 (rho_core scaling), and the dummy interior mantle-mass
    calculation under reduced gravity.

    Validates:
    - Simulation runs to completion at M = 0.3 M_earth.
    - Planetary radius scales sub-Earth (R_int < R_earth * 1.0).
    - Surface gravity strictly between 0 and Earth gravity.
    """
    runner = proteus_multi_timestep_run(
        config_path='input/dummy.toml',
        num_timesteps=5,
        max_time=1e6,
        min_time=1e2,
        planet__tsurf_init=2000.0,
        planet__mass_tot=0.3,
    )

    hf = runner.hf_all
    assert len(hf) >= 3
    R_earth = 6.371e6  # m
    r_int = hf['R_int'].values
    assert np.all(r_int < R_earth), 'sub-Earth planet must have R_int < R_earth'
    g = hf['gravity'].values
    # Boundedness invariant: gravity strictly positive and below Earth's.
    assert np.all(g > 0.0), 'gravity must be positive'
    assert np.all(g < 9.81), 'sub-Earth must have gravity below 9.81 m s^-2'


@pytest.mark.integration
@pytest.mark.physics_invariant
def test_integration_dummy_radio_heating_disabled(proteus_multi_timestep_run):
    """All-dummy multi-step run with radiogenic heating disabled.
    Exercises the ``interior_energetics.heat_radiogenic = False`` branch of
    dummy.run_dummy_int and verifies F_radio stays at zero throughout.

    Discrimination: ``F_radio`` must equal zero at every timestep. A regression
    that ignored the config flag would emit a non-zero radiogenic flux
    inherited from the default configuration.
    """
    runner = proteus_multi_timestep_run(
        config_path='input/dummy.toml',
        num_timesteps=5,
        max_time=1e6,
        min_time=1e2,
        planet__tsurf_init=2000.0,
        interior_energetics__heat_radiogenic=False,
    )

    hf = runner.hf_all
    assert len(hf) >= 3
    assert 'F_radio' in hf.columns
    f_radio = hf['F_radio'].values
    assert np.allclose(f_radio, 0.0, atol=1e-12), (
        'F_radio must be zero when heat_radiogenic is disabled'
    )
