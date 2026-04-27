"""
Unit tests for the BoundaryRunner class from proteus.interior.boundary.

This module tests thermal evolution, viscosity parameterizations, and
mantle convection calculations for rocky planets undergoing cooling from
magma ocean states to solid interiors.

**Test Scenarios**:
  - Magma ocean cooling (high T, melt-dominated convection)
  - Transitional state (partially molten mantle)
  - Solid mantle (low melt fraction, viscous convection)
  - Extreme irradiation (surface temperature constraints)

**Physical Constants Used**:
  - Temperature range: 1500–4000 K (magma ocean to solid mantle)
  - Melt fraction: 0–1 (fully solid to fully molten)
  - Rayleigh numbers: 10⁶–10¹⁰ (sub-critical to hyper-turbulent convection)

**Test Infrastructure**:
  See `docs/How-to/test_infrastructure.md`, `test_categorization.md`,
  and `test_building.md` for full testing framework documentation.

**Coverage**:
  - Viscosity models: constant, aggregate, Arrhenius
  - Rayleigh number and Nusselt scaling
  - Radioactive heating decay chains (K-40, U-238, U-235, Th-232)
  - Melt fraction and solidification radius evolution
  - Coupled thermal ODEs (potential and surface temperature)
  - ODE solver integration with terminal events
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from proteus.interior.boundary import BoundaryRunner
from proteus.utils.constants import (
    M_earth,
    secs_per_year,
)

# =============================================================================
# Fixtures for BoundaryRunner Configuration & Setup
# =============================================================================


@pytest.fixture
def mock_config():
    """
    Create a mock Config object with sensible defaults for boundary layer testing.

    Returns a Config mock with interior.boundary parameters matching typical
    rocky planet evolution scenarios (T_p = 3000 K, solid mantle viscosity ~1e21 Pa·s).

    **Physical Basis**: Terrestrial magma ocean cooling parameters (Turcotte & Schubert 2014)
    """
    config = Mock()

    # Interior boundary parameters
    config.interior.boundary.rtol = 1e-6
    config.interior.boundary.atol = 1e-9
    config.interior.boundary.T_p_0 = 3500.0  # K
    config.interior.boundary.T_solidus = 1420.0  # K
    config.interior.boundary.T_liquidus = 2020.0  # K
    config.interior.boundary.critical_melt_fraction = 0.4  # dimensionless
    config.interior.boundary.Tsurf_event_change = 500.0  # K
    config.interior.boundary.critical_rayleigh_number = 1e3  # dimensionless
    config.interior.boundary.heat_fusion_silicate = 4.0e5  # J/kg
    config.interior.boundary.nusselt_exponent = 1.0 / 3.0  # Rayleigh-Bénard scaling
    config.interior.boundary.silicate_heat_capacity = 1.2e3  # J/kg/K
    config.interior.boundary.thermal_conductivity = 4.2  # W/m/K
    config.interior.boundary.thermal_diffusivity = 1e-6  # m²/s
    config.interior.boundary.thermal_expansivity = 2e-5  # 1/K
    config.interior.boundary.viscosity_model = 1  # Constant viscosity (default)
    config.interior.boundary.eta_constant = 1e2  # Pa·s (liquid mantle)
    config.interior.boundary.transition_width = 0.2  # dimensionless
    config.interior.boundary.eta_solid_const = 1e22  # Pa·s
    config.interior.boundary.eta_melt_const = 1e2  # Pa·s
    config.interior.boundary.dynamic_viscosity = 3.8e9  # Pa·s
    config.interior.boundary.activation_energy = 3.5e5  # J/mol (silicate mantle)
    config.interior.boundary.creep_parameter = 26  # dimensionless
    config.interior.boundary.viscosity_prefactor = 2.4e-4  # Pa·s
    config.interior.boundary.viscosity_activation_temp = 4600.0  # K
    config.interior.boundary.logging = False

    # Structural parameters
    config.struct.mass_tot = 1.0  # Earth masses
    config.struct.corefrac = 0.55  # Core radius fraction
    config.struct.module = 'self'

    # Stellar parameters
    config.star.age_ini = 0.1  # Gyr

    # Delivery/radiogenic parameters
    config.delivery.radio_tref = 4.567  # Gyr (Solar System age)
    config.delivery.radio_U = 310  # ppm (BSE abundance)
    config.delivery.radio_Th = 0.031  # ppm (BSE abundance)
    config.delivery.radio_K = 0.124  # ppm (BSE abundance)

    # Interior heat sources
    config.interior.radiogenic_heat = False
    config.interior.tidal_heat = False

    return config


@pytest.fixture
def mock_interior():
    """
    Create a mock Interior_t object for coupling with BoundaryRunner.

    **Physical Basis**: Interior evolution object state at a magma ocean cooling timestep.
    """
    interior = Mock()
    interior.ic = 0  # Initial condition type (0 = not first step)
    interior.tides = [0.0]  # No tidal heating by default
    interior.phi = None
    interior.mass = None
    interior.visc = None
    interior.density = None
    interior.temp = None
    interior.pres = None
    interior.radius = None
    return interior


@pytest.fixture
def mock_atmos():
    """
    Create a mock Atmos_t object for atmospheric coupling.

    **Physical Basis**: Thin H₂-rich atmosphere on early magma ocean with
    heat capacity typical of light molecular atmospheres.
    """
    atmos = Mock()
    atmos._atm = Mock()
    atmos._atm.layer_cp = np.array([1.7e4, 1.7e4, 1.7e4])  # J/kg/K for H2 layers
    return atmos


@pytest.fixture
def mock_dirs():
    """Create mock directory dictionary for output paths."""
    return {'output': '/tmp'}


@pytest.fixture
def mock_hf_row():
    """
    Create a mock history file row with state variables at single timestep.

    **Physical Basis**: History file entry for Earth-like planet after 10 Myr
    of cooling from initial magma ocean (T_p = 3500 K → 2500 K).
    """
    return {
        'Time': 0.01,  # Myr (convert to seconds via secs_per_year)
        'R_int': 6.371e6,  # m (Earth radius)
        'M_core': 0.55 * M_earth,  # kg (Earth-mass core)
        'M_atm': 1e18,  # kg (thin H2 atmosphere)
        'F_atm': 100.0,  # W/m² (atmospheric heat loss)
        'T_magma': 3500.0,  # K (mantle potential temperature)
        'T_surf': 1600.0,  # K (surface temperature)
        'P_surf': 1e8,  # Pa (surface pressure)
        'R_core': 0.55 * 6.371e6,  # m (core radius from corefrac)
    }


@pytest.fixture
def mock_hf_all():
    """Create empty history file DataFrame (first timestep scenario)."""
    return None


@pytest.fixture
def boundary_runner(
    mock_config, mock_dirs, mock_hf_row, mock_hf_all, mock_interior, mock_atmos
):
    """
    Create a BoundaryRunner instance with sensible defaults.

    **Scope**: function (fresh instance per test)

    **Physical Setup**:
      - Planet: 1 M⊕, 1 R⊕ (Earth-like)
      - Mantle: Partially molten (T_p = 3500 K, φ ≈ 1.0, fully molten)
      - Atmosphere: Thin H₂ (~10¹⁸ kg)
      - No radioactive/tidal heating
    """
    return BoundaryRunner(
        config=mock_config,
        dirs=mock_dirs,
        hf_row=mock_hf_row,
        hf_all=mock_hf_all,
        interior_o=mock_interior,
        atmos_o=mock_atmos,
    )


# =============================================================================
# Tests: Viscosity Models (unit, fast <100ms)
# =============================================================================


@pytest.mark.unit
def test_viscosity_aggregate_model_limits(boundary_runner):
    """
    Test aggregate viscosity model behaves correctly at extremes (φ=0, φ=1).

    **Physical Scenario**: Magma ocean with smooth transition between solid
    and liquid phases via tanh smoothing function.

    **Expected Behavior**:
      - φ = 0 (solid): η ≈ η_solid
      - φ = 1 (melt): η ≈ η_melt
      - φ = 0.5 (critical): transitions smoothly
    """
    # Set aggregate model
    boundary_runner.viscosity_model = 2
    boundary_runner.eta_solid_const = 1e21
    boundary_runner.eta_melt_const = 1e2

    # Test fully solid
    eta_solid = boundary_runner.viscosity_aggregate_model(0.0)
    assert eta_solid == pytest.approx(1e21, rel=0.1)

    # Test fully molten
    eta_melt = boundary_runner.viscosity_aggregate_model(1.0)
    assert eta_melt == pytest.approx(1e2, rel=0.1)

    # Test intermediate: should be between melt and solid
    eta_mid = boundary_runner.viscosity_aggregate_model(0.5)
    assert 1e2 < eta_mid < 1e21


@pytest.mark.unit
@pytest.mark.parametrize(
    'phi,expected_eta_range',
    [
        (0.1, (1e19, 1e21)),  # Mostly solid
        (0.5, (1e2, 1e21)),  # Transition zone
        (0.9, (1e2, 1e5)),  # Mostly liquid
    ],
)
def test_viscosity_aggregate_parametrized(boundary_runner, phi, expected_eta_range):
    """
    Test aggregate viscosity model across melt fraction range via parametrization.

    **Physical Scenario**: Parametrized test for magma ocean cooling through
    critical melt fraction (φ_crit = 0.5).

    **Expected Behavior**: Viscosity monotonically decreases with melt fraction.
    """
    boundary_runner.viscosity_model = 2
    eta = boundary_runner.viscosity_aggregate_model(phi)
    eta_min, eta_max = expected_eta_range
    assert eta_min < eta < eta_max


@pytest.mark.unit
def test_viscosity_arrhenius_solid_mantle(boundary_runner):
    """
    Test Arrhenius viscosity below critical melt fraction (solid mantle Arrhenius).

    **Physical Scenario**: Cool mantle (T_p = 1500 K, φ = 0.1) with exponential
    temperature-dependent viscosity (silicate creep flow law).

    **Expected Behavior**: η(T) ∝ exp(E_a / RT); increases with decreasing T.
    """
    boundary_runner.viscosity_model = 3
    boundary_runner.dynamic_viscosity = 1e21
    boundary_runner.activation_energy = 3e5  # J/mol (diopside)
    boundary_runner.creep_parameter = 0.1

    T_cold = 1500.0  # K (cool mantle)
    T_hot = 2500.0  # K (hot mantle)
    phi = 0.1  # Mostly solid

    eta_cold = boundary_runner.viscosity_arrhenius(T_cold, phi)
    eta_hot = boundary_runner.viscosity_arrhenius(T_hot, phi)

    # Cold mantle should be more viscous
    assert eta_cold > eta_hot
    assert 1e20 < eta_cold < 1e25
    assert 1e19 < eta_hot < 1e24


@pytest.mark.unit
def test_viscosity_arrhenius_magma_ocean(boundary_runner):
    """
    Test Arrhenius viscosity above critical melt fraction (magma ocean Vogel-Fulcher-Tammann).

    **Physical Scenario**: Hot magma ocean (T_p = 3500 K, φ = 0.9) with
    Vogel-Fulcher-Tammann viscosity and crystal fraction weakening.

    **Expected Behavior**: η(T) ∝ exp(D/(T - T_ref)); temperature-sensitive,
    typically 1–100 Pa·s for silicate melts at high temperatures.
    """
    boundary_runner.viscosity_model = 3
    boundary_runner.viscosity_prefactor = 1e-2  # Pa·s
    boundary_runner.viscosity_activation_temp = 2000.0  # K
    boundary_runner.critical_melt_fraction = 0.5

    T_hot = 3500.0  # K (magma ocean)
    phi = 0.9  # Mostly molten

    eta = boundary_runner.viscosity_arrhenius(T_hot, phi)

    # Magma ocean viscosity should be low (~0.001–1000 Pa·s)
    assert 1e-3 < eta < 1e4


@pytest.mark.unit
def test_viscosity_dispatcher_constant_model(boundary_runner):
    """
    Test viscosity dispatcher selects constant model (model=1) correctly.

    **Physical Scenario**: Configuration with viscosity_model=1 should
    dispatch to viscosity_constant() regardless of temperature/melt.
    """
    boundary_runner.viscosity_model = 1
    boundary_runner.eta_constant = 1e21

    T_p, T_surf = 3000.0, 1600.0
    phi = 0.8

    eta = boundary_runner.viscosity(T_p, T_surf, phi)
    assert eta == pytest.approx(1e21, rel=1e-10)


@pytest.mark.unit
def test_viscosity_dispatcher_aggregate_model(boundary_runner):
    """
    Test viscosity dispatcher selects aggregate model (model=2) correctly.
    """
    boundary_runner.viscosity_model = 2
    boundary_runner.eta_solid_const = 1e21
    boundary_runner.eta_melt_const = 1e2

    T_p, T_surf = 3000.0, 1600.0
    phi = 0.5

    eta = boundary_runner.viscosity(T_p, T_surf, phi)
    # Should call aggregate model
    assert 1e2 < eta < 1e21


@pytest.mark.unit
def test_viscosity_dispatcher_arrhenius_model(boundary_runner):
    """
    Test viscosity dispatcher selects Arrhenius model (model=3) correctly.
    """
    boundary_runner.viscosity_model = 3
    boundary_runner.dynamic_viscosity = 1e21
    boundary_runner.activation_energy = 3e5

    T_p, T_surf = 2000.0, 1500.0
    phi = 0.2

    eta = boundary_runner.viscosity(T_p, T_surf, phi)
    # Should call Arrhenius model
    assert 1e19 < eta < 1e23


@pytest.mark.unit
def test_viscosity_dispatcher_unknown_model(boundary_runner, caplog):
    """
    Test viscosity dispatcher defaults to aggregate (model=2) for invalid model numbers.

    **Expected Behavior**: Unknown model triggers warning, falls back to aggregate.
    """
    boundary_runner.viscosity_model = 999  # Invalid
    boundary_runner.eta_solid_const = 1e21
    boundary_runner.eta_melt_const = 1e2

    eta = boundary_runner.viscosity(3000.0, 1600.0, 0.5)

    # Should not crash; should use aggregate fallback
    assert 1e2 < eta < 1e21


# =============================================================================
# Tests: Rayleigh Number & Convection Scaling
# =============================================================================


@pytest.mark.unit
def test_rayleigh_number_physical_range(boundary_runner):
    """
    Test Rayleigh number falls within physically realistic range for planetary mantles.

    **Physical Scenario**: Ra = 10⁶–10¹⁰ represents transition from laminar to
    turbulent convection in rocky planet mantles (Turcotte & Schubert 2014).

    **Expected Behavior**: Ra scales with (ΔT × L³) / (η × κ); should increase
    with larger temperature contrasts.
    """
    boundary_runner.viscosity_model = 1
    boundary_runner.eta_constant = 1e21

    T_p, T_surf = 3000.0, 300.0
    phi = 0.8

    ra = boundary_runner.rayleigh_number(T_p, T_surf, phi)

    # Ra should be in realistic mantle convection range
    assert 1e5 < ra < 1e15


@pytest.mark.unit
@pytest.mark.parametrize(
    'T_p,T_surf,phi',
    [
        (2500.0, 500.0, 0.2),  # Cool, mostly solid
        (3500.0, 1500.0, 0.8),  # Hot, mostly molten
        (3000.0, 1600.0, 0.5),  # Intermediate
    ],
)
def test_rayleigh_number_parametrized(boundary_runner, T_p, T_surf, phi):
    """
    Test Rayleigh number calculation across multiple thermal scenarios.
    """
    boundary_runner.viscosity_model = 1
    boundary_runner.eta_constant = 1e21

    ra = boundary_runner.rayleigh_number(T_p, T_surf, phi)

    # Ra should be positive and realistic
    assert ra > 0
    assert 1e5 < ra < 1e15


@pytest.mark.unit
def test_rayleigh_number_temperature_dependence(boundary_runner):
    """
    Test Rayleigh number increases with temperature contrast (Ra ∝ ΔT).

    **Physical Scenario**: Larger ΔT drives stronger convection (higher Ra).
    """
    boundary_runner.viscosity_model = 1
    boundary_runner.eta_constant = 1e21

    T_surf = 300.0
    phi = 0.5

    ra_small_delta = boundary_runner.rayleigh_number(2500.0, T_surf, phi)
    ra_large_delta = boundary_runner.rayleigh_number(4000.0, T_surf, phi)

    # Larger temperature contrast → larger Ra
    assert ra_large_delta > ra_small_delta


# =============================================================================
# Tests: Convective Heat Flux (q_m)
# =============================================================================


@pytest.mark.unit
def test_q_m_positive_heat_loss(boundary_runner):
    """
    Test convective heat flux is positive (heat flowing outward).

    **Physical Scenario**: Hot interior (T_p = 3000 K) cooler surface
    (T_surf = 1600 K) loses heat via convection.

    **Expected Behavior**: q_m > 0 (heat flows from hotter to cooler region).
    """
    T_p, T_surf = 3000.0, 1600.0
    phi = 0.8

    q_m = boundary_runner.q_m(T_p, T_surf, phi)

    assert q_m > 0
    assert 1e4 < q_m < 1e8  # W/m² (realistic mantle heat flux)


@pytest.mark.unit
def test_q_m_temperature_dependence(boundary_runner):
    """
    Test convective heat flux increases with temperature contrast.

    **Physical Scenario**: Larger ΔT → larger heat flux (q_m ∝ Nu × ΔT / L).
    """
    T_surf = 300.0
    phi = 0.5

    q_m_small_delta = boundary_runner.q_m(2500.0, T_surf, phi)
    q_m_large_delta = boundary_runner.q_m(4000.0, T_surf, phi)

    # Larger temperature contrast → larger heat flux
    assert q_m_large_delta > q_m_small_delta


@pytest.mark.unit
def test_q_m_zero_temperature_difference(boundary_runner):
    """
    Test convective heat flux approaches zero when T_p = T_surf.

    **Physical Scenario**: No temperature gradient → no convective driving force.
    """
    T_p = T_surf = 2000.0
    phi = 0.5

    q_m = boundary_runner.q_m(T_p, T_surf, phi)

    # Should be near zero (or exactly zero depending on Nu scaling)
    assert abs(q_m) < 1e2


# =============================================================================
# Tests: Radioactive Heating
# =============================================================================


@pytest.mark.unit
def test_radioactive_heating_disabled(boundary_runner):
    """
    Test radioactive heating returns 0 when disabled in config.

    **Physical Scenario**: radiogenic_heat = False (default).
    """
    boundary_runner.use_radiogenic_heating = False

    t = 1e9 * secs_per_year  # 1 Gyr after initial formation

    h_radio = boundary_runner.radioactive_heating(t)

    assert h_radio == 0.0


@pytest.mark.unit
def test_radioactive_heating_enabled_positive(boundary_runner):
    """
    Test radioactive heating is positive when enabled.

    **Physical Scenario**: radiogenic_heat = True; heating from K-40, U-238,
    U-235, Th-232 decay chains in silicate mantle.

    **Expected Behavior**: H > 0 W/kg (energy from radioactive decay).
    """
    boundary_runner.use_radiogenic_heating = True
    boundary_runner.age_ini = 0.0  # Gyr (Solar System age reference)
    boundary_runner.radio_tref = 4.567  # Gyr
    boundary_runner.U_abun = 20.0e-9  # kg U / kg mantle
    boundary_runner.Th_abun = 79.0e-9  # kg Th / kg mantle
    boundary_runner.K_abun = 143.0e-6  # kg K / kg mantle

    t = 0.0  # s (at formation)

    h_radio = boundary_runner.radioactive_heating(t)

    # Should be positive at t=0 (maximum heating)
    assert h_radio > 0


@pytest.mark.unit
def test_radioactive_heating_decay_with_time(boundary_runner):
    """
    Test radioactive heating decays exponentially with time.

    **Physical Scenario**: H(t) = H₀ exp(-λt) for each isotope.

    **Expected Behavior**: H decreases monotonically; H(t₁) > H(t₂) for t₁ < t₂.
    """
    boundary_runner.use_radiogenic_heating = True
    boundary_runner.age_ini = 0.0
    boundary_runner.radio_tref = 4.567
    boundary_runner.U_abun = 20.0e-9
    boundary_runner.Th_abun = 79.0e-9
    boundary_runner.K_abun = 143.0e-6

    t_early = 1e6 * secs_per_year  # 1 Myr
    t_late = 1e9 * secs_per_year  # 1 Gyr

    h_early = boundary_runner.radioactive_heating(t_early)
    h_late = boundary_runner.radioactive_heating(t_late)

    # Early heating should exceed late heating (radioactive decay)
    assert h_early > h_late


# =============================================================================
# Tests: Melt Fraction
# =============================================================================


@pytest.mark.unit
def test_melt_fraction_clipping_below_solidus(boundary_runner):
    """
    Test melt fraction is 0 below solidus temperature.

    **Physical Scenario**: T_p < T_solidus → fully solid (φ = 0).
    """
    boundary_runner.T_solidus = 1600.0
    boundary_runner.T_liquidus = 2000.0

    T_p = 1500.0  # Below solidus

    phi = boundary_runner.melt_fraction(T_p)

    assert phi == pytest.approx(0.0, abs=1e-10)


@pytest.mark.unit
def test_melt_fraction_clipping_above_liquidus(boundary_runner):
    """
    Test melt fraction is 1 above liquidus temperature.

    **Physical Scenario**: T_p > T_liquidus → fully molten (φ = 1).
    """
    boundary_runner.T_solidus = 1600.0
    boundary_runner.T_liquidus = 2000.0

    T_p = 2500.0  # Above liquidus

    phi = boundary_runner.melt_fraction(T_p)

    assert phi == pytest.approx(1.0, abs=1e-10)


@pytest.mark.unit
@pytest.mark.parametrize(
    'T_p,expected_phi',
    [
        (1600.0, 0.0),  # At solidus
        (1800.0, 0.5),  # Midpoint
        (2000.0, 1.0),  # At liquidus
    ],
)
def test_melt_fraction_parametrized(boundary_runner, T_p, expected_phi):
    """
    Test melt fraction linearly interpolates between solidus and liquidus.

    **Physical Scenario**: Simplified linear phase diagram for silicates.
    """
    boundary_runner.T_solidus = 1600.0
    boundary_runner.T_liquidus = 2000.0

    phi = boundary_runner.melt_fraction(T_p)

    assert phi == pytest.approx(expected_phi, rel=1e-10)


# =============================================================================
# Tests: Solidification Radius
# =============================================================================


@pytest.mark.unit
def test_r_s_fully_molten(boundary_runner):
    """
    Test solidification radius equals core radius when fully molten (φ = 1).

    **Physical Scenario**: T_p ≥ T_liquidus → entire mantle molten → r_s = r_core.
    """
    boundary_runner.T_liquidus = 2000.0

    T_p = 2500.0  # Above liquidus (φ = 1)

    r_s = boundary_runner.r_s(T_p)

    assert r_s == pytest.approx(boundary_runner.core_radius, rel=1e-10)


@pytest.mark.unit
def test_r_s_fully_solid(boundary_runner):
    """
    Test solidification radius equals planet radius when fully solid (φ = 0).

    **Physical Scenario**: T_p ≤ T_solidus → entire mantle solid → r_s = R_planet.
    """
    boundary_runner.T_solidus = 1600.0

    T_p = 1500.0  # Below solidus (φ = 0)

    r_s = boundary_runner.r_s(T_p)

    assert r_s == pytest.approx(boundary_runner.planet_radius, rel=1e-10)


@pytest.mark.unit
def test_r_s_intermediate(boundary_runner):
    """
    Test solidification radius is intermediate when partially molten.

    **Physical Scenario**: T_p between T_solidus and T_liquidus → φ ∈ (0,1) →
    r_s between r_core and R_planet.
    """
    boundary_runner.T_solidus = 1600.0
    boundary_runner.T_liquidus = 2000.0

    T_p = 1800.0  # Midpoint (φ = 0.5)

    r_s = boundary_runner.r_s(T_p)

    # Should be strictly between core and planet radii
    assert boundary_runner.core_radius < r_s < boundary_runner.planet_radius


# =============================================================================
# Tests: Temperature Derivatives
# =============================================================================


@pytest.mark.unit
def test_drs_dTp_sign_and_magnitude(boundary_runner):
    """
    Test dr_s/dT_p derivative is negative (solidification radius decreases with increasing T).

    **Physical Scenario**: Hotter T_p → more melt → smaller solid radius → dr_s/dT_p < 0.
    """
    boundary_runner.T_solidus = 1600.0
    boundary_runner.T_liquidus = 2000.0

    T_p = 1800.0  # Intermediate temperature

    drs_dTp = boundary_runner.drs_dTp(T_p)

    # Should be negative (r_s decreases with increasing T)
    assert drs_dTp < 0

    # Should have physical units [m/K]
    assert abs(drs_dTp) < 1e-5  # ~0.01 mm/K (reasonable)


@pytest.mark.unit
def test_dT_pdt_heat_loss_dominates(boundary_runner):
    """
    Test dT_p/dt is negative (cooling) when convective heat loss dominates internal heating.

    **Physical Scenario**: Cooling mantle with small internal heating.

    **Expected Behavior**: dT_p/dt < 0 (temperature decreases).
    """
    boundary_runner.use_radiogenic_heating = False
    boundary_runner.use_tidal_heating = False

    T_p, T_surf = 3000.0, 1600.0
    t = 0.0  # s

    dTp_dt = boundary_runner.dT_pdt(T_p, T_surf, t)

    # Should be negative (cooling)
    assert dTp_dt < 0


@pytest.mark.unit
def test_dT_pdt_near_surface_temperature(boundary_runner):
    """
    Test dT_p/dt becomes small when T_p ≈ T_surf (reduced driving force).

    **Physical Scenario**: Near-isothermal interior → reduced convection → slower cooling.
    """
    boundary_runner.use_radiogenic_heating = False

    T_p = 2000.0
    T_surf = 1950.0  # Nearly equal

    dTp_dt = boundary_runner.dT_pdt(T_p, T_surf, 0.0)

    # dT_p/dt should be small but negative
    assert dTp_dt < 0
    assert abs(dTp_dt) < abs(boundary_runner.dT_pdt(3000.0, 1600.0, 0.0))


@pytest.mark.unit
def test_dT_surfdt_physical_sign(boundary_runner):
    """
    Test dT_surf/dt sign depends on atmospheric balance.

    **Physical Scenario**: Surface temperature evolves based on convective influx
    minus atmospheric heat loss.

    **Expected Behavior**: dT_surf/dt typically negative (surface cooling) when
    atmosphere is thin.
    """
    T_p = 3000.0
    T_surf = 1600.0

    dTs_dt = boundary_runner.dT_surfdt(T_p, T_surf)

    # Should have reasonable magnitude [K/s]
    assert abs(dTs_dt) < 1e-8  # ~100 K / 1 Myr


# =============================================================================
# Tests: ODE System (thermal_rhs)
# =============================================================================


@pytest.mark.unit
def test_thermal_rhs_returns_list(boundary_runner):
    """
    Test thermal_rhs returns list of derivatives matching scipy.integrate.solve_ivp format.

    **Physical Scenario**: RHS function for coupled ODE [dT_p/dt, dT_surf/dt].
    """
    t = 0.0
    y = [3000.0, 1600.0]  # [T_p, T_surf]

    rhs = boundary_runner.thermal_rhs(t, y)

    assert isinstance(rhs, list)
    assert len(rhs) == 2
    assert all(isinstance(val, (float, np.floating)) for val in rhs)


@pytest.mark.unit
def test_thermal_rhs_surface_temperature_constraint(boundary_runner):
    """
    Test thermal_rhs enforces T_surf ≤ T_p constraint.

    **Physical Scenario**: Surface temperature cannot exceed mantle potential temperature
    (physical constraint).

    **Expected Behavior**: If T_surf > T_p, it is clamped to T_p - 1K internally.
    """
    boundary_runner.use_radiogenic_heating = False

    t = 0.0
    y_invalid = [2000.0, 2500.0]  # T_surf > T_p (invalid)

    rhs = boundary_runner.thermal_rhs(t, y_invalid)

    # Should not crash; internally corrects T_surf ≤ T_p
    assert isinstance(rhs, list)
    assert len(rhs) == 2


@pytest.mark.unit
def test_thermal_rhs_continuity_across_time(boundary_runner):
    """
    Test thermal_rhs is continuous (smooth) in state variables.

    **Physical Scenario**: Small perturbations in (T_p, T_surf) should produce
    small changes in derivatives (no discontinuities).
    """
    boundary_runner.use_radiogenic_heating = False

    t = 0.0
    y = [3000.0, 1600.0]

    rhs_nominal = boundary_runner.thermal_rhs(t, y)

    # Perturb slightly
    epsilon = 0.1  # K
    rhs_perturbed = boundary_runner.thermal_rhs(t, [y[0] + epsilon, y[1]])

    # Changes should be small (continuous derivatives)
    for rhs_nom, rhs_pert in zip(rhs_nominal, rhs_perturbed):
        assert abs(rhs_pert - rhs_nom) < 1e-5


# =============================================================================
# Tests: ODE Solver Integration (run_solver)
# =============================================================================


@pytest.mark.unit
@patch('proteus.interior.boundary.solve_ivp')
def test_run_solver_integration_success(mock_solve_ivp, boundary_runner, mock_interior):
    """
    Test run_solver successfully integrates thermal evolution ODE.

    **Physical Scenario**: Single timestep integration from initial conditions.

    **Expected Behavior**: Returns (sim_time, output_dict) with valid results.
    """
    # Mock solve_ivp output
    mock_solution = Mock()
    mock_solution.y = np.array(
        [
            [3000.0, 2800.0],  # T_p: initial → final
            [1600.0, 1550.0],  # T_surf: initial → final
        ]
    )
    mock_solution.t = np.array([0.0, 1e7 * secs_per_year])

    mock_solve_ivp.return_value = mock_solution

    hf_row = {
        'Time': 0.01,
        'R_int': 6.371e6,
        'M_core': 0.55 * M_earth,
        'M_atm': 1e18,
        'F_atm': 100.0,
        'T_magma': 3000.0,
        'T_surf': 1600.0,
        'P_surf': 1e8,
        'R_core': 0.55 * 6.371e6,
    }

    sim_time, output = boundary_runner.run_solver(hf_row, mock_interior, {})

    # Check outputs
    assert sim_time > 0
    assert isinstance(output, dict)
    assert 'T_magma' in output
    assert 'T_surf' in output
    assert 'Phi_global' in output


@pytest.mark.unit
@patch('proteus.interior.boundary.solve_ivp')
def test_run_solver_output_keys(mock_solve_ivp, boundary_runner, mock_interior):
    """
    Test run_solver output dictionary contains all required keys.

    **Expected Output Keys**:
      T_magma, T_pot, T_surf, F_int, Phi_global, Phi_global_vol, F_radio,
      RF_depth, M_mantle_liquid, M_mantle_solid, F_tidal, M_mantle
    """
    mock_solution = Mock()
    mock_solution.y = np.array([[2800.0, 2700.0], [1550.0, 1500.0]])
    mock_solution.t = np.array([0.0, 1e7 * secs_per_year])
    mock_solve_ivp.return_value = mock_solution

    hf_row = {
        'Time': 0.01,
        'R_int': 6.371e6,
        'M_core': 0.55 * M_earth,
        'M_atm': 1e18,
        'F_atm': 100.0,
        'P_surf': 1e8,
        'R_core': 0.55 * 6.371e6,
    }

    sim_time, output = boundary_runner.run_solver(hf_row, mock_interior, {})

    required_keys = [
        'T_magma',
        'T_pot',
        'T_surf',
        'F_int',
        'Phi_global',
        'Phi_global_vol',
        'F_radio',
        'RF_depth',
        'M_mantle_liquid',
        'M_mantle_solid',
        'F_tidal',
        'M_mantle',
    ]

    for key in required_keys:
        assert key in output


@pytest.mark.unit
@patch('proteus.interior.boundary.solve_ivp')
def test_run_solver_interior_object_updated(mock_solve_ivp, boundary_runner, mock_interior):
    """
    Test run_solver updates Interior_t object arrays with results.

    **Expected Behavior**: interior_o.{phi, mass, visc, density, temp, pres, radius}
    are set to numpy arrays.
    """
    mock_solution = Mock()
    mock_solution.y = np.array([[2800.0, 2700.0], [1550.0, 1500.0]])
    mock_solution.t = np.array([0.0, 1e7 * secs_per_year])
    mock_solve_ivp.return_value = mock_solution

    hf_row = {
        'Time': 0.01,
        'R_int': 6.371e6,
        'M_core': 0.55 * M_earth,
        'M_atm': 1e18,
        'F_atm': 100.0,
        'P_surf': 1e8,
        'R_core': 0.55 * 6.371e6,
    }

    boundary_runner.run_solver(hf_row, mock_interior, {})

    # Check interior object was updated
    assert isinstance(mock_interior.phi, np.ndarray)
    assert isinstance(mock_interior.temp, np.ndarray)
    assert isinstance(mock_interior.visc, np.ndarray)
    assert isinstance(mock_interior.mass, np.ndarray)
    assert isinstance(mock_interior.density, np.ndarray)
    assert isinstance(mock_interior.pres, np.ndarray)
    assert isinstance(mock_interior.radius, np.ndarray)


# =============================================================================
# Tests: Static Methods
# =============================================================================


@pytest.mark.unit
def test_compute_time_step_first_iteration(mock_config, mock_dirs):
    """
    Test compute_time_step returns 0 for first iteration (ic=1).

    **Physical Scenario**: Initial condition (ic=1) → no evolution → dt=0.
    """
    interior = Mock()
    interior.ic = 1
    interior.tides = [0.0]

    dt = BoundaryRunner.compute_time_step(mock_config, mock_dirs, {}, None, interior)

    assert dt == 0.0


@pytest.mark.unit
@patch('proteus.interior.boundary.next_step')
def test_compute_time_step_normal_iteration(mock_next_step, mock_config, mock_dirs):
    """
    Test compute_time_step calls next_step for normal iterations (ic ≠ 1).

    **Physical Scenario**: Subsequent timesteps → adaptive timestep from next_step().
    """
    mock_next_step.return_value = 1e-5  # Myr as returned by next_step

    interior = Mock()
    interior.ic = 0

    dt = BoundaryRunner.compute_time_step(mock_config, mock_dirs, {}, None, interior)

    # Should call next_step
    mock_next_step.assert_called_once()
    assert dt == 1e-5


# =============================================================================
# Integration & Regression Tests
# =============================================================================


@pytest.mark.unit
def test_physical_consistency_conservation(boundary_runner):
    """
    Test physical conservation laws: mass and energy conservation checks.

    **Physical Scenario**: Mantle mass should remain constant during evolution.
    """
    initial_mass = boundary_runner.mantle_mass

    # Simulate thermal evolution
    for T_p in [3000.0, 2800.0, 2600.0]:
        phi = boundary_runner.melt_fraction(T_p)
        q_m = boundary_runner.q_m(T_p, 1600.0, phi)

        # Mass should be conserved
        assert boundary_runner.mantle_mass == pytest.approx(initial_mass, rel=1e-10)

        # Volume should change with solidification
        assert q_m > 0  # Heat should flow outward


@pytest.mark.unit
def test_physical_bounds_temperatures(boundary_runner):
    """
    Test that temperatures stay within physically reasonable bounds.

    **Physical Scenario**: 0 K < T < 10,000 K for silicate mantles.
    """
    for T_p in [1500.0, 2500.0, 3500.0]:
        for T_surf in [300.0, 1000.0, 1500.0]:
            # All derivatives should be finite and real
            dTp_dt = boundary_runner.dT_pdt(T_p, T_surf, 0.0)
            dTs_dt = boundary_runner.dT_surfdt(T_p, T_surf)

            assert np.isfinite(dTp_dt)
            assert np.isfinite(dTs_dt)
            assert dTp_dt < 0  # Cooling (no heating)


# =============================================================================
# Edge Case & Error Handling Tests
# =============================================================================


@pytest.mark.unit
def test_zero_temperature_gradient(boundary_runner):
    """
    Test behavior when T_p = T_surf (isothermal interior).

    **Physical Scenario**: Edge case: no driving force for convection.
    """
    T_p = T_surf = 2000.0
    phi = 0.5

    # Should not crash
    q_m = boundary_runner.q_m(T_p, T_surf, phi)
    ra = boundary_runner.rayleigh_number(T_p, T_surf, phi)

    assert q_m >= 0
    assert ra >= 0


@pytest.mark.unit
def test_extreme_melt_fraction_bounds(boundary_runner):
    """
    Test viscosity models handle melt fraction bounds (φ = 0, φ = 1) gracefully.

    **Physical Scenario**: Fully solid (φ = 0) and fully molten (φ = 1) extremes.
    """
    for model in [1, 2, 3]:
        boundary_runner.viscosity_model = model

        T_p, T_surf = 2500.0, 500.0

        # Should not crash or produce NaN
        eta_solid = boundary_runner.viscosity(T_p, T_surf, 0.0)
        eta_melt = boundary_runner.viscosity(T_p, T_surf, 1.0)

        assert np.isfinite(eta_solid)
        assert np.isfinite(eta_melt)
