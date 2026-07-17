"""
Unit tests for the BoundaryRunner class from proteus.interior_energetics.boundary.

This module tests thermal evolution, viscosity parameterizations, and
mantle convection calculations for rocky planets undergoing cooling from
magma ocean states to solid interiors.

**Test Scenarios**:
  - Magma ocean cooling (high T, melt-dominated convection)
  - Transitional state (partially molten mantle)
  - Solid mantle (low melt fraction, viscous convection)
  - Extreme irradiation (surface temperature constraints)

**Physical Constants Used**:
  - Temperature range: 1500-4000 K (magma ocean to solid mantle)
  - Melt fraction: 0-1 (fully solid to fully molten)
  - Rayleigh numbers: 10^6-10^10 (sub-critical to hyper-turbulent convection)

**Test Infrastructure**:
  See `docs/How-to/testing.md` and `docs/Explanations/test_framework.md` for full testing
  framework documentation.

**Coverage**:
  - Viscosity models: constant, aggregate, Arrhenius
  - Rayleigh number and Nusselt scaling
  - Radioactive heating decay chains (K-40, U-238, U-235, Th-232)
  - Melt fraction and solidification radius evolution
  - Coupled thermal ODEs (potential and surface temperature)
  - ODE solver integration with terminal events
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from proteus.interior_energetics.boundary import BoundaryRunner
from proteus.utils.constants import (
    M_earth,
    secs_per_year,
)

# Tests run in the fast PR check. The config, interior and atmos
# fixtures are built from plain SimpleNamespace objects with every
# attribute the BoundaryRunner reads pinned to a concrete value, so
# no attribute lookup can auto-create a Mock that bypasses a guard
# or trips a downstream conversion. The solver tests patch solve_ivp
# at the local binding ``proteus.interior_energetics.boundary.solve_ivp``
# (the module does ``from scipy.integrate import solve_ivp`` and then
# calls it by bare name, so the in-module binding is the one that
# must be replaced). The earlier hang on hosted CI runners was a Mock
# auto-attribute leak: ``Mock()`` returns a truthy Mock for any unset
# attribute, so a getattr-with-default guard on the source side could
# silently descend into a real-physics path on a runner whose scipy
# and numpy versions differ from the local environment. SimpleNamespace
# raises AttributeError on any missing attribute, surfacing the leak
# loudly instead. The 30 s per-test timeout is a defensive ceiling
# against any future regression of the same shape.
pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# =============================================================================
# Fixtures for BoundaryRunner Configuration & Setup
# =============================================================================


@pytest.fixture
def mock_config():
    """
    Create a Config-shaped SimpleNamespace with sensible defaults for boundary layer testing.

    Returns a nested SimpleNamespace tree with interior.boundary parameters
    matching typical rocky planet evolution scenarios (T_p = 3000 K, solid
    mantle viscosity ~1e21 Pa s). SimpleNamespace, in contrast to Mock(),
    raises AttributeError on any unset attribute, so a downstream lookup
    the fixture forgot to pin surfaces loudly instead of returning a
    truthy Mock that could bypass a guard.

    **Physical Basis**: Terrestrial magma ocean cooling parameters (Turcotte & Schubert 2014)
    """
    boundary = SimpleNamespace(
        T_solidus=1420.0,  # K
        T_liquidus=2020.0,  # K
        critical_rayleigh_number=1e3,  # dimensionless
        nusselt_exponent=1.0 / 3.0,  # Rayleigh-Benard scaling
        silicate_heat_capacity=1.2e3,  # J/kg/K
        atm_heat_capacity_const=False,
        atm_heat_capacity=1.7e4,  # J/kg/K
        silicate_density=4500.0,  # kg/m^3
        thermal_conductivity=4.2,  # W/m/K
        thermal_diffusivity=1e-6,  # m^2/s
        thermal_expansivity=2e-5,  # 1/K
        viscosity_model=1,  # Constant viscosity (default)
        eta_constant=1e2,  # Pa s (liquid mantle)
        eta_solid_const=1e22,  # Pa s
        eta_melt_const=1e2,  # Pa s
        dynamic_viscosity=3.8e9,  # Pa s
        activation_energy=3.5e5,  # J/mol (silicate mantle)
        creep_parameter=26,  # dimensionless
        viscosity_prefactor=2.4e-4,  # Pa s
        viscosity_activation_temp=4600.0,  # K
        logging=False,
    )
    interior_energetics = SimpleNamespace(
        rtol=1e-6,
        atol=1e-9,
        boundary=boundary,
        rfront_loc=0.4,  # dimensionless
        phase_transition_width=0.2,
        heat_radiogenic=False,
        heat_tidal=False,
        tmagma_atol=500.0,  # K
        latent_heat_of_fusion=4.0e5,  # J/kg
        const_log10visc=2.0,
        solid_log10visc=22.0,
        melt_log10visc=2.0,
        radio_tref=4.567,  # Gyr (Solar System age)
        radio_U=0.031,  # ppm (BSE abundance)
        radio_Th=0.124,  # ppm (BSE abundance)
        radio_K=310,  # ppm (BSE abundance)
    )
    interior_struct = SimpleNamespace(
        core_frac=0.55,  # Core radius fraction
        core_frac_mode='radius',
        core_density=11000.0,  # kg/m^3
        module='self',
    )
    planet = SimpleNamespace(mass_tot=1.0, tsurf_init=3500.0)  # Earth masses, K
    star = SimpleNamespace(age_ini=0.1)  # Gyr
    return SimpleNamespace(
        interior_energetics=interior_energetics,
        interior_struct=interior_struct,
        planet=planet,
        star=star,
    )


@pytest.fixture
def mock_interior():
    """
    Create an Interior_t-shaped SimpleNamespace for coupling with BoundaryRunner.

    SimpleNamespace prevents auto-attribute leaks: any field the source
    reads that the fixture has not explicitly set raises AttributeError
    rather than silently returning a Mock that could trip a comparison
    or a numpy conversion.

    **Physical Basis**: Interior evolution object state at a magma ocean cooling timestep.
    """
    return SimpleNamespace(
        ic=0,  # Initial condition type (0 = not first step)
        tides=[0.0],  # No tidal heating by default
        phi=None,
        mass=None,
        visc=None,
        density=None,
        temp=None,
        pres=None,
        radius=None,
    )


@pytest.fixture
def mock_atmos():
    """
    Create an Atmos_t-shaped SimpleNamespace for atmospheric coupling.

    The source code uses ``getattr(getattr(atmos_o, '_atm', None),
    'layer_cp', None)`` to read the per-layer heat capacity. With a
    SimpleNamespace, the two getattr calls return the explicit
    attributes; a missing ``_atm`` correctly resolves to None.

    **Physical Basis**: Thin H2-rich atmosphere on early magma ocean with
    heat capacity typical of light molecular atmospheres.
    """
    return SimpleNamespace(
        _atm=SimpleNamespace(
            layer_cp=np.array([1.7e4, 1.7e4, 1.7e4]),  # J/kg/K for H2 layers
            layer_σ=np.array([1e18, 1e18, 1e18]),  # kg for H2 layers
        ),
    )


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

    ``next_step`` is patched out so the fixture does not require a fully-
    populated ``config.params.dt.*`` block on the Mock.
    """
    with patch('proteus.interior_energetics.boundary.next_step', return_value=1.0e3):
        yield BoundaryRunner(
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
    assert eta_solid == pytest.approx(1e21, rel=0.6)

    # Test fully molten
    eta_melt = boundary_runner.viscosity_aggregate_model(1.0)
    assert eta_melt == pytest.approx(1e2, rel=0.2)

    # Test intermediate: should be between melt and solid
    eta_mid = boundary_runner.viscosity_aggregate_model(0.5)
    assert 1e2 < eta_mid < 1e21


@pytest.mark.parametrize(
    'phi,expected_eta_range',
    [
        (0.1, (1e19, 1e22)),  # Mostly solid
        (0.5, (1e2, 1e21)),  # Transition zone
        (0.9, (1e2, 1e5)),  # Mostly liquid
    ],
)
@pytest.mark.physics_invariant
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
    # Positivity invariant: viscosity is computed as 10**(log_eta), a
    # log-linear blend of two strictly positive bounds; the result is
    # always positive. A sign-error regression that emitted a negative
    # log_eta from a swapped (z, 1-z) blend would still yield a
    # positive 10**, but a regression that returned a raw negative
    # blend (forgetting the exponentiation) would fail this.
    assert eta > 0.0


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
    assert eta_cold > 0
    assert eta_hot > 0


@pytest.mark.physics_invariant
def test_viscosity_arrhenius_magma_ocean(boundary_runner):
    """
    Test Arrhenius viscosity above critical melt fraction (magma ocean Vogel-Fulcher-Tammann).

    **Physical Scenario**: Hot magma ocean (T_p = 3500 K, φ = 0.9) with
    Vogel-Fulcher-Tammann viscosity and crystal fraction weakening.

    **Expected Behavior**: η(T) ∝ exp(D/(T - T_ref)); temperature-sensitive,
    typically 1-100 Pa*s for silicate melts at high temperatures.
    """
    boundary_runner.viscosity_model = 3
    boundary_runner.viscosity_prefactor = 1e-2  # Pa·s
    boundary_runner.viscosity_activation_temp = 2000.0  # K
    boundary_runner.critical_melt_fraction = 0.5

    T_hot = 3500.0  # K (magma ocean)
    phi = 0.9  # Mostly molten

    eta = boundary_runner.viscosity_arrhenius(T_hot, phi)

    # Magma ocean viscosity should be low (~0.001-1000 Pa*s)
    assert 1e-3 < eta < 1e4
    # Positivity invariant: viscosity is a physical resistance, so
    # must be positive everywhere. A regression that flipped the
    # sign of the crystal-weakening exponent could yield a negative
    # eta from a denominator going negative.
    assert eta > 0.0


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
    # T-independence guard: model==1 must ignore T_p / T_surf / phi.
    # A regression that fell through to the aggregate model would
    # blend toward eta_melt_const (1e2) at phi=0.8 and miss 1e21 by
    # nineteen orders of magnitude.
    eta_other = boundary_runner.viscosity(1500.0, 300.0, 0.1)
    assert eta_other == pytest.approx(eta, rel=1e-12)


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
    # Identity guard: when model==2, the dispatcher must return the
    # aggregate-model value, not the constant-model value (1e2 here
    # via the default eta_constant from the fixture). The two are
    # well-separated in this regime.
    expected = boundary_runner.viscosity_aggregate_model(phi)
    assert eta == pytest.approx(expected, rel=1e-12)


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
    expected_eta = boundary_runner.viscosity_arrhenius(T_p, phi)
    assert eta == pytest.approx(expected_eta, rel=1e-12)
    # Discriminator vs the constant-model fallback: with the fixture
    # defaults eta_constant=1e2, the Arrhenius branch at T_p=2000 K,
    # E_a=3e5 J/mol returns a value many orders of magnitude away
    # from 1e2. A dispatcher regression that routed model==3 to the
    # constant path would land at 1e2 and fail this check.
    assert eta != pytest.approx(boundary_runner.eta_constant, rel=1e-2)


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
    # Fallback-identity guard: the unknown-model branch must dispatch
    # to the aggregate model, not to the constant-eta_constant path
    # or to a raised exception. Compare to the aggregate output
    # directly so a future fallback change has to be intentional.
    expected = boundary_runner.viscosity_aggregate_model(0.5)
    assert eta == pytest.approx(expected, rel=1e-12)


# =============================================================================
# Tests: Rayleigh Number & Convection Scaling
# =============================================================================


@pytest.mark.physics_invariant
def test_rayleigh_number_physical_range(boundary_runner):
    """
    Test Rayleigh number falls within physically realistic range for planetary mantles.

    **Physical Scenario**: Ra = 10^6-10^10 represents transition from laminar to
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
    # Positivity invariant: every factor in the Rayleigh number is
    # strictly positive for this regime (rho, g, alpha, |ΔT|, L^3,
    # 1/(eta kappa)). A sign error in any factor would flip Ra
    # negative; the |T_p - T_surf| in the source must absorb any
    # ordering convention.
    assert ra > 0.0


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


@pytest.mark.physics_invariant
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
    # Pinned ratio: with constant viscosity, Ra is linear in |ΔT|.
    # ΔT_large / ΔT_small = (4000 - 300) / (2500 - 300) = 3700/2200
    # = 1.6818..., not 1.0 (constant-Ra bug) and not 2.56 (squared).
    expected_ratio = (4000.0 - 300.0) / (2500.0 - 300.0)
    assert ra_large_delta / ra_small_delta == pytest.approx(expected_ratio, rel=1e-10)


# =============================================================================
# Tests: Convective Heat Flux (q_m)
# =============================================================================


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


@pytest.mark.physics_invariant
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
    # Positivity invariant: heat flux is computed from |T_p - T_surf|
    # and Nu>0, so both must be strictly positive. A sign-error
    # regression that returned -|q_m| would fail this guard.
    assert q_m_small_delta > 0.0 and q_m_large_delta > 0.0


@pytest.mark.physics_invariant
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
    # Positivity invariant: q_m derives from |T_p - T_surf| * Nu, both
    # non-negative, so q_m must be non-negative even in the ΔT=0 limit.
    # A sign-error regression would land here as a small negative.
    assert q_m >= 0.0


# =============================================================================
# Tests: Radioactive Heating
# =============================================================================


def test_radioactive_heating_disabled(boundary_runner):
    """
    Test radioactive heating returns 0 when disabled in config.

    **Physical Scenario**: radiogenic_heat = False (default).
    """
    boundary_runner.use_radiogenic_heating = False

    t = 1e9 * secs_per_year  # 1 Gyr after initial formation

    h_radio = boundary_runner.radioactive_heating(t)

    assert h_radio == pytest.approx(0.0, abs=1e-12)
    # Discriminating guard: the disabled branch must short-circuit at
    # any time argument, not only at t=1 Gyr. A regression that gated
    # on time rather than the flag would fire for at least one of
    # these two probes.
    assert boundary_runner.radioactive_heating(0.0) == pytest.approx(0.0, abs=1e-12)


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
    # Discriminating scale check: H_radio at formation is ~1e-11 W/kg for
    # Earth-like crustal abundances (Turcotte & Schubert 2014, ch. 4.5). A
    # regression that returned a kg/g unit-mismatched value would land outside
    # the 1e-13 to 1e-9 W/kg envelope and fail this pin.
    assert 1e-13 < h_radio < 1e-9


@pytest.mark.physics_invariant
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
    # Positivity invariant: each isotope contributes a non-negative
    # decay term, so the sum is strictly positive while any half-life
    # > 0. A sign-flip in the exp(-lambda t) factor would invert the
    # primary monotonicity check; positivity catches the case where
    # both endpoints drop below zero together.
    assert h_late > 0.0


# =============================================================================
# Tests: Melt Fraction
# =============================================================================


@pytest.mark.physics_invariant
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
    # Boundedness invariant: without the np.clip lower bound the raw
    # linear formula gives phi = (1500 - 1600) / 400 = -0.25, a
    # negative melt fraction. The bound is the contract.
    assert phi >= 0.0


@pytest.mark.physics_invariant
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
    # Boundedness invariant: without the np.clip upper bound the raw
    # linear formula gives phi = (2500 - 1600) / (2000 - 1600) = 2.25,
    # well outside [0, 1]. The bound is the contract.
    assert phi <= 1.0


@pytest.mark.parametrize(
    'T_p,expected_phi',
    [
        (1600.0, 0.0),  # At solidus
        (1800.0, 0.5),  # Midpoint
        (2000.0, 1.0),  # At liquidus
    ],
)
@pytest.mark.physics_invariant
def test_melt_fraction_parametrized(boundary_runner, T_p, expected_phi):
    """
    Test melt fraction linearly interpolates between solidus and liquidus.

    **Physical Scenario**: Simplified linear phase diagram for silicates.
    """
    boundary_runner.T_solidus = 1600.0
    boundary_runner.T_liquidus = 2000.0

    phi = boundary_runner.melt_fraction(T_p)

    assert phi == pytest.approx(expected_phi, rel=1e-10)
    # Boundedness invariant: phi must always live in [0, 1]; the three
    # parametrized inputs span the closed interval and each must respect
    # the bound. A regression that returned the raw (unclipped) ratio
    # would fail this on the endpoints if T_p drifted across them.
    assert 0.0 <= phi <= 1.0


# =============================================================================
# Tests: Solidification Radius
# =============================================================================


@pytest.mark.physics_invariant
def test_r_s_fully_molten(boundary_runner):
    """
    Test solidification radius equals the core radius when fully molten.

    **Physical Scenario**: T_p ≥ T_liquidus → the solidification front retreats to
    the branch's fully molten lower bound.
    """
    boundary_runner.T_liquidus = 2000.0

    T_p = 2500.0  # Above liquidus (φ = 1)

    r_s = boundary_runner.r_s(T_p)

    assert r_s == pytest.approx(boundary_runner.core_radius, rel=1e-10)
    # Boundedness invariant: r_s never exceeds the planet radius and
    # never falls below the fully molten lower bound.
    assert boundary_runner.core_radius <= r_s <= boundary_runner.planet_radius


@pytest.mark.physics_invariant
def test_r_s_fully_solid(boundary_runner):
    """
    Test solidification radius equals planet radius when fully solid (φ = 0).

    **Physical Scenario**: T_p ≤ T_solidus → entire mantle solid → r_s = R_planet.
    """
    boundary_runner.T_solidus = 1600.0

    T_p = 1500.0  # Below solidus (φ = 0)

    r_s = boundary_runner.r_s(T_p)

    assert r_s == pytest.approx(boundary_runner.planet_radius, rel=1e-10)
    # Boundedness invariant: r_s must lie in [r_core, R_planet]; the
    # fully-solid limit pins the upper bound. A swapped-branch
    # regression that returned the core radius for phi=0 would
    # fail both this and the primary assertion.
    assert boundary_runner.core_radius <= r_s <= boundary_runner.planet_radius


@pytest.mark.physics_invariant
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

    # Should be strictly between the fully molten lower bound and the surface.
    assert boundary_runner.core_radius < r_s < boundary_runner.planet_radius
    # Monotonicity invariant: increasing T_p melts more mantle so the
    # solidification front retreats inward. r_s(T_higher) must be
    # strictly less than r_s(T_lower) in the partially-molten regime.
    r_s_hotter = boundary_runner.r_s(1900.0)
    assert r_s_hotter < r_s


# =============================================================================
# Tests: Temperature Derivatives
# =============================================================================


def test_r_s_decreases_with_temperature_in_partially_molten_regime(boundary_runner):
    """
    Test solidification radius decreases as potential temperature rises.

    **Physical Scenario**: Hotter T_p → more melt → smaller solid radius.
    """
    boundary_runner.T_solidus = 1600.0
    boundary_runner.T_liquidus = 2000.0

    r_s_cooler = boundary_runner.r_s(1700.0)
    r_s_mid = boundary_runner.r_s(1800.0)
    r_s_hotter = boundary_runner.r_s(1900.0)

    assert r_s_cooler > r_s_mid > r_s_hotter
    assert boundary_runner.core_radius < r_s_hotter < boundary_runner.planet_radius


@pytest.mark.physics_invariant
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
    # Finiteness invariant: a regression that returned NaN (div-by-zero
    # in denominator when the mantle goes fully solid) or inf (degenerate
    # latent-heat term) would pass the sign check (NaN < 0 is False, but
    # inf-cooling is < 0) and silently corrupt the ODE integration.
    assert np.isfinite(dTp_dt)


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

    q_m = boundary_runner.q_m(T_p, T_surf, boundary_runner.melt_fraction(T_p))
    assert np.sign(dTs_dt) == np.sign(q_m - boundary_runner.f_atm)
    assert np.isfinite(dTs_dt)


# =============================================================================
# Tests: ODE System (thermal_rhs)
# =============================================================================


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


@pytest.mark.physics_invariant
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
    # Finiteness invariant: continuous derivatives are useless if the
    # base evaluation diverged. Pin that both components are finite
    # so an inf/NaN regression cannot pass the smallness check by
    # subtracting two infinities.
    assert all(np.isfinite(v) for v in rhs_nominal)


# =============================================================================
# Tests: ODE Solver Integration (run_solver)
# =============================================================================


@patch('proteus.interior_energetics.boundary.solve_ivp')
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


@patch('proteus.interior_energetics.boundary.solve_ivp')
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
    # Conservation of mantle mass: M_mantle = M_mantle_liquid + M_mantle_solid.
    # A regression that double-counted or lost mass between phases
    # would break this closure beyond float-roundoff. This is the
    # mass-closure invariant for the solid-liquid mantle split.
    assert output['M_mantle'] == pytest.approx(
        output['M_mantle_liquid'] + output['M_mantle_solid'], rel=1e-10
    )


@patch('proteus.interior_energetics.boundary.solve_ivp')
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


def test_compute_time_step_first_iteration(mock_config, mock_dirs):
    """
    Test compute_time_step returns 0 for first iteration (ic=1).

    **Physical Scenario**: Initial condition (ic=1) → no evolution → dt=0.
    """
    interior = SimpleNamespace(ic=1, tides=[0.0])

    dt = BoundaryRunner.compute_time_step(mock_config, mock_dirs, {}, None, interior)

    assert dt == pytest.approx(0.0, abs=1e-12)
    # Type pin: the ic=1 branch returns the literal 0.0 (float); a regression
    # that called next_step and got 0 back would yield int 0 only on the
    # contrived case of next_step returning int(0). The two-part check
    # discriminates "early return path taken" from "next_step path with a
    # zero return", since the latter would also have called next_step and
    # any side effect there would be reflected via mock_next_step assertions
    # in the sister test test_compute_time_step_normal_iteration.
    assert isinstance(dt, float)


@patch('proteus.interior_energetics.boundary.next_step')
def test_compute_time_step_normal_iteration(mock_next_step, mock_config, mock_dirs):
    """
    Test compute_time_step calls next_step for normal iterations (ic ≠ 1).

    **Physical Scenario**: Subsequent timesteps → adaptive timestep from next_step().
    """
    mock_next_step.return_value = 1e-5  # Myr as returned by next_step

    interior = SimpleNamespace(ic=0)

    dt = BoundaryRunner.compute_time_step(mock_config, mock_dirs, {}, None, interior)

    # Should call next_step
    mock_next_step.assert_called_once()
    assert dt == pytest.approx(1e-5, rel=1e-12)


# =============================================================================
# Integration & Regression Tests
# =============================================================================


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
            assert dTp_dt <= 0  # Cooling or neutral when ΔT -> 0


# =============================================================================
# Edge Case & Error Handling Tests
# =============================================================================


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


# ============================================================================
# Regression: all-NaN layer_cp must fall back to atm_heat_capacity
# ============================================================================


def test_boundary_runner_all_nan_layer_cp_falls_back_to_atm_heat_capacity(
    mock_config, mock_atmos, mock_hf_row, mock_hf_all, mock_interior, mock_dirs
):
    """Regression for the const_heat_capacity AttributeError audit.

    Before the fix, an all-NaN ``atmos_o._atm.layer_cp`` array would
    fall through to ``config.interior_energetics.boundary.const_heat_capacity``,
    which is not a field of ``InteriorBoundary``. The only configured
    field is ``atm_heat_capacity``. Reading the wrong attribute name
    raised AttributeError at __init__, breaking the boundary backend
    on any first-call-after-restart where AGNI's layer_cp had not yet
    been populated to finite values.

    This test mutates the fixture's ``layer_cp`` to all-NaN before
    constructing the runner and confirms (a) no AttributeError, and
    (b) the configured atm_heat_capacity is used as the fallback.
    """
    mock_config.interior_energetics.boundary.atm_heat_capacity = 1234.5

    # Force the all-NaN branch.
    mock_atmos._atm.layer_cp = np.array([np.nan, np.nan, np.nan])
    mock_atmos._atm.layer_σ = np.array([1.0, 1.0, 1.0])

    with patch('proteus.interior_energetics.boundary.next_step', return_value=1.0e3):
        runner = BoundaryRunner(
            mock_config, mock_dirs, mock_hf_row, mock_hf_all, mock_interior, mock_atmos
        )

    assert runner.atmosphere_heat_capacity == pytest.approx(1234.5), (
        'all-NaN layer_cp must fall back to config.interior_energetics.boundary.'
        'atm_heat_capacity, not the non-existent const_heat_capacity'
    )
    # Positivity invariant: a heat capacity must be strictly positive.
    # A regression that fell back to 0.0 (e.g. silently swallowed
    # AttributeError) would produce a div-by-zero in dT_surfdt downstream.
    assert runner.atmosphere_heat_capacity > 0.0


def test_boundary_runner_partial_nan_layer_cp_averages_finite_values(
    mock_config, mock_atmos, mock_hf_row, mock_hf_all, mock_interior, mock_dirs
):
    """When ``layer_cp`` has both NaN and finite entries, the runner
    must use the mean of the finite ones. This is the most common
    realistic case: a few layers fail Cp evaluation but most succeed."""
    mock_config.interior_energetics.boundary.atm_heat_capacity = 9999.9

    mock_atmos._atm.layer_cp = np.array([1.0e4, np.nan, 2.0e4, np.nan, 3.0e4])
    mock_atmos._atm.layer_σ = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    with patch('proteus.interior_energetics.boundary.next_step', return_value=1.0e3):
        runner = BoundaryRunner(
            mock_config, mock_dirs, mock_hf_row, mock_hf_all, mock_interior, mock_atmos
        )

    # Mean of [1e4, 2e4, 3e4] = 2e4. The atm_heat_capacity fallback
    # value (9999.9) must NOT be used because at least one finite
    # layer exists.
    assert runner.atmosphere_heat_capacity == pytest.approx(2.0e4)
    # Discriminating guard: the fallback constant 9999.9 differs from
    # the 2e4 mean by 1e4. A regression that took the fallback branch
    # by treating any NaN as triggering "all-NaN" would fail this.
    assert abs(runner.atmosphere_heat_capacity - 9999.9) > 5e3


# =============================================================================
# Tests: BoundaryRunner init edge cases + run_solver logging path.
# Targets lines 66-100, 106, 378, 567, 658-720 in boundary.py.
# =============================================================================


def test_boundary_runner_init_uses_history_temperatures_for_ic2(
    mock_config, mock_dirs, mock_hf_row, mock_hf_all, mock_interior, mock_atmos
):
    """When ``interior_o.ic == 2``, BoundaryRunner initializes from the
    history row rather than the configured surface temperature.

    The branch now copies the stored magma and surface temperatures
    directly, so the test should pin that contract instead of the
    removed constructor clamp.
    """
    mock_config.interior_struct.module = 'self'
    mock_interior.ic = 2
    mock_hf_row['T_magma'] = 3495.0
    mock_hf_row['T_surf'] = 3500.0

    with patch('proteus.interior_energetics.boundary.next_step', return_value=1.0e3):
        runner = BoundaryRunner(
            mock_config, mock_dirs, mock_hf_row, mock_hf_all, mock_interior, mock_atmos
        )

    assert runner.T_p_0 == pytest.approx(3495.0, rel=1e-12)
    assert runner.T_surf_0 == pytest.approx(3500.0, rel=1e-12)
    assert runner.T_surf_0 > runner.T_p_0


def test_boundary_layer_thickness_returns_tiny_delta_at_equal_temperatures(
    boundary_runner,
):
    """At T_p == T_surf the temperature gradient vanishes and the
    boundary_layer_thickness formula divides by zero. The source
    guards this with a tiny-delta (1e-3) fallback at line 378.

    Edge: limit-input case. Discriminating: a regression that removed
    the guard would raise ZeroDivisionError or return NaN; both fail
    the finite-positive assertion below.
    """
    delta = boundary_runner.boundary_layer_thickness(T_p=1500.0, T_surf=1500.0, phi=1.0)
    assert delta == pytest.approx(1e-3, rel=1e-12)
    # Sign + scale guards: a tiny but strictly positive value.
    assert delta > 0
    assert delta < 1.0


@pytest.mark.physics_invariant
def test_boundary_layer_thickness_inverse_to_heat_flux(boundary_runner):
    """When T_p > T_surf, delta = k * (T_p - T_surf) / q_m. The
    thickness scales inversely with the heat flux: doubling the
    temperature difference roughly doubles delta while q_m grows.

    Discriminating: this is a non-trivial physical relationship,
    not a closed-form pin. Pin only the sign and the bounded range
    so the formula's structural correctness is verified without
    over-constraining the viscosity model output.
    """
    delta_small = boundary_runner.boundary_layer_thickness(T_p=1800.0, T_surf=1600.0, phi=1.0)
    delta_big = boundary_runner.boundary_layer_thickness(T_p=3000.0, T_surf=1600.0, phi=1.0)
    # Sign + boundedness: both thicknesses positive and well above
    # the tiny-delta fallback (1e-3 m); both below the mantle depth
    # (~3000 km).
    assert delta_small > 1e-3
    assert delta_big > 1e-3
    assert delta_small < 3e6
    assert delta_big < 3e6


@pytest.mark.physics_invariant
def test_dT_pdt_uses_silicate_heat_capacity_for_fully_solid_mantle(boundary_runner):
    """In the fully-solid regime (T_p below the solidus), the
    denominator of dT_p/dt reduces to silicate_heat_capacity *
    mantle_mass; the latent-heat term in the partially-molten branch
    is exactly zero because r_s_val == planet_radius.

    Discriminating: re-derive the numerator and denominator at the
    test parameters and pin dT_pdt to the closed-form ratio. A
    regression that flipped the if-condition (`<=` instead of `<`)
    would take the partial-melt branch and produce a denominator
    with the latent-heat term, landing at a different value.
    """
    import numpy as np

    runner = boundary_runner
    T_p_solid = 1300.0  # below configured solidus (1420 K) -> phi = 0
    T_surf = 1200.0
    t = 0.0
    dT_pdt = runner.dT_pdt(T_p_solid, T_surf, t=t)
    # Sanity: a real cooling rate (negative, finite, mantle-physics scale).
    assert np.isfinite(dT_pdt)
    assert dT_pdt < 0

    # Re-derive the expected value from the closed-form else-branch.
    phi = runner.melt_fraction(T_p_solid)
    assert phi == pytest.approx(0.0, abs=1e-12)
    q_m_val = runner.q_m(T_p_solid, T_surf, phi)
    Q_val = runner.radioactive_heating(t)
    numerator = -4 * np.pi * runner.planet_radius**2 * q_m_val + runner.mantle_mass * (
        Q_val + runner.tidal_term
    )
    denominator_else = runner.mantle_mass * runner.silicate_heat_capacity
    expected = numerator / denominator_else
    assert dT_pdt == pytest.approx(expected, rel=1e-12)
    # The fully solid branch must drop the latent-heat correction.
    volume_diff = runner.planet_radius**3 - runner.core_radius**3
    T_range = runner.T_liquidus - runner.T_solidus
    latent_heat_term = (
        -4
        * np.pi
        * runner.heat_fusion_silicate
        * runner.mantle_bulk_density
        * volume_diff
        / (3 * T_range)
    )
    denominator_wrong_branch = denominator_else - latent_heat_term
    wrong_partial = numerator / denominator_wrong_branch
    assert abs(dT_pdt - wrong_partial) > 1e-12


def test_run_solver_writes_csv_log_when_logging_enabled(
    mock_config, mock_dirs, mock_hf_row, mock_hf_all, mock_interior, mock_atmos, tmp_path
):
    """When logging is enabled, run_solver writes a CSV row per call.
    The header is written on first call (csv_needs_header path at
    line 671). Lines 657-720 of boundary.py are gated on
    self.logging.

    Discriminating: confirm the CSV exists, has the expected header,
    and contains exactly the rows we expect after running once. A
    regression that broke the header-write would leave only data
    rows (no leading header), failing the field-name check.
    """
    mock_config.interior_energetics.boundary.logging = True
    mock_dirs_local = {'output': str(tmp_path)}
    with patch('proteus.interior_energetics.boundary.next_step', return_value=1.0e3):
        runner = BoundaryRunner(
            mock_config,
            mock_dirs_local,
            mock_hf_row,
            mock_hf_all,
            mock_interior,
            mock_atmos,
        )
    fake_sol = SimpleNamespace(
        t=np.array([0.0, 1.0e3]),
        y=np.array([[3500.0, 3490.0], [1600.0, 1605.0]]),
        status=0,
    )
    with patch('proteus.interior_energetics.boundary.solve_ivp', return_value=fake_sol):
        # run_solver signature: (hf_row, interior_o, dirs)
        runner.run_solver(mock_hf_row, mock_interior, mock_dirs_local)
    csv_path = tmp_path / 'boundary_solver_debug.csv'
    assert csv_path.exists()
    text = csv_path.read_text()
    # Header line must list the expected columns; pin three of them
    # so a column-rename regression surfaces here.
    assert 'Time_years' in text
    assert 'T_p_K' in text
    assert 'rayleigh_number' in text
    # At least one data row was written.
    assert text.count('\n') >= 2


# =============================================================================
# Tests: atm_heat_capacity_const=True branch (new config flag)
# =============================================================================


def test_boundary_runner_uses_config_heat_capacity_when_const_flag_true(
    mock_config, mock_dirs, mock_hf_row, mock_hf_all, mock_interior, mock_atmos
):
    """When atm_heat_capacity_const=True, BoundaryRunner must read
    atmosphere_heat_capacity directly from config and ignore the
    per-layer arrays in atmos_o._atm.
    """
    mock_config.interior_energetics.boundary.atm_heat_capacity_const = True
    mock_config.interior_energetics.boundary.atm_heat_capacity = 3.0e4

    with patch('proteus.interior_energetics.boundary.next_step', return_value=1.0e3):
        runner = BoundaryRunner(
            mock_config, mock_dirs, mock_hf_row, mock_hf_all, mock_interior, mock_atmos
        )

    # Must use the config value, not the per-layer average (1.7e4).
    assert runner.atmosphere_heat_capacity == pytest.approx(3.0e4)

    # Discrimination: the per-layer average would be 1.7e4; the gap is
    # 1.3e4, more than 10 % of the expected value.
    assert abs(runner.atmosphere_heat_capacity - 1.7e4) > 1e4


# =============================================================================
# Tests: run_solver solver-status warnings
# =============================================================================


def test_run_solver_logs_warning_when_tsurf_event_terminates_step(
    boundary_runner, mock_interior, mock_dirs, mock_hf_row, caplog
):
    """run_solver logs a warning when the ODE integrator stops early because
    the surface-temperature event fired (sol.status == 1).
    """
    fake_sol = SimpleNamespace(
        t=np.array([0.0, 500.0]),
        y=np.array([[3500.0, 3490.0], [1600.0, 1650.0]]),
        status=1,  # event triggered
        message='A termination event occurred.',
    )
    import logging

    with patch('proteus.interior_energetics.boundary.solve_ivp', return_value=fake_sol):
        with caplog.at_level(
            logging.WARNING, logger='fwl.proteus.interior_energetics.boundary'
        ):
            boundary_runner.run_solver(mock_hf_row, mock_interior, mock_dirs)

    warning_texts = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any('Tsurf event' in msg for msg in warning_texts), (
        'Expected a warning about Tsurf event termination when sol.status==1'
    )
    # Discrimination: the status==-1 error message must NOT appear here.
    assert not any('failed to integrate' in msg for msg in warning_texts)


def test_run_solver_logs_warning_when_integration_fails(
    boundary_runner, mock_interior, mock_dirs, mock_hf_row, caplog
):
    """run_solver logs a warning when solve_ivp reports a solver error
    (sol.status == -1).

    The status==-1 branch at boundary.py:731-733 records a two-line warning
    that includes the solver message. Verify both lines appear.

    Discriminating: status==1 and status==0 each take different branches;
    pin the -1 warning text and confirm the status==1 message is absent.
    """
    fake_sol = SimpleNamespace(
        t=np.array([0.0, 100.0]),
        y=np.array([[3500.0, 3499.0], [1600.0, 1600.5]]),
        status=-1,
        message='Required step size is less than spacing between numbers.',
    )
    import logging

    with patch('proteus.interior_energetics.boundary.solve_ivp', return_value=fake_sol):
        with caplog.at_level(
            logging.WARNING, logger='fwl.proteus.interior_energetics.boundary'
        ):
            boundary_runner.run_solver(mock_hf_row, mock_interior, mock_dirs)

    warning_texts = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any('failed to integrate' in msg for msg in warning_texts)
    assert any('Required step size' in msg for msg in warning_texts), (
        'Expected the solver message to appear in the second warning line'
    )
    # Discrimination: the Tsurf-event message must NOT appear for status==-1.
    assert not any('Tsurf event' in msg for msg in warning_texts)


# =============================================================================
# Tests: tsurf_change_event function (closure defined inside run_solver)
# =============================================================================


def test_tsurf_change_event_returns_zero_at_threshold(
    boundary_runner, mock_interior, mock_dirs, mock_hf_row
):
    """The tsurf_change_event callback inside run_solver evaluates to zero
    exactly when |T_surf - T_surf_0| == Tsurf_event_change, and to a
    negative value below the threshold.

    Physics: the event triggers solve_ivp termination when the surface
    temperature deviates from its recently-initialised value by more than Tsurf_event_change.
    This prevents large changes during a single integration of the BL model.
    """
    captured = {}

    fake_sol = SimpleNamespace(
        t=np.array([0.0, 1.0e3]),
        y=np.array([[3500.0, 3490.0], [1600.0, 1605.0]]),
        status=0,
        message='',
    )

    def _capture_solve_ivp(fun, t_span, y0, **kwargs):
        captured['events'] = kwargs.get('events')
        return fake_sol

    with patch(
        'proteus.interior_energetics.boundary.solve_ivp',
        side_effect=_capture_solve_ivp,
    ):
        boundary_runner.run_solver(mock_hf_row, mock_interior, mock_dirs)

    event_fn = captured.get('events')
    assert event_fn is not None, 'solve_ivp was not called with an events kwarg'

    T_surf_0 = boundary_runner.T_surf_0
    delta = boundary_runner.Tsurf_event_change

    # At the threshold: |T_surf - T_surf_0| == delta → event value == 0.
    val_at_threshold = event_fn(0.0, [3000.0, T_surf_0 + delta])
    assert val_at_threshold == pytest.approx(0.0, abs=1e-12)

    # Below threshold: event value is negative (event has not fired).
    val_below = event_fn(0.0, [3000.0, T_surf_0 + 0.5 * delta])
    assert val_below < 0.0

    # Above threshold: event value is positive (event fires).
    val_above = event_fn(0.0, [3000.0, T_surf_0 + 2.0 * delta])
    assert val_above > 0.0

    # Discrimination: the event must depend on y[1] (T_surf), not y[0] (T_p).
    # Varying T_p while keeping T_surf fixed must leave the event value unchanged.
    val_tp_high = event_fn(0.0, [5000.0, T_surf_0 + 0.5 * delta])
    assert val_tp_high == pytest.approx(val_below, abs=1e-12)


# =============================================================================
# Tests: core_frac_mode dispatch in BoundaryRunner.__init__
# (boundary.py lines 93-110 – the three branches reached only when
# hf_row does not contain 'R_core')
# =============================================================================


def test_boundary_runner_init_core_frac_mode_radius_computes_from_config(
    mock_config, mock_dirs, mock_hf_row, mock_hf_all, mock_interior, mock_atmos
):
    """When hf_row lacks R_core and core_frac_mode='radius', the core radius
    equals core_frac * planet_radius.

    Physical scenario: rocky planet with a prescribed core-radius fraction
    (0.55 × R_earth) and no Zalmoxis CMB radius in the history row yet.
    The formula is purely geometric: R_core = f * R_planet.

    Discriminating: the mass-based formula at the same M_core and rho_core
    produces ~4.15 Mm, whereas the radius-fraction formula gives
    0.55 * 6.371 Mm = 3.504 Mm. The two differ by ~640 km, well outside
    the 1e-6 relative tolerance of the primary pin.
    """
    hf_row_no_r_core = {k: v for k, v in mock_hf_row.items() if k != 'R_core'}
    mock_config.interior_struct.core_frac_mode = 'radius'
    mock_config.interior_struct.core_frac = 0.55

    with patch('proteus.interior_energetics.boundary.next_step', return_value=1.0e3):
        runner = BoundaryRunner(
            mock_config, mock_dirs, hf_row_no_r_core, mock_hf_all, mock_interior, mock_atmos
        )

    expected = 0.55 * 6.371e6  # 3.50405e6 m
    assert runner.core_radius == pytest.approx(expected, rel=1e-6)
    # Boundedness: core must sit strictly inside the planet.
    assert 0.0 < runner.core_radius < runner.planet_radius
    # core_frac must equal the configured fraction exactly (no mass rescaling).
    assert runner.core_frac == pytest.approx(0.55, rel=1e-10)
    # Exponent-error / formula-swap guard: the mass-based formula gives ~4.15 Mm;
    # the separation from the radius-fraction result (~3.50 Mm) is > 600 km.
    r_mass_based = (3.0 * 0.55 * M_earth / (4.0 * np.pi * 11000.0)) ** (1.0 / 3.0)
    assert abs(runner.core_radius - r_mass_based) > 5e5


@pytest.mark.physics_invariant
def test_boundary_runner_init_core_frac_mode_mass_computes_from_sphere_volume(
    mock_config, mock_dirs, mock_hf_row, mock_hf_all, mock_interior, mock_atmos
):
    """When hf_row lacks R_core and core_frac_mode='mass', the core radius
    is derived from the sphere-volume formula: r = (3M / (4 pi rho))^(1/3).

    Physical scenario: iron-core planet where the CMB radius follows from
    core mass (0.55 M_earth) and a fixed core density (11 000 kg/m^3).

    Discriminating: the expected value is pinned to the closed-form result.
    A regression that dropped the cube-root (wrong exponent = 1 instead of 1/3)
    would return ~7e19 m; a unit-conversion bug (kg vs g) would shift by 1000^(1/3)
    ≈ 10x. Both land > 1e10 m from the correct answer.
    """
    hf_row_no_r_core = {k: v for k, v in mock_hf_row.items() if k != 'R_core'}
    mock_config.interior_struct.core_frac_mode = 'mass'
    mock_config.interior_struct.core_density = 11000.0  # kg/m^3

    with patch('proteus.interior_energetics.boundary.next_step', return_value=1.0e3):
        runner = BoundaryRunner(
            mock_config, mock_dirs, hf_row_no_r_core, mock_hf_all, mock_interior, mock_atmos
        )

    M_core_val = 0.55 * M_earth
    rho = 11000.0
    expected = (3.0 * M_core_val / (4.0 * np.pi * rho)) ** (1.0 / 3.0)

    assert runner.core_radius == pytest.approx(expected, rel=1e-10)
    # Sign guard: radius is always positive.
    assert runner.core_radius > 0.0
    # Scale guard: iron-core radius for ~0.55 M_earth should be 3.5–5 Mm.
    assert 3e6 < runner.core_radius < 5e6
    # Exponent-error guard: missing the cube-root exponent would give ~7e19 m.
    wrong_no_exponent = 3.0 * M_core_val / (4.0 * np.pi * rho)
    assert abs(runner.core_radius - wrong_no_exponent) > 1e10
    # Boundedness invariant: core radius must be < planet radius.
    assert runner.core_radius < runner.planet_radius


@pytest.mark.physics_invariant
def test_boundary_runner_init_core_frac_mode_mass_self_density_uses_boundary_config(
    mock_config, mock_dirs, mock_hf_row, mock_hf_all, mock_interior, mock_atmos
):
    """When core_frac_mode='mass' and core_density='self', the code falls back to
    config.interior_energetics.boundary.core_density instead of the
    interior_struct density.

    Physical scenario: the interior-struct module owns its own EOS density; the
    boundary module uses a separately configured reference iron density (8000 kg/m^3
    here) to compute the CMB radius.

    Discriminating: rho=8000 gives r≈4.61 Mm, rho=11000 gives r≈4.15 Mm.
    The separation is ~460 km — any regression that read the wrong density
    source fails the primary pin at rel=1e-10.
    """
    hf_row_no_r_core = {k: v for k, v in mock_hf_row.items() if k != 'R_core'}
    mock_config.interior_struct.core_frac_mode = 'mass'
    mock_config.interior_struct.core_density = 'self'  # triggers the self-density branch
    mock_config.interior_energetics.boundary.core_density = 8000.0  # kg/m^3

    with patch('proteus.interior_energetics.boundary.next_step', return_value=1.0e3):
        runner = BoundaryRunner(
            mock_config, mock_dirs, hf_row_no_r_core, mock_hf_all, mock_interior, mock_atmos
        )

    M_core_val = 0.55 * M_earth
    rho_boundary = 8000.0
    expected = (3.0 * M_core_val / (4.0 * np.pi * rho_boundary)) ** (1.0 / 3.0)

    assert runner.core_radius == pytest.approx(expected, rel=1e-10)
    # Sign guard: physical radius must be strictly positive.
    assert runner.core_radius > 0.0
    # Discrimination: rho=8000 vs rho=11000 shifts the radius by ~460 km.
    # A regression that used interior_struct.core_density (the string 'self')
    # would fail to parse and crash, while one using the 11000 fallback would
    # produce a radius that differs by > 4e5 m from the expected value.
    r_wrong_density = (3.0 * M_core_val / (4.0 * np.pi * 11000.0)) ** (1.0 / 3.0)
    assert abs(runner.core_radius - r_wrong_density) > 4e5
    # Boundedness invariant: CMB must be inside the planet.
    assert runner.core_radius < runner.planet_radius


def test_boundary_runner_init_invalid_core_frac_mode_raises_value_error(
    mock_config, mock_dirs, mock_hf_row, mock_hf_all, mock_interior, mock_atmos
):
    """BoundaryRunner.__init__ raises ValueError when core_frac_mode is neither
    'radius' nor 'mass' and hf_row contains no R_core key.

    Error contract: the message must name both valid modes and reproduce the
    offending value so users can diagnose config mistakes without reading source.

    Edge: the error branch is only reachable when R_core is absent from hf_row.
    A supplied CMB radius short-circuits the mode check, so the same invalid
    mode must construct in that case; this pairs the raise with the condition
    that makes it reachable.
    """
    hf_row_no_r_core = {k: v for k, v in mock_hf_row.items() if k != 'R_core'}
    mock_config.interior_struct.core_frac_mode = 'volume'  # neither 'radius' nor 'mass'

    with pytest.raises(ValueError, match=r"core_frac_mode must be 'mass' or 'radius'"):
        with patch('proteus.interior_energetics.boundary.next_step', return_value=1.0e3):
            BoundaryRunner(
                mock_config,
                mock_dirs,
                hf_row_no_r_core,
                mock_hf_all,
                mock_interior,
                mock_atmos,
            )

    # With R_core present the mode is never consulted: the same invalid 'volume'
    # constructs and the CMB radius is taken from the row. This is what makes
    # the raise above attributable to the missing R_core rather than to the
    # config value alone. The row carries a Zalmoxis-style CMB radius at 0.42
    # R_int, away from the config's core_frac of 0.55, so the radius pins that
    # the value came from the row and not from the core_frac * R_int formula;
    # at 0.55 the two paths would agree bit-for-bit and prove nothing. This
    # path never consults M_core, so the row's core mass and this radius are
    # not meant to imply a core density.
    hf_row_zalmoxis_core = {**mock_hf_row, 'R_core': 0.42 * mock_hf_row['R_int']}
    with patch('proteus.interior_energetics.boundary.next_step', return_value=1.0e3):
        runner = BoundaryRunner(
            mock_config, mock_dirs, hf_row_zalmoxis_core, mock_hf_all, mock_interior, mock_atmos
        )
    assert runner.core_radius == pytest.approx(0.42 * mock_hf_row['R_int'])
    assert runner.core_frac == pytest.approx(0.42)


def test_boundary_runner_init_invalid_core_frac_mode_message_includes_bad_value(
    mock_config, mock_dirs, mock_hf_row, mock_hf_all, mock_interior, mock_atmos
):
    """The ValueError raised for an invalid core_frac_mode must include the
    offending value so the user can identify the misconfigured field.

    The message is built from the config value rather than hard-coded: two
    different invalid values each quote themselves. A message that quoted one
    fixed example value would satisfy only one of the two matches, and would
    send the other user looking for a field they never set.
    """
    hf_row_no_r_core = {k: v for k, v in mock_hf_row.items() if k != 'R_core'}
    mock_config.interior_struct.core_frac_mode = 'fraction'

    with pytest.raises(ValueError, match=r"got 'fraction'"):
        with patch('proteus.interior_energetics.boundary.next_step', return_value=1.0e3):
            BoundaryRunner(
                mock_config,
                mock_dirs,
                hf_row_no_r_core,
                mock_hf_all,
                mock_interior,
                mock_atmos,
            )

    # A different invalid value quotes itself, so the message tracks the config.
    mock_config.interior_struct.core_frac_mode = 'volume'
    with pytest.raises(ValueError, match=r"got 'volume'"):
        with patch('proteus.interior_energetics.boundary.next_step', return_value=1.0e3):
            BoundaryRunner(
                mock_config,
                mock_dirs,
                hf_row_no_r_core,
                mock_hf_all,
                mock_interior,
                mock_atmos,
            )
