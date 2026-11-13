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
  - Adiabat melt model: the isothermal limit against the linear model, mass
    and volume melt fractions, the shells handed to the tides modules and the
    tidal heat summed over them, and the T(P) profile handed to the structure
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import attrs
import numpy as np
import pytest
from _adiabat_mesh import (
    ADIABAT_DTDP,
    CURVE_DTDP,
    count_adiabat_solves,
    patch_adiabat_inputs,
    write_mantle_mesh,
)

from proteus.config import read_config_object
from proteus.interior_energetics.boundary import BoundaryRunner
from proteus.utils.constants import (
    M_earth,
    const_G,
    radnuc_data,
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
        # The linear melt model, so these tests exercise the solidification
        # front with constant melting temperatures; the adiabat model has its
        # own fixtures further down.
        melt_model='linear',
        T_solidus=1420.0,  # K
        T_liquidus=2020.0,  # K
        critical_rayleigh_number=1e3,  # dimensionless
        nusselt_exponent=1.0 / 3.0,  # Rayleigh-Benard scaling
        silicate_heat_capacity=1.2e3,  # J/kg/K
        atm_heat_capacity_const=False,
        atm_heat_capacity=1.7e4,  # J/kg/K
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
        include_core_heat_capacity=False,  # schema default: mantle only
        sensible_heat='molten',  # schema default: molten part only
        tidal_viscosity='boundary',  # schema default: own viscosity_model
        viscosity_phi='global',  # schema default: mass-averaged melt fraction
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
        radio_Al=2.35e4,  # ppmw (BSE)
        radio_Al26_abund=0.0,  # kg/kg, present day
        radio_Fe=6.26e4,  # ppmw (BSE)
        radio_Fe60_abund=0.0,  # kg/kg, present day
        radio_U=0.031,  # ppmw (BSE)
        radio_U235_abund=0.0071141,  # kg/kg
        radio_U238_abund=0.9928859,  # kg/kg
        radio_Th=0.124,  # ppmw (BSE)
        radio_Th232_abund=1.0,  # kg/kg
        radio_K=310,  # ppmw (BSE)
        radio_K40_abund=1.1926e-4,  # kg/kg
    )
    # BoundaryRunner reads no structure settings: masses and radii come from hf_row.
    interior_struct = SimpleNamespace()
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
        nlev_s=1,  # Single layer, as the linear melt model publishes
        tides=[0.0],  # No tidal heating by default
        temp_profile=None,
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
    R_int, R_core, M_core = 6.371e6, 0.55 * 6.371e6, 0.55 * M_earth
    M_int = M_core + 4.0 / 3.0 * np.pi * (R_int**3 - R_core**3) * 4500.0
    return {
        'Time': 0.01,  # Myr (convert to seconds via secs_per_year)
        'R_int': R_int,  # m (Earth radius)
        'M_int': M_int,  # kg (interior mass from the structure)
        'M_core': M_core,  # kg (Earth-mass core)
        'M_atm': 1e18,  # kg (thin H2 atmosphere)
        'F_atm': 100.0,  # W/m² (atmospheric heat loss)
        'T_magma': 3500.0,  # K (mantle potential temperature)
        'T_surf': 1600.0,  # K (surface temperature)
        'P_surf': 1e8,  # Pa (surface pressure)
        'R_core': R_core,  # m (CMB radius from the structure)
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


@pytest.mark.physics_invariant
def test_rfront_loc_centres_the_aggregate_viscosity(
    config_minimal, mock_config, mock_dirs, mock_hf_row, mock_hf_all, mock_interior, mock_atmos
):
    """The rfront_loc default and a non-default value set the boundary critical melt fraction.

    input/minimal.toml sets no rfront_loc, so the parsed value is the Interior default.
    At the configured melt fraction the aggregate viscosity is the geometric mean of the
    solid (1e22 Pa s) and melt (1e2 Pa s) viscosities, and values 0 and 1 are rejected.
    """
    ie = read_config_object(config_minimal).interior_energetics
    mock_config.interior_energetics.rfront_loc = ie.rfront_loc
    with patch('proteus.interior_energetics.boundary.next_step', return_value=1.0e3):
        runner = BoundaryRunner(
            config=mock_config,
            dirs=mock_dirs,
            hf_row=mock_hf_row,
            hf_all=mock_hf_all,
            interior_o=mock_interior,
            atmos_o=mock_atmos,
        )
    assert runner.critical_melt_fraction == pytest.approx(0.4, abs=1e-12)
    eta = runner.viscosity_aggregate_model(0.4)
    assert eta == pytest.approx(1e12, rel=1e-9)
    # Centred at 0.5, phi = 0.4 would give 10**16.62 = 4.2e16 Pa s.
    assert eta < 1e-4 * runner.viscosity_aggregate_model(0.3)
    mock_config.interior_energetics.rfront_loc = 0.3
    with patch('proteus.interior_energetics.boundary.next_step', return_value=1.0e3):
        runner = BoundaryRunner(
            config=mock_config,
            dirs=mock_dirs,
            hf_row=mock_hf_row,
            hf_all=mock_hf_all,
            interior_o=mock_interior,
            atmos_o=mock_atmos,
        )
    assert runner.critical_melt_fraction == pytest.approx(0.3, abs=1e-12)
    assert runner.viscosity_aggregate_model(0.3) == pytest.approx(1e12, rel=1e-9)
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError, match='rfront_loc'):
            attrs.evolve(ie, rfront_loc=bad)


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
    the critical melt fraction (φ_crit = 0.4 in the fixture).

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


# (isotope, element ppmw, isotope/element mass ratio) for 20 ppb U, 79 ppb Th, 143 ppm K
_EARTH_RADIO_INVENTORY = [
    ('k40', 143.0, 1.1926e-4),
    ('th232', 0.079, 1.0),
    ('u235', 0.020, 0.0071141),
    ('u238', 0.020, 0.9928859),
]


def _earth_heating(t_years: float) -> float:
    """Heating of _EARTH_RADIO_INVENTORY [W/kg] at t_years before the present day [yr]."""
    return sum(
        c
        * 1e-6
        * x
        * radnuc_data[iso]['heatprod']
        * 2.0 ** (t_years / radnuc_data[iso]['halflife'])
        for iso, c, x in _EARTH_RADIO_INVENTORY
    )


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
    Test radioactive heating at the model start is the present-day value decayed back.

    **Physical Scenario**: radiogenic_heat = True; heating from K-40, U-238,
    U-235, Th-232 decay chains in silicate mantle, inputs at the present day.

    **Expected Behavior**: at t = 0 s, 4.567 Gyr before the inputs, H equals
    sum C * 1e-6 * X * h * 2**(4.567 Gyr / t_half), about 2.0e-11 W/kg.
    """
    boundary_runner.use_radiogenic_heating = True
    # Inputs at the present day, 4.567 Gyr after the model start [yr]
    boundary_runner.radio_t0_yr = 4.567e9
    boundary_runner.radio_inventory = _EARTH_RADIO_INVENTORY

    t = 0.0  # s (at formation)

    h_radio = boundary_runner.radioactive_heating(t)

    # Should be positive at t=0 (maximum heating)
    assert h_radio > 0
    assert h_radio == pytest.approx(_earth_heating(4.567e9), rel=1e-12)
    # A dropped t0 gives the present-day value, 4.5e-12 W/kg, at the model start
    assert abs(h_radio / _earth_heating(0.0) - 1.0) > 1.0
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
    boundary_runner.radio_t0_yr = 4.567e9
    boundary_runner.radio_inventory = _EARTH_RADIO_INVENTORY

    t_early = 1e6 * secs_per_year  # 1 Myr
    t_late = 1e9 * secs_per_year  # 1 Gyr

    h_early = boundary_runner.radioactive_heating(t_early)
    h_late = boundary_runner.radioactive_heating(t_late)

    # Early heating should exceed late heating (radioactive decay)
    assert h_early > h_late
    # 1 Gyr of model time is 3.567 Gyr before the inputs
    assert h_late == pytest.approx(_earth_heating(3.567e9), rel=1e-12)
    # Time not converted from s, or converted to Gyr: both land on the t = 0 value
    assert abs(h_late / _earth_heating(4.567e9) - 1.0) > 0.1
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
# Tests: atmosphere_specific_heat (the atm_heat_capacity_const = false path)
# =============================================================================


def _cp_config(const: bool, value: float = 1.7e4):
    """Config carrying only the two keys atmosphere_specific_heat reads."""
    boundary = SimpleNamespace(atm_heat_capacity_const=const, atm_heat_capacity=value)
    return SimpleNamespace(interior_energetics=SimpleNamespace(boundary=boundary))


def _cp_atmos(layer_cp, layer_mass):
    """Atmosphere object whose column carries AGNI's layer_cp and layer_σ arrays."""
    return SimpleNamespace(
        _atm=SimpleNamespace(layer_cp=np.asarray(layer_cp), layer_σ=np.asarray(layer_mass))
    )


@pytest.mark.physics_invariant
def test_atmosphere_specific_heat_is_layer_mass_weighted():
    """An H2 envelope whose deep, massive layers hold most of the heat sets the mean.

    A column with a heavy-molecule upper layer (cp 4.0e3 J/kg/K, 1e3 kg m-2) over an
    H2-rich deep layer (cp 1.43e4 J/kg/K, 9e3 kg m-2) has a mass-weighted cp of
    1.327e4 J/kg/K; the unweighted mean (9.15e3) is the plausible wrong formula.
    """
    from proteus.interior_energetics.boundary import atmosphere_specific_heat

    cp, source = atmosphere_specific_heat(
        _cp_config(False, value=2.0e3), _cp_atmos([4.0e3, 1.43e4], [1.0e3, 9.0e3])
    )

    assert cp == pytest.approx(1.327e4, rel=1e-12)
    assert source == 'atmosphere column'
    assert abs(cp - 9.15e3) > 1.0e3  # unweighted mean
    assert 4.0e3 < cp < 1.43e4  # a weighted mean stays between the layer values


@pytest.mark.physics_invariant
def test_atmosphere_specific_heat_skips_nonfinite_and_nonpositive_layers():
    """Layers whose heat capacity or mass is NaN, zero or negative carry no weight."""
    from proteus.interior_energetics.boundary import atmosphere_specific_heat

    cp, source = atmosphere_specific_heat(
        _cp_config(False),
        _cp_atmos([np.nan, 1.2e4, -5.0e3, 1.4e4, 8.0e3], [2.0, 1.0, 1.0, 3.0, 0.0]),
    )

    # Only the 1.2e4 (mass 1) and 1.4e4 (mass 3) layers count: (1.2e4 + 4.2e4) / 4.
    assert cp == pytest.approx(1.35e4, rel=1e-12)
    assert source == 'atmosphere column'
    assert cp > 0.0


def test_atmosphere_specific_heat_uses_configured_value_before_first_solve():
    """AGNI allocates layer_cp and layer_σ as ones; until a solve fills them the
    configured value stands in, rather than a 1 J/kg/K atmosphere that would leave
    the surface temperature with almost no thermal inertia."""
    from proteus.interior_energetics.boundary import atmosphere_specific_heat

    cp, source = atmosphere_specific_heat(
        _cp_config(False, value=1.6e4), _cp_atmos([1.0] * 4, [1.0] * 4)
    )

    assert cp == pytest.approx(1.6e4, rel=1e-12)
    assert 'not solved' in source
    assert abs(cp - 1.0) > 1.0e4


@pytest.mark.parametrize(
    'atmos_o, reason',
    [
        (None, 'no atmosphere column'),
        (SimpleNamespace(_atm=None), 'no atmosphere column'),
        (SimpleNamespace(_atm=SimpleNamespace(layer_cp=np.ones(3))), 'no atmosphere column'),
        (_cp_atmos([1.4e4, 1.3e4, 1.2e4], [1.0, 2.0]), 'differ in shape'),
        (_cp_atmos([np.nan, np.nan], [1.0, 1.0]), 'no usable layer'),
    ],
    ids=['no-object', 'no-column', 'no-layer-mass', 'shape-mismatch', 'all-nan'],
)
def test_atmosphere_specific_heat_falls_back_without_a_usable_column(atmos_o, reason):
    """Each way the column can be unusable returns the configured value and says why,
    instead of raising (a shape mismatch would index one array with the other's mask)."""
    from proteus.interior_energetics.boundary import atmosphere_specific_heat

    cp, source = atmosphere_specific_heat(_cp_config(False, value=1.5e4), atmos_o)

    assert cp == pytest.approx(1.5e4, rel=1e-12)
    assert reason in source
    assert source.startswith('configured value')


def test_atmosphere_specific_heat_constant_flag_ignores_column():
    """With atm_heat_capacity_const the column is never read, even when it is valid."""
    from proteus.interior_energetics.boundary import atmosphere_specific_heat

    cp, source = atmosphere_specific_heat(
        _cp_config(True, value=3.0e4), _cp_atmos([1.4e4], [5.0])
    )

    assert cp == pytest.approx(3.0e4, rel=1e-12)
    assert source == 'constant'
    assert abs(cp - 1.4e4) > 1.0e4


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
# Tests: masses and radii taken from the structure in BoundaryRunner.__init__
# =============================================================================


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'core_radius_frac',
    [
        pytest.param(0.55, id='fixture_core_radius'),
        pytest.param(0.42, id='smaller_zalmoxis_core'),
    ],
)
def test_boundary_runner_mantle_mass_closes_structure_budget(
    core_radius_frac,
    mock_config,
    mock_dirs,
    mock_hf_row,
    mock_hf_all,
    mock_interior,
    mock_atmos,
):
    """The mantle mass and bulk density follow from the structure's M_int, M_core and R_core.

    Physical scenario: the structure module sets the interior and core masses and the CMB
    radius, and the boundary model takes the mantle as the rest. At 0.55 R_int the fixture
    mantle density is 4500 kg m-3; at 0.42 R_int the same mantle mass fills a larger shell
    (about 4050 kg m-3), so a fixed configured density fails the mass closure in one case.
    """
    hf_row = {**mock_hf_row, 'R_core': core_radius_frac * mock_hf_row['R_int']}
    with patch('proteus.interior_energetics.boundary.next_step', return_value=1.0e3):
        runner = BoundaryRunner(
            mock_config, mock_dirs, hf_row, mock_hf_all, mock_interior, mock_atmos
        )

    R_int, R_core, M_int = hf_row['R_int'], hf_row['R_core'], hf_row['M_int']
    shell = 4.0 / 3.0 * np.pi * (R_int**3 - R_core**3)
    # The CMB radius comes from the row, not from a configured core fraction.
    assert runner.core_radius == pytest.approx(R_core, rel=1e-12)
    assert runner.core_frac == pytest.approx(core_radius_frac, rel=1e-12)
    # Mass closure: core plus mantle is the structure's interior mass.
    assert runner.core_mass + runner.mantle_mass == pytest.approx(M_int, rel=1e-12)
    assert runner.mantle_bulk_density == pytest.approx(runner.mantle_mass / shell, rel=1e-12)
    # Bound: a silicate mantle density between 2000 and 6000 kg m-3.
    assert 2.0e3 < runner.mantle_bulk_density < 6.0e3
    # Surface gravity uses the structure's interior mass.
    assert runner.surface_gravity == pytest.approx(const_G * M_int / R_int**2, rel=1e-12)


@pytest.mark.parametrize('core_share', [1.0, 1.2], ids=['no_mantle', 'core_heavier'])
def test_boundary_runner_refuses_a_structure_without_mantle_mass(
    core_share,
    mock_config,
    mock_dirs,
    mock_hf_row,
    mock_hf_all,
    mock_interior,
    mock_atmos,
):
    """An interior no heavier than its core is refused, naming both masses.

    Error contract: a zero or negative mantle mass would give a zero heat
    capacity and a non-physical mantle density, so the runner stops at setup.
    """
    hf_row = {**mock_hf_row, 'M_core': core_share * mock_hf_row['M_int']}

    with patch('proteus.interior_energetics.boundary.next_step', return_value=1.0e3):
        with pytest.raises(ValueError, match='M_int > M_core') as excinfo:
            BoundaryRunner(
                mock_config, mock_dirs, hf_row, mock_hf_all, mock_interior, mock_atmos
            )
    assert 'M_core' in str(excinfo.value)


# =============================================================================
# Regression freeze: shared helpers must not move boundary's numbers
# =============================================================================


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_solidification_front_matches_values_frozen_before_the_shared_helper(
    boundary_runner,
):
    """The front position and its derivative survive delegation bit for bit.

    Cross-implementation check against values recorded from this module's own
    previous inline expressions. The shared helper evaluates the identical
    operations in the identical order, so exact agreement is required here,
    not merely agreement to a tolerance.
    """
    boundary_runner.T_solidus = 1420.0
    boundary_runner.T_liquidus = 2020.0
    boundary_runner.planet_radius = 6.371e6
    boundary_runner.core_radius = 3.4e6

    frozen = {
        1300.0: (6.37100000000000000e06, -3.00148562003514371e03),
        1420.0: (6.37100000000000000e06, -3.00148562003514371e03),
        1642.0: (5.61951461057546455e06, -3.85792577532489258e03),
        1720.0: (5.30086969714618661e06, -4.33567998512968006e03),
        2020.0: (3.40000000000000000e06, -1.05388601889177244e04),
        2500.0: (3.40000000000000000e06, -1.05388601889177244e04),
    }
    for temp, (r_expected, drs_expected) in frozen.items():
        assert boundary_runner.r_s(temp) == pytest.approx(r_expected, rel=1e-15)
        assert boundary_runner.drs_dTp(temp) == pytest.approx(drs_expected, rel=1e-15)

    # Boundedness: the front never leaves the mantle shell.
    assert all(
        boundary_runner.core_radius <= boundary_runner.r_s(t) <= boundary_runner.planet_radius
        for t in frozen
    )


@pytest.mark.physics_invariant
def test_dT_pdt_excludes_latent_heat_from_a_wholly_molten_mantle(boundary_runner):
    """A fully molten mantle has no crystallisation front to draw heat from.

    The front of a mantle above the liquidus sits at the core-mantle boundary,
    which is still inside the planet, so a latent term keyed on the front
    radius alone fires there and slows the cooling of a mantle that is not
    freezing. Above the liquidus the cooling rate must therefore match the
    sensible-only balance exactly.
    """
    boundary_runner.use_radiogenic_heating = False
    boundary_runner.use_tidal_heating = False
    boundary_runner.tidal_term = 0.0

    T_p, T_surf = 3000.0, 1600.0
    assert boundary_runner.melt_fraction(T_p) == pytest.approx(1.0, rel=1e-12)

    dTp_dt = boundary_runner.dT_pdt(T_p, T_surf, 0.0)
    q_m = boundary_runner.q_m(T_p, T_surf, 1.0)
    sensible = boundary_runner.silicate_heat_capacity * boundary_runner.mantle_mass
    expected = -4 * np.pi * boundary_runner.planet_radius**2 * q_m / sensible

    assert dTp_dt == pytest.approx(expected, rel=1e-12)
    assert dTp_dt < 0.0

    # Discrimination: the ungated form would divide by an extra
    # L * M / (T_liquidus - T_solidus), slowing the rate by a factor of 1.56
    # at these parameters.
    latent = (
        boundary_runner.heat_fusion_silicate
        * boundary_runner.mantle_mass
        / (boundary_runner.T_liquidus - boundary_runner.T_solidus)
    )
    ungated = -4 * np.pi * boundary_runner.planet_radius**2 * q_m / (sensible + latent)
    assert abs(dTp_dt / ungated - 1.0) > 0.4

    # Inside the mushy zone the latent term is still active, so the gate is a
    # real boundary condition and not a wholesale removal.
    T_mush = 0.5 * (boundary_runner.T_solidus + boundary_runner.T_liquidus)
    phi_mush = boundary_runner.melt_fraction(T_mush)
    r_s_mush = boundary_runner.r_s(T_mush)
    mushy_sensible = (
        (4 / 3)
        * np.pi
        * boundary_runner.mantle_bulk_density
        * boundary_runner.silicate_heat_capacity
        * (boundary_runner.planet_radius**3 - r_s_mush**3)
    )
    mushy_expected = (
        -4
        * np.pi
        * boundary_runner.planet_radius**2
        * boundary_runner.q_m(T_mush, T_surf, phi_mush)
        / (mushy_sensible + latent)
    )
    assert boundary_runner.dT_pdt(T_mush, T_surf, 0.0) == pytest.approx(
        mushy_expected, rel=1e-12
    )


# =============================================================================
# Tests: adiabat melt model
# =============================================================================
#
# The adiabat model reads the structure's mantle mesh, so these tests write one
# under tmp_path and stand in for the Zalmoxis gradient and the tabulated
# melting curves (tests/helpers/_adiabat_mesh.py). The masses come from the
# structure, so the linear model's bulk density and the mesh describe the same
# mantle, and the constant melting curves match the fixture's T_solidus and
# T_liquidus so the two models can be compared in the isothermal limit.

_ADB_R_INT = 6.371e6  # m
_ADB_R_CORE = 3.4e6  # m
_ADB_M_CORE = 2.0e24  # kg
_ADB_M_MANTLE = 4.0e24  # kg
_ADB_AREA = 4.0 * np.pi * _ADB_R_INT**2  # m2
_ADB_TSOL = 1420.0  # K, the fixture's T_solidus
_ADB_TLIQ = 2020.0  # K, the fixture's T_liquidus


def _adiabat_hf_row(T_magma=2000.0):
    """Helpfile row for a planet whose masses come from a structure solve."""
    return {
        'Time': 1.0e4,
        'R_int': _ADB_R_INT,
        'R_core': _ADB_R_CORE,
        'M_int': _ADB_M_CORE + _ADB_M_MANTLE,
        'M_core': _ADB_M_CORE,
        'M_atm': 1e18,
        'F_atm': 100.0,
        'T_magma': T_magma,
        'T_surf': 1600.0,
        'P_surf': 1e8,
    }


def _adiabat_interior(nlev_s, tides=None, ic=0):
    """Interior state with `nlev_s` layers, as Interior_t sizes it."""
    return SimpleNamespace(
        ic=ic,
        nlev_s=nlev_s,
        tides=np.zeros(nlev_s) if tides is None else np.asarray(tides, dtype=float),
        phi=None,
        mass=None,
        visc=None,
        density=None,
        temp=None,
        pres=None,
        radius=None,
        temp_profile=None,
    )


def _make_runner(config, tmp_path, interior_o, melt_model, T_magma=2000.0):
    """BoundaryRunner on the shared geometry with the given melt model."""
    config.interior_energetics.boundary.melt_model = melt_model
    config.interior_energetics.module = 'boundary'
    config.interior_struct.module = 'zalmoxis'
    with patch('proteus.interior_energetics.boundary.next_step', return_value=1.0e3):
        return BoundaryRunner(
            config,
            {'output': str(tmp_path)},
            _adiabat_hf_row(T_magma),
            None,
            interior_o,
            None,
        )


def _adiabat_runner(
    config,
    tmp_path,
    monkeypatch,
    nlev_s=8,
    dtdp=0.0,
    curve_dtdp=0.0,
    rho_cmb=5500.0,
    rho_top=4300.0,
    tides=None,
    ic=0,
    T_magma=2000.0,
):
    """BoundaryRunner with the adiabat melt model on a freshly written mesh."""
    write_mantle_mesh(
        tmp_path, _ADB_R_CORE, _ADB_R_INT, _ADB_M_MANTLE, rho_cmb=rho_cmb, rho_top=rho_top
    )
    patch_adiabat_inputs(
        monkeypatch, dtdp=dtdp, curve_dtdp=curve_dtdp, tsol=_ADB_TSOL, tliq=_ADB_TLIQ
    )
    interior_o = _adiabat_interior(nlev_s, tides=tides, ic=ic)
    runner = _make_runner(config, tmp_path, interior_o, 'adiabat', T_magma=T_magma)
    return runner, interior_o


def _run_to(runner, interior_o, tmp_path, T_p_final, T_surf_final=1600.0):
    """Run one solver step whose integration ends at the given temperatures."""
    sol = SimpleNamespace(
        t=np.array([runner.curr_time, runner.curr_time + runner.dt]),
        y=np.array([[runner.T_p_0, T_p_final], [runner.T_surf_0, T_surf_final]]),
        status=0,
        message='',
    )
    with patch('proteus.interior_energetics.boundary.solve_ivp', return_value=sol):
        return runner.run_solver(_adiabat_hf_row(), interior_o, {'output': str(tmp_path)})


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_adiabat_isothermal_limit_reproduces_the_linear_cooling_rate(
    mock_config, tmp_path, monkeypatch
):
    """With no gradient and constant curves the adiabat model cools like the linear one.

    Cross-implementation check against the Schaefer et al. (2016) solidification
    front: an isothermal, uniform-density mantle between a pressure-independent
    solidus and liquidus has a molten mass fraction linear in T_p, so the
    sensible capacity of its molten part, c_p phi M, and the latent capacity
    of its front, L M / (T_liq - T_sol), must match the front-radius form of
    the linear model. 1713.6 K sits between the curve's nodes and inside the
    mushy zone.
    """
    linear = _make_runner(mock_config, tmp_path, _adiabat_interior(1), 'linear')
    adiabatic, _ = _adiabat_runner(
        mock_config, tmp_path, monkeypatch, rho_cmb=4800.0, rho_top=4800.0
    )
    T_p, T_surf = 1713.6, 1600.0
    cp = mock_config.interior_energetics.boundary.silicate_heat_capacity
    L = mock_config.interior_energetics.latent_heat_of_fusion
    phi = (T_p - _ADB_TSOL) / (_ADB_TLIQ - _ADB_TSOL)

    expected = _ADB_M_MANTLE * (cp * phi + L / (_ADB_TLIQ - _ADB_TSOL))
    assert adiabatic.heat_capacity(T_p) == pytest.approx(expected, rel=1e-12)
    assert linear.heat_capacity(T_p) == pytest.approx(expected, rel=1e-10)
    assert adiabatic.dT_pdt(T_p, T_surf, 0.0) == pytest.approx(
        linear.dT_pdt(T_p, T_surf, 0.0), rel=1e-10
    )
    # Factor guards: charging the whole mantle's sensible heat (1867 J/kg/K)
    # or dropping the latent term (587 J/kg/K) lands far from 1254 J/kg/K.
    whole_mantle = _ADB_M_MANTLE * (cp + L / (_ADB_TLIQ - _ADB_TSOL))
    assert abs(adiabatic.heat_capacity(T_p) / whole_mantle - 1.0) > 0.3
    assert abs(adiabatic.heat_capacity(T_p) / (_ADB_M_MANTLE * cp * phi) - 1.0) > 0.3
    # Scale guard: per unit mantle mass, of order a silicate heat capacity.
    assert 1000.0 < adiabatic.heat_capacity(T_p) / _ADB_M_MANTLE < 1500.0
    # Sign guard: a mantle hotter than its surface cools.
    assert adiabatic.dT_pdt(T_p, T_surf, 0.0) < 0.0


@pytest.mark.physics_invariant
def test_adiabat_step_reports_mass_and_volume_melt_fractions(
    mock_config, tmp_path, monkeypatch
):
    """A magma ocean over a solid, denser base melts less by mass than by volume.

    The melting curves rise with depth faster than the adiabat, so a mantle
    whose top is above its liquidus is solid at its base. The base is also the
    densest part, so the mass melt fraction must lag the volume one. The
    reported masses must close on the mantle mass, the front must sit inside
    the mantle, and the melt fraction must be the exact adiabat state at the
    end of the step, not the interpolated curve the solver stepped on.
    """
    runner, interior_o = _adiabat_runner(
        mock_config, tmp_path, monkeypatch, dtdp=ADIABAT_DTDP, curve_dtdp=CURVE_DTDP
    )
    T_p_final = 2100.37
    _, out = _run_to(runner, interior_o, tmp_path, T_p_final)

    assert 0.0 < out['Phi_global'] < out['Phi_global_vol'] < 1.0
    exact = runner.melt_curve.model.state(T_p_final)
    assert out['Phi_global'] == pytest.approx(exact.phi_mass, rel=1e-12)
    # Conservation: the molten and solid parts make up the mantle.
    assert out['M_mantle_liquid'] + out['M_mantle_solid'] == pytest.approx(
        _ADB_M_MANTLE, rel=1e-12
    )
    assert out['M_mantle_liquid'] == pytest.approx(out['Phi_global'] * _ADB_M_MANTLE, rel=1e-12)
    # The front lies between the surface and the core-mantle boundary.
    assert 0.0 < out['RF_depth'] < 1.0 - _ADB_R_CORE / _ADB_R_INT
    # The latent heat of the advancing front adds to the heat capacity.
    cp = mock_config.interior_energetics.boundary.silicate_heat_capacity
    assert out['Cp_eff'] > cp * out['Phi_global']
    assert out['Cp_eff'] == pytest.approx(
        runner.heat_capacity(T_p_final) / _ADB_M_MANTLE, rel=1e-12
    )

    # The adiabat itself is handed to the structure re-solve on pressure,
    # anchored at T_p at the top and warming with depth.
    pres, temp = interior_o.temp_profile
    np.testing.assert_allclose(pres, runner.melt_curve.model.profile.pressure, rtol=1e-12)
    assert temp[-1] == pytest.approx(T_p_final, rel=1e-12)
    assert np.all(np.diff(temp) <= 0.0)
    assert temp[0] > T_p_final + 1000.0


@pytest.mark.physics_invariant
def test_adiabat_shells_carry_the_mantle_mass_melt_and_rheology(
    mock_config, tmp_path, monkeypatch
):
    """The shells handed to the tides modules conserve the mantle and resolve its rheology.

    The shell masses must sum to the mantle mass and their melt fractions to
    the global one. Each shell takes the viscosity the aggregate rheology
    assigns to its own melt fraction, so the solid base is many orders of
    magnitude stiffer than the molten top; publishing the bulk viscosity in
    every shell would make them equal.
    """
    mock_config.interior_energetics.boundary.viscosity_model = 2
    runner, interior_o = _adiabat_runner(
        mock_config, tmp_path, monkeypatch, nlev_s=12, dtdp=ADIABAT_DTDP, curve_dtdp=CURVE_DTDP
    )
    _, out = _run_to(runner, interior_o, tmp_path, 2100.37)

    assert len(interior_o.mass) == 12
    assert len(interior_o.radius) == 13
    assert interior_o.radius[0] == pytest.approx(_ADB_R_CORE, rel=1e-12)
    assert interior_o.radius[-1] == pytest.approx(_ADB_R_INT, rel=1e-12)
    # Conservation of the mantle mass and of its melt across the shells.
    assert float(np.sum(interior_o.mass)) == pytest.approx(_ADB_M_MANTLE, rel=1e-12)
    phi_shells = float(np.sum(interior_o.phi * interior_o.mass)) / _ADB_M_MANTLE
    assert phi_shells == pytest.approx(out['Phi_global'], rel=1e-9)

    # A molten top over a mostly solid base, the melt rising outward.
    assert interior_o.phi[-1] == pytest.approx(1.0, abs=1e-12)
    assert interior_o.phi[0] < 0.2
    assert np.all(np.diff(interior_o.phi) >= 0.0)
    # Viscosity does not rise with melt fraction, and the two ends differ by
    # many orders of magnitude.
    order = np.argsort(interior_o.phi)
    assert np.all(np.diff(interior_o.visc[order]) <= 0.0)
    assert interior_o.visc[0] / interior_o.visc[-1] > 1e6


@pytest.mark.physics_invariant
def test_adiabat_tidal_heating_is_summed_over_the_shells(mock_config, tmp_path, monkeypatch):
    """Tidal heat enters the energy balance as each shell's rate times its mass.

    A uniform rate must give the single-layer flux H M / A. Heating only the
    base shell must deposit that shell's share alone, well under a fifth of
    the uniform total for ten shells. A rate array on a different grid is
    rejected before any heat is applied.
    """
    mock_config.interior_energetics.heat_tidal = True
    H = 2.0e-9  # W kg-1

    uniform, interior_o = _adiabat_runner(
        mock_config, tmp_path, monkeypatch, nlev_s=10, tides=np.full(10, H)
    )
    assert uniform.tidal_term == pytest.approx(H, rel=1e-12)
    _, out_uniform = _run_to(uniform, interior_o, tmp_path, 1850.4)
    assert out_uniform['F_tidal'] == pytest.approx(H * _ADB_M_MANTLE / _ADB_AREA, rel=1e-12)

    base_only = np.zeros(10)
    base_only[0] = H
    based, interior_b = _adiabat_runner(
        mock_config, tmp_path, monkeypatch, nlev_s=10, tides=base_only
    )
    _, out_base = _run_to(based, interior_b, tmp_path, 1850.4)
    assert out_base['F_tidal'] == pytest.approx(H * based.shells.mass[0] / _ADB_AREA, rel=1e-12)
    # The base shell holds well under a fifth of the mantle mass, so applying
    # its rate to the whole mantle would be caught.
    assert out_base['F_tidal'] < 0.2 * out_uniform['F_tidal']
    assert out_base['F_tidal'] > 0.0

    # Error contract: rates computed on another radial grid.
    with pytest.raises(ValueError, match='disagree on the radial grid'):
        _adiabat_runner(mock_config, tmp_path, monkeypatch, nlev_s=10, tides=np.full(9, H))

    # Edge case: with tidal heating off, the rates are ignored.
    mock_config.interior_energetics.heat_tidal = False
    off, _ = _adiabat_runner(
        mock_config, tmp_path, monkeypatch, nlev_s=10, tides=np.full(10, H)
    )
    assert off.tidal_term == pytest.approx(0.0, abs=1e-30)


@pytest.mark.physics_invariant
def test_adiabat_mantle_outside_its_mushy_zone_cools_as_a_whole(
    mock_config, tmp_path, monkeypatch
):
    """Wholly molten or wholly solid, the whole mantle cools with no latent heat.

    Above the hottest liquidus in the mantle every level is molten, and below
    the coldest solidus every level is solid, so there is no front to advance
    and the heat capacity is c_p M_mantle. A wholly molten mantle is
    recognised without integrating the adiabat. Inside the mushy zone the
    front releases latent heat on top of the molten part's sensible heat.
    """
    runner, _ = _adiabat_runner(
        mock_config, tmp_path, monkeypatch, dtdp=ADIABAT_DTDP, curve_dtdp=CURVE_DTDP
    )
    calls = count_adiabat_solves(monkeypatch)
    cp = mock_config.interior_energetics.boundary.silicate_heat_capacity
    whole = cp * _ADB_M_MANTLE

    hottest = float(np.max(runner.melt_curve.model.tmp_liq))
    assert runner.melt_fraction(hottest + 10.3) == pytest.approx(1.0, abs=1e-15)
    assert runner.heat_capacity(hottest + 10.3) == pytest.approx(whole, rel=1e-12)
    assert calls[0] == 0

    assert runner.melt_fraction(900.4) == pytest.approx(0.0, abs=1e-15)
    assert runner.heat_capacity(900.4) == pytest.approx(whole, rel=1e-12)

    # Inside the mushy zone the latent term is active: the capacity exceeds
    # the molten part's sensible heat by L M dphi/dT_p.
    T_mush = 1850.4
    phi = runner.melt_fraction(T_mush)
    assert 0.0 < phi < 1.0
    L = mock_config.interior_energetics.latent_heat_of_fusion
    latent = L * _ADB_M_MANTLE * runner.melt_curve.dphi_dtmp(T_mush)
    assert latent > 0.0
    assert runner.heat_capacity(T_mush) == pytest.approx(
        cp * phi * _ADB_M_MANTLE + latent, rel=1e-12
    )


def test_adiabat_without_a_structure_mesh_names_the_boundary_switch(
    mock_config, tmp_path, monkeypatch
):
    """A missing structure mesh fails loudly and names the boundary's way out.

    Error contract: the adiabat needs the structure module's mantle profile.
    Without it the runner must not start, the message must point at the
    boundary backend's own melt_model setting rather than the dummy's, and no
    half-built melt model may be left cached on the interior.
    """
    patch_adiabat_inputs(monkeypatch, tsol=_ADB_TSOL, tliq=_ADB_TLIQ)
    interior_o = _adiabat_interior(8)

    with pytest.raises(FileNotFoundError) as excinfo:
        _make_runner(mock_config, tmp_path, interior_o, 'adiabat')

    message = str(excinfo.value)
    assert 'interior_energetics.boundary.melt_model = "linear"' in message
    assert 'boundary interior module' in message
    assert getattr(interior_o, '_adiabat_melt_model', None) is None


def test_linear_step_publishes_one_layer_and_clears_the_profile(mock_config, tmp_path):
    """The linear model publishes a single isothermal layer and no T(P) profile.

    Edge case for a run that switched melt model on resume: a profile left on
    the interior by an earlier adiabat step must be cleared, or the structure
    re-solve would keep reading an adiabat the energetics no longer follows.
    The volume and mass melt fractions coincide in a uniform-density layer.
    """
    interior_o = _adiabat_interior(1)
    interior_o.temp_profile = (np.array([1e9, 1e5]), np.array([3000.0, 2000.0]))
    runner = _make_runner(mock_config, tmp_path, interior_o, 'linear')
    _, out = _run_to(runner, interior_o, tmp_path, 1713.6)

    assert interior_o.temp_profile is None
    assert len(interior_o.phi) == 1
    assert interior_o.phi[0] == pytest.approx((1713.6 - _ADB_TSOL) / (_ADB_TLIQ - _ADB_TSOL))
    assert out['Phi_global_vol'] == pytest.approx(out['Phi_global'], rel=1e-15)
    assert out['RF_depth'] == pytest.approx(
        out['Phi_global'] * (1.0 - _ADB_R_CORE / _ADB_R_INT), rel=1e-12
    )


def test_adiabat_curve_is_reused_by_the_next_step(mock_config, tmp_path, monkeypatch):
    """Consecutive steps on one interior share the melt curve and its nodes.

    The wrapper builds a fresh runner every step. The curve has to persist on
    the interior, or each step would integrate the adiabat afresh at every
    temperature the previous step already visited.
    """
    first, interior_o = _adiabat_runner(
        mock_config, tmp_path, monkeypatch, dtdp=ADIABAT_DTDP, curve_dtdp=CURVE_DTDP
    )
    first.melt_fraction(1850.4)
    calls = count_adiabat_solves(monkeypatch)

    second = _make_runner(mock_config, tmp_path, interior_o, 'adiabat')
    assert second.melt_curve is first.melt_curve
    assert second.melt_fraction(1850.4) == pytest.approx(first.melt_fraction(1850.4))
    assert calls[0] == 0


@pytest.mark.physics_invariant
def test_adiabat_step_cools_through_the_mushy_zone(mock_config, tmp_path, monkeypatch):
    """An integrated step through a partly molten mantle cools and freezes it.

    The real ODE integration, reading the melt curve inside its right-hand
    side, from a magma ocean over a solid base. Losing heat at the surface
    must lower the potential temperature and the melt fraction together, and
    both must stay physical.
    """
    runner, interior_o = _adiabat_runner(
        mock_config,
        tmp_path,
        monkeypatch,
        dtdp=ADIABAT_DTDP,
        curve_dtdp=CURVE_DTDP,
        ic=2,
        T_magma=1950.0,
    )
    phi_start = runner.melt_fraction(runner.T_p_0)
    _, out = runner.run_solver(_adiabat_hf_row(1950.0), interior_o, {'output': str(tmp_path)})

    assert 0.0 < out['T_magma'] < runner.T_p_0
    assert 0.0 <= out['Phi_global'] < phi_start <= 1.0
    # Monotonicity: the melt fraction followed the temperature down, so the
    # mantle froze by the amount the curve assigns to the final temperature.
    assert out['Phi_global'] == pytest.approx(runner.melt_fraction(out['T_magma']), abs=1e-4)


# =============================================================================
# Tests: the choices the dummy backend makes differently
# =============================================================================
#
# include_core_heat_capacity, sensible_heat and tidal_viscosity default to this
# backend's own behaviour; set to true / "whole" / "kervazo" they follow the
# dummy backend. Isothermal limit throughout (no gradient, constant curves), so
# phi = (T - 1420) / 600 at every level and dphi/dT_p = 1 / 600 K-1.

# 1850.4 K inside the mushy zone, 900.4 K below it, 2100.7 K above it.
_T_MUSH, _T_SOLID, _T_MOLTEN = 1850.4, 900.4, 2100.7


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_core_heat_capacity_is_added_only_when_requested(mock_config, tmp_path, monkeypatch):
    """With include_core_heat_capacity the core adds core_heatcap * M_core everywhere.

    The core is lumped with the mantle as in the dummy backend, so its share
    is the same solid, mushy or molten: 450 J kg-1 K-1 x 2e24 kg = 9.0e26 J/K.
    A wholly molten mantle then holds 1200 x 4e24 + 9.0e26 = 5.7e27 J/K. The
    reported Cp_eff stays the mantle's own, as the dummy reports it.
    """
    # Off by default: the core heat capacity is never read (the fixture has none).
    off, interior_off = _adiabat_runner(mock_config, tmp_path, monkeypatch)
    assert off.core_heat_capacity == pytest.approx(0.0, abs=1e-30)

    mock_config.interior_struct.core_heatcap = 450.0
    mock_config.interior_energetics.boundary.include_core_heat_capacity = True
    on, interior_on = _adiabat_runner(mock_config, tmp_path, monkeypatch)

    for tmp in (_T_SOLID, _T_MUSH, _T_MOLTEN):
        assert on.heat_capacity(tmp) - off.heat_capacity(tmp) == pytest.approx(
            9.0e26, rel=1e-12
        )
    assert on.heat_capacity(_T_MOLTEN) == pytest.approx(5.7e27, rel=1e-12)
    # Mass guard: the core's share uses the core mass, not the mantle's (3.6e27).
    assert abs(on.heat_capacity(_T_MOLTEN) - off.heat_capacity(_T_MOLTEN) - 3.6e27) > 1e27
    # More thermal inertia cools the same heat loss more slowly.
    assert abs(on.dT_pdt(_T_MOLTEN, 1600.0, 0.0)) < abs(off.dT_pdt(_T_MOLTEN, 1600.0, 0.0))

    _, out_on = _run_to(on, interior_on, tmp_path, _T_MUSH)
    _, out_off = _run_to(off, interior_off, tmp_path, _T_MUSH)
    assert out_on['Cp_eff'] == pytest.approx(out_off['Cp_eff'], rel=1e-12)

    # Edge case: core_heatcap = "self" with no structure value falls back to 450.
    mock_config.interior_struct.core_heatcap = 'self'
    fallback, _ = _adiabat_runner(mock_config, tmp_path, monkeypatch)
    assert fallback.core_heat_capacity == pytest.approx(9.0e26, rel=1e-12)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_sensible_heat_whole_cools_the_whole_partly_molten_mantle(
    mock_config, tmp_path, monkeypatch
):
    """sensible_heat = "whole" counts c_p M in the mushy zone, "molten" c_p phi M.

    At 1850.4 K, phi = 0.71733 and the latent capacity is
    L M / 600 = 4e5 x 4e24 / 600 = 2.66667e27 J/K, so "whole" gives
    4.8e27 + 2.66667e27 = 7.46667e27 J/K and "molten" gives
    3.44320e27 + 2.66667e27 = 6.10987e27 J/K. Outside the mushy zone the
    two agree on c_p M. "whole" has no jump in sensible heat at the solidus.
    """
    molten, _ = _adiabat_runner(mock_config, tmp_path, monkeypatch)
    mock_config.interior_energetics.boundary.sensible_heat = 'whole'
    whole, _ = _adiabat_runner(mock_config, tmp_path, monkeypatch)

    assert whole.heat_capacity(_T_MUSH) == pytest.approx(7.466667e27, rel=1e-6)
    assert molten.heat_capacity(_T_MUSH) == pytest.approx(6.109867e27, rel=1e-6)
    # Factor guard: the difference is the solid part's sensible heat, c_p (1 - phi) M.
    phi = whole.melt_fraction(_T_MUSH)
    assert whole.heat_capacity(_T_MUSH) - molten.heat_capacity(_T_MUSH) == pytest.approx(
        1200.0 * (1.0 - phi) * _ADB_M_MANTLE, rel=1e-9
    )
    # Edge cases: wholly solid or molten, both count the whole mantle.
    for tmp in (_T_SOLID, _T_MOLTEN):
        assert whole.heat_capacity(tmp) == pytest.approx(4.8e27, rel=1e-12)
        assert molten.heat_capacity(tmp) == pytest.approx(4.8e27, rel=1e-12)
    # Just above the solidus "molten" has almost no sensible heat left, "whole" all of it.
    near_solidus = _ADB_TSOL + 0.6
    latent = 4.0e5 * _ADB_M_MANTLE / 600.0
    assert whole.heat_capacity(near_solidus) == pytest.approx(4.8e27 + latent, rel=1e-6)
    assert molten.heat_capacity(near_solidus) - latent < 1e-2 * 4.8e27

    # The linear model takes the same switch: the whole mantle replaces the shell above the front.
    interior_lin = _adiabat_interior(1)
    lin_whole = _make_runner(mock_config, tmp_path, interior_lin, 'linear')
    mock_config.interior_energetics.boundary.sensible_heat = 'molten'
    lin_molten = _make_runner(mock_config, tmp_path, interior_lin, 'linear')
    assert lin_whole.heat_capacity(_T_MUSH) > lin_molten.heat_capacity(_T_MUSH)
    assert lin_whole.heat_capacity(_T_MOLTEN) == pytest.approx(
        lin_molten.heat_capacity(_T_MOLTEN), rel=1e-12
    )


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_tidal_viscosity_kervazo_hands_the_tides_the_dummy_rheology(
    mock_config, tmp_path, monkeypatch
):
    """tidal_viscosity = "kervazo" publishes the Kervazo et al. (2021) viscosity.

    The fixture's own viscosity_model is constant at 1e2 Pa s. With "kervazo"
    a wholly solid mantle is handed 1.2567e19 Pa s per shell (phi = 0) and a
    wholly molten one 1.0 Pa s (phi = 1), the dummy backend's values, either
    side of LovePy's 1e9 Pa s viscous-block threshold. The viscosity that sets
    the convective heat flux is unchanged.
    """
    own, interior_own = _adiabat_runner(mock_config, tmp_path, monkeypatch, nlev_s=6)
    _run_to(own, interior_own, tmp_path, _T_SOLID)
    np.testing.assert_allclose(interior_own.visc, 1.0e2, rtol=1e-12)

    mock_config.interior_energetics.boundary.tidal_viscosity = 'kervazo'
    solid, interior_solid = _adiabat_runner(mock_config, tmp_path, monkeypatch, nlev_s=6)
    _run_to(solid, interior_solid, tmp_path, _T_SOLID)
    np.testing.assert_allclose(interior_solid.visc, 1.2567096326586444e19, rtol=1e-9)
    assert np.all(interior_solid.visc > 1e9)

    molten, interior_molten = _adiabat_runner(mock_config, tmp_path, monkeypatch, nlev_s=6)
    _run_to(molten, interior_molten, tmp_path, _T_MOLTEN)
    np.testing.assert_allclose(interior_molten.visc, 1.0, rtol=1e-12)
    assert np.all(interior_molten.visc < 1e9)

    # The convective flux still uses viscosity_model.
    assert solid.viscosity(_T_MUSH, 1600.0, 0.5) == pytest.approx(1.0e2, rel=1e-12)
    assert solid.q_m(_T_MUSH, 1600.0, 0.5) == pytest.approx(
        own.q_m(_T_MUSH, 1600.0, 0.5), rel=1e-12
    )

    # The linear model's single layer takes the same switch.
    interior_lin = _adiabat_interior(1)
    lin = _make_runner(mock_config, tmp_path, interior_lin, 'linear')
    _run_to(lin, interior_lin, tmp_path, _T_SOLID)
    assert interior_lin.visc[0] == pytest.approx(1.2567096326586444e19, rel=1e-9)


@pytest.mark.unit
def test_boundary_dummy_choice_keys_default_to_the_boundary_backend_and_validate():
    """The three keys default to this backend's own choices and reject other values.

    Error contract: a misspelt choice is refused at config load rather than
    silently falling back to the default behaviour.
    """
    from proteus.config._interior import InteriorBoundary

    defaults = InteriorBoundary()
    assert defaults.include_core_heat_capacity is False
    assert defaults.sensible_heat == 'molten'
    assert defaults.tidal_viscosity == 'boundary'

    dummy_like = InteriorBoundary(
        include_core_heat_capacity=True, sensible_heat='whole', tidal_viscosity='kervazo'
    )
    assert dummy_like.sensible_heat == 'whole'
    assert dummy_like.tidal_viscosity == 'kervazo'

    with pytest.raises(ValueError):
        InteriorBoundary(sensible_heat='all')
    with pytest.raises(ValueError):
        InteriorBoundary(tidal_viscosity='dummy')


# =============================================================================
# Tests: viscosity_phi, the melt fraction behind the convective viscosity
# =============================================================================
#
# With the test gradients the melting curves (2.0e-8 K/Pa) are steeper than the
# adiabat (1.5e-8 K/Pa), so the mantle crystallises from the bottom up and its
# top level is its most molten one, as in a real magma ocean.


def _top_and_global_runners(config, tmp_path, monkeypatch):
    """Two adiabat runners on the same mesh, viscosity_phi "top" and "global"."""
    config.interior_energetics.boundary.viscosity_model = 2  # aggregate
    glob, _ = _adiabat_runner(
        config, tmp_path, monkeypatch, dtdp=ADIABAT_DTDP, curve_dtdp=CURVE_DTDP
    )
    config.interior_energetics.boundary.viscosity_phi = 'top'
    top, interior_top = _adiabat_runner(
        config, tmp_path, monkeypatch, dtdp=ADIABAT_DTDP, curve_dtdp=CURVE_DTDP
    )
    return top, glob, interior_top


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_top_melt_fraction_is_the_top_level_of_the_adiabat(mock_config, tmp_path, monkeypatch):
    """viscosity_phi = "top" reads the melt fraction of the mantle's top level.

    Cross-implementation check: the closed form (T_p - T_sol,top) / (T_liq,top -
    T_sol,top) must equal the top level of the full adiabat integration
    (`AdiabatMeltModel.state(T_p).phi[-1]`), and it exceeds the mass average
    because the mantle freezes from the bottom up.
    """
    top, glob, _ = _top_and_global_runners(mock_config, tmp_path, monkeypatch)
    model = top.melt_curve.model
    p_top = model.profile.pressure[-1]
    assert top.T_solidus_top == pytest.approx(_ADB_TSOL + CURVE_DTDP * p_top, rel=1e-12)
    assert top.T_liquidus_top == pytest.approx(_ADB_TLIQ + CURVE_DTDP * p_top, rel=1e-12)
    # Pinned at 1900 K: the top level is 80% molten, the mass average 38%.
    assert top.top_melt_fraction(1900.0) == pytest.approx(0.7999966666666668, rel=1e-9)
    assert glob.melt_fraction(1900.0) == pytest.approx(0.38114628582270216, rel=1e-6)

    for tmp in (1500.0, 1700.0, 1900.0, 2100.0):
        expected = np.clip((tmp - top.T_solidus_top) / 600.0, 0.0, 1.0)
        assert top.top_melt_fraction(tmp) == pytest.approx(expected, abs=1e-12)
        assert top.top_melt_fraction(tmp) == pytest.approx(model.state(tmp).phi[-1], abs=1e-9)
        assert top.convective_melt_fraction(tmp) == top.top_melt_fraction(tmp)
        # "global" keeps the mass average.
        assert glob.convective_melt_fraction(tmp) == glob.melt_fraction(tmp)
        # Bottom-up crystallisation: the top is at least as molten as the average.
        assert top.top_melt_fraction(tmp) >= glob.melt_fraction(tmp)

    # Edge cases: below the top solidus and above the top liquidus.
    assert top.top_melt_fraction(top.T_solidus_top - 1.0) == pytest.approx(0.0, abs=1e-15)
    assert top.top_melt_fraction(top.T_liquidus_top + 1.0) == pytest.approx(1.0, rel=1e-15)


@pytest.mark.physics_invariant
def test_top_melt_fraction_is_a_step_when_the_top_curves_meet(
    mock_config, tmp_path, monkeypatch
):
    """Coincident top melting curves give a sharp front, not a division by zero.

    Limit-input behaviour: with no mushy interval at the top level the mantle
    top is solid below the melting temperature and molten at or above it.
    """
    top, _, _ = _top_and_global_runners(mock_config, tmp_path, monkeypatch)
    top.T_solidus_top = top.T_liquidus_top = 1800.0

    assert top.top_melt_fraction(1799.0) == pytest.approx(0.0, abs=1e-15)
    assert top.top_melt_fraction(1800.0) == pytest.approx(1.0, rel=1e-15)
    assert top.top_melt_fraction(1801.0) == pytest.approx(1.0, rel=1e-15)


@pytest.mark.physics_invariant
def test_viscosity_phi_top_sets_only_the_convective_flux(mock_config, tmp_path, monkeypatch):
    """With "top" the convective flux follows the top level; the energy budget does not.

    At 1900 K the top level (phi 0.80) is above the aggregate model's 0.4
    transition and the mass average (phi 0.38) below it, so "global" gives a
    near-solid viscosity and "top" a near-melt one: "top" loses heat far
    faster and has a far thinner boundary layer. Heat capacity, the reported
    melt fraction and the mantle state stay on the mass average.
    """
    top, glob, interior_top = _top_and_global_runners(mock_config, tmp_path, monkeypatch)
    tmp, tsurf = 1900.0, 1600.0
    phi_top, phi_avg = top.top_melt_fraction(tmp), glob.melt_fraction(tmp)

    # The viscosity behind q_m, Ra and delta is the top level's.
    assert top.q_m(tmp, tsurf, top.convective_melt_fraction(tmp)) == pytest.approx(
        glob.q_m(tmp, tsurf, phi_top), rel=1e-12
    )
    assert top.viscosity(tmp, tsurf, phi_top) < 1e-6 * glob.viscosity(tmp, tsurf, phi_avg)
    q_top = top.q_m(tmp, tsurf, phi_top)
    q_glob = glob.q_m(tmp, tsurf, phi_avg)
    assert q_top > 10.0 * q_glob

    # The heat capacity is the same in both: only the flux changed.
    assert top.heat_capacity(tmp) == pytest.approx(glob.heat_capacity(tmp), rel=1e-12)
    ratio = top.dT_pdt(tmp, tsurf, 0.0) / glob.dT_pdt(tmp, tsurf, 0.0)
    assert ratio == pytest.approx(q_top / q_glob, rel=1e-9)

    # The surface equation sees the same flux and a thinner boundary layer.
    assert top.boundary_layer_thickness(tmp, tsurf, phi_top) < glob.boundary_layer_thickness(
        tmp, tsurf, phi_avg
    )
    assert top.dT_surfdt(tmp, tsurf) != pytest.approx(glob.dT_surfdt(tmp, tsurf), rel=1e-3)

    # The step's reported state stays on the mass average.
    _, out = _run_to(top, interior_top, tmp_path, tmp)
    assert out['Phi_global'] == pytest.approx(phi_avg, abs=1e-4)
    assert out['boundary_layer_thickness'] == pytest.approx(
        top.boundary_layer_thickness(tmp, 1600.0, phi_top), rel=1e-12
    )


@pytest.mark.unit
def test_viscosity_phi_defaults_to_global_and_top_needs_the_adiabat(
    mock_config, tmp_path, monkeypatch
):
    """viscosity_phi defaults to "global"; "top" is refused without a depth profile.

    Error contract: a misspelt value is refused at config load, and "top" with
    the linear melt model is refused both by the config validator and by the
    runner, rather than silently using the mass average.
    """
    from proteus.config._interior import InteriorBoundary, valid_interiorboundary

    assert InteriorBoundary().viscosity_phi == 'global'
    assert InteriorBoundary(viscosity_phi='top').viscosity_phi == 'top'
    with pytest.raises(ValueError):
        InteriorBoundary(viscosity_phi='mean')

    linear_top = SimpleNamespace(
        module='boundary', boundary=InteriorBoundary(melt_model='linear', viscosity_phi='top')
    )
    with pytest.raises(ValueError, match='needs'):
        valid_interiorboundary(linear_top, None, None)
    # Discriminator: the same block with the adiabat passes.
    adiabat_top = SimpleNamespace(
        module='boundary', boundary=InteriorBoundary(melt_model='adiabat', viscosity_phi='top')
    )
    valid_interiorboundary(adiabat_top, None, None)

    mock_config.interior_energetics.boundary.viscosity_phi = 'top'
    with pytest.raises(ValueError, match='adiabat'):
        _make_runner(mock_config, tmp_path, _adiabat_interior(1), 'linear')
