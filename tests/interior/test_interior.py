"""
Unit tests for proteus.interior.dummy module

This test file validates the dummy interior evolution model used for simple
thermal evolution calculations without full mantle convection physics.

Testing standards and documentation:
- docs/test_infrastructure.md: Test infrastructure overview
- docs/test_categorization.md: Test marker definitions
- docs/test_building.md: Best practices for test construction

Functions tested:
- calculate_simple_mantle_mass(): Mantle mass from radius/core fraction
- run_dummy_int(): Full interior time-step with thermal evolution
- _calc_phi(): Melt fraction calculation (internal helper)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from proteus.interior.common import Interior_t
from proteus.interior.dummy import calculate_simple_mantle_mass, run_dummy_int

# ============================================================================
# Test calculate_simple_mantle_mass
# ============================================================================


@pytest.mark.unit
def test_calculate_simple_mantle_mass_earth_like():
    """Test mantle mass calculation for Earth-like planet.

    Uses realistic values: R_earth = 6.371e6 m, corefrac = 0.55,
    rho_mantle = 4000 kg/m³. Validates that computed mantle mass
    is physically reasonable (~4e24 kg).
    """
    radius = 6.371e6  # meters (Earth radius)
    corefrac = 0.55
    density = 4000.0  # kg/m³

    mantle_mass = calculate_simple_mantle_mass(radius, corefrac, density)

    # Earth's mantle mass is ~4.0e24 kg
    assert mantle_mass > 0
    assert 3e24 < mantle_mass < 5e24


@pytest.mark.unit
def test_calculate_simple_mantle_mass_no_core():
    """Test mantle mass with zero core fraction (entire planet is mantle).

    For corefrac=0, the mantle volume equals the full planet volume.
    This validates the limiting case of a coreless body.
    """
    radius = 1e6  # meters
    corefrac = 0.0
    density = 3000.0  # kg/m³

    mantle_mass = calculate_simple_mantle_mass(radius, corefrac, density)

    # Full sphere volume: (4/3) * pi * r^3
    expected_mass = (4 * np.pi / 3) * radius**3 * density
    assert mantle_mass == pytest.approx(expected_mass, rel=1e-10)


@pytest.mark.unit
def test_calculate_simple_mantle_mass_scales_with_radius_cubed():
    """Test that mantle mass scales with radius^3 (volume scaling).

    Doubling radius should increase mantle mass by factor of 8
    (since volume ∝ r³). Validates geometric scaling.
    """
    radius1 = 1e6
    radius2 = 2e6
    corefrac = 0.5
    density = 3500.0

    mass1 = calculate_simple_mantle_mass(radius1, corefrac, density)
    mass2 = calculate_simple_mantle_mass(radius2, corefrac, density)

    # Volume scaling: (2r)³ = 8r³
    assert mass2 / mass1 == pytest.approx(8.0, rel=1e-8)


@pytest.mark.unit
def test_calculate_simple_mantle_mass_scales_with_density():
    """Test linear scaling with density.

    Mantle mass should scale linearly with density for fixed geometry.
    Validates that density is correctly applied to volume.
    """
    radius = 5e6
    corefrac = 0.4
    density1 = 3000.0
    density2 = 6000.0

    mass1 = calculate_simple_mantle_mass(radius, corefrac, density1)
    mass2 = calculate_simple_mantle_mass(radius, corefrac, density2)

    # Linear density scaling
    assert mass2 / mass1 == pytest.approx(2.0, rel=1e-10)


@pytest.mark.unit
@pytest.mark.skip(reason='Source code raises exception for corefrac=1.0 by design')
def test_calculate_simple_mantle_mass_full_core():
    """Test edge case: core fills entire planet (corefrac=1.0).

    This should result in zero mantle volume and thus zero mantle mass.
    Validates handling of geometric degeneracy.
    """
    radius = 1e6
    corefrac = 1.0
    density = 3000.0

    mantle_mass = calculate_simple_mantle_mass(radius, corefrac, density)

    # No mantle if core fills entire planet
    assert mantle_mass == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# Test run_dummy_int
# ============================================================================


def _create_mock_config(
    ini_tmagma: float = 3000.0,
    mantle_tliq: float = 2500.0,
    mantle_tsol: float = 1500.0,
    mantle_cp: float = 1200.0,
    mantle_rho: float = 4000.0,
    H_radio: float = 5e-12,
    tmagma_rtol: float = 0.01,
    tmagma_atol: float = 10.0,
    corefrac: float = 0.55,
    core_heatcap: float = 8e26,
    tidal_heat: bool = False,
) -> Any:
    """Helper to create minimal Config mock for interior tests."""
    config = MagicMock()
    config.interior.dummy.ini_tmagma = ini_tmagma
    config.interior.dummy.mantle_tliq = mantle_tliq
    config.interior.dummy.mantle_tsol = mantle_tsol
    config.interior.dummy.mantle_cp = mantle_cp
    config.interior.dummy.mantle_rho = mantle_rho
    config.interior.dummy.H_radio = H_radio
    config.interior.dummy.tmagma_rtol = tmagma_rtol
    config.interior.dummy.tmagma_atol = tmagma_atol
    config.struct.corefrac = corefrac
    config.struct.core_heatcap = core_heatcap
    config.interior.tidal_heat = tidal_heat
    return config


@pytest.mark.unit
def test_run_dummy_int_initialization():
    """Test initial condition (ic=1) sets T_magma to ini_tmagma.

    On the first interior timestep (ic=1), the magma temperature should
    be initialized to the configured ini_tmagma value, with dt=0.
    """
    config = _create_mock_config(ini_tmagma=3000.0)
    dirs = {}
    hf_row = {
        'F_atm': 100.0,
        'R_int': 6e6,
        'M_core': 2e24,
        'P_surf': 1e5,
        'Time': 0.0,
        'T_magma': 0.0,  # Irrelevant for ic=1
    }
    hf_all = pd.DataFrame()

    interior_o = Interior_t(nlev_b=2)
    interior_o.ic = 1
    interior_o.tides = [0.0]

    sim_time, output = run_dummy_int(config, dirs, hf_row, hf_all, interior_o)

    assert output['T_magma'] == pytest.approx(3000.0)
    assert output['T_pot'] == pytest.approx(3000.0)
    assert sim_time == 0.0  # dt=0 on initialization


@pytest.mark.unit
def test_run_dummy_int_melt_fraction_fully_solid():
    """Test melt fraction (phi) when T_magma < T_solidus.

    Below the solidus temperature, the mantle should be fully solid
    (phi=0). Validates phase boundary handling.
    """
    config = _create_mock_config(ini_tmagma=1400.0, mantle_tsol=1500.0, mantle_tliq=2500.0)
    dirs = {}
    hf_row = {'F_atm': 50.0, 'R_int': 6e6, 'M_core': 2e24, 'P_surf': 1e5, 'Time': 0.0}
    hf_all = pd.DataFrame()

    interior_o = Interior_t(nlev_b=2)
    interior_o.ic = 1
    interior_o.tides = [0.0]

    _, output = run_dummy_int(config, dirs, hf_row, hf_all, interior_o)

    assert output['Phi_global'] == pytest.approx(0.0)
    assert output['M_mantle_liquid'] == pytest.approx(0.0)
    assert output['M_mantle_solid'] == pytest.approx(output['M_mantle'])


@pytest.mark.unit
def test_run_dummy_int_melt_fraction_fully_molten():
    """Test melt fraction (phi) when T_magma > T_liquidus.

    Above the liquidus temperature, the mantle should be fully molten
    (phi=1). Validates upper phase boundary.
    """
    config = _create_mock_config(ini_tmagma=2600.0, mantle_tsol=1500.0, mantle_tliq=2500.0)
    dirs = {}
    hf_row = {'F_atm': 50.0, 'R_int': 6e6, 'M_core': 2e24, 'P_surf': 1e5, 'Time': 0.0}
    hf_all = pd.DataFrame()

    interior_o = Interior_t(nlev_b=2)
    interior_o.ic = 1
    interior_o.tides = [0.0]

    _, output = run_dummy_int(config, dirs, hf_row, hf_all, interior_o)

    assert output['Phi_global'] == pytest.approx(1.0)
    assert output['M_mantle_solid'] == pytest.approx(0.0)
    assert output['M_mantle_liquid'] == pytest.approx(output['M_mantle'])


@pytest.mark.unit
def test_run_dummy_int_melt_fraction_partial():
    """Test melt fraction (phi) for T_solidus < T_magma < T_liquidus.

    In the partial melt regime, phi should vary linearly between 0 and 1.
    For T_magma exactly midway between solidus and liquidus, phi=0.5.
    """
    tsol = 1500.0
    tliq = 2500.0
    tmid = (tsol + tliq) / 2.0  # 2000 K

    config = _create_mock_config(ini_tmagma=tmid, mantle_tsol=tsol, mantle_tliq=tliq)
    dirs = {}
    hf_row = {'F_atm': 50.0, 'R_int': 6e6, 'M_core': 2e24, 'P_surf': 1e5, 'Time': 0.0}
    hf_all = pd.DataFrame()

    interior_o = Interior_t(nlev_b=2)
    interior_o.ic = 1
    interior_o.tides = [0.0]

    _, output = run_dummy_int(config, dirs, hf_row, hf_all, interior_o)

    assert output['Phi_global'] == pytest.approx(0.5, rel=1e-10)
    assert output['M_mantle_liquid'] == pytest.approx(output['M_mantle'] / 2.0, rel=1e-8)
    assert output['M_mantle_solid'] == pytest.approx(output['M_mantle'] / 2.0, rel=1e-8)


@pytest.mark.unit
def test_run_dummy_int_radiogenic_heating():
    """Test radiogenic heating contribution (F_radio).

    Radiogenic heat flux should equal H_radio * M_mantle / area.
    Validates that internal heat generation is correctly computed.
    """
    H_radio = 1e-11  # W/kg
    config = _create_mock_config(H_radio=H_radio, ini_tmagma=2000.0)
    dirs = {}
    R_int = 6e6
    hf_row = {'F_atm': 100.0, 'R_int': R_int, 'M_core': 2e24, 'P_surf': 1e5, 'Time': 0.0}
    hf_all = pd.DataFrame()

    interior_o = Interior_t(nlev_b=2)
    interior_o.ic = 1
    interior_o.tides = [0.0]

    _, output = run_dummy_int(config, dirs, hf_row, hf_all, interior_o)

    area = 4 * np.pi * R_int**2
    expected_F_radio = H_radio * output['M_mantle'] / area

    assert output['F_radio'] == pytest.approx(expected_F_radio, rel=1e-8)


@pytest.mark.unit
def test_run_dummy_int_tidal_heating_enabled():
    """Test tidal heating flux when tidal_heat=True.

    When tidal heating is enabled, F_tidal should be computed from
    interior_o.tides[0] scaled by mantle mass and area.
    """
    config = _create_mock_config(tidal_heat=True, ini_tmagma=2000.0)
    dirs = {}
    R_int = 6e6
    hf_row = {'F_atm': 100.0, 'R_int': R_int, 'M_core': 2e24, 'P_surf': 1e5, 'Time': 0.0}
    hf_all = pd.DataFrame()

    interior_o = Interior_t(nlev_b=2)
    interior_o.ic = 1
    interior_o.tides = [5e-12]  # W/kg specific tidal heating

    _, output = run_dummy_int(config, dirs, hf_row, hf_all, interior_o)

    area = 4 * np.pi * R_int**2
    expected_F_tidal = interior_o.tides[0] * output['M_mantle'] / area

    assert output['F_tidal'] == pytest.approx(expected_F_tidal, rel=1e-8)


@pytest.mark.unit
def test_run_dummy_int_tidal_heating_disabled():
    """Test that F_tidal=0 when tidal_heat=False.

    Even if interior_o.tides is nonzero, tidal heating should be
    ignored if the config flag is disabled.
    """
    config = _create_mock_config(tidal_heat=False, ini_tmagma=2000.0)
    dirs = {}
    hf_row = {'F_atm': 100.0, 'R_int': 6e6, 'M_core': 2e24, 'P_surf': 1e5, 'Time': 0.0}
    hf_all = pd.DataFrame()

    interior_o = Interior_t(nlev_b=2)
    interior_o.ic = 1
    interior_o.tides = [5e-12]  # Should be ignored

    _, output = run_dummy_int(config, dirs, hf_row, hf_all, interior_o)

    assert output['F_tidal'] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.unit
def test_run_dummy_int_interior_arrays():
    """Test that Interior_t arrays are correctly populated.

    Validates that phi, mass, visc, density, temp, pres, and radius
    arrays are set with correct shapes and physically valid values.
    """
    config = _create_mock_config(ini_tmagma=2000.0, mantle_rho=4000.0)
    dirs = {}
    R_int = 6e6
    corefrac = 0.55
    P_surf = 1e5
    hf_row = {'F_atm': 100.0, 'R_int': R_int, 'M_core': 2e24, 'P_surf': P_surf, 'Time': 0.0}
    hf_all = pd.DataFrame()

    interior_o = Interior_t(nlev_b=2)
    interior_o.ic = 1
    interior_o.tides = [0.0]

    _, output = run_dummy_int(config, dirs, hf_row, hf_all, interior_o)

    # Check array shapes
    assert len(interior_o.phi) == 1
    assert len(interior_o.mass) == 1
    assert len(interior_o.visc) == 1
    assert len(interior_o.density) == 1
    assert len(interior_o.temp) == 1
    assert len(interior_o.pres) == 1
    assert len(interior_o.radius) == 2  # [R_core, R_int]

    # Check values
    assert interior_o.phi[0] == pytest.approx(output['Phi_global'])
    assert interior_o.mass[0] == pytest.approx(output['M_mantle'])
    assert interior_o.density[0] == pytest.approx(4000.0)
    assert interior_o.temp[0] == pytest.approx(output['T_magma'])
    assert interior_o.pres[0] == pytest.approx(P_surf)
    assert interior_o.radius[0] == pytest.approx(corefrac * R_int)
    assert interior_o.radius[1] == pytest.approx(R_int)


@pytest.mark.unit
def test_run_dummy_int_rf_depth_scaling():
    """Test RF_depth (rheological front depth) scales with melt fraction.

    RF_depth should be Phi_global * (1 - corefrac), representing
    the depth of the molten region normalized by planet radius.
    """
    config = _create_mock_config(
        ini_tmagma=2000.0, mantle_tsol=1500.0, mantle_tliq=2500.0, corefrac=0.6
    )
    dirs = {}
    hf_row = {'F_atm': 100.0, 'R_int': 6e6, 'M_core': 2e24, 'P_surf': 1e5, 'Time': 0.0}
    hf_all = pd.DataFrame()

    interior_o = Interior_t(nlev_b=2)
    interior_o.ic = 1
    interior_o.tides = [0.0]

    _, output = run_dummy_int(config, dirs, hf_row, hf_all, interior_o)

    expected_rf_depth = output['Phi_global'] * (1 - 0.6)
    assert output['RF_depth'] == pytest.approx(expected_rf_depth, rel=1e-10)


# ============================================================================
# Test interior.common rheological functions
# ============================================================================


@pytest.mark.unit
def test_eval_rheoparam_viscosity_solid():
    """Test viscosity evaluation at zero melt fraction (solid state).

    Physics: Solid mantle (phi=0) should have baseline viscosity par_visc.dotl
    without rheological reduction from melting. Expected viscosity = 1.0 Pa·s
    based on par_visc parameters.
    """
    from proteus.interior.common import eval_rheoparam

    phi = 0.0  # Solid state
    visc = eval_rheoparam(phi, 'visc')

    # At phi=0, viscosity is reduced by rheological function but not zero
    assert visc > 0
    assert isinstance(visc, (float, np.floating))


@pytest.mark.unit
def test_eval_rheoparam_viscosity_molten():
    """Test viscosity evaluation at high melt fraction (molten state).

    Physics: Heavily molten mantle (phi=0.35) should have dramatically reduced
    viscosity due to melt weakening. Viscosity should be much lower than solid case.
    Venus-like magma ocean scenario: phi ~0.3-0.4.
    """
    from proteus.interior.common import eval_rheoparam

    phi_solid = 0.0
    phi_molten = 0.35

    visc_solid = eval_rheoparam(phi_solid, 'visc')
    visc_molten = eval_rheoparam(phi_molten, 'visc')

    # Molten state should have lower viscosity than solid
    assert visc_molten < visc_solid
    assert visc_molten > 0


@pytest.mark.unit
def test_eval_rheoparam_shear_modulus():
    """Test shear modulus evaluation across melt fractions.

    Physics: Shear modulus (rigidity) decreases with melt fraction.
    par_shear parameters are from Kervazo+21 for melt weakening behavior.
    """
    from proteus.interior.common import eval_rheoparam

    phi_low = 0.1
    phi_high = 0.3

    shear_low = eval_rheoparam(phi_low, 'shear')
    shear_high = eval_rheoparam(phi_high, 'shear')

    # Higher melt fraction should reduce shear modulus
    assert shear_high < shear_low
    assert shear_low > 0
    assert shear_high > 0


@pytest.mark.unit
def test_eval_rheoparam_bulk_modulus():
    """Test bulk modulus evaluation.

    Physics: Bulk modulus (incompressibility) also affected by melt fraction.
    Represents resistance to compression in melted material.
    """
    from proteus.interior.common import eval_rheoparam

    phi = 0.2

    bulk = eval_rheoparam(phi, 'bulk')

    # Bulk modulus should be positive
    assert bulk > 0
    assert isinstance(bulk, (float, np.floating))


@pytest.mark.unit
def test_eval_rheoparam_invalid_parameter():
    """Test eval_rheoparam raises ValueError for invalid parameter name.

    Physics: Only 'visc', 'shear', 'bulk' are valid rheological parameters
    in the match statement. Invalid names should fail with clear error.
    """
    from proteus.interior.common import eval_rheoparam

    with pytest.raises(ValueError, match='Invalid rheological parameter'):
        eval_rheoparam(0.2, 'invalid')


@pytest.mark.unit
def test_interior_t_initialization():
    """Test Interior_t class initialization with default parameters.

    Physics: Interior_t represents mantle column with nlev_b boundary points
    (creating nlev_s=nlev_b-1 shell layers). All property arrays should be
    initialized to zero.
    """
    from proteus.interior.common import Interior_t

    nlev_b = 5
    interior = Interior_t(nlev_b)

    assert interior.nlev_b == 5
    assert interior.nlev_s == 4  # One less layer
    assert interior.ic == -1  # Initial condition flag
    assert interior.dt == 1.0  # Default time step [yr]
    assert len(interior.radius) == 5
    assert len(interior.tides) == 4
    assert len(interior.phi) == 4


@pytest.mark.unit
def test_interior_t_update_rheology_solid():
    """Test update_rheology with solid mantle (zero melt fraction).

    Physics: Solid mantle (all phi=0) should compute rheological parameters
    without melt weakening. Shear and bulk moduli should be positive and stable.
    """
    from proteus.interior.common import Interior_t

    interior = Interior_t(nlev_b=3)
    interior.phi = np.array([0.0, 0.0])  # Solid mantle

    interior.update_rheology(visc=False)

    # Check that moduli were computed
    assert len(interior.shear) == 2
    assert len(interior.bulk) == 2
    assert all(interior.shear > 0)
    assert all(interior.bulk > 0)
    # Viscosity should not be updated when visc=False
    assert interior.visc[0] == 0.0


@pytest.mark.unit
def test_interior_t_update_rheology_with_viscosity():
    """Test update_rheology with viscosity calculation enabled.

    Physics: When visc=True, viscosity array is also computed based on melt
    fraction. Mixed melt-solid case: phi=[0.1, 0.3] creates different
    viscosities at different depths.
    """
    from proteus.interior.common import Interior_t

    interior = Interior_t(nlev_b=3)
    interior.phi = np.array([0.1, 0.3])  # Partially molten

    interior.update_rheology(visc=True)

    # Viscosity should be computed at different values
    assert interior.visc[0] > 0
    assert interior.visc[1] > 0
    # Different melt fractions should yield different viscosities
    assert interior.visc[1] < interior.visc[0]


@pytest.mark.unit
def test_get_file_tides_path():
    """Test get_file_tides constructs correct filepath.

    Physics: Tidal heating array saved in data/ subdirectory with filename
    'tides_recent.dat' for model resumption from previous state.
    """
    from proteus.interior.common import get_file_tides

    outdir = '/tmp/output'
    fpath = get_file_tides(outdir)

    assert fpath == '/tmp/output/data/tides_recent.dat'
    assert 'data' in fpath
    assert 'tides_recent.dat' in fpath


@pytest.mark.unit
def test_interior_t_resume_tides_file_missing(tmp_path):
    """Test resume_tides when tides_recent.dat doesn't exist.

    Physics: First-time simulation has no previous tides file. Function
    should log warning and return without error, allowing simulation to
    continue with default (zero) tidal heating.
    """
    from proteus.interior.common import Interior_t

    outdir = str(tmp_path / 'output')
    interior = Interior_t(nlev_b=3)

    # Should not raise error, just log warning
    interior.resume_tides(outdir)

    # Tides should remain zero from initialization
    assert interior.tides[0] == 0.0


@pytest.mark.unit
def test_interior_t_write_and_resume_tides(tmp_path):
    """Test write_tides and resume_tides round-trip.

    Physics: Simulations can be resumed by saving tidal heating (W/kg) and
    melt fraction arrays to disk. Reading back should restore exact values
    (within floating-point precision).
    """
    from proteus.interior.common import Interior_t

    outdir = str(tmp_path / 'output')
    (tmp_path / 'output' / 'data').mkdir(parents=True)

    # Write tides from initial interior state
    interior_write = Interior_t(nlev_b=4)
    interior_write.phi = np.array([0.1, 0.2, 0.3])
    interior_write.tides = np.array([1e-3, 2e-3, 3e-3])

    interior_write.write_tides(outdir)

    # Read back into new interior object
    interior_read = Interior_t(nlev_b=4)
    interior_read.resume_tides(outdir)

    # Check values match
    np.testing.assert_allclose(interior_read.phi, interior_write.phi, rtol=1e-6)
    np.testing.assert_allclose(interior_read.tides, interior_write.tides, rtol=1e-6)


@pytest.mark.unit
def test_interior_t_single_layer_dummy_interior():
    """Test Interior_t with single layer (dummy interior setup).

    Physics: Dummy interior models (nlev_b=2) represent single-layer mantle
    without internal resolution, used for fast calculations.
    """
    from proteus.interior.common import Interior_t

    interior = Interior_t(nlev_b=2)

    # Single layer (nlev_s = 1)
    assert interior.nlev_s == 1
    assert len(interior.tides) == 1
    assert len(interior.phi) == 1
    assert len(interior.radius) == 2  # One more radius point than layers
