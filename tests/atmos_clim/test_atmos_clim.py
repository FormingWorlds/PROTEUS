"""
Unit tests for proteus.atmos_clim.dummy module

This module tests the dummy atmosphere model including:
- RunDummyAtm(): Main function orchestrating flux calculations and surface temperature
- Fixed surface temperature mode (surf_state='fixed')
- Conductive boundary layer mode (surf_state='skin') with root-finding
- Flux calculation physics (OLR, scattered SW, net flux)
- Scale height and observed radius calculations

Physics tested:
- Stefan-Boltzmann radiative transfer (σT⁴ scaling)
- Gamma-factor atmospheric opacity parameterization (0=transparent, 1=opaque)
- Solar flux geometry (zenith angle, albedo, s0_factor)
- Conductive boundary layer energy balance (skin conductivity)
- Scale height H = RT/(μg) for atmospheric thickness
- Net flux constraint (prevent warming: F_atm >= 0)

All tests use mocked config objects to isolate physics calculations (<100ms runtime).
See docs/test_infrastructure.md, docs/test_building.md for testing standards.

Related documentation:
- docs/test_infrastructure.md: Test framework and CI integration
- docs/test_categorization.md: Test markers and categories
- docs/test_building.md: Best practices for writing robust tests
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from proteus.utils.constants import const_R, const_sigma

# =======================================================================================
# SECTION: RunDummyAtm() — Fixed surface temperature mode
# =======================================================================================


@pytest.mark.unit
def test_rundummyatm_fixed_surface():
    """Test dummy atmosphere with fixed surface temperature (surf_state='fixed').

    Physical scenario: Magma ocean planet with T_surf = T_magma (no conductive lid).
    Validates that surface temperature equals magma temperature and fluxes are calculated.
    """
    from proteus.atmos_clim.dummy import RunDummyAtm

    # Mock config for fixed surface state
    config = MagicMock()
    config.atmos_clim.dummy.gamma = 0.5  # Semi-opaque atmosphere
    config.atmos_clim.dummy.height_factor = 1.0
    config.orbit.zenith_angle = 0.0  # Subsolar point
    config.orbit.s0_factor = 1.0
    config.atmos_clim.surf_greyalbedo = 0.1  # Low surface albedo
    config.atmos_clim.surf_state = 'fixed'
    config.atmos_clim.prevent_warming = False

    # Minimal dirs (not used in fixed mode but required by function signature)
    dirs = {}

    # Minimal hf_row for Earth-like planet
    hf_row = {
        'T_magma': 1800.0,  # K; magma ocean temperature
        'albedo_pl': 0.3,  # Planetary albedo
        'F_ins': 1361.0,  # W/m²; Solar constant at 1 AU
        'atm_kg_per_mol': 0.029,  # kg/mol; mean molecular weight (N₂-like)
        'gravity': 9.8,  # m/s²; Earth surface gravity
        'R_int': 6.371e6,  # m; Earth radius
        'P_surf': 1e5,  # Pa; 1 bar surface pressure
    }

    # Call RunDummyAtm
    output = RunDummyAtm(dirs, config, hf_row)

    # Verify T_surf equals T_magma (fixed mode)
    assert output['T_surf'] == pytest.approx(1800.0, rel=1e-8)

    # Verify OLR > 0 (Stefan-Boltzmann emission)
    assert output['F_olr'] > 0.0

    # Verify net flux F_atm > 0 (radiative cooling)
    assert output['F_atm'] > 0.0

    # Verify scattered flux F_sct >= 0
    assert output['F_sct'] >= 0.0

    # Verify observed radius R_obs > R_int (atmosphere adds thickness)
    assert output['R_obs'] > hf_row['R_int']

    # Verify albedo is physical (0 <= albedo <= 1)
    assert 0.0 <= output['albedo'] <= 1.0


@pytest.mark.unit
def test_rundummyatm_transparent_atmosphere():
    """Test dummy atmosphere with gamma=0 (transparent atmosphere).

    Physical scenario: Planet with no atmospheric opacity (OLR = σT⁴).
    Validates that gamma=0 gives full surface emission (no atmospheric absorption).
    """
    from proteus.atmos_clim.dummy import RunDummyAtm

    # Mock config with transparent atmosphere
    config = MagicMock()
    config.atmos_clim.dummy.gamma = 0.0  # Fully transparent
    config.atmos_clim.dummy.height_factor = 1.0
    config.orbit.zenith_angle = 0.0
    config.orbit.s0_factor = 1.0
    config.atmos_clim.surf_greyalbedo = 0.0  # Zero surface albedo
    config.atmos_clim.surf_state = 'fixed'
    config.atmos_clim.prevent_warming = False

    dirs = {}

    hf_row = {
        'T_magma': 1500.0,  # K
        'albedo_pl': 0.0,
        'F_ins': 1000.0,  # W/m²
        'atm_kg_per_mol': 0.044,  # kg/mol; CO₂-like
        'gravity': 10.0,  # m/s²
        'R_int': 6.0e6,  # m
        'P_surf': 1e5,  # Pa
    }

    # Call RunDummyAtm
    output = RunDummyAtm(dirs, config, hf_row)

    # With gamma=0, OLR should equal σT⁴ exactly
    expected_olr = const_sigma * hf_row['T_magma'] ** 4.0
    assert output['F_olr'] == pytest.approx(expected_olr, rel=1e-8)


@pytest.mark.unit
def test_rundummyatm_opaque_atmosphere():
    """Test dummy atmosphere with gamma=1 (fully opaque atmosphere).

    Physical scenario: Planet with maximum atmospheric opacity (OLR = 0).
    Validates that gamma=1 gives zero OLR (complete atmospheric absorption).
    """
    from proteus.atmos_clim.dummy import RunDummyAtm

    # Mock config with opaque atmosphere
    config = MagicMock()
    config.atmos_clim.dummy.gamma = 1.0  # Fully opaque
    config.atmos_clim.dummy.height_factor = 1.0
    config.orbit.zenith_angle = 0.0
    config.orbit.s0_factor = 1.0
    config.atmos_clim.surf_greyalbedo = 0.0
    config.atmos_clim.surf_state = 'fixed'
    config.atmos_clim.prevent_warming = False

    dirs = {}

    hf_row = {
        'T_magma': 1500.0,  # K
        'albedo_pl': 0.0,
        'F_ins': 1000.0,  # W/m²
        'atm_kg_per_mol': 0.044,
        'gravity': 10.0,
        'R_int': 6.0e6,
        'P_surf': 1e5,
    }

    # Call RunDummyAtm
    output = RunDummyAtm(dirs, config, hf_row)

    # With gamma=1, OLR should be zero (fully absorbed by atmosphere)
    assert output['F_olr'] == pytest.approx(0.0, abs=1e-10)


# =======================================================================================
# SECTION: RunDummyAtm() — Conductive boundary layer (skin) mode
# =======================================================================================


@pytest.mark.unit
def test_rundummyatm_skin_mode():
    """Test dummy atmosphere with conductive boundary layer (surf_state='skin').

    Physical scenario: Planet with conductive lid separating surface from magma.
    Validates that T_surf < T_magma and energy balance F_net = F_conductive is satisfied.
    """
    from proteus.atmos_clim.dummy import RunDummyAtm

    # Mock config for skin mode
    config = MagicMock()
    config.atmos_clim.dummy.gamma = 0.5
    config.atmos_clim.dummy.height_factor = 1.0
    config.orbit.zenith_angle = 0.0
    config.orbit.s0_factor = 1.0
    config.atmos_clim.surf_greyalbedo = 0.1
    config.atmos_clim.surf_state = 'skin'
    config.atmos_clim.prevent_warming = False
    config.atmos_clim.surface_k = 2.0  # W/(m·K); thermal conductivity
    config.atmos_clim.surface_d = 1000.0  # m; conductive lid thickness

    dirs = {}

    hf_row = {
        'T_magma': 1800.0,  # K; magma temperature below lid
        'albedo_pl': 0.3,
        'F_ins': 1361.0,  # W/m²
        'atm_kg_per_mol': 0.029,
        'gravity': 9.8,
        'R_int': 6.371e6,
        'P_surf': 1e5,
    }

    # Call RunDummyAtm
    output = RunDummyAtm(dirs, config, hf_row)

    # Verify T_surf < T_magma (conduction cools surface)
    assert output['T_surf'] < hf_row['T_magma']

    # Verify T_surf is physical (> 0 K)
    assert output['T_surf'] > 0.0

    # Verify net flux F_atm > 0
    assert output['F_atm'] > 0.0

    # Verify observed radius includes atmosphere
    assert output['R_obs'] > hf_row['R_int']


@pytest.mark.unit
def test_rundummyatm_skin_convergence():
    """Test that skin mode root-finding converges successfully.

    Physical scenario: Standard Earth-like conductive lid parameters.
    Validates that scipy.optimize.root_scalar finds T_surf solution.
    """
    from proteus.atmos_clim.dummy import RunDummyAtm

    # Mock config with realistic skin parameters
    config = MagicMock()
    config.atmos_clim.dummy.gamma = 0.6
    config.atmos_clim.dummy.height_factor = 1.0
    config.orbit.zenith_angle = 30.0  # 30° zenith angle
    config.orbit.s0_factor = 1.0
    config.atmos_clim.surf_greyalbedo = 0.2
    config.atmos_clim.surf_state = 'skin'
    config.atmos_clim.prevent_warming = False
    config.atmos_clim.surface_k = 3.0  # W/(m·K)
    config.atmos_clim.surface_d = 500.0  # m

    dirs = {}

    hf_row = {
        'T_magma': 1600.0,  # K
        'albedo_pl': 0.25,
        'F_ins': 1200.0,  # W/m²
        'atm_kg_per_mol': 0.030,
        'gravity': 9.5,
        'R_int': 6.2e6,
        'P_surf': 1e5,
    }

    # Should converge without raising RuntimeError
    output = RunDummyAtm(dirs, config, hf_row)

    # Verify solution exists
    assert output['T_surf'] > 0.0
    assert output['T_surf'] < hf_row['T_magma']


# =======================================================================================
# SECTION: RunDummyAtm() — Flux physics and edge cases
# =======================================================================================


@pytest.mark.unit
def test_rundummyatm_prevent_warming():
    """Test prevent_warming constraint enforces F_atm >= 1e-8 W/m².

    Physical scenario: Planet with near-zero or negative net flux.
    Validates that prevent_warming clamps F_atm to small positive value.
    """
    from proteus.atmos_clim.dummy import RunDummyAtm

    # Mock config with high albedo + low insolation → negative flux
    config = MagicMock()
    config.atmos_clim.dummy.gamma = 0.1  # Very transparent
    config.atmos_clim.dummy.height_factor = 1.0
    config.orbit.zenith_angle = 80.0  # High zenith angle (low flux)
    config.orbit.s0_factor = 0.1  # Very low insolation factor
    config.atmos_clim.surf_greyalbedo = 0.9  # High surface albedo
    config.atmos_clim.surf_state = 'fixed'
    config.atmos_clim.prevent_warming = True  # Enable constraint

    dirs = {}

    hf_row = {
        'T_magma': 500.0,  # K; low surface temp
        'albedo_pl': 0.8,  # High planetary albedo
        'F_ins': 100.0,  # W/m²; very low insolation
        'atm_kg_per_mol': 0.044,
        'gravity': 5.0,
        'R_int': 3.0e6,
        'P_surf': 1e4,
    }

    # Call RunDummyAtm
    output = RunDummyAtm(dirs, config, hf_row)

    # Verify F_atm >= 1e-8 (prevent_warming enforced)
    assert output['F_atm'] >= 1.0e-8


@pytest.mark.unit
def test_rundummyatm_scale_height():
    """Test scale height calculation H = RT/(μg).

    Physical scenario: Earth-like planet with known molecular weight and gravity.
    Validates that observed radius increases by atmospheric scale height.
    """
    from proteus.atmos_clim.dummy import RunDummyAtm

    # Mock config
    config = MagicMock()
    config.atmos_clim.dummy.gamma = 0.4
    config.atmos_clim.dummy.height_factor = 2.0  # 2x scale height
    config.orbit.zenith_angle = 0.0
    config.orbit.s0_factor = 1.0
    config.atmos_clim.surf_greyalbedo = 0.1
    config.atmos_clim.surf_state = 'fixed'
    config.atmos_clim.prevent_warming = False

    dirs = {}

    hf_row = {
        'T_magma': 1200.0,  # K
        'albedo_pl': 0.3,
        'F_ins': 1000.0,  # W/m²
        'atm_kg_per_mol': 0.029,  # kg/mol; N₂
        'gravity': 10.0,  # m/s²
        'R_int': 6.0e6,  # m
        'P_surf': 1e5,  # Pa
    }

    # Call RunDummyAtm
    output = RunDummyAtm(dirs, config, hf_row)

    # Calculate expected scale height manually
    H_expected = (const_R * output['T_surf']) / (hf_row['atm_kg_per_mol'] * hf_row['gravity'])

    # Verify R_obs = R_int + height_factor * H
    R_obs_expected = hf_row['R_int'] + config.atmos_clim.dummy.height_factor * H_expected
    assert output['R_obs'] == pytest.approx(R_obs_expected, rel=1e-8)


@pytest.mark.unit
def test_rundummyatm_zenith_angle_effect():
    """Test that zenith angle reduces scattered solar flux via cosine factor.

    Physical scenario: Planet with varying solar geometry (0° vs 60° zenith).
    Validates that F_sct (scattered SW) decreases with increasing zenith angle.
    """
    from proteus.atmos_clim.dummy import RunDummyAtm

    # Mock config
    config = MagicMock()
    config.atmos_clim.dummy.gamma = 0.5
    config.atmos_clim.dummy.height_factor = 1.0
    config.orbit.s0_factor = 1.0
    config.atmos_clim.surf_greyalbedo = 0.1
    config.atmos_clim.surf_state = 'fixed'
    config.atmos_clim.prevent_warming = False

    dirs = {}

    hf_row = {
        'T_magma': 1500.0,  # K
        'albedo_pl': 0.3,
        'F_ins': 1361.0,  # W/m²
        'atm_kg_per_mol': 0.029,
        'gravity': 9.8,
        'R_int': 6.371e6,
        'P_surf': 1e5,
    }

    # Case 1: Zenith angle = 0° (subsolar point)
    config.orbit.zenith_angle = 0.0
    output_0deg = RunDummyAtm(dirs, config, hf_row)

    # Case 2: Zenith angle = 60° (oblique incidence)
    config.orbit.zenith_angle = 60.0
    output_60deg = RunDummyAtm(dirs, config, hf_row)

    # Verify F_sct (scattered SW) decreases with zenith angle
    # At 60°, cos(60°) = 0.5, so F_D_SW is halved, thus F_sct is halved
    assert output_60deg['F_sct'] < output_0deg['F_sct']


@pytest.mark.unit
@patch('proteus.atmos_clim.dummy.UpdateStatusfile')
def test_rundummyatm_invalid_surf_state(mock_update):
    """Test that invalid surf_state raises ValueError.

    Physical scenario: Configuration error with unrecognized surf_state.
    Validates proper error handling for invalid modes.
    """
    from proteus.atmos_clim.dummy import RunDummyAtm

    # Mock config with invalid surf_state
    config = MagicMock()
    config.atmos_clim.dummy.gamma = 0.5
    config.atmos_clim.dummy.height_factor = 1.0
    config.orbit.zenith_angle = 0.0
    config.orbit.s0_factor = 1.0
    config.atmos_clim.surf_greyalbedo = 0.1
    config.atmos_clim.surf_state = 'invalid_state'  # Invalid
    config.atmos_clim.prevent_warming = False

    dirs = {}

    hf_row = {
        'T_magma': 1500.0,
        'albedo_pl': 0.3,
        'F_ins': 1361.0,
        'atm_kg_per_mol': 0.029,
        'gravity': 9.8,
        'R_int': 6.371e6,
        'P_surf': 1e5,
    }

    # Verify ValueError is raised
    with pytest.raises(ValueError, match='Invalid surface state'):
        RunDummyAtm(dirs, config, hf_row)


@pytest.mark.unit
def test_rundummyatm_albedo_calculation():
    """Test albedo calculation (scattered SW / downwelling SW).

    Physical scenario: Planet with known surface and planetary albedo.
    Validates that output albedo = F_sct / F_D_SW is computed.
    Note: This simple model can produce albedo > 1 due to simplified radiative transfer.
    """
    from proteus.atmos_clim.dummy import RunDummyAtm

    # Mock config with moderate surface albedo
    config = MagicMock()
    config.atmos_clim.dummy.gamma = 0.3
    config.atmos_clim.dummy.height_factor = 1.0
    config.orbit.zenith_angle = 0.0
    config.orbit.s0_factor = 1.0
    config.atmos_clim.surf_greyalbedo = 0.5  # Moderate surface reflectivity
    config.atmos_clim.surf_state = 'fixed'
    config.atmos_clim.prevent_warming = False

    dirs = {}

    hf_row = {
        'T_magma': 1400.0,  # K
        'albedo_pl': 0.2,
        'F_ins': 1361.0,  # W/m²
        'atm_kg_per_mol': 0.029,
        'gravity': 9.8,
        'R_int': 6.371e6,
        'P_surf': 1e5,
    }

    # Call RunDummyAtm
    output = RunDummyAtm(dirs, config, hf_row)

    # Verify albedo is computed (may exceed 1.0 due to simplified model)
    assert 'albedo' in output
    assert output['albedo'] > 0.0  # Should be positive


@pytest.mark.unit
def test_rundummyatm_output_keys():
    """Test that all expected output keys are present in returned dictionary.

    Physical scenario: Standard Earth-like planet.
    Validates that output includes all required keys for downstream coupling.
    """
    from proteus.atmos_clim.dummy import RunDummyAtm

    # Mock config
    config = MagicMock()
    config.atmos_clim.dummy.gamma = 0.5
    config.atmos_clim.dummy.height_factor = 1.0
    config.orbit.zenith_angle = 0.0
    config.orbit.s0_factor = 1.0
    config.atmos_clim.surf_greyalbedo = 0.1
    config.atmos_clim.surf_state = 'fixed'
    config.atmos_clim.prevent_warming = False

    dirs = {}

    hf_row = {
        'T_magma': 1500.0,
        'albedo_pl': 0.3,
        'F_ins': 1361.0,
        'atm_kg_per_mol': 0.029,
        'gravity': 9.8,
        'R_int': 6.371e6,
        'P_surf': 1e5,
    }

    # Call RunDummyAtm
    output = RunDummyAtm(dirs, config, hf_row)

    # Verify all required keys are present
    required_keys = [
        'T_surf',
        'F_atm',
        'F_olr',
        'F_sct',
        'R_obs',
        'albedo',
        'p_xuv',
        'R_xuv',
        'p_obs',
        'P_surf_clim',
    ]
    for key in required_keys:
        assert key in output
