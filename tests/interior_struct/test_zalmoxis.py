"""
Unit tests for proteus.interior_struct.zalmoxis module.

Validates SPIDER mesh file generation from Zalmoxis mantle profiles,
Zalmoxis configuration building, and solidus/liquidus loading.

Testing standards and documentation:
- docs/test_infrastructure.md: Test infrastructure overview
- docs/test_categorization.md: Test marker definitions
- docs/test_building.md: Best practices for test construction

Functions tested:
- write_spider_mesh_file(): Interpolate Zalmoxis profiles onto SPIDER mesh
- load_zalmoxis_configuration(): Build config dict from PROTEUS config
- load_zalmoxis_solidus_liquidus_functions(): Load melting curves by EOS type
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from proteus.interior_struct.zalmoxis import write_spider_mesh_file


def _make_synthetic_mantle_profiles(
    r_cmb: float = 3.48e6,
    r_surf: float = 6.37e6,
    n_zalmoxis: int = 200,
):
    """Create synthetic Zalmoxis-like mantle profiles (CMB to surface, ascending r).

    Parameters
    ----------
    r_cmb : float
        Core-mantle boundary radius [m]. Default ~Earth CMB.
    r_surf : float
        Surface radius [m]. Default ~Earth surface.
    n_zalmoxis : int
        Number of radial points in the Zalmoxis profile.

    Returns
    -------
    radii : np.ndarray
        Ascending radii from CMB to surface [m].
    pressure : np.ndarray
        Pressure decreasing from ~135 GPa at CMB to ~0 at surface [Pa].
    density : np.ndarray
        Density decreasing from ~5500 at CMB to ~3300 at surface [kg/m^3].
    gravity : np.ndarray
        Gravity magnitude, roughly 10 m/s^2 throughout (positive) [m/s^2].
    """
    radii = np.linspace(r_cmb, r_surf, n_zalmoxis)

    # Pressure: hydrostatic-like decrease from CMB to surface
    frac = (radii - r_cmb) / (r_surf - r_cmb)
    pressure = 135e9 * (1.0 - frac)  # Pa, ~135 GPa at CMB to 0 at surface

    # Density: linear decrease from ~5500 to ~3300 kg/m^3
    density = 5500.0 - 2200.0 * frac

    # Gravity: roughly constant ~10 m/s^2 with slight decrease toward surface
    gravity = 10.5 - 1.0 * frac  # positive values

    return radii, pressure, density, gravity


@pytest.mark.unit
def test_write_spider_mesh_file(tmp_path):
    """Verify SPIDER mesh file from synthetic Earth-like mantle profiles.

    Creates a Zalmoxis-like mantle column (CMB at 3.48e6 m, surface at
    6.37e6 m) with physically plausible pressure, density, and gravity
    profiles. Writes a SPIDER mesh with num_basic=50 and checks:

    - File exists at the expected path (outdir/data/spider_mesh.dat)
    - Header format matches SPIDER convention: ``# <num_basic> <num_staggered>``
    - Total line count = 1 (header) + 50 (basic) + 49 (staggered) = 100
    - Basic nodes ordered surface-first (largest r) to CMB-last (smallest r)
    - Staggered nodes follow the same descending-r ordering
    - Gravity is negative throughout (SPIDER inward-pointing convention)
    - Pressure and density are positive throughout
    - Each staggered node radius lies between its bounding basic node radii
    """
    # Setup: create data/ subdirectory inside tmp_path
    outdir = str(tmp_path)
    (tmp_path / 'data').mkdir()

    num_basic = 50
    num_staggered = num_basic - 1

    radii, pressure, density, gravity = _make_synthetic_mantle_profiles()

    # Call the function under test
    mesh_path = write_spider_mesh_file(
        outdir=outdir,
        mantle_radii=radii,
        mantle_pressure=pressure,
        mantle_density=density,
        mantle_gravity=gravity,
        num_basic=num_basic,
    )

    # --- File existence ---
    assert mesh_path == str(tmp_path / 'data' / 'spider_mesh.dat')
    assert (tmp_path / 'data' / 'spider_mesh.dat').exists()

    # --- Parse file contents ---
    with open(mesh_path) as f:
        lines = f.readlines()

    # --- Total line count: header + basic + staggered ---
    expected_total_lines = 1 + num_basic + num_staggered
    assert len(lines) == expected_total_lines, (
        f'Expected {expected_total_lines} lines, got {len(lines)}'
    )

    # --- Header format ---
    header = lines[0].strip()
    assert header == f'# {num_basic} {num_staggered}'

    # --- Parse data lines ---
    basic_data = np.array([list(map(float, line.split())) for line in lines[1 : 1 + num_basic]])
    staggered_data = np.array(
        [list(map(float, line.split())) for line in lines[1 + num_basic :]]
    )

    assert basic_data.shape == (num_basic, 4)
    assert staggered_data.shape == (num_staggered, 4)

    r_basic = basic_data[:, 0]
    p_basic = basic_data[:, 1]
    rho_basic = basic_data[:, 2]
    g_basic = basic_data[:, 3]

    r_stag = staggered_data[:, 0]
    p_stag = staggered_data[:, 1]
    rho_stag = staggered_data[:, 2]
    g_stag = staggered_data[:, 3]

    # --- Basic nodes: surface first (largest r) to CMB last (smallest r) ---
    assert r_basic[0] > r_basic[-1], 'Basic nodes should descend from surface to CMB'
    # Verify strictly descending
    assert np.all(np.diff(r_basic) < 0), 'Basic node radii must be strictly descending'

    # --- Staggered nodes: same descending-r ordering ---
    assert r_stag[0] > r_stag[-1], 'Staggered nodes should descend from surface to CMB'
    assert np.all(np.diff(r_stag) < 0), 'Staggered node radii must be strictly descending'

    # --- Gravity is negative throughout (SPIDER convention) ---
    assert np.all(g_basic < 0), 'Basic node gravity must be negative (SPIDER convention)'
    assert np.all(g_stag < 0), 'Staggered node gravity must be negative (SPIDER convention)'

    # --- Pressure and density are positive throughout ---
    assert np.all(p_basic >= 0), 'Basic node pressure must be non-negative'
    assert np.all(p_stag >= 0), 'Staggered node pressure must be non-negative'
    assert np.all(rho_basic > 0), 'Basic node density must be positive'
    assert np.all(rho_stag > 0), 'Staggered node density must be positive'

    # --- Staggered radii lie between consecutive basic radii ---
    # Since both are descending: r_basic[i] > r_stag[i] > r_basic[i+1]
    for i in range(num_staggered):
        assert r_basic[i] > r_stag[i] > r_basic[i + 1], (
            f'Staggered node {i} radius {r_stag[i]:.6e} not between '
            f'basic nodes {r_basic[i]:.6e} and {r_basic[i + 1]:.6e}'
        )

    # --- Staggered radii are midpoints of basic radii ---
    expected_midpoints = 0.5 * (r_basic[:-1] + r_basic[1:])
    np.testing.assert_allclose(
        r_stag,
        expected_midpoints,
        rtol=1e-12,
        err_msg='Staggered radii should be midpoints of basic radii',
    )

    # --- Boundary radii match input profile ---
    r_surf = radii[-1]
    r_cmb = radii[0]
    assert r_basic[0] == pytest.approx(r_surf, rel=1e-10)
    assert r_basic[-1] == pytest.approx(r_cmb, rel=1e-10)


# ============================================================================
# test load_zalmoxis_configuration
# ============================================================================


@pytest.mark.unit
def test_zalmoxis_config_with_ice_layer():
    """Config dict includes ice_layer when ice_layer_eos is set."""
    from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration

    config = MagicMock()
    config.planet.planet_mass_tot = 1.0
    config.interior_struct.zalmoxis.core_eos = 'Seager2007:iron'
    config.interior_struct.zalmoxis.mantle_eos = 'Seager2007:silicate'
    config.interior_struct.zalmoxis.ice_layer_eos = 'Seager2007:water'
    config.interior_struct.core_frac = 0.325
    config.interior_struct.zalmoxis.mantle_mass_fraction = 0.0
    config.interior_struct.zalmoxis.temperature_mode = 'isothermal'
    config.interior_struct.zalmoxis.surface_temperature = 300
    config.interior_struct.zalmoxis.center_temperature = 5000
    config.interior_struct.zalmoxis.temperature_profile_file = None
    config.interior_struct.zalmoxis.num_levels = 200
    config.interior_struct.zalmoxis.max_iterations_outer = 50
    config.interior_struct.zalmoxis.tolerance_outer = 1e-3
    config.interior_struct.zalmoxis.max_iterations_inner = 100
    config.interior_struct.zalmoxis.tolerance_inner = 1e-6
    config.interior_struct.zalmoxis.relative_tolerance = 1e-8
    config.interior_struct.zalmoxis.absolute_tolerance = 1e-12
    config.interior_struct.zalmoxis.maximum_step = 100.0
    config.interior_struct.zalmoxis.adaptive_radial_fraction = 0.01
    config.interior_struct.zalmoxis.max_center_pressure_guess = 1e14
    config.interior_struct.zalmoxis.target_surface_pressure = 1e5
    config.interior_struct.zalmoxis.pressure_tolerance = 0.01
    config.interior_struct.zalmoxis.max_iterations_pressure = 20
    config.interior_struct.zalmoxis.verbose = False
    config.interior_struct.zalmoxis.iteration_profiles_enabled = False

    hf_row = {
        f'{e}_kg_total': 0 for e in ('H', 'O', 'C', 'N', 'S', 'Si', 'Mg', 'Fe', 'Na', 'He')
    }

    result = load_zalmoxis_configuration(config, hf_row)
    assert 'ice_layer' in result['layer_eos_config']
    assert result['layer_eos_config']['ice_layer'] == 'Seager2007:water'


@pytest.mark.unit
def test_zalmoxis_config_no_ice_layer():
    """Config dict omits ice_layer when ice_layer_eos is empty."""
    from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration

    config = MagicMock()
    config.planet.planet_mass_tot = 1.0
    config.interior_struct.zalmoxis.core_eos = 'Seager2007:iron'
    config.interior_struct.zalmoxis.mantle_eos = 'Seager2007:silicate'
    config.interior_struct.zalmoxis.ice_layer_eos = ''
    config.interior_struct.core_frac = 0.325
    config.interior_struct.zalmoxis.mantle_mass_fraction = 0.0
    config.interior_struct.zalmoxis.temperature_mode = 'isothermal'
    config.interior_struct.zalmoxis.surface_temperature = 300
    config.interior_struct.zalmoxis.center_temperature = 5000
    config.interior_struct.zalmoxis.temperature_profile_file = None
    config.interior_struct.zalmoxis.num_levels = 200
    config.interior_struct.zalmoxis.max_iterations_outer = 50
    config.interior_struct.zalmoxis.tolerance_outer = 1e-3
    config.interior_struct.zalmoxis.max_iterations_inner = 100
    config.interior_struct.zalmoxis.tolerance_inner = 1e-6
    config.interior_struct.zalmoxis.relative_tolerance = 1e-8
    config.interior_struct.zalmoxis.absolute_tolerance = 1e-12
    config.interior_struct.zalmoxis.maximum_step = 100.0
    config.interior_struct.zalmoxis.adaptive_radial_fraction = 0.01
    config.interior_struct.zalmoxis.max_center_pressure_guess = 1e14
    config.interior_struct.zalmoxis.target_surface_pressure = 1e5
    config.interior_struct.zalmoxis.pressure_tolerance = 0.01
    config.interior_struct.zalmoxis.max_iterations_pressure = 20
    config.interior_struct.zalmoxis.verbose = False
    config.interior_struct.zalmoxis.iteration_profiles_enabled = False

    hf_row = {
        f'{e}_kg_total': 0 for e in ('H', 'O', 'C', 'N', 'S', 'Si', 'Mg', 'Fe', 'Na', 'He')
    }

    result = load_zalmoxis_configuration(config, hf_row)
    assert 'ice_layer' not in result['layer_eos_config']


# ============================================================================
# test load_zalmoxis_solidus_liquidus_functions
# ============================================================================


@pytest.mark.unit
def test_solidus_liquidus_non_tdep():
    """Non-T-dependent EOS (Seager2007) returns None."""

    from proteus.interior_struct.zalmoxis import load_zalmoxis_solidus_liquidus_functions

    result = load_zalmoxis_solidus_liquidus_functions('Seager2007:silicate', MagicMock())
    assert result is None


@pytest.mark.unit
def test_solidus_liquidus_rtpress():
    """RTPress100TPa prefix triggers melting curve loading."""
    from unittest.mock import patch

    from proteus.interior_struct.zalmoxis import load_zalmoxis_solidus_liquidus_functions

    with patch(
        'proteus.interior_struct.zalmoxis.get_zalmoxis_melting_curves',
        return_value=('solidus_fn', 'liquidus_fn'),
    ) as mock_mc:
        result = load_zalmoxis_solidus_liquidus_functions('RTPress100TPa_MgSiO3', MagicMock())

    assert result == ('solidus_fn', 'liquidus_fn')
    mock_mc.assert_called_once()
