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
    config.planet.mass_tot = 1.0
    config.interior_struct.zalmoxis.core_eos = 'Seager2007:iron'
    config.interior_struct.zalmoxis.mantle_eos = 'Seager2007:silicate'
    config.interior_struct.zalmoxis.ice_layer_eos = 'Seager2007:water'
    config.interior_struct.core_frac = 0.325
    config.interior_struct.zalmoxis.mantle_mass_fraction = 0.0
    config.planet.temperature_mode = 'isothermal'
    config.planet.tsurf_init = 300
    config.planet.tcenter_init = 5000
    config.interior_struct.zalmoxis.num_levels = 200

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
    config.planet.mass_tot = 1.0
    config.interior_struct.zalmoxis.core_eos = 'Seager2007:iron'
    config.interior_struct.zalmoxis.mantle_eos = 'Seager2007:silicate'
    config.interior_struct.zalmoxis.ice_layer_eos = None
    config.interior_struct.core_frac = 0.325
    config.interior_struct.zalmoxis.mantle_mass_fraction = 0.0
    config.planet.temperature_mode = 'isothermal'
    config.planet.tsurf_init = 300
    config.planet.tcenter_init = 5000
    config.interior_struct.zalmoxis.num_levels = 200

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


# ============================================================================
# T1.3: zalmoxis_output.dat schema check at file handover boundary
# ============================================================================


def _write_synthetic_zalmoxis_output(
    path,
    r_cmb=3.48e6,
    r_surf=6.371e6,
    n_layers=80,
    rho_const=4500.0,
):
    """Write a synthetic zalmoxis_output.dat with uniform-density mantle.

    Returns (R_int_top, M_mantle_shellsum) where M_mantle_shellsum is the
    integrated mantle mass matching the file's shell-sum
    (4/3 pi (r2^3 - r1^3) * rho_avg). Uniform density makes the
    shell-sum trivially exact.
    """
    r = np.linspace(r_cmb, r_surf, n_layers)
    rho = np.full_like(r, rho_const)
    P = np.linspace(140e9, 0.0, n_layers)  # placeholder; not validated
    g = np.full_like(r, 9.81)
    T = np.linspace(4000.0, 2000.0, n_layers)
    with open(path, 'w') as f:
        for i in range(n_layers):
            f.write(
                f'{r[i]:.17e} {P[i]:.17e} {rho[i]:.17e} {g[i]:.17e} {T[i]:.17e}\n'
            )
    shells = (4.0 / 3.0) * np.pi * (
        r[1:] ** 3 - r[:-1] ** 3
    ) * 0.5 * (rho[1:] + rho[:-1])
    return float(r[-1]), float(np.sum(shells))


@pytest.mark.unit
def test_validate_zalmoxis_output_schema_consistent(tmp_path):
    """A correctly-written file with matching hf_row scalars must NOT raise."""
    from proteus.interior_struct.zalmoxis import validate_zalmoxis_output_schema

    output_path = str(tmp_path / 'zalmoxis_output.dat')
    R_int, M_mantle = _write_synthetic_zalmoxis_output(output_path)
    M_core = 1.94e24
    M_int = M_mantle + M_core
    hf_row = {'R_int': R_int, 'M_int': M_int, 'M_core': M_core}

    validate_zalmoxis_output_schema(output_path, hf_row)


@pytest.mark.unit
def test_validate_zalmoxis_output_schema_radius_mismatch(tmp_path):
    """File top-r off by >1e-6 from hf_row['R_int'] -> raise."""
    from proteus.interior_struct.zalmoxis import validate_zalmoxis_output_schema

    output_path = str(tmp_path / 'zalmoxis_output.dat')
    R_int, M_mantle = _write_synthetic_zalmoxis_output(output_path)
    M_core = 1.94e24
    M_int = M_mantle + M_core
    # 1e-3 relative error in claimed R_int -> raise.
    hf_row_over = {'R_int': R_int * 1.001, 'M_int': M_int, 'M_core': M_core}
    with pytest.raises(RuntimeError, match='top-of-mantle'):
        validate_zalmoxis_output_schema(output_path, hf_row_over)

    # Edge: 5e-7 drift (sub-tolerance) must NOT raise.
    hf_row_under = {'R_int': R_int * (1 + 5e-7), 'M_int': M_int, 'M_core': M_core}
    validate_zalmoxis_output_schema(output_path, hf_row_under)


@pytest.mark.unit
def test_validate_zalmoxis_output_schema_mass_mismatch(tmp_path):
    """File mantle mass off by >rtol_mass (5e-2 default) raises.

    Default rtol_mass=5e-2 reflects two stacked legitimate-noise
    sources: (a) integrator-method difference between Zalmoxis' RK45
    ODE-state mass and grid-trapezoidal shell-sum (~0.8-2 %), and
    (b) blend_mesh_files post-write modification of the file (up to
    ~5 % drift from unblended hf_row, see T2.5). Genuine corruption
    (column swap, truncation) shows up at >>5 %.

    Edge cases covered:
    - 10 % mismatch (clear corruption signature) raises.
    - 3 % mismatch (within blend-induced noise floor) passes.
    - 10 % mismatch with explicit tighter tol (1e-3) raises.
    """
    from proteus.interior_struct.zalmoxis import validate_zalmoxis_output_schema

    output_path = str(tmp_path / 'zalmoxis_output.dat')
    R_int, M_mantle = _write_synthetic_zalmoxis_output(output_path)
    M_core = 1.94e24

    # 10% inflation -> clearly above 5e-2 default -> raise.
    M_int_10pct = (M_mantle * 1.10) + M_core
    hf_row_10pct = {'R_int': R_int, 'M_int': M_int_10pct, 'M_core': M_core}
    with pytest.raises(RuntimeError, match='mantle mass'):
        validate_zalmoxis_output_schema(output_path, hf_row_10pct)

    # 3% mismatch -> below 5e-2 noise floor -> pass.
    M_int_3pct = (M_mantle * 1.03) + M_core
    hf_row_3pct = {'R_int': R_int, 'M_int': M_int_3pct, 'M_core': M_core}
    validate_zalmoxis_output_schema(output_path, hf_row_3pct)

    # Caller can pass a tighter tolerance and recover the strict check.
    with pytest.raises(RuntimeError, match='mantle mass'):
        validate_zalmoxis_output_schema(
            output_path, hf_row_3pct, rtol_mass=1e-3
        )


@pytest.mark.unit
def test_validate_zalmoxis_output_schema_corrupt_file(tmp_path):
    """File missing / wrong-shape -> raise.

    Edge cases for genuine I/O corruption that the schema check
    must catch before Aragog ingests garbage.
    """
    from proteus.interior_struct.zalmoxis import validate_zalmoxis_output_schema

    hf_row = {'R_int': 6.371e6, 'M_int': 5.972e24, 'M_core': 1.94e24}

    # Case 1: file does not exist.
    output_missing = str(tmp_path / 'missing.dat')
    with pytest.raises(RuntimeError, match='could not reload'):
        validate_zalmoxis_output_schema(output_missing, hf_row)

    # Case 2: wrong number of columns (3 instead of 5).
    output_3col = str(tmp_path / 'wrong_cols.dat')
    with open(output_3col, 'w') as f:
        f.write('1.0 2.0 3.0\n4.0 5.0 6.0\n')
    with pytest.raises(RuntimeError, match='unexpected shape'):
        validate_zalmoxis_output_schema(output_3col, hf_row)


@pytest.mark.unit
def test_validate_zalmoxis_output_schema_skips_when_hf_row_unset(tmp_path):
    """Degenerate hf_row inputs (zero scalars) must skip silently.

    Matches the wrapper-level mass-anchor guard's degenerate-input
    contract: callers without populated R_int / mass scalars
    (e.g. very-early init paths) must not see spurious schema
    raises. The file-shape check still runs.
    """
    from proteus.interior_struct.zalmoxis import validate_zalmoxis_output_schema

    output_path = str(tmp_path / 'zalmoxis_output.dat')
    R_int_top, _ = _write_synthetic_zalmoxis_output(output_path)

    # All scalars zero: both checks skipped, file-shape passes.
    hf_row_empty = {'R_int': 0.0, 'M_int': 0.0, 'M_core': 0.0}
    validate_zalmoxis_output_schema(output_path, hf_row_empty)

    # Only mass info missing: radius check still runs against R_int_top.
    hf_row_no_mass = {'R_int': R_int_top, 'M_int': 0.0, 'M_core': 0.0}
    validate_zalmoxis_output_schema(output_path, hf_row_no_mass)
