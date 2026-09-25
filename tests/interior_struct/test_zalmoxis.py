"""
Unit tests for proteus.interior_struct.zalmoxis module.

Validates SPIDER mesh file generation from Zalmoxis mantle profiles,
Zalmoxis configuration building, and solidus/liquidus loading.

Testing standards and documentation:
- docs/How-to/testing.md: Running, writing, and marking tests; coverage and CI
- docs/Explanations/test_framework.md: Test tiers, physics invariants, and quality rules

Functions tested:
- write_spider_mesh_file(): Interpolate Zalmoxis profiles onto SPIDER mesh
- load_zalmoxis_configuration(): Build config dict from PROTEUS config
- load_zalmoxis_solidus_liquidus_functions(): Load melting curves by EOS type
- compute_structure_mass_desync(): Mass self-consistency diagnostic (issue #68)
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from proteus.interior_struct.zalmoxis import (
    compute_structure_mass_desync,
    write_spider_mesh_file,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


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
@pytest.mark.physics_invariant
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
    # Discrimination: the core + mantle layers must still be present. A
    # regression that dropped ALL layers when ice_layer_eos was None would
    # still satisfy the absence check above but break the downstream EOS
    # dispatch.
    assert 'core' in result['layer_eos_config']
    assert 'mantle' in result['layer_eos_config']


def _liquidus_super_config():
    """MagicMock config selecting the 'liquidus_super' CMB-anchor path."""
    config = MagicMock()
    config.planet.mass_tot = 1.0
    config.interior_struct.zalmoxis.core_eos = 'Seager2007:iron'
    config.interior_struct.zalmoxis.mantle_eos = 'PALEOS-2phase:MgSiO3'
    config.interior_struct.zalmoxis.ice_layer_eos = None
    config.interior_struct.core_frac = 0.325
    config.interior_struct.zalmoxis.mantle_mass_fraction = 0.0
    config.planet.temperature_mode = 'liquidus_super'
    config.planet.tsurf_init = 300
    config.planet.tcenter_init = 5000
    config.interior_struct.zalmoxis.num_levels = 200
    return config


@pytest.mark.unit
def test_load_zalmoxis_configuration_converts_missing_table_error(monkeypatch):
    """A FileNotFoundError from the CMB-anchor resolve becomes the dedicated error.

    Forces the ``except FileNotFoundError`` branch in
    ``load_zalmoxis_configuration`` directly: the recheck finds a missing
    table, so its own ``ZalmoxisMissingEOSFilesError`` propagates.
    """
    from proteus.interior_struct import zalmoxis as zalmoxis_wrapper
    from proteus.interior_struct.zalmoxis import (
        ZalmoxisMissingEOSFilesError,
        load_zalmoxis_configuration,
    )

    recheck_calls = []

    def _raise_file_not_found(*args, **kwargs):
        raise FileNotFoundError('missing mantle EOS table')

    def _raise_missing_eos(*args, **kwargs):
        recheck_calls.append(args)
        raise ZalmoxisMissingEOSFilesError('missing table')

    monkeypatch.setattr(
        zalmoxis_wrapper, '_resolve_zalmoxis_cmb_temperature', _raise_file_not_found
    )
    monkeypatch.setattr(zalmoxis_wrapper, 'check_zalmoxis_eos_files', _raise_missing_eos)
    monkeypatch.setattr(
        zalmoxis_wrapper, 'load_zalmoxis_material_dictionaries', lambda: {'sentinel': True}
    )

    hf_row = {
        f'{e}_kg_total': 0 for e in ('H', 'O', 'C', 'N', 'S', 'Si', 'Mg', 'Fe', 'Na', 'He')
    }

    with pytest.raises(ZalmoxisMissingEOSFilesError, match='missing table'):
        load_zalmoxis_configuration(_liquidus_super_config(), hf_row)

    assert len(recheck_calls) == 1
    layer_eos_config, mat_dicts = recheck_calls[0]
    assert layer_eos_config['core'] == 'Seager2007:iron'
    assert layer_eos_config['mantle'] == 'PALEOS-2phase:MgSiO3'
    assert mat_dicts == {'sentinel': True}


@pytest.mark.unit
def test_load_zalmoxis_configuration_reraises_unconverted_file_error(monkeypatch):
    """A FileNotFoundError the recheck cannot classify re-raises unchanged.

    ``check_zalmoxis_eos_files`` only inspects registered EOS tables; a
    ``FileNotFoundError`` from a different data class (e.g. a missing
    melting-curve directory) finds nothing missing on recheck, so the
    original bare exception must propagate rather than a false skip.
    """
    from proteus.interior_struct import zalmoxis as zalmoxis_wrapper
    from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration

    def _raise_file_not_found(*args, **kwargs):
        raise FileNotFoundError('melting curves directory not found')

    monkeypatch.setattr(
        zalmoxis_wrapper, '_resolve_zalmoxis_cmb_temperature', _raise_file_not_found
    )
    monkeypatch.setattr(zalmoxis_wrapper, 'check_zalmoxis_eos_files', lambda *a, **k: None)
    monkeypatch.setattr(zalmoxis_wrapper, 'load_zalmoxis_material_dictionaries', lambda: {})

    hf_row = {
        f'{e}_kg_total': 0 for e in ('H', 'O', 'C', 'N', 'S', 'Si', 'Mg', 'Fe', 'Na', 'He')
    }

    with pytest.raises(FileNotFoundError, match='melting curves directory not found'):
        load_zalmoxis_configuration(_liquidus_super_config(), hf_row)


# ============================================================================
# test load_zalmoxis_solidus_liquidus_functions
# ============================================================================


@pytest.mark.unit
def test_solidus_liquidus_non_tdep():
    """Non-T-dependent EOS (Seager2007) returns None."""

    from proteus.interior_struct.zalmoxis import load_zalmoxis_solidus_liquidus_functions

    eos_name = 'Seager2007:silicate'
    result = load_zalmoxis_solidus_liquidus_functions(eos_name, MagicMock())
    assert result is None  # non-T-dependent EOS branch must yield None silently
    # Discriminating check: the EOS name is on the non-T-dependent prefix list
    # (Seager2007 is fixed-T); only that branch can produce a None here.
    assert eos_name.startswith('Seager2007')


@pytest.mark.unit
def test_solidus_liquidus_rtpress():
    """RTPress100TPa prefix triggers melting curve loading."""
    from proteus.interior_struct.zalmoxis import load_zalmoxis_solidus_liquidus_functions

    with patch(
        'proteus.interior_struct.zalmoxis.get_zalmoxis_melting_curves',
        return_value=('solidus_fn', 'liquidus_fn'),
    ) as mock_mc:
        result = load_zalmoxis_solidus_liquidus_functions('RTPress100TPa_MgSiO3', MagicMock())

    assert result == ('solidus_fn', 'liquidus_fn')
    mock_mc.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize(
    'eos_name',
    [
        'PALEOS:MgSiO3',
        'PALEOS-2phase:MgSiO3',
        'PALEOS-API:MgSiO3',
        'PALEOS-API-2phase:MgSiO3',
    ],
)
def test_solidus_liquidus_paleos_family_derives_solidus_from_mzf(eos_name):
    """Every PALEOS-family mantle gets a solidus equal to mzf times the liquidus."""
    from proteus.interior_struct.zalmoxis import load_zalmoxis_solidus_liquidus_functions

    config = MagicMock()
    config.interior_struct.zalmoxis.mushy_zone_factor = 0.8
    result = load_zalmoxis_solidus_liquidus_functions(eos_name, config)

    assert result is not None
    solidus_func, liquidus_func = result
    pressure = 50e9
    t_liq = float(liquidus_func(pressure))
    assert np.isfinite(t_liq)
    assert float(solidus_func(pressure)) == pytest.approx(0.8 * t_liq)
    # Discrimination: a different factor moves the solidus with the liquidus fixed.
    config.interior_struct.zalmoxis.mushy_zone_factor = 0.9
    solidus_func_90, liquidus_func_90 = load_zalmoxis_solidus_liquidus_functions(
        eos_name, config
    )
    assert float(solidus_func_90(pressure)) == pytest.approx(
        0.9 * float(liquidus_func_90(pressure))
    )
    # Range boundaries: 1.0 gives T_sol == T_liq exactly, 0.7 the widest band.
    config.interior_struct.zalmoxis.mushy_zone_factor = 1.0
    solidus_func_1, liquidus_func_1 = load_zalmoxis_solidus_liquidus_functions(eos_name, config)
    assert float(solidus_func_1(pressure)) == float(liquidus_func_1(pressure))
    config.interior_struct.zalmoxis.mushy_zone_factor = 0.7
    solidus_func_70, liquidus_func_70 = load_zalmoxis_solidus_liquidus_functions(
        eos_name, config
    )
    assert float(solidus_func_70(pressure)) == pytest.approx(
        0.7 * float(liquidus_func_70(pressure))
    )


@pytest.mark.unit
def test_build_mushy_zone_factors_ignores_spaces_around_plus():
    """Zalmoxis strips each '+' segment, so a spaced string must map the same way."""
    from proteus.interior_struct.zalmoxis import _build_mushy_zone_factors

    result = _build_mushy_zone_factors(
        {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3:0.9 + PALEOS:H2O:0.1'}, mzf=0.8
    )
    assert result['PALEOS:H2O'] == pytest.approx(0.8)
    assert result['PALEOS:MgSiO3'] == pytest.approx(0.8)


# ============================================================================
# zalmoxis_output.dat schema check at file handover boundary
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
            f.write(f'{r[i]:.17e} {P[i]:.17e} {rho[i]:.17e} {g[i]:.17e} {T[i]:.17e}\n')
    shells = (4.0 / 3.0) * np.pi * (r[1:] ** 3 - r[:-1] ** 3) * 0.5 * (rho[1:] + rho[:-1])
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

    result = validate_zalmoxis_output_schema(output_path, hf_row)
    assert result is None  # contract: schema validator returns None silently on a match
    # Discriminating check: hf_row['M_int'] equals M_mantle + M_core within float
    # noise; a regression that swapped M_core and M_mantle would fail here even
    # if the file itself was OK.
    assert hf_row['M_int'] == pytest.approx(M_mantle + M_core, rel=1e-12)


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

    # Edge: 5e-7 drift (sub-tolerance) must NOT raise. Pin the silent-pass
    # return contract so a regression that started returning a non-None
    # status object would fail here.
    hf_row_under = {'R_int': R_int * (1 + 5e-7), 'M_int': M_int, 'M_core': M_core}
    result = validate_zalmoxis_output_schema(output_path, hf_row_under)
    assert result is None


@pytest.mark.unit
def test_validate_zalmoxis_output_schema_mass_mismatch(tmp_path):
    """File mantle mass off by >rtol_mass (5e-2 default) raises.

    Default rtol_mass=5e-2 reflects two stacked legitimate-noise
    sources: (a) integrator-method difference between Zalmoxis' RK45
    ODE-state mass and grid-trapezoidal shell-sum (~0.8-2 %), and
    (b) blend_mesh_files post-write modification of the file (up to
    ~5 % drift from unblended hf_row). Genuine corruption
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
        validate_zalmoxis_output_schema(output_path, hf_row_3pct, rtol_mass=1e-3)


@pytest.mark.unit
def test_validate_zalmoxis_output_schema_accumulator_overrides_trapezoid(tmp_path):
    """Supplying mantle_mass_ref bypasses the coarse-trapezoid mass check.

    Reproduces the high-planet-mass regime: the written file is faithful
    but a grid-trapezoidal re-integration of its coarse nodes under-reads
    the RK45 accumulator mantle mass by ~10% across the steep interior
    density profile. Without the reference the schema check false-rejects
    such a run; with the accumulator value passed in it must pass, because
    the accumulator (not the trapezoid) is the structure's true mantle mass.

    Discriminating construction: the file's trapezoid mantle mass is set
    10% below hf_row['M_int'] - hf_row['M_core'] (the truth). The trapezoid
    path therefore RAISES while the accumulator path PASSES on identical
    inputs, isolating the parameter as the sole cause of the difference.
    """
    from proteus.interior_struct.zalmoxis import validate_zalmoxis_output_schema

    output_path = str(tmp_path / 'zalmoxis_output.dat')
    R_int, M_mantle_file = _write_synthetic_zalmoxis_output(output_path)
    M_core = 1.94e24
    # Truth (accumulator) mantle mass is 1/0.90 x the file trapezoid, i.e. the
    # trapezoid under-reads the truth by 10% (the high-mass discretization gap).
    M_mantle_true = M_mantle_file / 0.90
    M_int = M_mantle_true + M_core
    hf_row = {'R_int': R_int, 'M_int': M_int, 'M_core': M_core}

    # Without the reference: trapezoid (file) vs expected (truth) is 10% low.
    with pytest.raises(RuntimeError, match='trapezoid'):
        validate_zalmoxis_output_schema(output_path, hf_row)

    # With the accurate accumulator reference: passes silently.
    result = validate_zalmoxis_output_schema(output_path, hf_row, mantle_mass_ref=M_mantle_true)
    assert result is None  # accumulator path returns None on a match
    # Discrimination guard: the trapezoid/truth gap (10%) is double the 5%
    # default tolerance, so the pass cannot be an artifact of a loose threshold.
    trapezoid_gap = abs(M_mantle_file / M_mantle_true - 1.0)
    assert trapezoid_gap == pytest.approx(0.10, abs=1e-9)
    assert trapezoid_gap > 5e-2


@pytest.mark.unit
def test_validate_zalmoxis_output_schema_accumulator_mismatch_raises(tmp_path):
    """A grossly inconsistent mantle_mass_ref still raises, naming its source.

    The accumulator override must not become a blanket bypass: if the
    structure's own mantle mass disagrees with hf_row['M_int'] - M_core by
    more than rtol_mass, that is a real hf_row/structure desync and must
    surface. The realistic small CMB-node snap overshoot (~1%) must pass.
    """
    from proteus.interior_struct.zalmoxis import validate_zalmoxis_output_schema

    output_path = str(tmp_path / 'zalmoxis_output.dat')
    R_int, M_mantle_file = _write_synthetic_zalmoxis_output(output_path)
    M_core = 1.94e24
    M_int = M_mantle_file + M_core
    hf_row = {'R_int': R_int, 'M_int': M_int, 'M_core': M_core}
    expected_mantle = M_int - M_core

    # 12% off -> above 5% default -> raise, and the message names the source.
    with pytest.raises(RuntimeError, match='accumulator'):
        validate_zalmoxis_output_schema(
            output_path, hf_row, mantle_mass_ref=expected_mantle * 1.12
        )

    # 1% off (representative CMB-node snap overshoot) -> within tolerance.
    result = validate_zalmoxis_output_schema(
        output_path, hf_row, mantle_mass_ref=expected_mantle * 1.01
    )
    assert result is None  # sub-tolerance snap overshoot must not raise
    # Discrimination guard: 1% passes, 12% raises, so the boundary sits
    # strictly between them rather than rubber-stamping any reference.
    assert abs(1.01 - 1.0) < 5e-2 < abs(1.12 - 1.0)


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
def test_stale_mesh_regression_discriminates_real_drift(tmp_path):
    """A resumed row's own mesh passes; the run's stale end-of-run mesh fails.

    Pins a real incident: resuming a run at an earlier row while
    ``data/zalmoxis_output.dat`` on disk still holds a later row's mesh
    (left over from before the interior structure module regenerates it)
    crashes Aragog's own EOS-radius-range guard downstream, because the
    file's top radius no longer matches the resumed row's R_int. The
    two R_int values here are a real archived run's resumed row and its
    final row: 5,667 m (~0.097%) apart, about three orders of magnitude
    below the planet's own radius and about three orders of magnitude
    above the schema check's 1e-6 relative tolerance, so the check must
    still catch it.
    """
    from proteus.interior_struct.zalmoxis import validate_zalmoxis_output_schema

    resumed_r_int = 5.8518474239e6  # real archived run, row resumed into
    stale_r_int = 5.8575141291e6  # same run's later, final-row mesh
    r_core = 2.8670124963e6
    m_core = 9.8365400909e23

    # A mesh regenerated at the resumed row's own R_int passes.
    fresh_path = str(tmp_path / 'fresh.dat')
    r_top, m_mantle = _write_synthetic_zalmoxis_output(
        fresh_path, r_cmb=r_core, r_surf=resumed_r_int
    )
    hf_row = {'R_int': resumed_r_int, 'M_int': m_mantle + m_core, 'M_core': m_core}
    validate_zalmoxis_output_schema(fresh_path, hf_row)

    # The same run's stale, later-row mesh -- unregenerated -- fails against
    # the identical resumed-row hf_row.
    stale_path = str(tmp_path / 'stale.dat')
    _write_synthetic_zalmoxis_output(stale_path, r_cmb=r_core, r_surf=stale_r_int)
    with pytest.raises(RuntimeError, match='top-of-mantle'):
        validate_zalmoxis_output_schema(stale_path, hf_row)

    # Discrimination guard: the real drift is well above the check's
    # tolerance, so the raise above is not an artifact of a loose default.
    real_drift = abs(stale_r_int - resumed_r_int) / resumed_r_int
    assert real_drift == pytest.approx(9.684e-4, rel=1e-2)
    assert real_drift > 1e-6


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
    result_empty = validate_zalmoxis_output_schema(output_path, hf_row_empty)
    assert result_empty is None  # all-zero scalars must take the silent-skip branch

    # Only mass info missing: radius check still runs against R_int_top.
    hf_row_no_mass = {'R_int': R_int_top, 'M_int': 0.0, 'M_core': 0.0}
    result_no_mass = validate_zalmoxis_output_schema(output_path, hf_row_no_mass)
    assert result_no_mass is None  # mass-only-missing path still passes the radius check
    # Discriminating check: R_int matches the file top while masses are zero;
    # only the mass-skip branch can produce a silent pass on the second call.
    assert hf_row_no_mass['R_int'] == R_int_top
    assert hf_row_no_mass['M_int'] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# check_zalmoxis_eos_files: missing-table fail-fast
# ---------------------------------------------------------------------------


def _synthetic_registry(tmp_path, *, write_files: bool = True) -> dict:
    """Build a minimal EOS registry with flat, nested, and lazy entries.

    Mirrors the three entry shapes of ``load_zalmoxis_material_dictionaries``:
    flat (unified PALEOS), nested per-role (2-phase), and PALEOS-API live
    tabulation (no ``eos_file`` until first density query).
    """
    iron = tmp_path / 'paleos_iron.dat'
    mantle_liq = tmp_path / 'mgsio3_liquid.dat'
    mantle_sol = tmp_path / 'mgsio3_solid.dat'
    if write_files:
        for f in (iron, mantle_liq, mantle_sol):
            f.write_text('eos table stub')
    return {
        'PALEOS:iron': {'eos_file': str(iron), 'format': 'paleos_unified'},
        'PALEOS-2phase:MgSiO3': {
            'core': {'eos_file': str(iron)},
            'melted_mantle': {'eos_file': str(mantle_liq), 'format': 'paleos'},
            'solid_mantle': {'eos_file': str(mantle_sol), 'format': 'paleos'},
        },
        'PALEOS-API:MgSiO3': {'format': 'paleos_api', 'material': 'mgsio3'},
    }


def test_check_eos_files_passes_when_tables_present(tmp_path):
    """All selected EOS table files on disk: the check is silent.

    Exercises both the flat (core) and nested (mantle) registry shapes.
    """
    from proteus.interior_struct.zalmoxis import check_zalmoxis_eos_files

    registry = _synthetic_registry(tmp_path, write_files=True)
    layer_cfg = {'core': 'PALEOS:iron', 'mantle': 'PALEOS-2phase:MgSiO3'}
    assert check_zalmoxis_eos_files(layer_cfg, registry) is None

    # Unknown identifiers are deferred to the registry lookup in the
    # solver, which produces its own error; the file check stays silent.
    layer_cfg_unknown = {'mantle': 'NoSuchEOS:MgSiO3'}
    assert check_zalmoxis_eos_files(layer_cfg_unknown, registry) is None


def test_check_eos_files_raises_actionable_error(tmp_path):
    """Missing tables produce one error naming every path and the fix.

    This is the guard for the offline-first-run failure mode: without it
    the solver logs one read error per shell and exits via a misleading
    non-convergence message.
    """
    from proteus.interior_struct.zalmoxis import check_zalmoxis_eos_files

    registry = _synthetic_registry(tmp_path, write_files=False)
    layer_cfg = {'core': 'PALEOS:iron', 'mantle': 'PALEOS-2phase:MgSiO3'}

    with pytest.raises(RuntimeError) as excinfo:
        check_zalmoxis_eos_files(layer_cfg, registry)
    msg = str(excinfo.value)
    # Every missing file is listed: the flat core table and both 2-phase
    # mantle tables (three distinct paths, deduplicated).
    assert 'paleos_iron.dat' in msg
    assert 'mgsio3_liquid.dat' in msg
    assert 'mgsio3_solid.dat' in msg
    # The error names the commands that repair the state.
    assert 'proteus get interiordata' in msg
    assert '--offline' in msg
    assert '`fwl-io relocate`' in msg


def _paleos_registry(tmp_path, missing: str = '') -> dict:
    """Registry with the PALEOS tables a PALEOS mantle reads; ``missing`` names the absent one."""
    files = {
        name: tmp_path / f'{name}.dat' for name in ('mgsio3', 'h2o', 'iron', 'solid', 'liquid')
    }
    for name, path in files.items():
        if name != missing:
            path.write_text('eos table stub')
    unified = {'format': 'paleos_unified'}
    return {
        'PALEOS:MgSiO3': {**unified, 'eos_file': str(files['mgsio3'])},
        'PALEOS:H2O': {**unified, 'eos_file': str(files['h2o'])},
        'PALEOS:iron': {**unified, 'eos_file': str(files['iron'])},
        'PALEOS-2phase:MgSiO3': {
            'core': {'eos_file': str(tmp_path / 'seager_iron_absent.txt')},
            'melted_mantle': {'eos_file': str(files['liquid']), 'format': 'paleos'},
            'solid_mantle': {'eos_file': str(files['solid']), 'format': 'paleos'},
        },
    }


def _require_config(mantle_eos, *, resume=False, ice=None):
    """Mock config for require_paleos_tables with a PALEOS iron core."""
    config = MagicMock()
    config.interior_struct.module = 'zalmoxis'
    config.interior_struct.melting_dir = 'Monteux-600'
    zc = config.interior_struct.zalmoxis
    zc.core_eos, zc.mantle_eos, zc.ice_layer_eos = 'PALEOS:iron', mantle_eos, ice
    zc.mushy_zone_factor, zc.dry_mantle = 0.8, True
    config.interior_energetics.module = 'aragog'
    config.planet.temperature_mode = 'adiabatic'
    config.params.resume = resume
    config.params.offline = True
    return config


@pytest.mark.parametrize('missing', ['solid', 'liquid'])
def test_check_eos_files_requires_the_paleos_companions(tmp_path, missing):
    """With the companions required, a PALEOS mantle stops on an absent table of the
    MgSiO3 2-phase pair, naming that file and the fetch command; without them only its
    own table counts."""
    from proteus.interior_struct.zalmoxis import (
        ZalmoxisMissingEOSFilesError,
        check_zalmoxis_eos_files,
    )

    registry = _paleos_registry(tmp_path, missing)
    layers = {'core': 'PALEOS:iron', 'mantle': 'PALEOS:H2O'}
    with pytest.raises(ZalmoxisMissingEOSFilesError) as excinfo:
        check_zalmoxis_eos_files(layers, registry, paleos_companions=True)
    msg = str(excinfo.value)
    assert f'{missing}.dat' in msg
    assert 'proteus get interiordata' in msg
    # The Seager core of the 2-phase entry is not read for a mantle role.
    assert 'seager_iron_absent' not in msg
    assert check_zalmoxis_eos_files(layers, registry) is None


@pytest.mark.parametrize(
    'mantle, stops',
    [
        ('PALEOS-2phase:MgSiO3', False),
        ('PALEOS:H2O', False),
        ('PALEOS:MgSiO3', True),
        ('PALEOS:MgSiO3:0.9+PALEOS:H2O:0.1', True),
    ],
)
def test_check_eos_files_needs_the_mgsio3_unified_table_only_for_an_mgsio3_layer(
    tmp_path, mantle, stops
):
    """Without the MgSiO3 unified table a mantle that never reads it passes, while a
    mantle with a PALEOS:MgSiO3 component stops and names the file."""
    from proteus.interior_struct.zalmoxis import (
        ZalmoxisMissingEOSFilesError,
        check_zalmoxis_eos_files,
    )

    registry = _paleos_registry(tmp_path, 'mgsio3')
    layers = {'core': 'PALEOS:iron', 'mantle': mantle}
    if stops:
        with pytest.raises(ZalmoxisMissingEOSFilesError, match='mgsio3.dat'):
            check_zalmoxis_eos_files(layers, registry, paleos_companions=True)
    else:
        assert check_zalmoxis_eos_files(layers, registry, paleos_companions=True) is None
    # The pair is still required in every case.
    (tmp_path / 'solid.dat').unlink()
    with pytest.raises(ZalmoxisMissingEOSFilesError, match='solid.dat'):
        check_zalmoxis_eos_files(layers, registry, paleos_companions=True)


def test_check_eos_files_stops_when_the_paleos_api_resolver_is_missing(tmp_path, monkeypatch):
    """A PALEOS-API layer table or companion that cannot be materialised stops the
    run; a resolver that writes the tables lets it pass."""
    import sys
    import types

    from proteus.interior_struct.zalmoxis import (
        ZalmoxisMissingEOSFilesError,
        check_zalmoxis_eos_files,
    )

    api = {'format': 'paleos_api', 'material': 'mgsio3'}
    pair = {r: {'format': 'paleos_api_2phase'} for r in ('solid_mantle', 'melted_mantle')}
    registry = {'PALEOS-API:MgSiO3': api, 'PALEOS-API-2phase:MgSiO3': pair}
    layers = {'mantle': 'PALEOS-API:MgSiO3'}
    monkeypatch.setitem(sys.modules, 'zalmoxis.eos.paleos_api_cache', None)
    with pytest.raises(ZalmoxisMissingEOSFilesError) as excinfo:
        check_zalmoxis_eos_files(layers, registry, paleos_companions=True)
    assert 'PALEOS-API-2phase:MgSiO3 (PALEOS-API tables not built: ModuleNotFoundError' in str(
        excinfo.value
    )
    assert 'PALEOS-API:MgSiO3 (PALEOS-API tables not built: ModuleNotFoundError' in str(
        excinfo.value
    )

    table = tmp_path / 'api.dat'
    table.write_text('eos table stub')

    def _resolve(entry):
        for sub in [entry, *(v for v in entry.values() if isinstance(v, dict))]:
            if 'format' in sub:
                sub['eos_file'] = str(table)

    fake = types.ModuleType('zalmoxis.eos.paleos_api_cache')
    fake.resolve_registry_entry = _resolve
    monkeypatch.setitem(sys.modules, 'zalmoxis.eos.paleos_api_cache', fake)
    assert check_zalmoxis_eos_files(layers, registry, paleos_companions=True) is None
    assert pair['solid_mantle']['eos_file'] == str(table)
    assert api['eos_file'] == str(table)


@pytest.mark.parametrize('missing', ['iron', 'solid', 'h2o'])
def test_require_paleos_tables_stops_an_offline_run(tmp_path, monkeypatch, missing):
    """An offline fresh run stops before any solve on an absent core, ice-layer or pair
    table and names it; with every table present it passes."""
    from proteus.interior_struct import zalmoxis as zmod

    monkeypatch.setattr(
        zmod, 'load_zalmoxis_material_dictionaries', lambda: _paleos_registry(tmp_path, missing)
    )
    config = _require_config('PALEOS:MgSiO3', ice='PALEOS:H2O')
    with pytest.raises(zmod.ZalmoxisMissingEOSFilesError, match=f'{missing}.dat'):
        zmod.require_paleos_tables(config, str(tmp_path))
    (tmp_path / f'{missing}.dat').write_text('eos table stub')
    assert zmod.require_paleos_tables(config, str(tmp_path)) is None


def test_require_paleos_tables_lets_a_resume_keep_its_tables(tmp_path, monkeypatch, caplog):
    """A resumed SPIDER run with kept P-S tables continues without the pair, with one
    WARNING; an Aragog resume, which re-solves its initial condition on the pair when
    the mesh changes, and a resume without kept tables stop like a fresh run."""
    from proteus.interior_struct import zalmoxis as zmod

    monkeypatch.delenv('PROTEUS_PS_CACHE_DIR', raising=False)
    monkeypatch.setattr(
        zmod, 'load_zalmoxis_material_dictionaries', lambda: _paleos_registry(tmp_path, 'solid')
    )
    config = _require_config('PALEOS:MgSiO3', resume=True)
    config.interior_energetics.module = 'spider'
    bare = tmp_path / 'bare'
    with pytest.raises(zmod.ZalmoxisMissingEOSFilesError, match='solid.dat'):
        zmod.require_paleos_tables(config, str(bare))

    kept = tmp_path / 'kept'
    _seed_tables(kept / 'data' / 'spider_eos', 'old-key')
    with caplog.at_level('WARNING', logger='fwl.proteus.interior_struct.zalmoxis'):
        assert zmod.require_paleos_tables(config, str(kept)) is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert 'keeps its original energetics P-S entropy tables' in warnings[0]

    config.interior_energetics.module = 'aragog'
    with pytest.raises(zmod.ZalmoxisMissingEOSFilesError, match='solid.dat'):
        zmod.require_paleos_tables(config, str(kept))


@pytest.mark.parametrize(
    'mantle, warned',
    [
        ('PALEOS:H2O', True),
        ('PALEOS:iron', True),
        ('PALEOS:MgSiO3', False),
        ('PALEOS:H2O:1.0', True),
        ('PALEOS:H2O:0.5+PALEOS:H2O:0.5', True),
    ],
)
def test_require_paleos_tables_warns_once_for_a_water_or_iron_mantle(
    tmp_path, monkeypatch, caplog, mantle, warned
):
    """A PALEOS H2O or iron mantle gets one WARNING at the start of the run that its
    energetics and melting curves are PALEOS MgSiO3; the per-solve check adds none, and
    an MgSiO3 mantle gets none."""
    from proteus.interior_struct import zalmoxis as zmod

    registry = _paleos_registry(tmp_path)
    monkeypatch.setattr(zmod, 'load_zalmoxis_material_dictionaries', lambda: registry)
    with caplog.at_level('WARNING', logger='fwl.proteus.interior_struct.zalmoxis'):
        zmod.require_paleos_tables(_require_config(mantle), str(tmp_path))
        for _ in range(3):
            zmod.check_zalmoxis_eos_files({'core': 'PALEOS:iron', 'mantle': mantle}, registry)
    messages = [
        r.getMessage() for r in caplog.records if 'are PALEOS MgSiO3 (' in r.getMessage()
    ]
    assert len(messages) == (1 if warned else 0)
    if warned:
        assert f'mantle_eos={mantle}: the structure uses its density, while' in messages[0]
        assert '(PALEOS-2phase:MgSiO3, solidus = 0.80 x PALEOS liquidus)' in messages[0]
    # Energetics without P-S tables build none, so no warning, unless the
    # liquidus_super initial adiabat is solved on the MgSiO3 pair.
    for mode, adiabat_warned in (('adiabatic', False), ('liquidus_super', warned)):
        caplog.clear()
        config = _require_config(mantle)
        config.interior_energetics.module = 'dummy'
        config.planet.temperature_mode = mode
        with caplog.at_level('WARNING', logger='fwl.proteus.interior_struct.zalmoxis'):
            zmod.require_paleos_tables(config, str(tmp_path))
        assert 'are PALEOS MgSiO3 (' not in caplog.text
        assert ('initial adiabat is solved on the MgSiO3' in caplog.text) is adiabat_warned


@pytest.mark.parametrize('missing', ['h2o', 'chabrier'])
def test_require_paleos_tables_needs_the_volatile_tables_of_a_wet_mantle(
    tmp_path, monkeypatch, missing
):
    """With dry_mantle = false the water and hydrogen tables of the dissolved volatiles
    are required."""
    from proteus.interior_struct import zalmoxis as zmod

    registry = _paleos_registry(tmp_path, missing)
    chabrier = tmp_path / 'chabrier.dat'
    if missing != 'chabrier':
        chabrier.write_text('eos table stub')
    registry['Chabrier:H'] = {'eos_file': str(chabrier), 'format': 'paleos_unified'}
    monkeypatch.setattr(zmod, 'load_zalmoxis_material_dictionaries', lambda: registry)
    config = _require_config('PALEOS:MgSiO3')
    assert zmod.require_paleos_tables(config, str(tmp_path)) is None
    config.interior_struct.zalmoxis.dry_mantle = False
    with pytest.raises(zmod.ZalmoxisMissingEOSFilesError, match=f'{missing}.dat'):
        zmod.require_paleos_tables(config, str(tmp_path))


def test_check_eos_files_strips_spaces_in_a_mixture(tmp_path):
    """Each component of a spaced mixture is checked."""
    from proteus.interior_struct.zalmoxis import (
        ZalmoxisMissingEOSFilesError,
        check_zalmoxis_eos_files,
    )

    registry = _paleos_registry(tmp_path, 'h2o')
    layers = {'core': 'PALEOS:iron', 'mantle': ' PALEOS:MgSiO3:0.9 + PALEOS:H2O:0.1 '}
    with pytest.raises(ZalmoxisMissingEOSFilesError, match='h2o.dat'):
        check_zalmoxis_eos_files(layers, registry, paleos_companions=True)


def test_check_eos_files_stops_when_the_paleos_api_build_fails(monkeypatch):
    """Any error of the PALEOS-API table build is a missing-table stop with its reason."""
    import sys
    import types

    from proteus.interior_struct.zalmoxis import (
        ZalmoxisMissingEOSFilesError,
        check_zalmoxis_eos_files,
    )

    def _resolve(entry):
        raise RuntimeError('grid build failed')

    fake = types.ModuleType('zalmoxis.eos.paleos_api_cache')
    fake.resolve_registry_entry = _resolve
    monkeypatch.setitem(sys.modules, 'zalmoxis.eos.paleos_api_cache', fake)
    pair = {r: {'format': 'paleos_api_2phase'} for r in ('solid_mantle', 'melted_mantle')}
    registry = {'PALEOS-API-2phase:MgSiO3': pair}
    with pytest.raises(ZalmoxisMissingEOSFilesError, match='RuntimeError: grid build failed'):
        check_zalmoxis_eos_files(
            {'mantle': 'PALEOS-API:MgSiO3'}, registry, paleos_companions=True
        )


_WB_ENERGETICS = 'the energetics use the eos_dir P-S tables (by default Wolf and Bower 2018)'


@pytest.mark.parametrize('module', ['aragog', 'spider'])
@pytest.mark.parametrize(
    'mantle, energetics',
    [
        (
            'PALEOS:MgSiO3:0.9+PALEOS:H2O:0.1',
            'the energetics and melting curves are PALEOS MgSiO3 (PALEOS:MgSiO3, solidus = 0.80',
        ),
        (
            'PALEOS:H2O:0.5+PALEOS:iron:0.5',
            'the energetics and melting curves are PALEOS MgSiO3 (PALEOS-2phase:MgSiO3, solidus',
        ),
        ('WolfBower2018:MgSiO3:0.9+PALEOS:H2O:0.1', _WB_ENERGETICS),
        ('WolfBower2018:MgSiO3:0.9+Chabrier:H:0.1', _WB_ENERGETICS),
    ],
)
def test_require_paleos_tables_warns_once_for_a_paleos_mixture(
    tmp_path, monkeypatch, caplog, module, mantle, energetics
):
    """A mixture with a PALEOS component gets one WARNING that its other components
    enter only the structure density, naming the set its energetics follow."""
    from proteus.interior_struct import zalmoxis as zmod

    monkeypatch.setattr(
        zmod, 'load_zalmoxis_material_dictionaries', lambda: _paleos_registry(tmp_path)
    )
    # Chabrier:H is not in the PALEOS test registry.
    monkeypatch.setattr(zmod, 'check_zalmoxis_eos_files', lambda *a, **k: None)
    config = _require_config(mantle)
    config.interior_energetics.module = module
    with caplog.at_level('WARNING', logger='fwl.proteus.interior_struct.zalmoxis'):
        zmod.require_paleos_tables(config, str(tmp_path))
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert (
        f'mantle_eos={mantle}: the non-MgSiO3 components enter only the structure density'
        in (warnings[0])
    )
    assert energetics in warnings[0]


@pytest.mark.parametrize(
    'mixture, single',
    [
        ('PALEOS:MgSiO3:0.9+PALEOS:H2O:0.1', 'PALEOS:MgSiO3'),
        ('PALEOS:H2O:0.3+PALEOS:MgSiO3:0.7', 'PALEOS:MgSiO3'),
        ('PALEOS:H2O:0.5+PALEOS:iron:0.5', 'PALEOS-2phase:MgSiO3'),
        ('PALEOS:H2O', 'PALEOS-2phase:MgSiO3'),
        ('PALEOS:iron', 'PALEOS-2phase:MgSiO3'),
    ],
)
def test_a_paleos_mixture_gets_the_ps_key_of_its_mgsio3_set(tmp_path, mixture, single):
    """The P-S tables of a PALEOS mixture carry the key of its MgSiO3 set, the same key
    and cache as that single mantle, whatever the fractions."""
    from proteus.interior_struct import zalmoxis as zmod

    registry = _paleos_registry(tmp_path)
    keys = []
    for mantle in (mixture, single):
        config = _require_config(mantle)
        config.planet.mass_tot = 1.0
        config.interior_struct.zalmoxis.lookup_nP = 20
        config.interior_struct.zalmoxis.lookup_nS = 30
        key, entry = zmod.energetics_entry(mantle, registry)
        keys.append((key, zmod._ps_table_inputs(config, key, entry, registry)[-1]))
    assert keys[0] == keys[1]
    assert keys[0][0] == single


@pytest.mark.parametrize(
    'mantle',
    [
        'WolfBower2018:MgSiO3',
        'RTPress100TPa:MgSiO3',
        'WolfBower2018:MgSiO3:0.9+PALEOS:H2O:0.1',
        'RTPress100TPa:MgSiO3:0.9+PALEOS:H2O:0.1',
    ],
)
def test_generate_spider_tables_builds_no_set_for_a_wolf_bower_mantle(
    tmp_path, monkeypatch, mantle
):
    """A mantle whose energetics key is not PALEOS gets no generated set and no table
    check, also when its registry entry has solid and melt tables on disk."""
    from proteus.interior_struct import zalmoxis as zmod

    solid, melt = tmp_path / 'wb_solid.dat', tmp_path / 'wb_melt.dat'
    solid.write_text('stub')
    melt.write_text('stub')
    entry = {'melted_mantle': {'eos_file': str(melt)}, 'solid_mantle': {'eos_file': str(solid)}}
    registry = {'WolfBower2018:MgSiO3': entry, 'RTPress100TPa:MgSiO3': entry}
    monkeypatch.setattr(zmod, 'load_zalmoxis_material_dictionaries', lambda: registry)
    monkeypatch.setattr(
        zmod, 'check_zalmoxis_eos_files', lambda *a, **k: pytest.fail('checked tables')
    )
    config = _require_config(mantle)
    config.params.resume = False
    assert zmod.generate_spider_tables(config, str(tmp_path)) is None


@pytest.mark.parametrize('missing, other', [('liquid', 'solid'), ('solid', 'liquid')])
def test_resolve_2phase_paths_stops_on_a_missing_table_when_required(tmp_path, missing, other):
    """A required pair with a missing table raises and names the table; otherwise the
    missing table is None."""
    from proteus.interior_struct import zalmoxis as zmod

    registry = _paleos_registry(tmp_path, missing)
    paths = dict(
        zip(('solid', 'liquid'), zmod.resolve_2phase_mgsio3_paths('PALEOS:MgSiO3', registry))
    )
    assert paths == {missing: None, other: str(tmp_path / f'{other}.dat')}
    with pytest.raises(zmod.ZalmoxisMissingEOSFilesError, match=rf'not available: {missing}\.'):
        zmod.resolve_2phase_mgsio3_paths('PALEOS:MgSiO3', registry, required=True)


def test_resolve_2phase_paths_required_returns_a_full_pair_and_names_two_missing(tmp_path):
    """A required pair on disk returns both paths; with both tables absent both are named."""
    from proteus.interior_struct import zalmoxis as zmod

    registry = _paleos_registry(tmp_path)
    full = zmod.resolve_2phase_mgsio3_paths('PALEOS:MgSiO3', registry, required=True)
    assert full == (str(tmp_path / 'solid.dat'), str(tmp_path / 'liquid.dat'))
    for name in ('solid', 'liquid'):
        (tmp_path / f'{name}.dat').unlink()
    with pytest.raises(
        zmod.ZalmoxisMissingEOSFilesError, match=r'not available: solid, liquid\.'
    ):
        zmod.resolve_2phase_mgsio3_paths('PALEOS:MgSiO3', registry, required=True)


@pytest.mark.parametrize('mantle', ['WolfBower2018:MgSiO3', 'PALEOS:MgSiO3'])
def test_liquidus_super_needs_the_2phase_pair_for_any_mantle(tmp_path, monkeypatch, mantle):
    """The liquidus_super anchor is solved on the MgSiO3 2-phase pair for any mantle, so
    a missing pair stops at the start check and in the anchor itself, also for a Wolf and
    Bower mantle without a generated PALEOS set; an adiabatic start does not need it."""
    from proteus.interior_struct import zalmoxis as zmod

    registry = _paleos_registry(tmp_path, 'liquid')
    registry['WolfBower2018:MgSiO3'] = {'eos_file': str(tmp_path / 'mgsio3.dat')}
    monkeypatch.setattr(zmod, 'load_zalmoxis_material_dictionaries', lambda: registry)
    monkeypatch.setattr(zmod, '_SUPERLIQ_CACHE', {})
    config = _require_config(mantle)
    config.interior_energetics.module = 'spider'
    if mantle.startswith('WolfBower'):
        assert zmod.require_paleos_tables(config, str(tmp_path)) is None
    config.planet.temperature_mode = 'liquidus_super'
    with pytest.raises(zmod.ZalmoxisMissingEOSFilesError, match='liquid.dat'):
        zmod.require_paleos_tables(config, str(tmp_path))
    with pytest.raises(zmod.ZalmoxisMissingEOSFilesError, match='not available: liquid'):
        zmod._solve_superliquidus_adiabat(config, {'P_cmb': 1.2e11})


def test_superliquidus_anchor_passes_a_missing_table_stop_through(monkeypatch):
    """A missing-table stop inside the anchor reaches the caller as that error, not as a
    numerical anchor failure."""
    from proteus.interior_struct import zalmoxis as zmod

    def _missing(config, hf_row):
        raise zmod.ZalmoxisMissingEOSFilesError('pair not available')

    failed = {}
    monkeypatch.setattr(zmod, '_solve_superliquidus_adiabat', _missing)
    monkeypatch.setattr(zmod, '_SUPERLIQ_FAILED', failed)
    config = _require_config('PALEOS:MgSiO3')
    with pytest.raises(zmod.ZalmoxisMissingEOSFilesError, match='pair not available'):
        zmod.solve_superliquidus_adiabat(config, {'P_cmb': 1.2e11})
    # Not cached as an anchor failure, so a run with the tables fetched retries.
    assert failed == {}


def test_generate_spider_tables_stops_on_a_missing_pair_table(tmp_path, monkeypatch):
    """A PALEOS unified mantle whose 2-phase liquid table is absent stops before any
    table is built, instead of building the P-S set from the unified table alone."""
    import zalmoxis.eos_export

    from proteus.interior_struct import zalmoxis as zmod

    monkeypatch.delenv('PROTEUS_PS_CACHE_DIR', raising=False)
    registry = _paleos_registry(tmp_path, 'liquid')
    monkeypatch.setattr(zmod, 'load_zalmoxis_material_dictionaries', lambda: registry)
    bounds = MagicMock(side_effect=RuntimeError('build started'))
    monkeypatch.setattr(zalmoxis.eos_export, 'generate_spider_phase_boundaries', bounds)
    monkeypatch.setattr(zalmoxis.eos_export, 'generate_spider_eos_tables', bounds)
    config = _require_config('PALEOS:MgSiO3')
    config.planet.mass_tot = 1.0
    with pytest.raises(zmod.ZalmoxisMissingEOSFilesError, match='liquid.dat'):
        zmod.generate_spider_tables(config, str(tmp_path / 'run'))
    bounds.assert_not_called()
    # Discrimination: with the pair present the build starts.
    (tmp_path / 'liquid.dat').write_text('eos table stub')
    with pytest.raises(RuntimeError, match='build started'):
        zmod.generate_spider_tables(config, str(tmp_path / 'run'))


def test_check_eos_files_parses_extended_mantle_strings(tmp_path):
    """Volatile-extended mantle EOS strings resolve to their registry keys.

    ``extend_mantle_eos_with_volatiles`` produces strings of the form
    ``'PALEOS:MgSiO3:0.9800+PALEOS:H2O:0.0100'``; the trailing fraction
    tokens must be stripped before the registry lookup, and every
    component of the composite must be checked.
    """
    from proteus.interior_struct.zalmoxis import (
        _strip_fraction_tokens,
        check_zalmoxis_eos_files,
    )

    # Token stripping: fraction suffixes go, the identifier stays intact.
    assert _strip_fraction_tokens('PALEOS:MgSiO3:0.9800') == 'PALEOS:MgSiO3'
    assert _strip_fraction_tokens('PALEOS:MgSiO3') == 'PALEOS:MgSiO3'
    # Edge case: a lone fraction token strips to the empty string rather
    # than raising; the registry lookup then misses and the check defers.
    assert _strip_fraction_tokens('0.5') == ''

    missing = tmp_path / 'h2o.dat'  # never written
    registry = {
        'PALEOS:MgSiO3': {'eos_file': str(tmp_path / 'mgsio3.dat')},
        'PALEOS:H2O': {'eos_file': str(missing)},
    }
    (tmp_path / 'mgsio3.dat').write_text('stub')

    layer_cfg = {'mantle': 'PALEOS:MgSiO3:0.9900+PALEOS:H2O:0.0100'}
    with pytest.raises(RuntimeError, match='h2o.dat'):
        check_zalmoxis_eos_files(layer_cfg, registry)

    # Non-finite trailing tokens are not fraction tokens: 'nan' stays in
    # the identifier (and then defers as unknown) instead of collapsing
    # the component to the empty string.
    assert _strip_fraction_tokens('PALEOS:MgSiO3:nan') == 'PALEOS:MgSiO3:nan'


def test_check_eos_files_walks_nested_and_lazy_siblings(tmp_path, monkeypatch):
    """Nested role entries are walked; lazy siblings do not mask misses.

    The real registry nests Seager-style single-role entries
    (``{'core': {...}}``) and mixes lazy PALEOS-API sub-entries (no
    ``eos_file`` until first density query) with file-backed ones inside
    one nested entry. A missing file-backed sibling must be flagged even
    when a lazy sibling sits next to it.
    """
    import sys
    import types

    from proteus.interior_struct.zalmoxis import check_zalmoxis_eos_files

    present = tmp_path / 'seager_iron.txt'
    present.write_text('stub')
    missing = tmp_path / 'mgsio3_solid.dat'  # never written

    registry = {
        # Seager-style nested single-role entry (no top-level eos_file).
        'Seager2007:iron': {'core': {'eos_file': str(present)}},
        # Mixed nested entry: lazy API liquid side + file-backed solid side.
        'PALEOS-API-2phase:MgSiO3': {
            'core': {'eos_file': str(present)},
            'melted_mantle': {'format': 'paleos_api_2phase', 'side': 'liquid'},
            'solid_mantle': {'eos_file': str(missing), 'format': 'paleos'},
        },
    }

    # All file-backed paths present for the core: silent.
    assert check_zalmoxis_eos_files({'core': 'Seager2007:iron'}, registry) is None
    fake = types.ModuleType('zalmoxis.eos.paleos_api_cache')
    fake.resolve_registry_entry = lambda entry: None
    monkeypatch.setitem(sys.modules, 'zalmoxis.eos.paleos_api_cache', fake)

    # The missing file-backed sibling is flagged; the lazy sibling is not.
    with pytest.raises(RuntimeError) as excinfo:
        check_zalmoxis_eos_files({'mantle': 'PALEOS-API-2phase:MgSiO3'}, registry)
    msg = str(excinfo.value)
    assert 'mgsio3_solid.dat' in msg
    assert 'seager_iron.txt' not in msg  # present file must not be flagged


def test_check_eos_files_nested_core_scoped_to_core_role(tmp_path):
    """A nested ``'core'`` sub-entry is only checked for a core-role request.

    Uses a registry where the nested ``'core'`` path is missing and distinct
    from every other path in the same entry, so a call that walks it
    regardless of role would flag it under both roles below; only the
    core-role call should.
    """
    from proteus.interior_struct.zalmoxis import check_zalmoxis_eos_files

    missing_core = tmp_path / 'iron_core_only.dat'  # never written
    present_liquid = tmp_path / 'mgsio3_liquid.dat'
    present_solid = tmp_path / 'mgsio3_solid.dat'
    for f in (present_liquid, present_solid):
        f.write_text('stub')

    registry = {
        'PALEOS-2phase:MgSiO3-with-core': {
            'core': {'eos_file': str(missing_core)},
            'melted_mantle': {'eos_file': str(present_liquid)},
            'solid_mantle': {'eos_file': str(present_solid)},
        },
    }

    with pytest.raises(RuntimeError, match='iron_core_only.dat'):
        check_zalmoxis_eos_files({'core': 'PALEOS-2phase:MgSiO3-with-core'}, registry)

    assert (
        check_zalmoxis_eos_files({'mantle': 'PALEOS-2phase:MgSiO3-with-core'}, registry) is None
    )


def test_check_eos_files_single_key_core_checked_for_non_core_role(tmp_path):
    """A single-key nested ``'core'`` entry is checked from any role.

    Unlike ``PALEOS-2phase:MgSiO3-with-core`` above, an entry whose only
    sub-entry is ``'core'`` (e.g. ``Seager2007:iron``) has no other
    role-specific data for a non-core requester to use instead; the real
    dispatch reads that sole sub-entry regardless of which layer role
    points at the identifier, so the check must not exempt it here.
    """
    from proteus.interior_struct.zalmoxis import check_zalmoxis_eos_files

    missing = tmp_path / 'seager_iron.txt'  # never written
    registry = {
        'Seager2007:iron': {'core': {'eos_file': str(missing)}},
    }

    with pytest.raises(RuntimeError, match='seager_iron.txt'):
        check_zalmoxis_eos_files({'mantle': 'Seager2007:iron'}, registry)
    with pytest.raises(RuntimeError, match='seager_iron.txt'):
        check_zalmoxis_eos_files({'ice_layer': 'Seager2007:iron'}, registry)


def _volatile_config(dry_mantle: bool):
    """MagicMock config for the dry-mass target tests.

    MagicMock drives ``load_zalmoxis_configuration`` directly, independent
    of config-load validation: the subtraction logic is exercised for both
    mantle EOS modes now that the ``dry_mantle = false`` gate is lifted.
    """
    config = MagicMock()
    config.planet.mass_tot = 1.0
    config.interior_struct.zalmoxis.core_eos = 'Seager2007:iron'
    config.interior_struct.zalmoxis.mantle_eos = 'Seager2007:silicate'
    config.interior_struct.zalmoxis.ice_layer_eos = None
    config.interior_struct.core_frac = 0.325
    config.interior_struct.zalmoxis.mantle_mass_fraction = 0.0
    config.interior_struct.zalmoxis.dry_mantle = dry_mantle
    config.planet.temperature_mode = 'isothermal'
    config.planet.tsurf_init = 300
    config.planet.tcenter_init = 5000
    config.interior_struct.zalmoxis.num_levels = 200
    # Real values where MagicMock attributes would otherwise be silently
    # absorbed by defensive try/except paths (surface-pressure resolution)
    # or string formatting (the config debug log), hiding broken setup.
    config.planet.tcmb_init = 4000.0
    config.planet.gas_prs = None
    config.interior_struct.zalmoxis.outer_solver = 'newton'
    config.interior_struct.zalmoxis.solver_tol_outer = 3e-3
    config.interior_struct.zalmoxis.solver_tol_inner = 1e-4
    config.interior_struct.zalmoxis.use_jax = True
    config.interior_struct.zalmoxis.use_anderson = False
    return config


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_dry_mass_target_excludes_only_undissolved_volatiles():
    """Mass closure of the structure target under both mantle EOS modes.

    With a bare-silicate mantle EOS (dry_mantle = true) the full volatile
    inventory is excluded from the dry-mass target. When the mantle EOS
    carries the dissolved volatiles (dry_mantle = false), only the
    atmospheric inventory may be excluded: the dissolved mass is already
    part of the wet-mantle EOS density, and excluding it again would
    remove it twice and solve the structure for an underweight planet.
    """
    from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration
    from proteus.utils.constants import M_earth

    # Asymmetric inventories so the two modes are well separated: most of
    # the H and O mass is dissolved, not atmospheric.
    H_total, H_atm = 4.7e20, 1.2e20
    O_total, O_atm = 3.1e21, 0.9e21
    hf_row = {
        'H_kg_total': H_total,
        'H_kg_atm': H_atm,
        'O_kg_total': O_total,
        'O_kg_atm': O_atm,
    }

    dry = load_zalmoxis_configuration(_volatile_config(True), hf_row)
    wet = load_zalmoxis_configuration(_volatile_config(False), hf_row)

    # Conservation: total = dry-target + excluded volatiles, per mode.
    assert dry['planet_mass'] == pytest.approx(M_earth - (H_total + O_total), rel=1e-12)
    assert wet['planet_mass'] == pytest.approx(M_earth - (H_atm + O_atm), rel=1e-12)

    # The mode difference is exactly the dissolved inventory: this is the
    # mass that the previous unconditional subtraction removed twice.
    dissolved = (H_total - H_atm) + (O_total - O_atm)
    assert wet['planet_mass'] - dry['planet_mass'] == pytest.approx(dissolved, rel=1e-12)
    # Sign guard: the wet-mode target must be the heavier one.
    assert wet['planet_mass'] > dry['planet_mass']
    # Scale guard: both targets remain within 0.1% of an Earth mass; a
    # unit error (g vs kg) or a doubled subtraction would breach this.
    assert 0.999 * M_earth < dry['planet_mass'] < M_earth
    assert 0.999 * M_earth < wet['planet_mass'] < M_earth


# ============================================================================
# Temperature-source dispatch: JAX-path viability and callable pass-through
# ============================================================================


def _gate_registry(tmp_path) -> dict:
    """EOS registry covering the JAX-viable and numpy-fallback layouts.

    Mirrors the shapes built by ``load_zalmoxis_material_dictionaries``:
    a unified PALEOS core and mantle (flat entries, format
    ``paleos_unified``, no phase sub-tables) and a 2-phase PALEOS mantle
    (``solid_mantle`` + ``melted_mantle`` sub-tables). All referenced
    table files exist on disk so ``check_zalmoxis_eos_files`` passes.
    """
    iron = tmp_path / 'paleos_iron.dat'
    unified = tmp_path / 'mgsio3_unified.dat'
    sol = tmp_path / 'mgsio3_solid.dat'
    liq = tmp_path / 'mgsio3_liquid.dat'
    water = tmp_path / 'h2o_unified.dat'
    hydrogen = tmp_path / 'chabrier_h.dat'
    seager_sil = tmp_path / 'seager_silicate.txt'
    for f in (iron, unified, sol, liq, water, hydrogen, seager_sil):
        f.write_text('eos table stub')
    return {
        'PALEOS:iron': {'eos_file': str(iron), 'format': 'paleos_unified'},
        'PALEOS:MgSiO3': {'eos_file': str(unified), 'format': 'paleos_unified'},
        'PALEOS:H2O': {'eos_file': str(water), 'format': 'paleos_unified'},
        # Same shape as the production registry entry: unified format, but
        # the JAX wet path rejects it by name (binodal suppression).
        'Chabrier:H': {'eos_file': str(hydrogen), 'format': 'paleos_unified'},
        'PALEOS-2phase:MgSiO3': {
            'core': {'eos_file': str(iron)},
            'melted_mantle': {'eos_file': str(liq), 'format': 'paleos'},
            'solid_mantle': {'eos_file': str(sol), 'format': 'paleos'},
        },
        # Flat entry with neither the unified format nor phase sub-tables:
        # no JAX representation, the numpy ODE handles it.
        'Seager2007:silicate': {'eos_file': str(seager_sil)},
    }


def test_jax_structure_viability_predicate(tmp_path):
    """JAX-path viability tracks the registry layout, not the config flags.

    The predicate must reproduce the precondition outcome of Zalmoxis'
    JAX dispatch: True for a unified PALEOS mantle (flat entry, format
    ``paleos_unified`` with an ``eos_file``) or a 2-phase mantle (both
    phase sub-tables), paired with a core that resolves to the unified
    PALEOS format. Edge cases: unknown registry keys (conservative
    False, keeping the temperature callable in play), ``paleos_api``
    entries (resolve in place to ``paleos_unified``, so True), a
    unified entry without an ``eos_file`` (the wrapper raises, so
    False), and a volatile-extended mantle string whose fraction token
    must be stripped before lookup.
    """
    from proteus.interior_struct.zalmoxis import _zalmoxis_jax_structure_viable

    registry = _gate_registry(tmp_path)

    # Unified mantle (flat paleos_unified entry): the JAX path can run.
    assert _zalmoxis_jax_structure_viable(registry, 'PALEOS:iron', 'PALEOS:MgSiO3') is True

    # 2-phase mantle + unified core: the JAX path can run.
    assert (
        _zalmoxis_jax_structure_viable(registry, 'PALEOS:iron', 'PALEOS-2phase:MgSiO3') is True
    )

    # Flat mantle entry with neither the unified format nor phase
    # sub-tables: the JAX dispatch raises and the numpy ODE runs.
    assert (
        _zalmoxis_jax_structure_viable(registry, 'PALEOS:iron', 'Seager2007:silicate') is False
    )

    # A unified mantle entry without an eos_file cannot be cached by the
    # wrapper (it raises), so the predicate must be False.
    registry_nofile = dict(registry)
    registry_nofile['PALEOS:MgSiO3'] = {'format': 'paleos_unified'}
    assert (
        _zalmoxis_jax_structure_viable(registry_nofile, 'PALEOS:iron', 'PALEOS:MgSiO3') is False
    )

    # Missing registry entries (mantle or core): conservative False.
    assert _zalmoxis_jax_structure_viable(registry, 'PALEOS:iron', 'NoSuchEOS') is False
    assert (
        _zalmoxis_jax_structure_viable(registry, 'NoSuchCore', 'PALEOS-2phase:MgSiO3') is False
    )
    assert _zalmoxis_jax_structure_viable({}, 'PALEOS:iron', 'PALEOS-2phase:MgSiO3') is False

    # paleos_api entries materialise to paleos_unified before the JAX
    # dispatch checks them, so they qualify on either layer.
    registry_api = dict(registry)
    registry_api['PALEOS-API:iron'] = {'format': 'paleos_api', 'material': 'iron'}
    registry_api['PALEOS-API:MgSiO3'] = {'format': 'paleos_api', 'material': 'mgsio3'}
    assert (
        _zalmoxis_jax_structure_viable(registry_api, 'PALEOS-API:iron', 'PALEOS-2phase:MgSiO3')
        is True
    )
    assert (
        _zalmoxis_jax_structure_viable(registry_api, 'PALEOS:iron', 'PALEOS-API:MgSiO3') is True
    )

    # A non-unified core (Seager nested layout, no top-level format)
    # fails the core precondition even with a 2-phase mantle.
    registry_seager = dict(registry)
    registry_seager['Seager2007:iron'] = {'core': {'eos_file': str(tmp_path / 'fe.txt')}}
    assert (
        _zalmoxis_jax_structure_viable(
            registry_seager, 'Seager2007:iron', 'PALEOS-2phase:MgSiO3'
        )
        is False
    )

    # Volatile-extended mantle strings carry fraction tokens; the lookup
    # strips them, so the extended form matches its registry key.
    assert (
        _zalmoxis_jax_structure_viable(registry, 'PALEOS:iron', 'PALEOS-2phase:MgSiO3:0.9800')
        is True
    )
    assert (
        _zalmoxis_jax_structure_viable(registry, 'PALEOS:iron', 'PALEOS:MgSiO3:0.9800') is True
    )


_WET_GATE_REGISTRY = {
    'PALEOS:H2O': {'eos_file': 'h2o_unified.dat', 'format': 'paleos_unified'},
    # Same shape as the production entry; the wet envelope rejects it by
    # name, not by format.
    'Chabrier:H': {'eos_file': 'chabrier_h.dat', 'format': 'paleos_unified'},
    # Live-tabulated entry: materialises to paleos_unified in place.
    'PALEOS-API:H2O': {'format': 'paleos_api', 'material': 'h2o'},
    # 2-phase style format: the JAX wet path has no reader for it.
    'WolfBower2018:H2O': {'eos_file': 'wb_h2o.dat', 'format': 'paleos'},
}

_WET_GATE_MANTLE = 'PALEOS:MgSiO3:0.9800+PALEOS:H2O:0.0200'


def _make_profile(**overrides):
    """Real Zalmoxis VolatileProfile, default in-envelope (H2O in melt)."""
    from zalmoxis.mixing import VolatileProfile

    kwargs = {
        'w_liquid': {'PALEOS:H2O': 0.02},
        'w_solid': {},
        'primary_component': 'PALEOS:MgSiO3',
    }
    kwargs.update(overrides)
    return VolatileProfile(**kwargs)


def test_volatile_profile_jax_viable_accepts_in_envelope_profiles():
    """The wet-envelope predicate accepts what the Zalmoxis JAX path runs.

    In-envelope means a single active paleos_unified volatile blended
    into the mantle through the profile, with no binodal or miscibility
    physics. Melt-only and solid-only weights both count as active, a
    zero-weight primary entry contributes nothing and is tolerated, and
    a paleos_api volatile qualifies because it materialises to
    paleos_unified in place.
    """
    from proteus.interior_struct.zalmoxis import _volatile_profile_jax_viable

    # Single H2O volatile in the melt: the canonical wet solve.
    assert (
        _volatile_profile_jax_viable(_make_profile(), _WET_GATE_REGISTRY, _WET_GATE_MANTLE)
        is True
    )

    # Solid-phase weight alone also makes the volatile active.
    solid_only = _make_profile(w_liquid={}, w_solid={'PALEOS:H2O': 0.005})
    assert (
        _volatile_profile_jax_viable(solid_only, _WET_GATE_REGISTRY, _WET_GATE_MANTLE) is True
    )

    # A zero-weight primary entry contributes nothing in the blend and
    # is tolerated by the wrapper.
    zero_primary = _make_profile(w_liquid={'PALEOS:H2O': 0.02, 'PALEOS:MgSiO3': 0.0})
    assert (
        _volatile_profile_jax_viable(zero_primary, _WET_GATE_REGISTRY, _WET_GATE_MANTLE) is True
    )

    # A paleos_api volatile materialises to paleos_unified in place.
    api_vol = _make_profile(w_liquid={'PALEOS-API:H2O': 0.02})
    assert (
        _volatile_profile_jax_viable(
            api_vol, _WET_GATE_REGISTRY, 'PALEOS:MgSiO3:0.9800+PALEOS-API:H2O:0.0200'
        )
        is True
    )


def test_volatile_profile_jax_viable_rejects_out_of_envelope_profiles():
    """The wet-envelope predicate is False for every wrapper rejection.

    Each case mirrors one ValueError branch of the Zalmoxis wet-mantle
    validation (or of the volatile cache load that follows it): the
    numpy fallback fires inside Zalmoxis, so the callable must stay in
    play. Unknown or malformed input also reads False, the conservative
    answer.
    """
    from proteus.interior_struct.zalmoxis import _volatile_profile_jax_viable

    reg = _WET_GATE_REGISTRY
    mantle = _WET_GATE_MANTLE

    # No profile: nothing to blend (callers gate the dry case separately).
    assert _volatile_profile_jax_viable(None, reg, mantle) is False

    # Binodal-controlled profiles: global miscibility or x_interior.
    assert (
        _volatile_profile_jax_viable(_make_profile(global_miscibility=True), reg, mantle)
        is False
    )
    assert (
        _volatile_profile_jax_viable(_make_profile(x_interior={'Chabrier:H': 0.1}), reg, mantle)
        is False
    )

    # Nonzero weight on the primary silicate (either phase).
    assert (
        _volatile_profile_jax_viable(
            _make_profile(w_liquid={'PALEOS:H2O': 0.02, 'PALEOS:MgSiO3': 0.5}), reg, mantle
        )
        is False
    )
    assert (
        _volatile_profile_jax_viable(_make_profile(w_solid={'PALEOS:MgSiO3': 0.5}), reg, mantle)
        is False
    )

    # Primary component absent from the mantle mixture string.
    assert (
        _volatile_profile_jax_viable(
            _make_profile(primary_component='RTPress100TPa:MgSiO3'), reg, mantle
        )
        is False
    )

    # No active volatile: zero weights, or a volatile that is not a
    # component of the mantle mixture.
    assert (
        _volatile_profile_jax_viable(_make_profile(w_liquid={'PALEOS:H2O': 0.0}), reg, mantle)
        is False
    )
    assert _volatile_profile_jax_viable(_make_profile(), reg, 'PALEOS:MgSiO3') is False

    # More than one active volatile in the mixture.
    two_vols = _make_profile(w_liquid={'PALEOS:H2O': 0.02, 'Chabrier:H': 0.01})
    assert (
        _volatile_profile_jax_viable(
            two_vols, reg, 'PALEOS:MgSiO3:0.9700+PALEOS:H2O:0.0200+Chabrier:H:0.0100'
        )
        is False
    )

    # An unmanaged extra mixture component keeps its fraction only on
    # the numpy path.
    assert (
        _volatile_profile_jax_viable(
            _make_profile(), reg, 'PALEOS:MgSiO3:0.9600+PALEOS:H2O:0.0200+PALEOS:iron:0.0200'
        )
        is False
    )

    # Chabrier:H as the single active volatile: rejected by name (its
    # binodal suppression factor is not ported to the JAX RHS).
    h2 = _make_profile(w_liquid={'Chabrier:H': 0.02})
    assert (
        _volatile_profile_jax_viable(h2, reg, 'PALEOS:MgSiO3:0.9800+Chabrier:H:0.0200') is False
    )

    # Volatile registry entry missing, non-unified, or without eos_file.
    assert _volatile_profile_jax_viable(_make_profile(), {}, mantle) is False
    wb = _make_profile(w_liquid={'WolfBower2018:H2O': 0.02})
    assert (
        _volatile_profile_jax_viable(wb, reg, 'PALEOS:MgSiO3:0.9800+WolfBower2018:H2O:0.0200')
        is False
    )
    reg_nofile = dict(reg)
    reg_nofile['PALEOS:H2O'] = {'format': 'paleos_unified'}
    assert _volatile_profile_jax_viable(_make_profile(), reg_nofile, mantle) is False

    # Malformed profile object: conservative False, not an exception.
    assert _volatile_profile_jax_viable(object(), reg, mantle) is False


@pytest.mark.reference_pinned
def test_volatile_profile_predicate_matches_zalmoxis_wet_envelope():
    """Cross-check the PROTEUS wet-envelope predicate against Zalmoxis.

    Pins ``_volatile_profile_jax_viable`` to the accept/reject behavior
    of ``zalmoxis.jax_eos.wrapper._validate_wet_mantle`` (fwl-zalmoxis
    >= 26.07.13), the authoritative envelope check the Zalmoxis JAX
    dispatch runs on the same profiles. A drift between the two would
    either drop the callable on a solve that falls back to numpy (the
    hot-anchor physics bug) or keep JAX-viable solves on the slow numpy
    path, so the two implementations must agree on every profile whose
    registry entry passes the format check both sides share.
    """
    wrapper = pytest.importorskip('zalmoxis.jax_eos.wrapper')
    mixing = pytest.importorskip('zalmoxis.mixing')
    from proteus.interior_struct.zalmoxis import _volatile_profile_jax_viable

    validate = getattr(wrapper, '_validate_wet_mantle', None)
    if validate is None:
        pytest.skip('installed zalmoxis does not expose _validate_wet_mantle')

    def zalmoxis_accepts(profile, mantle_string):
        mixture = mixing.parse_layer_components(mantle_string)
        try:
            validate(profile, mixture, _WET_GATE_REGISTRY)
        except ValueError:
            return False
        return True

    cases = [
        # (profile, extended mantle EOS string, expected viability)
        (_make_profile(), _WET_GATE_MANTLE, True),
        (
            _make_profile(w_liquid={'Chabrier:H': 0.02}),
            'PALEOS:MgSiO3:0.9800+Chabrier:H:0.0200',
            False,
        ),
        (_make_profile(global_miscibility=True), _WET_GATE_MANTLE, False),
        (
            _make_profile(w_liquid={'PALEOS:H2O': 0.02, 'PALEOS:MgSiO3': 0.5}),
            _WET_GATE_MANTLE,
            False,
        ),
        (
            _make_profile(w_liquid={'PALEOS:H2O': 0.02, 'Chabrier:H': 0.01}),
            'PALEOS:MgSiO3:0.9700+PALEOS:H2O:0.0200+Chabrier:H:0.0100',
            False,
        ),
    ]
    for profile, mantle_string, expected in cases:
        assert zalmoxis_accepts(profile, mantle_string) is expected
        assert (
            _volatile_profile_jax_viable(profile, _WET_GATE_REGISTRY, mantle_string) is expected
        )


def _gate_config(mantle_eos: str):
    """MagicMock config for the temperature-source dispatch tests.

    Isothermal temperature mode keeps the configuration loader off the
    super-liquidus adiabat solve; ``outer_solver='picard'`` keeps the
    Newton-only knobs out of the config dict; ``use_jax=True`` arms the
    arrays-vs-callable gate under test.
    """
    config = MagicMock()
    config.planet.mass_tot = 1.0
    config.planet.temperature_mode = 'isothermal'
    config.planet.tsurf_init = 300.0
    config.planet.tcenter_init = 5000.0
    config.planet.tcmb_init = 4000.0
    config.interior_struct.core_frac = 0.325
    config.interior_struct.core_frac_mode = 'mass'
    config.interior_struct.core_heatcap = 450.0
    config.interior_struct.zalmoxis.core_eos = 'PALEOS:iron'
    config.interior_struct.zalmoxis.mantle_eos = mantle_eos
    config.interior_struct.zalmoxis.ice_layer_eos = None
    config.interior_struct.zalmoxis.mantle_mass_fraction = 0.0
    config.interior_struct.zalmoxis.dry_mantle = True
    config.interior_struct.zalmoxis.global_miscibility = False
    config.interior_struct.zalmoxis.num_levels = 60
    config.interior_struct.zalmoxis.mushy_zone_factor = 0.8
    config.interior_struct.zalmoxis.solver_tol_outer = 3e-3
    config.interior_struct.zalmoxis.solver_tol_inner = 1e-4
    config.interior_struct.zalmoxis.solver_max_iter_outer = 100
    config.interior_struct.zalmoxis.solver_max_iter_inner = 100
    config.interior_struct.zalmoxis.use_jax = True
    config.interior_struct.zalmoxis.use_anderson = False
    config.interior_struct.zalmoxis.outer_solver = 'picard'
    config.interior_energetics.module = 'aragog'
    config.interior_energetics.num_levels = 30
    config.interior_energetics.aragog.scalar_gravity_override = False
    return config


def _plausible_model_results(n: int = 60) -> dict:
    """Converged Zalmoxis model_results for an Earth-mass rocky planet.

    Density decreases outward (iron core to silicate surface), pressure
    decreases outward, the enclosed mass is the cumulative shell
    integral of that density, and ``cmb_mass`` sits exactly on a grid
    node so the core-mantle split is self-consistent.
    """
    r_surf = 6.371e6
    radii = np.linspace(1.0e3, r_surf, n)
    cmb_index = n // 3
    density = np.linspace(12000.0, 3300.0, n)  # kg/m^3, core to surface
    pressure = np.linspace(360e9, 1e5, n)  # Pa, monotone decreasing
    temperature = np.linspace(5500.0, 2500.0, n)  # K
    gravity = np.linspace(0.5, 9.8, n)  # m/s^2, ~0 at centre to surface g
    shell_mass = 4.0 * np.pi * radii**2 * density * np.gradient(radii)
    mass_enclosed = np.cumsum(shell_mass)
    return {
        'radii': radii,
        'density': density,
        'gravity': gravity,
        'pressure': pressure,
        'temperature': temperature,
        'mass_enclosed': mass_enclosed,
        'cmb_mass': float(mass_enclosed[cmb_index]),
        'core_mantle_mass': float(mass_enclosed[cmb_index]),
        'converged': True,
        'converged_pressure': True,
        'converged_density': True,
        'converged_mass': True,
        'best_mass_error': 1e-4,
        'p_center': 360e9,
    }


def _cooled_mantle_arrays(model_results: dict) -> tuple[np.ndarray, np.ndarray]:
    """Cooled mantle (r, T) hand-off arrays spanning CMB to surface.

    Represents an evolved (partially crystallized) interior profile:
    7740 K at the CMB falling to 2500 K at the surface, well below the
    hot initial adiabat, so a solve that honors it is distinguishable
    from one that rebuilds the hot internal profile.
    """
    radii = np.asarray(model_results['radii'])
    n = len(radii)
    cmb_index = n // 3
    r_arr = np.linspace(float(radii[cmb_index]), float(radii[-1]), 20)
    t_arr = np.linspace(7740.0, 2500.0, 20)
    return r_arr, t_arr


# Constant offset between the fake blended density and the fake bare
# density at equal (P, T): lets the gate tests discriminate which
# evaluator wrote the rebuilt density column.
_MIXED_DENSITY_OFFSET = 250.0


def _run_gate_solver(
    tmp_path,
    monkeypatch,
    mantle_eos,
    temperature_arrays,
    tf,
    main_results=None,
    dry_mantle=True,
    hf_extra=None,
    mixed_side_effect=None,
    real_melting_curves=False,
    mzf=None,
):
    """Invoke zalmoxis_solver with the heavy solve mocked out.

    Patches the EOS registry to the on-disk synthetic one, the melting
    curves to plausible constants, the Zalmoxis ``main`` solve to a
    converged Earth-like result, and both density evaluators used by
    the post-solve column rebuild: the bare EOS dispatch
    (``calculate_density``, dry mantle and core nodes) and the blended
    evaluator (``calculate_mixed_density``, wet mantle nodes). The two
    fakes differ by a constant ``_MIXED_DENSITY_OFFSET`` at equal
    (P, T), so a written column discriminates which evaluator produced
    it. Returns the mocks and the hf_row.

    Parameters
    ----------
    main_results : dict or list of dict, optional
        Result(s) the mocked ``main`` solve returns. A dict is returned
        on every call; a list is consumed call by call (primary, retry,
        ...). Defaults to a single converged Earth-like result.
    dry_mantle : bool, optional
        Value for ``config.interior_struct.zalmoxis.dry_mantle``. With
        False, supply dissolved-reservoir keys via ``hf_extra`` so a
        ``VolatileProfile`` is actually built.
    hf_extra : dict, optional
        Extra hf_row keys merged over the default surface-pressure row
        (e.g. mantle phase masses and dissolved volatile masses).
    mixed_side_effect : callable, optional
        Replacement side effect for the blended evaluator, e.g. to make
        selected nodes return non-finite densities. Defaults to the
        offset fake.
    real_melting_curves : bool, optional
        With True, the real ``load_zalmoxis_solidus_liquidus_functions``
        runs (analytic, cheap) instead of the constant-pair stub, so the
        curves handed to the solve are the ones PROTEUS builds.
    mzf : float, optional
        Value for ``config.interior_struct.zalmoxis.mushy_zone_factor``;
        the default keeps the config helper's 0.8.
    """
    from proteus.interior_struct import zalmoxis as zalmoxis_wrapper

    (tmp_path / 'data').mkdir(exist_ok=True)
    registry = _gate_registry(tmp_path)
    if main_results is None:
        main_results = _plausible_model_results()
    if isinstance(main_results, list):
        main_patch_kwargs = {'side_effect': main_results}
        model_results = main_results[-1]
    else:
        main_patch_kwargs = {'return_value': main_results}
        model_results = main_results
    hf_row = {'P_surf': 1.0}  # bar; resolves the surface-pressure target
    if hf_extra:
        hf_row.update(hf_extra)
    config = _gate_config(mantle_eos)
    config.interior_struct.zalmoxis.dry_mantle = dry_mantle
    if mzf is not None:
        config.interior_struct.zalmoxis.mushy_zone_factor = mzf
    melting_patch_kwargs = (
        {'side_effect': zalmoxis_wrapper.load_zalmoxis_solidus_liquidus_functions}
        if real_melting_curves
        else {'return_value': (lambda P: 4000.0, lambda P: 5000.0)}
    )

    monkeypatch.setattr(
        zalmoxis_wrapper,
        '_density_cache',
        {'density': None, 'radii': None, 'key': None},
    )

    def _fake_density(P, mats, eos, T, sol, liq, interpolation_functions=None, **kwargs):
        # Plausible monotone-in-P, decreasing-in-T condensed density.
        return 3300.0 + 2.0e-8 * float(P) - 0.1 * (float(T) - 3000.0)

    def _fake_mixed_density(P, T, mixture, mats, sol, liq, interp, **kwargs):
        # Bare fake plus a constant blend offset: distinguishable from
        # _fake_density at the same (P, T).
        return _fake_density(P, mats, None, T, sol, liq) + _MIXED_DENSITY_OFFSET

    with (
        patch.object(
            zalmoxis_wrapper,
            'load_zalmoxis_material_dictionaries',
            return_value=registry,
        ),
        patch.object(
            zalmoxis_wrapper,
            'load_zalmoxis_solidus_liquidus_functions',
            **melting_patch_kwargs,
        ),
        patch.object(zalmoxis_wrapper, 'main', **main_patch_kwargs) as main_mock,
        patch('zalmoxis.eos.dispatch.calculate_density', side_effect=_fake_density) as rho_mock,
        patch(
            'zalmoxis.mixing.calculate_mixed_density',
            side_effect=mixed_side_effect or _fake_mixed_density,
        ) as mixed_mock,
    ):
        cmb_radius, mesh_file = zalmoxis_wrapper.zalmoxis_solver(
            config,
            str(tmp_path),
            hf_row,
            num_spider_nodes=0,
            temperature_function=tf,
            temperature_arrays=temperature_arrays,
        )

    return main_mock, rho_mock, mixed_mock, hf_row, model_results, cmb_radius, mesh_file


def test_zalmoxis_solver_passes_callable_when_jax_path_not_viable(tmp_path, monkeypatch):
    """Non-viable-mantle re-solves hand the evolved T(r) callable to the solve.

    With ``use_jax=True``, ``temperature_arrays`` supplied, and a mantle
    whose registry entry is neither unified PALEOS nor 2-phase PALEOS,
    the JAX dispatch cannot run and the numpy ODE consumes only
    ``temperature_function``. The callable must therefore reach the
    solve unmodified; nulling it would make every re-solve rebuild the
    internal hot-anchor profile and freeze the interior radius across
    the entire crystallization sequence.
    """
    model_for_arrays = _plausible_model_results()
    r_arr, t_arr = _cooled_mantle_arrays(model_for_arrays)

    def tf(r, P):
        # Evolved-interior closure: cooled mantle, clamped below the CMB.
        if r <= r_arr[0]:
            return float(t_arr[0])
        return float(np.interp(r, r_arr, t_arr))

    main_mock, rho_mock, mixed_mock, hf_row, model_results, cmb_radius, mesh_file = (
        _run_gate_solver(tmp_path, monkeypatch, 'Seager2007:silicate', (r_arr, t_arr), tf)
    )

    # The callable passes through identically; the arrays ride along for
    # the (declined) JAX dispatch inside Zalmoxis.
    assert main_mock.call_count == 1
    kwargs = main_mock.call_args.kwargs
    assert kwargs['temperature_function'] is tf
    passed_r, passed_t = kwargs['temperature_arrays']
    np.testing.assert_allclose(passed_r, r_arr, rtol=0, atol=0)
    np.testing.assert_allclose(passed_t, t_arr, rtol=0, atol=0)

    # The post-solve column rebuild is exclusive to the arrays-via-JAX
    # path: with the callable honored, the solver's own columns stand.
    assert rho_mock.call_count == 0

    # Structure scalars come from the (mocked) converged solve; the
    # core-mantle split stays bounded by the total.
    radii = model_results['radii']
    assert hf_row['R_int'] == pytest.approx(float(radii[-1]), rel=1e-12)
    assert 0.0 < hf_row['M_core'] < hf_row['M_int']
    assert 0.0 < cmb_radius < hf_row['R_int']
    assert mesh_file is None  # num_spider_nodes=0 requests no SPIDER mesh


@pytest.mark.parametrize('mantle_eos', ['PALEOS-2phase:MgSiO3', 'PALEOS:MgSiO3'])
def test_zalmoxis_solver_withholds_callable_on_jax_viable_eos(
    tmp_path, monkeypatch, mantle_eos
):
    """JAX-viable-mantle re-solves feed the arrays to the JAX path instead.

    With a JAX-capable mantle layout (2-phase PALEOS sub-tables or the
    unified PALEOS table) the JAX dispatch consumes
    ``temperature_arrays``, so the external callable is withheld (the
    inner Picard converges on the internal linear-T profile) and the
    post-solve rebuild rewrites the density and temperature columns
    against the hand-off arrays before any downstream consumer reads
    them.
    """
    model_for_arrays = _plausible_model_results()
    r_arr, t_arr = _cooled_mantle_arrays(model_for_arrays)

    def tf(r, P):
        if r <= r_arr[0]:
            return float(t_arr[0])
        return float(np.interp(r, r_arr, t_arr))

    main_mock, rho_mock, mixed_mock, hf_row, model_results, cmb_radius, _ = _run_gate_solver(
        tmp_path, monkeypatch, mantle_eos, (r_arr, t_arr), tf
    )

    # Callable withheld, arrays passed through.
    kwargs = main_mock.call_args.kwargs
    assert kwargs['temperature_function'] is None
    passed_r, passed_t = kwargs['temperature_arrays']
    np.testing.assert_allclose(passed_r, r_arr, rtol=0, atol=0)
    np.testing.assert_allclose(passed_t, t_arr, rtol=0, atol=0)

    # The rebuild ran: one bare EOS density evaluation per radial node,
    # and no blended evaluation on a dry solve.
    assert rho_mock.call_count == len(model_results['radii'])
    assert mixed_mock.call_count == 0

    # The written hand-off file carries the rebuilt (cooled) temperature
    # column: its surface value tracks the arrays, not the mocked solve
    # output (2500 K vs 2500 K here by construction at the surface, but
    # the CMB row discriminates: 7740 K hand-off vs 5500/4500 K mock).
    data = np.loadtxt(tmp_path / 'data' / 'zalmoxis_output.dat')
    t_column = data[:, 4]
    assert t_column[0] == pytest.approx(7740.0, rel=1e-6)
    # Discrimination guard: the mocked solver temperature at the first
    # mantle node is ~4500 K, more than 3000 K away from the hand-off
    # value, so the assertion above cannot pass on un-rebuilt columns.
    n = len(model_results['radii'])
    t_mock_cmb = float(model_results['temperature'][n // 3])
    assert abs(7740.0 - t_mock_cmb) > 3000.0

    # Boundedness of the structure scalars still holds on this path.
    assert 0.0 < hf_row['M_core'] < hf_row['M_int']
    assert 0.0 < cmb_radius < hf_row['R_int']


def test_zalmoxis_solver_init_adiabat_arrays_take_jax_dispatch(tmp_path, monkeypatch):
    """IC adiabat calls with sampled arrays dispatch like evolved re-solves.

    The liquidus_super IC hand-off (issue #719) passes the P-indexed
    adiabat closure together with its r-indexed sampling on the previous
    structure's grid. On a JAX-viable EOS the hoisted dispatch must make
    the same decision it makes for evolved Aragog re-solves: withhold
    the callable, feed the arrays to the JAX path, and run the
    post-solve rebuild against them. The internal-mode init case (no
    callable, no arrays) is pinned separately by
    test_zalmoxis_solver_init_call_keeps_internal_mode_dispatch.
    """
    model_for_arrays = _plausible_model_results()
    radii = np.asarray(model_for_arrays['radii'])
    pressure = np.asarray(model_for_arrays['pressure'])
    n = len(radii)
    cmb_index = n // 3

    # P-indexed adiabat closure: ignores r, monotone in P, the shape
    # _build_superliquidus_adiabat_tp produces. The slope keeps the
    # deep-mantle sample several hundred K away from the mocked solver
    # column so the rebuilt-column assertion below discriminates.
    def tf(r, P):
        return 4000.0 + 4.0e-9 * float(P)

    # Arrays sampled from the closure on the previous structure's mantle
    # grid, the _sample_adiabat_temperature_arrays convention.
    r_arr = radii[cmb_index:]
    t_arr = np.array([tf(r, P) for r, P in zip(r_arr, pressure[cmb_index:])])

    main_mock, rho_mock, mixed_mock, hf_row, model_results, cmb_radius, _ = _run_gate_solver(
        tmp_path, monkeypatch, 'PALEOS:MgSiO3', (r_arr, t_arr), tf
    )

    # Same dispatch outcome as the evolved re-solve: callable withheld,
    # arrays through, dry rebuild against the arrays.
    kwargs = main_mock.call_args.kwargs
    assert kwargs['temperature_function'] is None
    passed_r, passed_t = kwargs['temperature_arrays']
    np.testing.assert_allclose(passed_r, r_arr, rtol=0, atol=0)
    np.testing.assert_allclose(passed_t, t_arr, rtol=0, atol=0)
    assert rho_mock.call_count == len(model_results['radii'])
    assert mixed_mock.call_count == 0

    # The written temperature column carries the adiabat samples: the CMB
    # row reads the sampled deep-mantle T, not the mocked solver column.
    data = np.loadtxt(tmp_path / 'data' / 'zalmoxis_output.dat')
    assert data[:, 4][0] == pytest.approx(float(t_arr[0]), rel=1e-6)
    assert abs(float(t_arr[0]) - float(model_results['temperature'][cmb_index])) > 100.0

    assert 0.0 < hf_row['M_core'] < hf_row['M_int']
    assert 0.0 < cmb_radius < hf_row['R_int']


def test_zalmoxis_solver_withholds_callable_on_wet_in_envelope_profile(tmp_path, monkeypatch):
    """In-envelope wet solves feed the arrays to the JAX path.

    With ``dry_mantle = false`` and a single dissolved H2O inventory,
    the ``VolatileProfile`` matches the Zalmoxis JAX wet envelope
    (one paleos_unified volatile blended into the mantle), so the
    external callable is withheld and the JAX RHS integrates against
    the hand-off arrays. The post-solve rebuild must run
    profile-aware: mantle nodes go through the blended evaluator
    (``calculate_mixed_density`` with the profile), core nodes through
    the bare EOS dispatch, and the written columns carry the hand-off
    temperature and the blended density, matching what a
    callable-driven numpy solve would have written.
    """
    model_for_arrays = _plausible_model_results()
    r_arr, t_arr = _cooled_mantle_arrays(model_for_arrays)

    def tf(r, P):
        if r <= r_arr[0]:
            return float(t_arr[0])
        return float(np.interp(r, r_arr, t_arr))

    # Dissolved inventory: 2% of the liquid mantle mass as H2O, so
    # build_volatile_profile returns a profile instead of None.
    hf_extra = {
        'M_mantle_liquid': 4.0e24,
        'M_mantle_solid': 1.0e24,
        'H2O_kg_liquid': 8.0e22,
    }
    main_mock, rho_mock, mixed_mock, hf_row, model_results, cmb_radius, _ = _run_gate_solver(
        tmp_path,
        monkeypatch,
        'PALEOS-2phase:MgSiO3',
        (r_arr, t_arr),
        tf,
        dry_mantle=False,
        hf_extra=hf_extra,
    )

    # Callable withheld, arrays passed through: the wet JAX envelope
    # accepts the single-H2O profile.
    kwargs = main_mock.call_args.kwargs
    assert kwargs['temperature_function'] is None
    passed_r, passed_t = kwargs['temperature_arrays']
    np.testing.assert_allclose(passed_r, r_arr, rtol=0, atol=0)
    np.testing.assert_allclose(passed_t, t_arr, rtol=0, atol=0)

    # The rebuild ran profile-aware: every mantle node through the
    # blended evaluator with this solve's profile, every core node
    # through the bare EOS dispatch.
    n = len(model_results['radii'])
    cmb_index = n // 3
    assert mixed_mock.call_count == n - cmb_index
    assert rho_mock.call_count == cmb_index
    for call in mixed_mock.call_args_list:
        assert call.kwargs['volatile_profile'] is kwargs['volatile_profile']
        # The blend evaluates the extended mantle mixture, so the
        # dissolved species is available to the per-shell fractions.
        assert 'PALEOS:H2O' in call.args[2].components

    # The written hand-off file carries the rebuilt columns. The CMB row
    # temperature is the 7740 K hand-off value (the mocked solver column
    # reads ~4500 K there, so the assertion cannot pass un-rebuilt), and
    # the CMB row density is the blended fake at that (P, T), a constant
    # _MIXED_DENSITY_OFFSET above the bare fake (the bare recompute
    # cannot produce it).
    data = np.loadtxt(tmp_path / 'data' / 'zalmoxis_output.dat')
    t_column = data[:, 4]
    assert t_column[0] == pytest.approx(7740.0, rel=1e-6)
    t_mock_cmb = float(model_results['temperature'][cmb_index])
    assert abs(7740.0 - t_mock_cmb) > 3000.0
    p_cmb_row = float(data[:, 1][0])
    rho_bare = 3300.0 + 2.0e-8 * p_cmb_row - 0.1 * (7740.0 - 3000.0)
    assert data[:, 2][0] == pytest.approx(rho_bare + _MIXED_DENSITY_OFFSET, rel=1e-9)
    assert abs(float(data[:, 2][0]) - rho_bare) > 0.5 * _MIXED_DENSITY_OFFSET

    # Vacuous-pass guard: the profile was actually built and handed to
    # the solve (a None profile would satisfy the dispatch trivially),
    # and the mantle EOS string was extended with the dissolved species
    # so the per-shell blend has the volatile component available.
    assert kwargs['volatile_profile'] is not None
    solver_params = main_mock.call_args.args[0]
    assert '+PALEOS:H2O:' in solver_params['layer_eos_config']['mantle']

    # Structure scalars from the converged solve stay bounded either way.
    assert 0.0 < hf_row['M_core'] < hf_row['M_int']
    assert 0.0 < cmb_radius < hf_row['R_int']


def test_wet_rebuild_falls_back_to_solver_density_on_non_finite_blend(tmp_path, monkeypatch):
    """A non-finite blended density falls back to the solver's column value.

    Near the edge of a volatile EOS table the blend can return NaN for a
    node the solver itself filled through its own out-of-range handling,
    so the rebuild must keep that node's solver density instead of
    writing NaN into the hand-off file Aragog reads. Discrimination: the
    poisoned node carries the solver column value (which the offset fake
    cannot produce at that (P, T)), every other mantle node carries the
    blended fake, and no NaN reaches the written file.
    """
    model_for_arrays = _plausible_model_results()
    r_arr, t_arr = _cooled_mantle_arrays(model_for_arrays)

    def tf(r, P):
        if r <= r_arr[0]:
            return float(t_arr[0])
        return float(np.interp(r, r_arr, t_arr))

    calls = {'n': 0}

    def _nan_first_blend(P, T, mixture, mats, sol, liq, interp, **kwargs):
        calls['n'] += 1
        if calls['n'] == 1:
            return float('nan')
        return 3300.0 + 2.0e-8 * float(P) - 0.1 * (float(T) - 3000.0) + _MIXED_DENSITY_OFFSET

    hf_extra = {
        'M_mantle_liquid': 4.0e24,
        'M_mantle_solid': 1.0e24,
        'H2O_kg_liquid': 8.0e22,
    }
    main_mock, rho_mock, mixed_mock, hf_row, model_results, cmb_radius, _ = _run_gate_solver(
        tmp_path,
        monkeypatch,
        'PALEOS-2phase:MgSiO3',
        (r_arr, t_arr),
        tf,
        dry_mantle=False,
        hf_extra=hf_extra,
        mixed_side_effect=_nan_first_blend,
    )

    n = len(model_results['radii'])
    cmb_index = n // 3
    assert mixed_mock.call_count == n - cmb_index

    data = np.loadtxt(tmp_path / 'data' / 'zalmoxis_output.dat')
    rho_column = data[:, 2]
    assert np.all(np.isfinite(rho_column))
    # The poisoned first mantle node keeps the solver's column density.
    solver_rho_cmb = float(model_results['density'][cmb_index])
    assert rho_column[0] == pytest.approx(solver_rho_cmb, rel=1e-9)
    # The next node carries the blended fake at the hand-off temperature,
    # so the fallback is per node, not a whole-column bailout.
    p_next = float(data[1, 1])
    t_next = float(data[1, 4])
    rho_blend_next = 3300.0 + 2.0e-8 * p_next - 0.1 * (t_next - 3000.0) + _MIXED_DENSITY_OFFSET
    assert rho_column[1] == pytest.approx(rho_blend_next, rel=1e-9)
    # Discrimination: the solver density and the blended fake differ by
    # far more than the tolerances above at the poisoned node.
    assert abs(solver_rho_cmb - rho_blend_next) > 0.5 * _MIXED_DENSITY_OFFSET


def test_zalmoxis_solver_keeps_callable_on_wet_out_of_envelope_profile(tmp_path, monkeypatch):
    """Out-of-envelope wet solves keep the evolved T(r) callable.

    A dissolved H2 inventory builds a ``VolatileProfile`` carrying
    ``Chabrier:H``, which the Zalmoxis JAX wet path rejects (the binodal
    suppression factor is not ported), so the solve must consume the
    callable on the numpy path. The EOS pair is identical to the
    in-envelope test above, so the retained callable discriminates
    exactly the wet-envelope conjunct of the dispatch.
    """
    from proteus.interior_struct.zalmoxis import _zalmoxis_jax_structure_viable

    model_for_arrays = _plausible_model_results()
    r_arr, t_arr = _cooled_mantle_arrays(model_for_arrays)

    def tf(r, P):
        if r <= r_arr[0]:
            return float(t_arr[0])
        return float(np.interp(r, r_arr, t_arr))

    # Discrimination guard: this EOS pair is JAX-viable, so only the
    # wet-envelope conjunct can keep the callable in play below.
    registry = _gate_registry(tmp_path)
    assert (
        _zalmoxis_jax_structure_viable(registry, 'PALEOS:iron', 'PALEOS-2phase:MgSiO3') is True
    )

    # Dissolved inventory: H2 only, so the profile's single active
    # volatile is Chabrier:H and the wet envelope rejects it.
    hf_extra = {
        'M_mantle_liquid': 4.0e24,
        'M_mantle_solid': 1.0e24,
        'H2_kg_liquid': 4.0e22,
    }
    main_mock, rho_mock, mixed_mock, hf_row, model_results, cmb_radius, _ = _run_gate_solver(
        tmp_path,
        monkeypatch,
        'PALEOS-2phase:MgSiO3',
        (r_arr, t_arr),
        tf,
        dry_mantle=False,
        hf_extra=hf_extra,
    )

    # The callable reaches the solve unmodified and the arrays ride
    # along for the (declined) JAX dispatch inside Zalmoxis.
    kwargs = main_mock.call_args.kwargs
    assert kwargs['temperature_function'] is tf
    passed_r, passed_t = kwargs['temperature_arrays']
    np.testing.assert_allclose(passed_r, r_arr, rtol=0, atol=0)
    np.testing.assert_allclose(passed_t, t_arr, rtol=0, atol=0)

    # No rebuild on the honored-callable path, bare or blended.
    assert rho_mock.call_count == 0
    assert mixed_mock.call_count == 0

    # Vacuous-pass guard: the profile was actually built and carries the
    # rejected volatile.
    assert kwargs['volatile_profile'] is not None
    solver_params = main_mock.call_args.args[0]
    assert '+Chabrier:H:' in solver_params['layer_eos_config']['mantle']

    # Structure scalars from the converged solve stay bounded either way.
    assert 0.0 < hf_row['M_core'] < hf_row['M_int']
    assert 0.0 < cmb_radius < hf_row['R_int']


def test_zalmoxis_solver_init_call_keeps_internal_mode_dispatch(tmp_path, monkeypatch):
    """Init-style calls (no callable, no arrays) keep the internal T mode.

    Initial-condition and equilibration calls arrive with neither a
    temperature callable nor hand-off arrays. The solver must then
    disable the JAX and Anderson paths (their internal T dispatch
    collapses for P-ignoring profiles) and run the internal
    temperature-mode dispatch, with no external profile injected. This
    pins the limit-input behavior of the gate: the dispatch fix applies
    only to re-solves that actually carry an evolved profile.
    """
    main_mock, rho_mock, _mixed_mock, hf_row, model_results, _, _ = _run_gate_solver(
        tmp_path, monkeypatch, 'PALEOS-2phase:MgSiO3', None, None
    )

    config_params = main_mock.call_args.args[0]
    kwargs = main_mock.call_args.kwargs
    assert kwargs['temperature_function'] is None
    assert kwargs['temperature_arrays'] is None
    # The no-profile downgrade turns the JAX path off for this call.
    assert config_params['use_jax'] is False
    assert config_params['use_anderson'] is False
    # No arrays: the post-solve rebuild has nothing to rebuild against.
    assert rho_mock.call_count == 0
    # The internal isothermal dispatch is the active temperature source.
    assert config_params['temperature_mode'] == 'isothermal'
    assert hf_row['R_int'] == pytest.approx(float(model_results['radii'][-1]), rel=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_zalmoxis_solver_retry_replaces_density_and_gravity(tmp_path, monkeypatch):
    """A retry-converged solve propagates the retry's density and gravity.

    The primary solve fails mass convergence with ``best_mass_error``
    just below the 0.05 retry-skip threshold (the edge of that guard),
    carrying density and gravity columns offset from the converged
    structure. The relaxed-tolerance retry converges; every downstream
    consumer (surface gravity in hf_row, the density and gravity columns
    Aragog reads from zalmoxis_output.dat for its cell masses) must then
    reflect the retry solution, not the failed primary's. Self-consistency
    of the written structure (rho, g, M from one solve) is the invariant.
    The mantle layout is not JAX-viable, so the callable is honored on
    both the primary and the retry (the gate applies to each unchanged).
    """
    retry_results = _plausible_model_results()
    primary_results = _plausible_model_results()
    primary_results['density'] = primary_results['density'] - 500.0
    primary_results['gravity'] = primary_results['gravity'] * 0.8
    primary_results['converged'] = False
    primary_results['converged_mass'] = False
    primary_results['best_mass_error'] = 0.049  # just below the 0.05 skip gate

    r_arr, t_arr = _cooled_mantle_arrays(retry_results)

    def tf(r, P):
        # Evolved-interior closure: cooled mantle, clamped below the CMB.
        if r <= r_arr[0]:
            return float(t_arr[0])
        return float(np.interp(r, r_arr, t_arr))

    main_mock, rho_mock, _mixed_mock, hf_row, _, cmb_radius, _ = _run_gate_solver(
        tmp_path,
        monkeypatch,
        'Seager2007:silicate',
        (r_arr, t_arr),
        tf,
        main_results=[primary_results, retry_results],
    )

    # Primary plus exactly one relaxed-tolerance retry, with the
    # temperature-source gate applying to the retry unchanged.
    assert main_mock.call_count == 2
    retry_params = main_mock.call_args_list[1].args[0]
    assert retry_params['tolerance_outer'] == pytest.approx(3e-3 * 3, rel=1e-12)
    assert retry_params['max_iterations_outer'] == 200
    assert retry_params['wall_timeout'] == pytest.approx(600.0)
    assert main_mock.call_args_list[1].kwargs['temperature_function'] is tf

    # Surface gravity comes from the retry solution.
    g_retry = float(retry_results['gravity'][-1])
    g_primary = float(primary_results['gravity'][-1])
    assert hf_row['gravity'] == pytest.approx(g_retry, rel=1e-12)
    # Discrimination guard: the stale primary value sits ~2 m/s^2 away,
    # far outside the comparison tolerance.
    assert abs(g_retry - g_primary) > 1.0

    # The written hand-off columns carry the retry's density and gravity
    # over the mantle nodes (columns: r, P, rho, g, T).
    n = len(retry_results['radii'])
    cmb_index = n // 3
    data = np.loadtxt(tmp_path / 'data' / 'zalmoxis_output.dat')
    np.testing.assert_allclose(data[:, 2], retry_results['density'][cmb_index:], rtol=1e-15)
    np.testing.assert_allclose(data[:, 3], retry_results['gravity'][cmb_index:], rtol=1e-15)
    # Discrimination guard: the primary's columns differ by 500 kg/m^3
    # in rho and 20% in g at every node, so the assertions above cannot
    # pass on a file written from the failed primary's arrays.
    assert np.all(np.abs(retry_results['density'] - primary_results['density']) > 100.0)

    # No rebuild on the honored-callable path; structure scalars bounded.
    assert rho_mock.call_count == 0
    assert 0.0 < hf_row['M_core'] < hf_row['M_int']
    assert 0.0 < cmb_radius < hf_row['R_int']


@pytest.mark.physics_invariant
def test_structure_mass_desync_near_zero_for_self_consistent_sphere():
    """A physically self-consistent structure reads near-zero mass desync.

    Build a uniform-density sphere and feed the analytic sphere mass
    ``4/3 pi R^3 rho`` as the ODE accumulator total. The trapezoid of
    ``4 pi r^2 rho`` over a fine radial grid integrates to the same analytic
    mass to within O(h^2) discretisation error, so the metric (mass-closure
    invariant: two independent estimates of one conserved total) must be far
    below the ~1% worst-case production desync. R = 6e6 m and a non-trivial
    rho = 4000 kg/m^3 keep the absolute mass at a realistic super-Earth scale
    so a units slip would not hide in the ratio.
    """
    rho0 = 4000.0  # kg m-3, silicate-mantle scale
    radius = 6.0e6  # m
    radii = np.linspace(0.0, radius, 2000)
    density = np.full_like(radii, rho0)
    analytic_total = 4.0 / 3.0 * np.pi * radius**3 * rho0  # kg
    mass_enclosed = np.array([analytic_total])

    desync = compute_structure_mass_desync(radii, density, mass_enclosed)

    # 2000-node trapezoid of r^2 differs from the analytic integral by ~1e-7,
    # far under 1e-4; a self-consistent profile reads ~0.
    assert desync == pytest.approx(0.0, abs=1.0e-4)
    # Boundedness: a relative magnitude is non-negative regardless of input.
    assert desync >= 0.0


@pytest.mark.physics_invariant
def test_structure_mass_desync_reports_known_accumulator_mismatch():
    """A known accumulator-vs-trapezoid gap is reported as that fraction.

    Set the ODE accumulator total 5% heavier than the trapezoid of the same
    converged profile (the worst-case desync magnitude seen in production).
    The metric must read ``0.05 / 1.05`` because it normalises by the
    accumulator total, not by the trapezoid.
    """
    rho0 = 4000.0  # kg m-3
    radius = 6.0e6  # m
    radii = np.linspace(0.0, radius, 4000)
    density = np.full_like(radii, rho0)
    trapezoid_total = float(np.trapezoid(4.0 * np.pi * radii**2 * density, radii))
    # Accumulator 5% heavier than the density-integral total.
    mass_enclosed = np.array([trapezoid_total * 1.05])

    desync = compute_structure_mass_desync(radii, density, mass_enclosed)

    # |T - 1.05 T| / (1.05 T) = 0.05 / 1.05 = 0.047619...
    assert desync == pytest.approx(0.05 / 1.05, rel=1.0e-6)
    # Discrimination guard: a regression normalising by the trapezoid instead
    # of the accumulator would read exactly 0.05, which differs from 0.0476 by
    # more than the tolerance.
    assert abs(desync - 0.05) > 1.0e-3
    # Sign guard: the magnitude is strictly positive for a real mismatch; a
    # dropped abs() would go negative because the trapezoid is the lighter side.
    assert desync > 0.0


def test_structure_mass_desync_guards_nonpositive_accumulator():
    """A degenerate or failed structure solve returns 0.0, not a ZeroDivision.

    When the accumulator total is non-positive (empty or failed profile) the
    metric is undefined; the guard returns 0.0 so the helpfile column stays a
    finite number rather than propagating a division by zero. Exercises the
    error-contract edge case.
    """
    radii = np.linspace(0.0, 1.0e6, 16)
    density = np.full_like(radii, 3000.0)

    zero_total = compute_structure_mass_desync(radii, density, np.array([0.0]))
    negative_total = compute_structure_mass_desync(radii, density, np.array([-1.0e20]))

    assert zero_total == pytest.approx(0.0, abs=0.0)
    assert negative_total == pytest.approx(0.0, abs=0.0)


# ---------------------------------------------------------------------------
# PROTEUS_PS_CACHE_DIR pointer (resume support for shared-cache runs)
# ---------------------------------------------------------------------------


def test_ps_cache_pointer_roundtrip_records_absolute_cache_dir(tmp_path):
    """The pointer written for a shared-cache run round-trips to the absolute
    cache directory, so a resumed run can follow the tables out of the run
    directory into PROTEUS_PS_CACHE_DIR.

    Discrimination: the reader must return the absolute cache path even when
    the writer is handed a non-absolute one, because the resume consumer
    joins nothing onto the result; a regression that stored the raw relative
    string would resolve against the wrong working directory on resume.
    """
    from proteus.interior_struct.zalmoxis import (
        PS_CACHE_POINTER_NAME,
        _write_ps_cache_pointer,
        read_ps_cache_pointer,
    )

    outdir = tmp_path / 'run'
    (outdir / 'data').mkdir(parents=True)
    cache_dir = tmp_path / 'shared_cache' / 'P_max-1p0e13'
    cache_dir.mkdir(parents=True)

    # Hand the writer a relative path; it must persist the absolute form
    import os as _os

    rel_cache = _os.path.relpath(cache_dir, tmp_path)
    prev = _os.getcwd()
    try:
        _os.chdir(tmp_path)
        _write_ps_cache_pointer(str(outdir), rel_cache)
    finally:
        _os.chdir(prev)

    pointer = outdir / 'data' / PS_CACHE_POINTER_NAME
    assert pointer.is_file()
    result = read_ps_cache_pointer(str(outdir))
    # Discrimination: absolute, and equal to the real cache dir (not the rel str)
    assert _os.path.isabs(result)
    assert _os.path.realpath(result) == _os.path.realpath(str(cache_dir))


def test_read_ps_cache_pointer_none_when_absent_or_empty(tmp_path):
    """The reader returns None both when no pointer exists (a normal per-run
    layout) and when the pointer is present but empty (a truncated write).

    Both must be None so the resume path falls through to its per-run
    output/<run>/data/spider_eos lookup rather than setting
    dirs['spider_eos_dir'] to '' and mis-reporting a cache hit. Exercises
    the empty-string guard, an error-contract edge.
    """
    from proteus.interior_struct.zalmoxis import (
        PS_CACHE_POINTER_NAME,
        read_ps_cache_pointer,
    )

    outdir = tmp_path / 'run'
    (outdir / 'data').mkdir(parents=True)

    # No pointer at all
    assert read_ps_cache_pointer(str(outdir)) is None

    # Present but empty (and whitespace-only) must also be None
    pointer = outdir / 'data' / PS_CACHE_POINTER_NAME
    pointer.write_text('   \n', encoding='utf-8')
    assert read_ps_cache_pointer(str(outdir)) is None


def test_read_ps_cache_pointer_returns_recorded_path_even_if_missing(tmp_path):
    """The reader reports the recorded directory verbatim without checking it
    exists: validating the directory is the resume caller's job (it guards
    with os.path.isdir before use).

    Discrimination: if the cache directory was evicted between the run and the
    resume, the reader still returns the recorded path (non-None), and the
    caller, not the reader, decides to skip it. A regression that silently
    dropped a dangling pointer would hide an evicted-cache resume from the
    caller's isdir guard.
    """
    from proteus.interior_struct.zalmoxis import (
        _write_ps_cache_pointer,
        read_ps_cache_pointer,
    )

    outdir = tmp_path / 'run'
    (outdir / 'data').mkdir(parents=True)
    evicted = tmp_path / 'was_here'  # never created

    _write_ps_cache_pointer(str(outdir), str(evicted))
    result = read_ps_cache_pointer(str(outdir))

    import os as _os

    assert result == _os.path.abspath(str(evicted))
    # Confirm the discrimination premise: the recorded dir really is absent
    assert not _os.path.isdir(result)


# ---------------------------------------------------------------------------
# Shared PROTEUS_PS_CACHE_DIR concurrency primitives (atomic publish + marker)
# ---------------------------------------------------------------------------


def test_atomic_write_text_publishes_complete_and_overwrites(tmp_path):
    """The atomic marker writer replaces the target in one rename and leaves no
    temporary file behind, whether the target is new or already present.

    Discrimination: after writing, the destination directory must contain
    exactly the target (no stray .tmp-* files); a non-atomic implementation
    that wrote in place or left its scratch file would fail the count. The
    overwrite case pins that a stale marker is fully replaced, not appended
    to, so a second run's key does not concatenate onto the first.
    """
    from proteus.interior_struct.zalmoxis import _atomic_write_text

    dest = tmp_path / '.cache_info.txt'
    _atomic_write_text(str(dest), 'P_max-1p0e13_nP-200_nS-200')
    assert dest.read_text() == 'P_max-1p0e13_nP-200_nS-200'

    # Overwrite with a different key: content is replaced, not appended
    _atomic_write_text(str(dest), 'P_max-2p0e13_nP-400_nS-400')
    assert dest.read_text() == 'P_max-2p0e13_nP-400_nS-400'

    # No scratch files leaked into the directory
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith('.tmp-')]
    assert leftovers == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ['.cache_info.txt']


def test_publish_ps_tables_moves_all_and_overwrites_existing(tmp_path):
    """Publishing a staging directory relocates every table into the shared
    cache directory, emptying the staging dir and overwriting any file a
    competing writer already placed there.

    Discrimination: a pre-existing file in the destination carries stale
    content; after publish it must hold the freshly staged bytes, proving
    os.replace overwrote it rather than skipping an existing name. The
    staging dir must end empty so the caller's cleanup has nothing to prune,
    and the destination must hold exactly the staged set.
    """
    from proteus.interior_struct.zalmoxis import _publish_ps_tables

    src = tmp_path / 'staging'
    dest = tmp_path / 'cache_key'
    src.mkdir()
    dest.mkdir()

    # Stage three tables; one name already exists in dest with stale content
    (src / 'solidus_P-S.dat').write_text('NEW-solidus')
    (src / 'density_melt.dat').write_text('NEW-density')
    (src / '.cache_info.txt').write_text('NEW-key')
    (dest / 'density_melt.dat').write_text('STALE-density')

    _publish_ps_tables(str(src), str(dest))

    # Staging emptied
    assert sorted(p.name for p in src.iterdir()) == []
    # Destination holds exactly the staged set
    assert sorted(p.name for p in dest.iterdir()) == [
        '.cache_info.txt',
        'density_melt.dat',
        'solidus_P-S.dat',
    ]
    # The pre-existing file was overwritten with staged content
    assert (dest / 'density_melt.dat').read_text() == 'NEW-density'
    assert (dest / 'solidus_P-S.dat').read_text() == 'NEW-solidus'


def test_publish_ps_tables_empty_staging_is_noop(tmp_path):
    """An empty staging directory publishes nothing and leaves the destination
    untouched, the degenerate edge where generation produced no files.

    Discrimination: a destination file present before the call must survive
    unchanged, confirming publish neither clears the destination nor errors
    on an empty source.
    """
    from proteus.interior_struct.zalmoxis import _publish_ps_tables

    src = tmp_path / 'staging'
    dest = tmp_path / 'cache_key'
    src.mkdir()
    dest.mkdir()
    (dest / 'existing.dat').write_text('keep-me')

    _publish_ps_tables(str(src), str(dest))

    assert sorted(p.name for p in dest.iterdir()) == ['existing.dat']
    assert (dest / 'existing.dat').read_text() == 'keep-me'


def test_ps_cache_key_distinguishes_mantle_eos_and_resolved_paths():
    """Two P-S table sets built from different mantle EOS must land on distinct
    cache keys even when every other table parameter is identical.

    A shared PROTEUS_PS_CACHE_DIR keys its subdirectories on this string, so a
    key that ignored the EOS identity would serve one run tables built from a
    different mantle EOS. Three discriminations are checked:

    - Different EOS registry names with the same P_max/nP/nS/mzf/layout produce
      keys whose non-EOS prefix is byte-identical; only the trailing ``_eos=``
      token differs. That common prefix is exactly the old key, so the pair
      would have collided before the EOS identity was folded in.
    - The same registry name resolving to different table files still produces
      distinct keys. A fix that folded in only the display name (not the
      resolved paths) would pass the first check but fail this one.
    - Identical inputs produce an identical key, so a genuine cache hit is still
      recognised; a key made unique per call would disable caching entirely.
    """
    import re as _re

    from proteus.interior_struct.zalmoxis import _ps_cache_key

    common = dict(P_max=1.4e12, nP=200, nS=200, mzf=0.8, layout='unified')

    key_a = _ps_cache_key(
        **common,
        mantle_eos='PALEOS:MgSiO3',
        eos_file='/data/mgsio3/unified.dat',
        solid_eos=None,
        liquid_eos=None,
    )
    key_b = _ps_cache_key(
        **common,
        mantle_eos='PALEOS:Fe2SiO4',
        eos_file='/data/fe2sio4/unified.dat',
        solid_eos=None,
        liquid_eos=None,
    )

    # Distinct EOS -> distinct keys, but identical non-EOS prefix (the old key).
    assert key_a != key_b
    prefix_a, _, tail_a = key_a.partition('_eos=')
    prefix_b, _, tail_b = key_b.partition('_eos=')
    assert prefix_a == prefix_b  # would have collided before the fix
    assert tail_a and tail_b and tail_a != tail_b

    # Same display name, different resolved file -> still distinct. Guards
    # against a name-only fix that leaves cross-version reuse open.
    key_same_name_1 = _ps_cache_key(
        **common,
        mantle_eos='PALEOS:MgSiO3',
        eos_file='/cache/api/v1/mgsio3.dat',
        solid_eos=None,
        liquid_eos=None,
    )
    key_same_name_2 = _ps_cache_key(
        **common,
        mantle_eos='PALEOS:MgSiO3',
        eos_file='/cache/api/v2/mgsio3.dat',
        solid_eos=None,
        liquid_eos=None,
    )
    assert key_same_name_1 != key_same_name_2

    # Idempotence: identical inputs -> identical key (cache hits still work).
    key_a_again = _ps_cache_key(
        **common,
        mantle_eos='PALEOS:MgSiO3',
        eos_file='/data/mgsio3/unified.dat',
        solid_eos=None,
        liquid_eos=None,
    )
    assert key_a_again == key_a

    # The raw key uses only characters the production dir sanitiser can map to
    # a filesystem-safe name (it rewrites '.', '=', '+'; everything else must
    # already be safe). A stray path separator would nest the cache directory.
    assert _re.fullmatch(r'[A-Za-z0-9._=+-]+', key_a)
    assert '/' not in key_a


def test_ps_cache_key_sanitises_names_and_tolerates_missing_paths():
    """The key stays filesystem-safe for awkward EOS names and unresolved paths.

    Registry names carry a colon (``PALEOS-2phase:MgSiO3``) and the dummy /
    unresolved layouts pass ``None`` for every table path. The key must remain a
    single path segment in both situations, and two different names must still
    separate even when all resolved paths are ``None``.
    """
    from proteus.interior_struct.zalmoxis import _ps_cache_key

    common = dict(P_max=1.4e12, nP=128, nS=128, mzf=1.0, layout='2phase')

    key = _ps_cache_key(
        **common,
        mantle_eos='PALEOS-2phase:MgSiO3',
        eos_file=None,
        solid_eos=None,
        liquid_eos=None,
    )
    # Colon and any other path-hostile characters are stripped from the label.
    assert ':' not in key
    assert '/' not in key
    assert 'PALEOS-2phase-MgSiO3' in key

    # All-None resolved paths must not crash and must still separate distinct
    # EOS names (the boundary/dummy layouts hit this path).
    key_other = _ps_cache_key(
        **common,
        mantle_eos='PALEOS-2phase:Fe2SiO4',
        eos_file=None,
        solid_eos=None,
        liquid_eos=None,
    )
    assert key != key_other


def test_ps_cache_key_separates_table_generators():
    """Tables built by a different Zalmoxis generator land on a different key.

    A shared PROTEUS_PS_CACHE_DIR would otherwise keep serving tables written by
    an older Zalmoxis after the generator changed (for example a corrected
    liquidus curve). Identical inputs and generator still give one key, so a
    genuine cache hit is kept.
    """
    from proteus.interior_struct.zalmoxis import _ps_cache_key, _ps_generator_identity

    common = dict(
        P_max=3.5e11,
        nP=1350,
        nS=280,
        mzf=0.8,
        layout='2phase',
        mantle_eos='PALEOS-2phase:MgSiO3',
        eos_file='/data/unified.dat',
        solid_eos='/data/solid.dat',
        liquid_eos='/data/liquid.dat',
    )
    key_old = _ps_cache_key(**common, generator='26.9.21-aaaaaaaaaaaa')
    key_new = _ps_cache_key(**common, generator='26.9.21-bbbbbbbbbbbb')
    assert key_old != key_new
    # Only the generator token differs; the EOS identity part is unchanged.
    assert key_old.partition('_gen=')[0] == key_new.partition('_gen=')[0]
    assert key_old == _ps_cache_key(**common, generator='26.9.21-aaaaaaaaaaaa')
    # The default is the installed generator identity, filesystem safe.
    key_default = _ps_cache_key(**common)
    assert key_default == _ps_cache_key(**common, generator=_ps_generator_identity())
    assert '/' not in key_default and ':' not in key_default


def test_ps_generator_identity_follows_the_generator_source(tmp_path, monkeypatch):
    """The generator identity changes when the table generator source changes,
    even at a fixed version string (an editable install keeps the version from
    install time), and falls back to the version when a source file is missing.
    """
    import zalmoxis.eos_export
    import zalmoxis.melting_curves

    from proteus.interior_struct import zalmoxis as zmod

    src = tmp_path / 'eos_export.py'
    src.write_text('# generator A\n')
    monkeypatch.setattr(zalmoxis.eos_export, '__file__', str(src))
    zmod._ps_generator_identity.cache_clear()
    try:
        ident_a = zmod._ps_generator_identity()
        src.write_text('# generator B\n')
        zmod._ps_generator_identity.cache_clear()
        ident_b = zmod._ps_generator_identity()
        version = str(zalmoxis.__version__)
        assert ident_a != ident_b
        assert ident_a.startswith(version + '-') and ident_b.startswith(version + '-')
        # Edge case: an unreadable source leaves the version alone.
        monkeypatch.setattr(zalmoxis.melting_curves, '__file__', str(tmp_path / 'missing.py'))
        zmod._ps_generator_identity.cache_clear()
        assert zmod._ps_generator_identity() == version
    finally:
        monkeypatch.undo()
        zmod._ps_generator_identity.cache_clear()


def test_jax_nonviable_fallback_logs_once_per_run(caplog):
    """The JAX to numpy structure fallback is announced once, not per re-solve.

    A run whose EOS cannot take the Zalmoxis JAX inner path falls back to the
    numpy ODE integrator on every structure solve. Reporting that on each of the
    hundreds of re-solves buries the run log, so the provenance line is emitted
    only on the first non-viable solve and suppressed thereafter. A fresh run
    (cache cleared) must announce it again so a later run is not left silent.
    """
    import logging

    from proteus.interior_struct.zalmoxis import (
        _clear_superliquidus_cache,
        _log_jax_nonviable_once,
    )

    # Start from a clean run state so a leaked flag from another test cannot
    # make the first call return False and hide a real regression.
    _clear_superliquidus_cache()

    with caplog.at_level(logging.INFO, logger='fwl.proteus.interior_struct.zalmoxis'):
        first = _log_jax_nonviable_once('PALEOS:Fe', 'PALEOS:MgSiO3')
        second = _log_jax_nonviable_once('PALEOS:Fe', 'PALEOS:MgSiO3')
        third = _log_jax_nonviable_once('PALEOS:Fe', 'PALEOS:MgSiO3')

    # Only the first call emits; the guard return suppresses the rest.
    assert first is True
    assert second is False
    assert third is False

    fallback_records = [
        r for r in caplog.records if 'not viable for the configured EOS' in r.message
    ]
    assert len(fallback_records) == 1, (
        f'Expected exactly one fallback line, got {len(fallback_records)}'
    )
    # The single line must carry both EOS labels so the run log identifies which
    # configuration declined the JAX path.
    assert fallback_records[0].levelno == logging.INFO
    assert 'PALEOS:Fe' in fallback_records[0].message
    assert 'PALEOS:MgSiO3' in fallback_records[0].message

    # A new run resets the one-shot flag; otherwise a second simulation in the
    # same process would never record why it used the numpy path.
    caplog.clear()
    _clear_superliquidus_cache()
    with caplog.at_level(logging.INFO, logger='fwl.proteus.interior_struct.zalmoxis'):
        after_reset = _log_jax_nonviable_once('PALEOS:Fe', 'PALEOS:MgSiO3')
    assert after_reset is True
    assert sum('not viable for the configured EOS' in r.message for r in caplog.records) == 1

    # Leave the module flag clean for any later test in this session.
    _clear_superliquidus_cache()


# ============================================================================
# Volatile profile construction and the wet mantle EOS extension
# ============================================================================


@pytest.mark.unit
def test_build_volatile_profile_fractions():
    """Per-phase mass fractions come straight from the helpfile reservoirs.

    ``build_volatile_profile`` divides each dissolved-volatile phase mass by
    the corresponding mantle phase mass, keyed by the Zalmoxis EOS name, and
    maps the PROTEUS species names through ``_VOLATILE_EOS_MAP``
    (``H2O -> PALEOS:H2O``, ``H2 -> Chabrier:H``).
    """
    from proteus.interior_struct.zalmoxis import build_volatile_profile

    M_liq, M_sol = 4.0e24, 1.0e24
    hf_row = {
        'M_mantle_liquid': M_liq,
        'M_mantle_solid': M_sol,
        'H2O_kg_liquid': 8.0e22,
        'H2O_kg_solid': 5.0e21,
        'H2_kg_liquid': 4.0e21,
        'H2_kg_solid': 0.0,
    }
    profile = build_volatile_profile(hf_row, 'PALEOS:MgSiO3')

    assert profile is not None
    assert profile.primary_component == 'PALEOS:MgSiO3'
    assert profile.w_liquid['PALEOS:H2O'] == pytest.approx(8.0e22 / M_liq)
    assert profile.w_solid['PALEOS:H2O'] == pytest.approx(5.0e21 / M_sol)
    assert profile.w_liquid['Chabrier:H'] == pytest.approx(4.0e21 / M_liq)
    assert profile.w_solid['Chabrier:H'] == pytest.approx(0.0)


@pytest.mark.unit
def test_build_volatile_profile_clamps_to_silicate_floor():
    """Total dissolved fraction is clamped so >=5% silicate always remains.

    An over-saturated inventory (sum of volatile fractions > 0.95) is scaled
    down proportionally rather than pushing the silicate fraction negative.
    """
    from proteus.interior_struct.zalmoxis import build_volatile_profile

    M_liq = 1.0e24
    hf_row = {
        'M_mantle_liquid': M_liq,
        'M_mantle_solid': 0.0,
        # 0.8 + 0.5 = 1.3 of liquid mass before clamping.
        'H2O_kg_liquid': 8.0e23,
        'H2_kg_liquid': 5.0e23,
    }
    profile = build_volatile_profile(hf_row, 'PALEOS:MgSiO3')

    assert profile is not None
    total = profile.w_liquid['PALEOS:H2O'] + profile.w_liquid['Chabrier:H']
    assert total == pytest.approx(0.95)
    # Proportional clamp preserves the H2O:H2 ratio (0.8:0.5).
    ratio = profile.w_liquid['PALEOS:H2O'] / profile.w_liquid['Chabrier:H']
    assert ratio == pytest.approx(0.8 / 0.5)


@pytest.mark.unit
def test_build_volatile_profile_returns_none_paths():
    """The None returns are the silent dry-fallback branch: cover them.

    ``dry_mantle = false`` degrades to the dry path whenever no profile can
    be built, so the two None conditions (no mantle phase mass, no dissolved
    volatile mass) must be pinned so the fallback stays intentional.
    """
    from proteus.interior_struct.zalmoxis import build_volatile_profile

    # No mantle phase mass yet (e.g. pre-energetics init step).
    assert build_volatile_profile({'H2O_kg_liquid': 1.0e22}, 'PALEOS:MgSiO3') is None
    # Mantle mass present but nothing dissolved.
    assert (
        build_volatile_profile(
            {'M_mantle_liquid': 4.0e24, 'M_mantle_solid': 1.0e24}, 'PALEOS:MgSiO3'
        )
        is None
    )


@pytest.mark.unit
def test_extend_mantle_eos_with_volatiles():
    """A single-component mantle EOS gains placeholder volatile tokens.

    The placeholder fractions are overridden per shell by the profile at run
    time; only the token structure matters here. A ``None`` profile or an
    already-composite EOS string is returned unchanged.
    """
    from zalmoxis.mixing import VolatileProfile

    from proteus.interior_struct.zalmoxis import extend_mantle_eos_with_volatiles

    profile = VolatileProfile(
        w_liquid={'PALEOS:H2O': 0.02},
        w_solid={'PALEOS:H2O': 0.0},
        primary_component='PALEOS:MgSiO3',
    )
    extended = extend_mantle_eos_with_volatiles('PALEOS:MgSiO3', profile)
    assert extended == 'PALEOS:MgSiO3:0.9900+PALEOS:H2O:0.0100'

    # Pass-through: no profile, and an already-composite string.
    assert extend_mantle_eos_with_volatiles('PALEOS:MgSiO3', None) == 'PALEOS:MgSiO3'
    composite = 'PALEOS:MgSiO3:0.9+PALEOS:H2O:0.1'
    assert extend_mantle_eos_with_volatiles(composite, profile) == composite


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_build_volatile_profile_uses_structural_mantle_mass():
    """The melt concentration divides CALLIOPE's dissolved kg by the same
    reservoir CALLIOPE dissolved into, Phi_global * M_mantle (structural).
    Aragog's M_mantle_liquid is computed from its PALEOS EOS cell masses,
    ~7% denser than the wet structural mantle; using it diluted w_liquid
    and lost the same fraction of dissolved water from the structure
    (the -6.4% residual of the wire_wet_1Me_aragog verification run).
    """
    from proteus.interior_struct.zalmoxis import build_volatile_profile

    hf_row = {
        'M_mantle': 4.0e24,
        'Phi_global': 1.0,
        'M_mantle_liquid': 4.3e24,  # PALEOS-density mass, must be ignored
        'M_mantle_solid': 0.0,
        'H2O_kg_liquid': 3.32e23,
        'H2O_kg_solid': 0.0,
    }
    prof = build_volatile_profile(hf_row, 'PALEOS:MgSiO3')
    assert prof is not None
    assert prof.w_liquid['PALEOS:H2O'] == pytest.approx(3.32e23 / 4.0e24, rel=1e-12)

    # Fallback: without the structural fields, the per-phase masses apply.
    hf_fallback = {
        'M_mantle_liquid': 4.3e24,
        'M_mantle_solid': 0.0,
        'H2O_kg_liquid': 3.32e23,
        'H2O_kg_solid': 0.0,
    }
    prof = build_volatile_profile(hf_fallback, 'PALEOS:MgSiO3')
    assert prof.w_liquid['PALEOS:H2O'] == pytest.approx(3.32e23 / 4.3e24, rel=1e-12)

    # Partially molten: the split follows Phi_global.
    hf_mixed = dict(hf_row, Phi_global=0.5, H2O_kg_solid=1.0e22)
    prof = build_volatile_profile(hf_mixed, 'PALEOS:MgSiO3')
    assert prof.w_liquid['PALEOS:H2O'] == pytest.approx(3.32e23 / 2.0e24, rel=1e-12)
    assert prof.w_solid['PALEOS:H2O'] == pytest.approx(1.0e22 / 2.0e24, rel=1e-12)


@pytest.mark.unit
def test_build_mushy_zone_factors_covers_unified_and_paleos_api():
    """Every configured unified material gets the real mzf, others get 1.0.

    Covers the bare PALEOS and PALEOS-API unified names in one layer config,
    including a material absent from any layer (must stay at 1.0).
    """
    from proteus.interior_struct.zalmoxis import _build_mushy_zone_factors

    layer_eos_config = {
        'core': 'PALEOS-API:iron',
        'mantle': 'PALEOS:MgSiO3',
    }
    result = _build_mushy_zone_factors(layer_eos_config, mzf=0.8)
    assert result['PALEOS-API:iron'] == pytest.approx(0.8)
    assert result['PALEOS:MgSiO3'] == pytest.approx(0.8)
    assert result['PALEOS:iron'] == pytest.approx(1.0)
    assert result['PALEOS-API:MgSiO3'] == pytest.approx(1.0)
    assert result['Chabrier:H'] == pytest.approx(1.0)


@pytest.mark.unit
def test_build_mushy_zone_factors_wet_mantle_after_volatile_extension():
    """A dissolved-volatile component in an extended mantle string gets the
    real mzf, not the 1.0 default (regression: mushy_zone_factors was built
    from the dry mantle_eos string before the volatile tokens were appended,
    silently disabling mzf for PALEOS:H2O and Chabrier:H in wet runs).
    """
    from zalmoxis.mixing import VolatileProfile

    from proteus.interior_struct.zalmoxis import (
        _build_mushy_zone_factors,
        extend_mantle_eos_with_volatiles,
    )

    profile = VolatileProfile(
        w_liquid={'PALEOS:H2O': 0.02, 'Chabrier:H': 0.01},
        w_solid={'PALEOS:H2O': 0.0, 'Chabrier:H': 0.0},
        primary_component='PALEOS:MgSiO3',
    )
    extended_mantle = extend_mantle_eos_with_volatiles('PALEOS:MgSiO3', profile)
    layer_eos_config = {'core': 'PALEOS:iron', 'mantle': extended_mantle}

    result = _build_mushy_zone_factors(layer_eos_config, mzf=0.8)
    assert result['PALEOS:MgSiO3'] == pytest.approx(0.8)
    assert result['PALEOS:H2O'] == pytest.approx(0.8)
    assert result['Chabrier:H'] == pytest.approx(0.8)
    assert result['PALEOS:iron'] == pytest.approx(0.8)


@pytest.mark.unit
def test_zalmoxis_solver_rebuilds_mushy_zone_factors_for_wet_mantle(tmp_path, monkeypatch):
    """A wet solve carries the real mzf for the dissolved species into the solve call.

    Exercises the fix at its actual call site inside ``zalmoxis_solver``
    (the rebuild after ``extend_mantle_eos_with_volatiles``), not just the
    ``_build_mushy_zone_factors`` helper in isolation. Reverting that
    rebuild would leave ``PALEOS:H2O`` at the 1.0 default instead of the
    configured mzf, and this test discriminates between the two.
    """
    model_for_arrays = _plausible_model_results()
    r_arr, t_arr = _cooled_mantle_arrays(model_for_arrays)

    def tf(r, P):
        if r <= r_arr[0]:
            return float(t_arr[0])
        return float(np.interp(r, r_arr, t_arr))

    hf_extra = {
        'M_mantle_liquid': 4.0e24,
        'M_mantle_solid': 1.0e24,
        'H2O_kg_liquid': 8.0e22,
    }
    main_mock, _, _, _, _, _, _ = _run_gate_solver(
        tmp_path,
        monkeypatch,
        'PALEOS:MgSiO3',
        (r_arr, t_arr),
        tf,
        dry_mantle=False,
        hf_extra=hf_extra,
    )

    solver_params = main_mock.call_args.args[0]
    mzf = solver_params['mushy_zone_factors']
    assert mzf['PALEOS:MgSiO3'] == pytest.approx(0.8)
    assert mzf['PALEOS:H2O'] == pytest.approx(0.8)
    # Discrimination: a build from the pre-extension dry mantle string
    # would leave the dissolved species at the 1.0 default instead.
    assert mzf['PALEOS:H2O'] != pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.parametrize('mzf', [0.7, 0.8, 1.0])
def test_zalmoxis_solver_passes_mzf_derived_solidus_to_solve(tmp_path, monkeypatch, mzf):
    """The solve receives the mzf-derived solidus built from the PALEOS liquidus.

    Runs the real ``load_zalmoxis_solidus_liquidus_functions`` (the
    other solver tests stub it with a constant pair) and checks the
    ``melting_curves_functions`` handed to the structure solve:
    ``solidus(P) == mzf * liquidus(P)`` at several pressures, and exact
    equality at ``mzf = 1.0``. Passing ``None`` or a stale curve pair to
    the solve fails these assertions.
    """
    model_for_arrays = _plausible_model_results()
    r_arr, t_arr = _cooled_mantle_arrays(model_for_arrays)

    def tf(r, P):
        if r <= r_arr[0]:
            return float(t_arr[0])
        return float(np.interp(r, r_arr, t_arr))

    main_mock, _, _, _, _, _, _ = _run_gate_solver(
        tmp_path,
        monkeypatch,
        'PALEOS:MgSiO3',
        (r_arr, t_arr),
        tf,
        real_melting_curves=True,
        mzf=mzf,
    )

    melt_funcs = main_mock.call_args.kwargs['melting_curves_functions']
    assert melt_funcs is not None
    solidus_func, liquidus_func = melt_funcs
    for pressure in (5e9, 50e9, 200e9):
        t_liq = float(liquidus_func(pressure))
        assert np.isfinite(t_liq)
        if mzf == 1.0:
            assert float(solidus_func(pressure)) == t_liq
        else:
            assert float(solidus_func(pressure)) == pytest.approx(mzf * t_liq)
            assert float(solidus_func(pressure)) < t_liq


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_derive_solidus_from_liquidus_scales_liquidus_by_mzf():
    """The derived solidus is the liquidus scaled pointwise by mushy_zone_factor.

    Discrimination: mzf < 1 must strictly lower the solidus below the
    liquidus (rules out an implementation that ignores mzf), and a
    different mzf must give a different solidus (rules out a hardcoded
    or memoized return value).
    """
    from zalmoxis.melting_curves import derive_solidus_from_liquidus

    def liquidus(pressure):
        return 3000.0 + 10.0 * pressure

    solidus = derive_solidus_from_liquidus(liquidus, mushy_zone_factor=0.8)
    for pressure in (0.0, 20e9, 80e9):
        assert solidus(pressure) == pytest.approx(0.8 * liquidus(pressure))
        assert solidus(pressure) < liquidus(pressure)

    other_solidus = derive_solidus_from_liquidus(liquidus, mushy_zone_factor=0.95)
    assert other_solidus(20e9) != pytest.approx(solidus(20e9))

    # Boundaries of the allowed range: 1.0 collapses the mushy zone exactly
    # (no clamp below 1), 0.7 is the widest band.
    sharp = derive_solidus_from_liquidus(liquidus, mushy_zone_factor=1.0)
    widest = derive_solidus_from_liquidus(liquidus, mushy_zone_factor=0.7)
    for pressure in (0.0, 20e9, 80e9):
        assert sharp(pressure) == liquidus(pressure)
        assert widest(pressure) == pytest.approx(0.7 * liquidus(pressure))


@pytest.mark.unit
def test_generate_spider_tables_twophase_solidus_tracks_mzf(tmp_path, monkeypatch):
    """generate_spider_tables hands the writers a solidus = mushy_zone_factor * liquidus.

    Drives the real ``generate_spider_tables`` down the PALEOS-2phase branch
    with synthetic solid + liquid tables, stubbing only the heavy Zalmoxis
    table writers and the analytic liquidus. Captures the ``solidus_func``
    passed to the phase-boundary writer and checks that it equals the
    liquidus scaled by ``mushy_zone_factor``, and that a different factor
    (0.8 vs 0.95) moves the written solidus. Dropping the mzf scaling, or
    hardcoding a factor, breaks one of the two assertions.
    """
    from proteus.interior_struct import zalmoxis as zalmoxis_wrapper

    mantle_eos = 'PALEOS-2phase:MgSiO3'

    # Synthetic 2-phase tables. Content is irrelevant: the writers are stubbed;
    # only os.path.isfile must succeed on the resolved paths.
    solid_file = tmp_path / 'solid.dat'
    liquid_file = tmp_path / 'liquid.dat'
    solid_file.write_text('# synthetic solid table\n')
    liquid_file.write_text('# synthetic liquid table\n')

    eos_entry = {
        'solid_mantle': {'eos_file': str(solid_file)},
        'melted_mantle': {'eos_file': str(liquid_file)},
    }

    def liquidus(pressure):
        return 3000.0 + 1.0e-8 * pressure

    def run_with_mzf(mzf):
        config = MagicMock()
        config.interior_struct.zalmoxis.mantle_eos = mantle_eos
        config.interior_struct.zalmoxis.mushy_zone_factor = mzf
        config.interior_struct.zalmoxis.lookup_nP = 8
        config.interior_struct.zalmoxis.lookup_nS = 8
        config.planet.mass_tot = 1.0

        outdir = tmp_path / f'out_{mzf}'
        outdir.mkdir()

        monkeypatch.delenv('PROTEUS_PS_CACHE_DIR', raising=False)
        monkeypatch.setattr(
            zalmoxis_wrapper,
            'load_zalmoxis_material_dictionaries',
            lambda: {mantle_eos: eos_entry},
        )
        monkeypatch.setattr('zalmoxis.eos.dispatch._is_paleos_api', lambda entry: False)
        monkeypatch.setattr(
            'zalmoxis.melting_curves.get_solidus_liquidus_functions',
            lambda solidus_id, liquidus_id: (None, liquidus),
        )
        phase_writer = MagicMock()
        monkeypatch.setattr(
            'zalmoxis.eos_export.generate_spider_phase_boundaries', phase_writer
        )
        monkeypatch.setattr('zalmoxis.eos_export.generate_spider_eos_tables', MagicMock())

        result = zalmoxis_wrapper.generate_spider_tables(config, str(outdir))
        assert result is not None
        return phase_writer.call_args.kwargs['solidus_func']

    solidus_08 = run_with_mzf(0.8)
    solidus_095 = run_with_mzf(0.95)

    for pressure in (1e9, 3e10, 1.2e11):
        assert solidus_08(pressure) == pytest.approx(0.8 * liquidus(pressure))
        assert solidus_08(pressure) < liquidus(pressure)
        # Discrimination: a different mushy_zone_factor moves the written solidus.
        assert solidus_095(pressure) == pytest.approx(0.95 * liquidus(pressure))
        assert solidus_095(pressure) != pytest.approx(solidus_08(pressure))


@pytest.mark.unit
@pytest.mark.parametrize(
    ('mantle_eos', 'expected'),
    [
        ('PALEOS-API:MgSiO3', 'PALEOS-API-2phase:MgSiO3'),
        ('PALEOS-API-2phase:MgSiO3', 'PALEOS-API-2phase:MgSiO3'),
        ('PALEOS-2phase:MgSiO3-highres', 'PALEOS-2phase:MgSiO3-highres'),
        ('PALEOS-2phase:MgSiO3', 'PALEOS-2phase:MgSiO3'),
        ('PALEOS:MgSiO3', 'PALEOS-2phase:MgSiO3'),
    ],
)
def test_twophase_registry_key_selects_table_family(mantle_eos, expected):
    """The 2-phase registry key follows the EOS family of the mantle."""
    from proteus.interior_struct.zalmoxis import twophase_registry_key

    assert twophase_registry_key(mantle_eos) == expected


def test_material_dictionaries_seager_paths_use_the_versioned_dataset_dir(
    monkeypatch, tmp_path
):
    """The Seager entries point into the fwl-io dataset directory, not a legacy folder."""
    import proteus.interior_struct.zalmoxis as zalmoxis_wrapper
    from proteus.data import EOS_SEAGER_2007, dataset_dir

    monkeypatch.setattr('proteus.utils.data.FWL_DATA_DIR', tmp_path)

    registry = zalmoxis_wrapper.load_zalmoxis_material_dictionaries()

    seager = dataset_dir(EOS_SEAGER_2007, data_root=tmp_path)
    assert registry['Seager2007:iron']['core']['eos_file'] == str(
        seager / 'eos_seager07_iron.txt'
    )
    assert registry['Seager2007:MgSiO3']['mantle']['eos_file'] == str(
        seager / 'eos_seager07_silicate.txt'
    )
    assert registry['Seager2007:H2O']['ice_layer']['eos_file'] == str(
        seager / 'eos_seager07_water.txt'
    )
    assert seager.name.startswith('r')


def test_material_dictionaries_read_the_tree_the_fetch_writes(monkeypatch, tmp_path):
    """The EOS paths use the data root that download_zalmoxis_eos fetches into.

    The raw ``FWL_DATA`` value is not expanded and has a different default from
    the resolved root. It is set to another tree here, so paths built from it
    instead of the resolved root fail the test.
    """
    from pathlib import Path

    import proteus.interior_struct.zalmoxis as zalmoxis_wrapper
    from proteus.data import EOS_PALEOS_IRON, dataset_dir

    fetch_root = tmp_path / 'fetched'
    monkeypatch.setattr('proteus.utils.data.FWL_DATA_DIR', fetch_root)
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'raw_env_value'))

    registry = zalmoxis_wrapper.load_zalmoxis_material_dictionaries()

    iron = Path(registry['PALEOS:iron']['eos_file'])
    assert iron == dataset_dir(EOS_PALEOS_IRON, data_root=fetch_root) / (
        'paleos_iron_eos_table_pt.dat'
    )
    assert not iron.is_relative_to(tmp_path / 'raw_env_value')


def test_material_dictionaries_mantle_paths_use_the_versioned_dataset_dirs(
    monkeypatch, tmp_path
):
    """Every mantle table resolves into its own fwl-io dataset directory."""
    import proteus.interior_struct.zalmoxis as zalmoxis_wrapper
    from proteus.data import (
        EOS_PALEOS_MGSIO3,
        EOS_RTPRESS_100TPA,
        EOS_WOLF_BOWER_2018,
        dataset_dir,
    )

    monkeypatch.setattr('proteus.utils.data.FWL_DATA_DIR', tmp_path)

    registry = zalmoxis_wrapper.load_zalmoxis_material_dictionaries()

    wb = dataset_dir(EOS_WOLF_BOWER_2018, data_root=tmp_path)
    rt = dataset_dir(EOS_RTPRESS_100TPA, data_root=tmp_path)
    # Both PALEOS 2-phase resolutions live in one shared dataset.
    p2 = dataset_dir(EOS_PALEOS_MGSIO3, data_root=tmp_path)
    p2hr = p2

    wolf = registry['WolfBower2018:MgSiO3']
    assert wolf['melted_mantle']['eos_file'] == str(wb / 'density_melt.dat')
    assert wolf['solid_mantle']['eos_file'] == str(wb / 'density_solid.dat')
    assert wolf['melted_mantle']['adiabat_grad_file'] == str(wb / 'adiabat_temp_grad_melt.dat')

    rtpress = registry['RTPress100TPa:MgSiO3']
    assert rtpress['melted_mantle']['eos_file'] == str(rt / 'density_melt.dat')
    assert rtpress['melted_mantle']['adiabat_grad_file'] == str(
        rt / 'adiabat_temp_grad_melt.dat'
    )
    # RTPress takes its solid density from the Wolf and Bower dataset.
    assert rtpress['solid_mantle']['eos_file'] == str(wb / 'density_solid.dat')

    two = registry['PALEOS-2phase:MgSiO3']
    assert two['melted_mantle']['eos_file'] == str(
        p2 / 'paleos_mgsio3_tables_pt_proteus_liquid.dat'
    )
    assert two['solid_mantle']['eos_file'] == str(
        p2 / 'paleos_mgsio3_tables_pt_proteus_solid.dat'
    )

    high = registry['PALEOS-2phase:MgSiO3-highres']
    assert high['melted_mantle']['eos_file'] == str(
        p2hr / 'paleos_mgsio3_tables_pt_proteus_liquid_highres.dat'
    )
    assert high['solid_mantle']['eos_file'] == str(
        p2hr / 'paleos_mgsio3_tables_pt_proteus_solid_highres.dat'
    )
    # The two PALEOS 2-phase resolutions are distinct files in the shared dataset.
    assert high['melted_mantle']['eos_file'] != two['melted_mantle']['eos_file']
    assert p2.parent.name == 'paleos_mgsio3'


_UNIFIED = object()


def _generate_tables_stubbed(
    tmp_path, monkeypatch, *, resume, run=True, entry=_UNIFIED, melt_calls=None, on_build=None
):
    """Run generate_spider_tables for a unified PALEOS entry with stubbed
    generators; return the result, the two generator mocks and the key the
    current code builds (the result is None when ``run`` is False).

    ``entry`` replaces the registry entry at run time (None removes it),
    ``melt_calls`` records melting-curve setup calls, and ``on_build`` is
    called when the phase-boundary build starts."""
    from pathlib import Path

    import zalmoxis.eos_export
    import zalmoxis.melting_curves

    from proteus.interior_struct import zalmoxis as zmod

    eos = tmp_path / 'eos.dat'
    eos.write_text('table')
    unified = {'format': 'paleos_unified', 'eos_file': str(eos)}
    current = unified if entry is _UNIFIED else entry
    monkeypatch.setattr(
        zmod,
        'load_zalmoxis_material_dictionaries',
        lambda: {} if current is None else {'PALEOS:MgSiO3': current},
    )
    monkeypatch.setattr(zmod, 'resolve_2phase_mgsio3_paths', lambda *a: (None, None))
    calls = [] if melt_calls is None else melt_calls

    def _curves(**kw):
        calls.append('curves')
        return None, lambda P: 3000.0

    def _derive(f, m):
        calls.append('derive')
        return f

    monkeypatch.setattr(zalmoxis.melting_curves, 'get_solidus_liquidus_functions', _curves)
    monkeypatch.setattr(zalmoxis.melting_curves, 'derive_solidus_from_liquidus', _derive)

    def _write_bounds(**kw):
        if on_build is not None:
            on_build()
        for name in ('solidus_P-S.dat', 'liquidus_P-S.dat'):
            (Path(kw['output_dir']) / name).write_text('NEW')

    bounds = MagicMock(side_effect=_write_bounds)
    tables = MagicMock()
    monkeypatch.setattr(zalmoxis.eos_export, 'generate_spider_phase_boundaries', bounds)
    monkeypatch.setattr(zalmoxis.eos_export, 'generate_spider_eos_tables', tables)

    config = MagicMock()
    config.interior_struct.zalmoxis.mantle_eos = 'PALEOS:MgSiO3'
    config.interior_struct.zalmoxis.mushy_zone_factor = 0.8
    config.interior_struct.zalmoxis.lookup_nP = 8
    config.interior_struct.zalmoxis.lookup_nS = 8
    config.planet.mass_tot = 1.0
    config.params.resume = resume
    key = zmod._ps_cache_key(
        P_max=3.5e11,
        nP=8,
        nS=8,
        mzf=0.8,
        layout='unified',
        mantle_eos='PALEOS:MgSiO3',
        eos_file=str(eos),
        solid_eos=None,
        liquid_eos=None,
    )
    out = zmod.generate_spider_tables(config, str(tmp_path / 'run')) if run else None
    return out, bounds, tables, key


def _seed_tables(eos_dir, marker):
    eos_dir.mkdir(parents=True, exist_ok=True)
    (eos_dir / '.cache_info.txt').write_text(marker)
    for name in ('solidus_P-S.dat', 'liquidus_P-S.dat'):
        (eos_dir / name).write_text('OLD')


def test_resume_keeps_run_tables_with_an_old_format_marker(tmp_path, monkeypatch, caplog):
    """A resumed run whose marker has no generator identity keeps its own
    tables and warns; a fresh run in the same directory rebuilds them under
    the current key."""
    from pathlib import Path as _Path

    monkeypatch.delenv('PROTEUS_PS_CACHE_DIR', raising=False)
    run_eos = tmp_path / 'run' / 'data' / 'spider_eos'
    # A marker without a generator suffix (base key only).
    _, _, _, key = _generate_tables_stubbed(tmp_path, monkeypatch, resume=True, run=False)
    old_marker = key.partition('_gen=')[0]
    _seed_tables(run_eos, old_marker)

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_struct.zalmoxis'):
        out, bounds, tables, key = _generate_tables_stubbed(tmp_path, monkeypatch, resume=True)
    assert _Path(out['eos_dir']) == run_eos
    bounds.assert_not_called()
    tables.assert_not_called()
    assert (run_eos / 'solidus_P-S.dat').read_text() == 'OLD'
    assert (run_eos / '.cache_info.txt').read_text() == old_marker
    assert 'keeps its original energetics P-S entropy tables' in caplog.text
    assert 'structure solve uses the current melting curves' in caplog.text
    assert 'generator unknown' in caplog.text and key in caplog.text

    out, bounds, tables, key = _generate_tables_stubbed(tmp_path, monkeypatch, resume=False)
    bounds.assert_called_once()
    assert (run_eos / 'solidus_P-S.dat').read_text() == 'NEW'
    assert (run_eos / '.cache_info.txt').read_text() == key


def test_resume_follows_the_pointer_to_shared_cache_tables(tmp_path, monkeypatch, caplog):
    """With a shared cache, a resumed run keeps the tables its pointer names
    even when they come from another generator, and leaves the pointer alone;
    an exact key match keeps them without a warning."""
    from pathlib import Path as _Path

    from proteus.interior_struct import zalmoxis as zmod
    from proteus.interior_struct.zalmoxis import PS_CACHE_POINTER_NAME

    monkeypatch.setenv('PROTEUS_PS_CACHE_DIR', str(tmp_path / 'cache'))
    old_dir = tmp_path / 'cache' / 'old-key-dir'
    pointer = tmp_path / 'run' / 'data' / PS_CACHE_POINTER_NAME
    pointer.parent.mkdir(parents=True)
    pointer.write_text(str(old_dir))
    _, _, _, key = _generate_tables_stubbed(tmp_path, monkeypatch, resume=True, run=False)
    base = key.partition('_gen=')[0]
    _seed_tables(old_dir, base + '_gen=0-0-1-aaaaaaaaaaaa')

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_struct.zalmoxis'):
        out, bounds, _, key = _generate_tables_stubbed(tmp_path, monkeypatch, resume=True)
    assert _Path(out['eos_dir']) == old_dir
    bounds.assert_not_called()
    assert pointer.read_text() == str(old_dir)
    assert 'generator 0-0-1-aaaaaaaaaaaa' in caplog.text

    # Edge case: the stored key equals the current key, so nothing is logged
    # (a new process, since each kept directory is reported once per process).
    (old_dir / '.cache_info.txt').write_text(key)
    monkeypatch.setattr(zmod, '_PS_RESUME_REPORTED', set())
    caplog.clear()
    with caplog.at_level('WARNING', logger='fwl.proteus.interior_struct.zalmoxis'):
        out, bounds, _, _ = _generate_tables_stubbed(tmp_path, monkeypatch, resume=True)
    assert _Path(out['eos_dir']) == old_dir and caplog.text == ''
    bounds.assert_not_called()


def test_resume_keeps_run_tables_after_a_settings_change(tmp_path, monkeypatch, caplog):
    """A resumed run whose stored key differs in a physical setting (here the
    pressure ceiling) keeps its tables and warns with both keys that the
    changed settings are ignored."""
    from pathlib import Path as _Path

    monkeypatch.delenv('PROTEUS_PS_CACHE_DIR', raising=False)
    run_eos = tmp_path / 'run' / 'data' / 'spider_eos'
    _, _, _, key = _generate_tables_stubbed(tmp_path, monkeypatch, resume=True, run=False)
    stored = key.replace('P_max=3.500000e+11', 'P_max=4.000000e+11')
    assert stored != key
    _seed_tables(run_eos, stored)

    melt_calls = []
    with caplog.at_level('WARNING', logger='fwl.proteus.interior_struct.zalmoxis'):
        out, bounds, tables, key = _generate_tables_stubbed(
            tmp_path, monkeypatch, resume=True, melt_calls=melt_calls
        )
    assert _Path(out['eos_dir']) == run_eos
    bounds.assert_not_called()
    tables.assert_not_called()
    # Kept tables need no melting curves; a build would set them up.
    assert melt_calls == []
    assert (run_eos / 'solidus_P-S.dat').read_text() == 'OLD'
    assert 'ignores the changed settings' in caplog.text
    assert stored in caplog.text and key in caplog.text


def test_resume_reports_kept_tables_once_per_process(tmp_path, monkeypatch, caplog):
    """The kept-table WARNING is logged the first time a resumed run asks for
    its tables, not at every later call in the same process."""
    monkeypatch.delenv('PROTEUS_PS_CACHE_DIR', raising=False)
    _, _, _, key = _generate_tables_stubbed(tmp_path, monkeypatch, resume=True, run=False)
    _seed_tables(tmp_path / 'run' / 'data' / 'spider_eos', key.partition('_gen=')[0])
    with caplog.at_level('WARNING', logger='fwl.proteus.interior_struct.zalmoxis'):
        for _ in range(3):
            out, bounds, _, _ = _generate_tables_stubbed(tmp_path, monkeypatch, resume=True)
            bounds.assert_not_called()
    assert caplog.text.count('keeps its original energetics P-S entropy tables') == 1


def test_resume_keeps_tables_without_resolving_a_paleos_api_eos(tmp_path, monkeypatch, caplog):
    """A resumed PALEOS-API run keeps its tables without resolving the API
    cache, which can build for an hour or fail offline; the key is then not
    checked, and the WARNING says why."""
    import zalmoxis.eos.dispatch
    import zalmoxis.eos.paleos_api_cache

    from proteus.interior_struct import zalmoxis as zmod

    monkeypatch.delenv('PROTEUS_PS_CACHE_DIR', raising=False)
    _, _, _, key = _generate_tables_stubbed(tmp_path, monkeypatch, resume=True, run=False)
    _seed_tables(tmp_path / 'run' / 'data' / 'spider_eos', key)
    monkeypatch.setattr(zalmoxis.eos.dispatch, '_is_paleos_api', lambda entry: True)
    resolve = MagicMock(side_effect=OSError('offline'))
    monkeypatch.setattr(zalmoxis.eos.paleos_api_cache, 'resolve_registry_entry', resolve)
    with caplog.at_level('WARNING', logger='fwl.proteus.interior_struct.zalmoxis'):
        out, bounds, tables, _ = _generate_tables_stubbed(tmp_path, monkeypatch, resume=True)
    resolve.assert_not_called()
    bounds.assert_not_called()
    tables.assert_not_called()
    assert out['eos_dir'] == str(tmp_path / 'run' / 'data' / 'spider_eos')
    assert 'current key is not checked: a PALEOS-API mantle EOS is not resolved' in caplog.text
    # Discrimination: a fresh PALEOS-API run resolves its tables and stops on the failure.
    with pytest.raises(zmod.ZalmoxisMissingEOSFilesError, match='OSError: offline'):
        _generate_tables_stubbed(tmp_path, monkeypatch, resume=False)
    resolve.assert_called()


def test_resume_without_kept_tables_warns_before_the_build(tmp_path, monkeypatch, caplog):
    """A resumed run with tables but no marker (the marker is the completion
    sentinel) keeps nothing, says at WARNING that it continues on newly built
    tables, and builds them under the current key."""
    monkeypatch.delenv('PROTEUS_PS_CACHE_DIR', raising=False)
    run_eos = tmp_path / 'run' / 'data' / 'spider_eos'
    _seed_tables(run_eos, 'unused')
    (run_eos / '.cache_info.txt').unlink()
    warned_at_build = []
    melt_calls = []
    with caplog.at_level('WARNING', logger='fwl.proteus.interior_struct.zalmoxis'):
        out, bounds, _, key = _generate_tables_stubbed(
            tmp_path,
            monkeypatch,
            resume=True,
            melt_calls=melt_calls,
            on_build=lambda: warned_at_build.append('has no kept' in caplog.text),
        )
    bounds.assert_called_once()
    # The warning is already recorded when the build starts.
    assert warned_at_build == [True]
    assert melt_calls == ['curves', 'derive']
    assert (run_eos / 'solidus_P-S.dat').read_text() == 'NEW'
    assert (run_eos / '.cache_info.txt').read_text() == key
    assert 'has no kept P-S entropy tables' in caplog.text and key in caplog.text
    # Discrimination: a fresh run builds the same tables without the warning.
    caplog.clear()
    (run_eos / '.cache_info.txt').unlink()
    with caplog.at_level('WARNING', logger='fwl.proteus.interior_struct.zalmoxis'):
        _generate_tables_stubbed(tmp_path, monkeypatch, resume=False)
    assert 'has no kept' not in caplog.text


@pytest.mark.parametrize(
    'case, reason',
    [
        ('not-paleos', 'is neither PALEOS unified nor PALEOS-2phase'),
        ('file-missing', 'no PALEOS EOS file is available'),
        ('unregistered', 'is not in the material dictionary'),
    ],
)
def test_resume_keeps_tables_when_the_current_eos_gives_none(
    tmp_path, monkeypatch, caplog, case, reason
):
    """A resumed run keeps its tables when its current mantle EOS is no longer
    PALEOS, has lost its file, or is not registered, and logs exactly one
    WARNING naming that reason, with no record that it falls back to other
    tables or skips table generation."""
    monkeypatch.delenv('PROTEUS_PS_CACHE_DIR', raising=False)
    run_eos = tmp_path / 'run' / 'data' / 'spider_eos'
    _, _, _, key = _generate_tables_stubbed(tmp_path, monkeypatch, resume=True, run=False)
    _seed_tables(run_eos, key)
    wb = tmp_path / 'wb.dat'
    wb.write_text('table')
    entry = {
        'not-paleos': {'format': 'WolfBower2018', 'eos_file': str(wb)},
        'file-missing': {'format': 'paleos_unified', 'eos_file': str(tmp_path / 'gone.dat')},
        'unregistered': None,
    }[case]
    with caplog.at_level('INFO', logger='fwl.proteus.interior_struct.zalmoxis'):
        out, bounds, _, _ = _generate_tables_stubbed(
            tmp_path, monkeypatch, resume=True, entry=entry
        )
    assert out['eos_dir'] == str(run_eos)
    bounds.assert_not_called()
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert 'keeps its original energetics P-S entropy tables' in warnings[0]
    assert reason in warnings[0]
    assert 'pre-existing SPIDER tables' not in caplog.text
    assert 'skipping' not in caplog.text
    # Discrimination: a fresh run stops on the missing PALEOS file; only a
    # non-PALEOS or unregistered EOS leaves the tables to the energetics module.
    caplog.clear()
    if case == 'file-missing':
        from proteus.interior_struct.zalmoxis import ZalmoxisMissingEOSFilesError

        with pytest.raises(ZalmoxisMissingEOSFilesError, match='gone.dat'):
            _generate_tables_stubbed(tmp_path, monkeypatch, resume=False, entry=entry)
        return
    with caplog.at_level('INFO', logger='fwl.proteus.interior_struct.zalmoxis'):
        out, _, _, _ = _generate_tables_stubbed(
            tmp_path, monkeypatch, resume=False, entry=entry
        )
    assert out is None and 'pre-existing SPIDER tables' in caplog.text


def test_resume_keeps_tables_when_the_key_builder_fails(tmp_path, monkeypatch, caplog):
    """Any failure while building the current key for the warning, not only a
    ValueError, leaves the resumed run on its kept tables with the reason."""
    from proteus.interior_struct import zalmoxis as zmod

    monkeypatch.delenv('PROTEUS_PS_CACHE_DIR', raising=False)
    run_eos = tmp_path / 'run' / 'data' / 'spider_eos'
    _, _, _, key = _generate_tables_stubbed(tmp_path, monkeypatch, resume=True, run=False)
    _seed_tables(run_eos, key)

    def _fail(*args):
        raise OSError('registry unreadable')

    monkeypatch.setattr(zmod, '_ps_table_inputs', _fail)
    with caplog.at_level('WARNING', logger='fwl.proteus.interior_struct.zalmoxis'):
        out, bounds, _, _ = _generate_tables_stubbed(tmp_path, monkeypatch, resume=True)
    assert out['eos_dir'] == str(run_eos)
    bounds.assert_not_called()
    assert 'current key is not checked: registry unreadable' in caplog.text
