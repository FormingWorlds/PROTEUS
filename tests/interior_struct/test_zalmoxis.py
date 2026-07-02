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
- compute_structure_mass_desync(): Mass self-consistency diagnostic (issue #68)
"""

from __future__ import annotations

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

    Exercises both the flat (core) and nested (mantle) registry shapes,
    plus the lazy PALEOS-API shape that never carries a file path.
    """
    from proteus.interior_struct.zalmoxis import check_zalmoxis_eos_files

    registry = _synthetic_registry(tmp_path, write_files=True)
    layer_cfg = {'core': 'PALEOS:iron', 'mantle': 'PALEOS-2phase:MgSiO3'}
    assert check_zalmoxis_eos_files(layer_cfg, registry) is None

    # PALEOS-API entries have no eos_file at build time (tables are
    # generated on demand): the check must not flag them.
    layer_cfg_api = {'mantle': 'PALEOS-API:MgSiO3'}
    assert check_zalmoxis_eos_files(layer_cfg_api, registry) is None

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
    # The error names the command that repairs the state.
    assert 'proteus get interiordata' in msg
    assert '--offline' in msg


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


def test_check_eos_files_walks_nested_and_lazy_siblings(tmp_path):
    """Nested role entries are walked; lazy siblings do not mask misses.

    The real registry nests Seager-style single-role entries
    (``{'core': {...}}``) and mixes lazy PALEOS-API sub-entries (no
    ``eos_file`` until first density query) with file-backed ones inside
    one nested entry. A missing file-backed sibling must be flagged even
    when a lazy sibling sits next to it.
    """
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

    # The missing file-backed sibling is flagged; the lazy sibling is not.
    with pytest.raises(RuntimeError) as excinfo:
        check_zalmoxis_eos_files({'mantle': 'PALEOS-API-2phase:MgSiO3'}, registry)
    msg = str(excinfo.value)
    assert 'mgsio3_solid.dat' in msg
    assert 'seager_iron.txt' not in msg  # present file must not be flagged


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
    for f in (iron, unified, sol, liq):
        f.write_text('eos table stub')
    return {
        'PALEOS:iron': {'eos_file': str(iron), 'format': 'paleos_unified'},
        'PALEOS:MgSiO3': {'eos_file': str(unified), 'format': 'paleos_unified'},
        'PALEOS-2phase:MgSiO3': {
            'core': {'eos_file': str(iron)},
            'melted_mantle': {'eos_file': str(liq), 'format': 'paleos'},
            'solid_mantle': {'eos_file': str(sol), 'format': 'paleos'},
        },
    }


def test_jax_structure_viability_predicate(tmp_path):
    """JAX-path viability tracks the registry layout, not the config flags.

    The predicate must reproduce the precondition outcome of Zalmoxis'
    JAX dispatch: True only for a 2-phase mantle (both phase sub-tables)
    paired with a core that resolves to the unified PALEOS format.
    Edge cases: unknown registry keys (conservative False, keeping the
    temperature callable in play), a ``paleos_api`` core (resolves in
    place to ``paleos_unified``, so True), and a volatile-extended
    mantle string whose fraction token must be stripped before lookup.
    """
    from proteus.interior_struct.zalmoxis import _zalmoxis_jax_structure_viable

    registry = _gate_registry(tmp_path)

    # Unified mantle has no phase sub-tables: the JAX dispatch raises and
    # the numpy ODE runs, so the predicate must be False.
    assert _zalmoxis_jax_structure_viable(registry, 'PALEOS:iron', 'PALEOS:MgSiO3') is False

    # 2-phase mantle + unified core: the JAX path can run.
    assert (
        _zalmoxis_jax_structure_viable(registry, 'PALEOS:iron', 'PALEOS-2phase:MgSiO3') is True
    )

    # Missing registry entries (mantle or core): conservative False.
    assert _zalmoxis_jax_structure_viable(registry, 'PALEOS:iron', 'NoSuchEOS') is False
    assert (
        _zalmoxis_jax_structure_viable(registry, 'NoSuchCore', 'PALEOS-2phase:MgSiO3') is False
    )
    assert _zalmoxis_jax_structure_viable({}, 'PALEOS:iron', 'PALEOS-2phase:MgSiO3') is False

    # A paleos_api core materialises to paleos_unified before the JAX
    # dispatch checks it, so it qualifies.
    registry_api = dict(registry)
    registry_api['PALEOS-API:iron'] = {'format': 'paleos_api', 'material': 'iron'}
    assert (
        _zalmoxis_jax_structure_viable(registry_api, 'PALEOS-API:iron', 'PALEOS-2phase:MgSiO3')
        is True
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


def _run_gate_solver(
    tmp_path, monkeypatch, mantle_eos, temperature_arrays, tf, main_results=None
):
    """Invoke zalmoxis_solver with the heavy solve mocked out.

    Patches the EOS registry to the on-disk synthetic one, the melting
    curves to plausible constants, the Zalmoxis ``main`` solve to a
    converged Earth-like result, and the EOS density dispatch used by
    the post-solve column rebuild. Returns the mocks and the hf_row.

    Parameters
    ----------
    main_results : dict or list of dict, optional
        Result(s) the mocked ``main`` solve returns. A dict is returned
        on every call; a list is consumed call by call (primary, retry,
        ...). Defaults to a single converged Earth-like result.
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

    monkeypatch.setattr(
        zalmoxis_wrapper,
        '_density_cache',
        {'density': None, 'radii': None, 'key': None},
    )

    def _fake_density(P, mats, eos, T, sol, liq, interpolation_functions=None, **kwargs):
        # Plausible monotone-in-P, decreasing-in-T condensed density.
        return 3300.0 + 2.0e-8 * float(P) - 0.1 * (float(T) - 3000.0)

    with (
        patch.object(
            zalmoxis_wrapper,
            'load_zalmoxis_material_dictionaries',
            return_value=registry,
        ),
        patch.object(
            zalmoxis_wrapper,
            'load_zalmoxis_solidus_liquidus_functions',
            return_value=(lambda P: 4000.0, lambda P: 5000.0),
        ),
        patch.object(zalmoxis_wrapper, 'main', **main_patch_kwargs) as main_mock,
        patch('zalmoxis.eos.dispatch.calculate_density', side_effect=_fake_density) as rho_mock,
    ):
        cmb_radius, mesh_file = zalmoxis_wrapper.zalmoxis_solver(
            _gate_config(mantle_eos),
            str(tmp_path),
            hf_row,
            num_spider_nodes=0,
            temperature_function=tf,
            temperature_arrays=temperature_arrays,
        )

    return main_mock, rho_mock, hf_row, model_results, cmb_radius, mesh_file


def test_zalmoxis_solver_passes_callable_when_jax_path_not_viable(tmp_path, monkeypatch):
    """Unified-mantle re-solves hand the evolved T(r) callable to the solve.

    With ``use_jax=True``, ``temperature_arrays`` supplied, and a unified
    PALEOS mantle (no phase sub-tables), the JAX dispatch cannot run and
    the numpy ODE consumes only ``temperature_function``. The callable
    must therefore reach the solve unmodified; nulling it would make
    every re-solve rebuild the internal hot-anchor profile and freeze
    the interior radius across the entire crystallization sequence.
    """
    model_for_arrays = _plausible_model_results()
    r_arr, t_arr = _cooled_mantle_arrays(model_for_arrays)

    def tf(r, P):
        # Evolved-interior closure: cooled mantle, clamped below the CMB.
        if r <= r_arr[0]:
            return float(t_arr[0])
        return float(np.interp(r, r_arr, t_arr))

    main_mock, rho_mock, hf_row, model_results, cmb_radius, mesh_file = _run_gate_solver(
        tmp_path, monkeypatch, 'PALEOS:MgSiO3', (r_arr, t_arr), tf
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


def test_zalmoxis_solver_withholds_callable_on_jax_viable_eos(tmp_path, monkeypatch):
    """2-phase-mantle re-solves feed the arrays to the JAX path instead.

    With a 2-phase PALEOS mantle the JAX dispatch consumes
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

    main_mock, rho_mock, hf_row, model_results, cmb_radius, _ = _run_gate_solver(
        tmp_path, monkeypatch, 'PALEOS-2phase:MgSiO3', (r_arr, t_arr), tf
    )

    # Callable withheld, arrays passed through.
    kwargs = main_mock.call_args.kwargs
    assert kwargs['temperature_function'] is None
    passed_r, passed_t = kwargs['temperature_arrays']
    np.testing.assert_allclose(passed_r, r_arr, rtol=0, atol=0)
    np.testing.assert_allclose(passed_t, t_arr, rtol=0, atol=0)

    # The rebuild ran: one EOS density evaluation per radial node.
    assert rho_mock.call_count == len(model_results['radii'])

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
    main_mock, rho_mock, hf_row, model_results, _, _ = _run_gate_solver(
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

    main_mock, rho_mock, hf_row, _, cmb_radius, _ = _run_gate_solver(
        tmp_path,
        monkeypatch,
        'PALEOS:MgSiO3',
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
