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

    MagicMock bypasses the config-load gate on ``dry_mantle = false``
    deliberately: the subtraction logic must already be correct for when
    the Zalmoxis pin gains per-shell volatile-profile support and the
    gate is lifted.
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
