"""
Unit tests for proteus.interior_energetics.spider module.

Tests entropy remapping, mesh blending, mesh file I/O, core-size
extraction, EOS range checking, JSON solution rewriting, and the
_try_spider() call-sequence builder (EOS path resolution, mesh handling).

Testing standards and documentation:
- docs/test_infrastructure.md: Test infrastructure overview
- docs/test_categorization.md: Test marker definitions
- docs/test_building.md: Best practices for test construction

Functions tested:
- _coresize_from_mesh(): Extract R_cmb/R_surf ratio from mesh file
- _check_eos_table_range(): Warn on EOS table limits
- _read_mesh_file(): Parse SPIDER external mesh file (all 8 columns)
- _write_mesh_file(): Write SPIDER mesh file with full precision
- _rewrite_json_solution(): Overwrite solution vector in SPIDER JSON
- blend_mesh_files(): Clamp mesh shift by linear blending
- remap_entropy_for_new_mesh(): Core remapping logic
- _try_spider(): SPIDER call-sequence building (EOS paths, mesh mode)
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from proteus.interior_energetics.spider import (
    RADIUS0,
    _check_eos_table_range,
    _coresize_from_mesh,
    _read_mesh_file,
    _rewrite_json_solution,
    _write_mesh_file,
    blend_mesh_files,
    remap_entropy_for_new_mesh,
)

# Path to SPIDER output used for testing
SPIDER_JSON = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    'SPIDER',
    'output_standard',
    '0.json',
)


def _make_mesh_file(
    path: str,
    r_b: np.ndarray,
    r_s: np.ndarray,
    P_b: np.ndarray | None = None,
    rho_b: np.ndarray | None = None,
    g_b: np.ndarray | None = None,
    P_s: np.ndarray | None = None,
    rho_s: np.ndarray | None = None,
    g_s: np.ndarray | None = None,
) -> None:
    """Write a SPIDER-format mesh file.

    Optional arrays default to dummy values for backward compatibility.
    """
    nb = len(r_b)
    ns = len(r_s)
    if P_b is None:
        P_b = np.full(nb, 1e9)
    if rho_b is None:
        rho_b = np.full(nb, 4000.0)
    if g_b is None:
        g_b = np.full(nb, -9.81)
    if P_s is None:
        P_s = np.full(ns, 1e9)
    if rho_s is None:
        rho_s = np.full(ns, 4000.0)
    if g_s is None:
        g_s = np.full(ns, -9.81)

    with open(path, 'w') as f:
        f.write(f'# {nb} {ns}\n')
        for i in range(nb):
            f.write(f'{r_b[i]:.6f} {P_b[i]:.6e} {rho_b[i]:.6f} {g_b[i]:.6f}\n')
        for i in range(ns):
            f.write(f'{r_s[i]:.6f} {P_s[i]:.6e} {rho_s[i]:.6f} {g_s[i]:.6f}\n')


@pytest.fixture
def spider_json_dir():
    """Create a temporary directory with a copy of the reference SPIDER JSON."""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp)


def _copy_json(spider_json_dir: str) -> str:
    """Copy the reference JSON to a temp dir and return the path."""
    dst = os.path.join(spider_json_dir, 'test.json')
    shutil.copy2(SPIDER_JSON, dst)
    return dst


def _get_mesh_arrays_from_json(json_path: str):
    """Extract physical radius and entropy arrays from a SPIDER JSON."""
    with open(json_path) as f:
        d = json.load(f)
    data = d['data']
    r_b = np.array([float(v) for v in data['radius_b']['values']]) * float(
        data['radius_b']['scaling']
    )
    r_s = np.array([float(v) for v in data['radius_s']['values']]) * float(
        data['radius_s']['scaling']
    )
    S_s = np.array([float(v) for v in data['S_s']['values']]) * float(data['S_s']['scaling'])
    return r_b, r_s, S_s


# ============================================================================
# test _read_mesh_file (all 8 columns)
# ============================================================================


@pytest.mark.unit
def test_read_mesh_file_roundtrip(spider_json_dir):
    """Write a mesh file and verify _read_mesh_file parses all 8 arrays back.

    Uses realistic Earth-like radii (6371 km surface, 3504 km CMB) with
    50 basic and 49 staggered nodes to match SPIDER's default grid.
    """
    N_b = 50
    N_s = N_b - 1
    r_b = np.linspace(6.371e6, 3.504e6, N_b)
    r_s = 0.5 * (r_b[:-1] + r_b[1:])
    P_b = np.linspace(1e5, 130e9, N_b)
    rho_b = np.linspace(3500, 5500, N_b)
    g_b = np.linspace(-9.81, -10.5, N_b)
    P_s = 0.5 * (P_b[:-1] + P_b[1:])
    rho_s = 0.5 * (rho_b[:-1] + rho_b[1:])
    g_s = 0.5 * (g_b[:-1] + g_b[1:])

    mesh_path = os.path.join(spider_json_dir, 'mesh.dat')
    _make_mesh_file(mesh_path, r_b, r_s, P_b, rho_b, g_b, P_s, rho_s, g_s)

    result = _read_mesh_file(mesh_path)
    assert len(result) == 8

    r_b_read, P_b_read, rho_b_read, g_b_read = result[:4]
    r_s_read, P_s_read, rho_s_read, g_s_read = result[4:]

    assert len(r_b_read) == N_b
    assert len(r_s_read) == N_s
    np.testing.assert_allclose(r_b_read, r_b, rtol=1e-6)
    np.testing.assert_allclose(r_s_read, r_s, rtol=1e-6)
    np.testing.assert_allclose(P_b_read, P_b, rtol=1e-5)
    np.testing.assert_allclose(rho_b_read, rho_b, rtol=1e-5)
    np.testing.assert_allclose(g_b_read, g_b, rtol=1e-5)
    np.testing.assert_allclose(P_s_read, P_s, rtol=1e-5)
    np.testing.assert_allclose(rho_s_read, rho_s, rtol=1e-5)
    np.testing.assert_allclose(g_s_read, g_s, rtol=1e-5)


@pytest.mark.unit
def test_read_mesh_file_all_columns(spider_json_dir):
    """Verify all 8 returned arrays have correct types and shapes.

    Uses a small 5-node mesh to ensure parsing works for minimal grids.
    """
    N_b = 5
    N_s = 4
    r_b = np.linspace(6.371e6, 3.504e6, N_b)
    r_s = 0.5 * (r_b[:-1] + r_b[1:])

    mesh_path = os.path.join(spider_json_dir, 'mesh.dat')
    _make_mesh_file(mesh_path, r_b, r_s)

    result = _read_mesh_file(mesh_path)
    assert len(result) == 8
    for arr in result[:4]:
        assert isinstance(arr, np.ndarray)
        assert len(arr) == N_b
    for arr in result[4:]:
        assert isinstance(arr, np.ndarray)
        assert len(arr) == N_s


# ============================================================================
# test blend_mesh_files
# ============================================================================


@pytest.mark.unit
def test_blend_mesh_large_shift(spider_json_dir):
    """A 39% radius shift should be clamped to 5% by blending.

    Verifies that the blended file has radii within 5% of the old mesh
    and that the function returns the actual (pre-blend) shift.
    """
    N_b = 20
    r_b_old = np.linspace(6.371e6, 3.504e6, N_b)
    r_s_old = 0.5 * (r_b_old[:-1] + r_b_old[1:])
    r_b_new = r_b_old * 1.39  # 39% shift
    r_s_new = r_s_old * 1.39

    old_path = os.path.join(spider_json_dir, 'old.dat')
    new_path = os.path.join(spider_json_dir, 'new.dat')
    _make_mesh_file(old_path, r_b_old, r_s_old)
    _make_mesh_file(new_path, r_b_new, r_s_new)

    actual_shift = blend_mesh_files(old_path, new_path, max_shift=0.05)
    assert pytest.approx(actual_shift, rel=1e-3) == 0.39

    # Read blended mesh and verify shift is clamped
    r_b_blended = _read_mesh_file(new_path)[0]
    blended_shift = np.max(np.abs(r_b_blended - r_b_old) / np.abs(r_b_old))
    assert pytest.approx(blended_shift, rel=1e-3) == 0.05


@pytest.mark.unit
def test_blend_mesh_small_shift(spider_json_dir):
    """A 2% shift below the 5% threshold should pass through unchanged.

    The new mesh file should not be modified.
    """
    N_b = 20
    r_b_old = np.linspace(6.371e6, 3.504e6, N_b)
    r_s_old = 0.5 * (r_b_old[:-1] + r_b_old[1:])
    r_b_new = r_b_old * 1.02
    r_s_new = r_s_old * 1.02

    old_path = os.path.join(spider_json_dir, 'old.dat')
    new_path = os.path.join(spider_json_dir, 'new.dat')
    _make_mesh_file(old_path, r_b_old, r_s_old)
    _make_mesh_file(new_path, r_b_new, r_s_new)

    # Record file content before blend
    with open(new_path) as f:
        content_before = f.read()

    actual_shift = blend_mesh_files(old_path, new_path, max_shift=0.05)
    assert pytest.approx(actual_shift, rel=1e-3) == 0.02

    # File should be untouched
    with open(new_path) as f:
        content_after = f.read()
    assert content_before == content_after


@pytest.mark.unit
def test_blend_mesh_no_old_file(spider_json_dir):
    """Missing old mesh file should return 0.0 and leave new file untouched."""
    N_b = 10
    r_b_new = np.linspace(6.371e6, 3.504e6, N_b)
    r_s_new = 0.5 * (r_b_new[:-1] + r_b_new[1:])

    old_path = os.path.join(spider_json_dir, 'nonexistent.dat')
    new_path = os.path.join(spider_json_dir, 'new.dat')
    _make_mesh_file(new_path, r_b_new, r_s_new)

    with open(new_path) as f:
        content_before = f.read()

    actual_shift = blend_mesh_files(old_path, new_path, max_shift=0.05)
    assert actual_shift == 0.0

    with open(new_path) as f:
        content_after = f.read()
    assert content_before == content_after


@pytest.mark.unit
def test_blend_mesh_format_correct(spider_json_dir):
    """Blended mesh file should have correct header, descending r, negative g,
    and staggered nodes between basic nodes.
    """
    N_b = 20
    N_s = N_b - 1
    r_b_old = np.linspace(6.371e6, 3.504e6, N_b)
    r_s_old = 0.5 * (r_b_old[:-1] + r_b_old[1:])
    g_b_old = np.linspace(-9.81, -10.5, N_b)
    g_s_old = 0.5 * (g_b_old[:-1] + g_b_old[1:])

    r_b_new = r_b_old * 1.20  # 20% shift
    r_s_new = r_s_old * 1.20
    g_b_new = g_b_old * 1.20
    g_s_new = g_s_old * 1.20

    old_path = os.path.join(spider_json_dir, 'old.dat')
    new_path = os.path.join(spider_json_dir, 'new.dat')
    _make_mesh_file(old_path, r_b_old, r_s_old, g_b=g_b_old, g_s=g_s_old)
    _make_mesh_file(new_path, r_b_new, r_s_new, g_b=g_b_new, g_s=g_s_new)

    blend_mesh_files(old_path, new_path, max_shift=0.05)

    # Parse blended file
    r_b_bl, P_b_bl, rho_b_bl, g_b_bl, r_s_bl, P_s_bl, rho_s_bl, g_s_bl = _read_mesh_file(
        new_path
    )

    # Header: correct counts
    assert len(r_b_bl) == N_b
    assert len(r_s_bl) == N_s

    # Radii should be descending (surface to CMB)
    assert np.all(np.diff(r_b_bl) < 0)
    assert np.all(np.diff(r_s_bl) < 0)

    # Gravity should be negative
    assert np.all(g_b_bl < 0)
    assert np.all(g_s_bl < 0)

    # Staggered nodes should be between adjacent basic nodes
    for i in range(N_s):
        assert r_b_bl[i] > r_s_bl[i] > r_b_bl[i + 1]


# ============================================================================
# test mesh convergence trigger
# ============================================================================


@pytest.mark.unit
def test_mesh_convergence_trigger():
    """Verify the mesh convergence trigger fires at the short interval
    when mesh_shift_active=True, and uses normal triggers when False.

    Tests the early-exit paths of update_structure_from_interior without
    invoking the Zalmoxis solver.  Case 1 (trigger fires) patches the
    solver at its source module since it is imported locally.
    """
    from unittest.mock import MagicMock, patch

    from proteus.interior_energetics.wrapper import update_structure_from_interior

    # Create minimal config mock
    config = MagicMock()
    config.interior_struct.zalmoxis.update_interval = 1000.0
    config.interior_struct.zalmoxis.update_min_interval = 100.0
    config.interior_struct.zalmoxis.update_dtmagma_frac = 0.03
    config.interior_struct.zalmoxis.update_dphi_abs = 0.05
    config.interior_struct.zalmoxis.mesh_max_shift = 0.05
    config.interior_struct.zalmoxis.mesh_convergence_interval = 10.0
    config.planet.temperature_mode = 'isothermal'
    config.interior_struct.zalmoxis.num_levels = 50
    config.interior_energetics.module = 'spider'

    dirs = {
        'output': '/tmp/test_output',
        'spider': '/tmp/spider',
        'spider_mesh': '/tmp/test_mesh.dat',
        'spider_mesh_prev': '/tmp/test_mesh.dat.prev',
    }

    hf_row = {
        'Time': 50.0,
        'T_magma': 3000.0,
        'Phi_global': 0.8,
        'R_int': 6.371e6,
        'gravity': 9.81,
    }
    interior_o = MagicMock()
    interior_o.radius = np.linspace(6.371e6, 3.504e6, 50)
    interior_o.temp = np.full(49, 3000.0)

    # Case 1: mesh_shift_active=True, elapsed=15 yr (> convergence_interval=10)
    # Should trigger despite being under floor (100 yr)
    dirs['mesh_shift_active'] = True
    last_struct_time = 35.0  # elapsed = 50 - 35 = 15 yr
    last_Tmagma = 3000.0
    last_Phi = 0.8

    # Patch zalmoxis_solver at its source module (locally imported)
    with (
        patch('proteus.interior_struct.zalmoxis.zalmoxis_solver', return_value=(3.504e6, None)),
        patch('proteus.interior_energetics.wrapper.np.savetxt'),
        patch('proteus.interior_energetics.wrapper.shutil.copy2'),
    ):
        result = update_structure_from_interior(
            dirs, config, hf_row, interior_o, last_struct_time, last_Tmagma, last_Phi
        )
    # Should have triggered (returned updated values)
    assert result[0] == 50.0  # last_struct_time updated to current

    # Case 2: mesh_shift_active=False, elapsed=15 yr (< floor 100 yr)
    # Should NOT trigger (floor blocks)
    dirs['mesh_shift_active'] = False
    last_struct_time = 35.0
    result = update_structure_from_interior(
        dirs, config, hf_row, interior_o, last_struct_time, last_Tmagma, last_Phi
    )
    assert result[0] == last_struct_time

    # Case 3: mesh_shift_active=True, elapsed=5 yr (< convergence_interval=10)
    # Should NOT trigger (below convergence interval)
    dirs['mesh_shift_active'] = True
    last_struct_time = 45.0  # elapsed = 50 - 45 = 5 yr
    result = update_structure_from_interior(
        dirs, config, hf_row, interior_o, last_struct_time, last_Tmagma, last_Phi
    )
    assert result[0] == last_struct_time


# ============================================================================
# test remap_entropy_for_new_mesh
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not os.path.isfile(SPIDER_JSON), reason='SPIDER output not available')
def test_remap_entropy_identity(spider_json_dir):
    """Identical old and new mesh should produce no change.

    When the mesh file has the same radii as the JSON, the max relative
    radius difference is below the 1e-10 threshold, so the function
    should return False and leave the JSON untouched.
    """
    test_json = _copy_json(spider_json_dir)
    r_b, r_s, _ = _get_mesh_arrays_from_json(test_json)

    mesh_path = os.path.join(spider_json_dir, 'mesh.dat')
    _make_mesh_file(mesh_path, r_b, r_s)

    result = remap_entropy_for_new_mesh(test_json, mesh_path, r_b[0])
    assert result is False

    # Solution should be bit-identical to original
    with open(SPIDER_JSON) as f:
        orig = json.load(f)
    with open(test_json) as f:
        after = json.load(f)

    orig_vals = orig['solution']['subdomain data'][0]['values']
    after_vals = after['solution']['subdomain data'][0]['values']
    assert orig_vals == after_vals


@pytest.mark.unit
@pytest.mark.skipif(not os.path.isfile(SPIDER_JSON), reason='SPIDER output not available')
def test_remap_entropy_small_shift(spider_json_dir):
    """A 1% mesh shift should produce a correctly interpolated entropy profile.

    After remapping, the new dS/dxi + S0, when reconstructed on the new xi
    grid, should match the entropy interpolated from old to new positions.
    """
    test_json = _copy_json(spider_json_dir)
    r_b_old, r_s_old, S_s_old = _get_mesh_arrays_from_json(test_json)

    # Create a 1% shifted mesh
    r_b_new = r_b_old * 1.01
    r_s_new = r_s_old * 1.01
    mesh_path = os.path.join(spider_json_dir, 'mesh.dat')
    _make_mesh_file(mesh_path, r_b_new, r_s_new)

    result = remap_entropy_for_new_mesh(test_json, mesh_path, r_b_old[0])
    assert result is True

    # Read back and reconstruct S(xi) using SPIDER's forward algorithm
    with open(test_json) as f:
        jdata = json.load(f)
    sd = jdata['solution']['subdomain data']
    dSdxi = np.array([float(v) for v in sd[0]['values']])
    S0 = float(sd[1]['values'][0])
    S0_scaling = float(sd[1]['scaling'])

    N_b = len(r_b_new)
    N_s = N_b - 1
    R_surf = r_b_new[0]
    R_cmb = r_b_new[-1]
    coresize = R_cmb / R_surf
    radius_nondim = R_surf / RADIUS0
    dx_b = -radius_nondim * (1.0 - coresize) / (N_b - 1)
    xi_s = np.array(
        [radius_nondim * coresize - 0.5 * dx_b - (N_s - 1 - i) * dx_b for i in range(N_s)]
    )

    # Forward reconstruction (from util.c set_entropy_reconstruction_from_ctx)
    S_recon = np.zeros(N_s)
    for i in range(1, N_b - 1):
        S_recon[i] = dSdxi[i] * (xi_s[i] - xi_s[i - 1]) + S_recon[i - 1]
    S_recon += S0
    S_recon_phys = S_recon * S0_scaling

    # Expected: interpolation of old S(r) onto new positions
    S_expected = np.interp(r_s_new[::-1], r_s_old[::-1], S_s_old[::-1])[::-1]

    np.testing.assert_allclose(S_recon_phys, S_expected, rtol=1e-12)


@pytest.mark.unit
@pytest.mark.skipif(not os.path.isfile(SPIDER_JSON), reason='SPIDER output not available')
def test_remap_entropy_preserves_volatiles(spider_json_dir):
    """Subdomains beyond 0 and 1 must be left untouched by the remap.

    Injects a fake subdomain 2 (volatile partial pressures) and verifies
    that it survives the remap unchanged.
    """
    test_json = _copy_json(spider_json_dir)
    r_b_old, r_s_old, _ = _get_mesh_arrays_from_json(test_json)

    # Inject a fake volatile subdomain
    with open(test_json) as f:
        jdata = json.load(f)
    fake_volatile = {
        'subdomain': 2,
        'description': 'volatile partial pressures',
        'units': 'Pa',
        'size': 3,
        'scaling': '1.0e5',
        'scaled': 'false',
        'values': ['0.123456789', '0.987654321', '0.555555555'],
    }
    jdata['solution']['subdomain data'].append(fake_volatile)
    jdata['solution']['subdomains'] = 3
    with open(test_json, 'w') as f:
        json.dump(jdata, f, indent=2)

    # Create shifted mesh and remap
    r_b_new = r_b_old * 1.005
    r_s_new = r_s_old * 1.005
    mesh_path = os.path.join(spider_json_dir, 'mesh.dat')
    _make_mesh_file(mesh_path, r_b_new, r_s_new)

    result = remap_entropy_for_new_mesh(test_json, mesh_path, r_b_old[0])
    assert result is True

    # Verify subdomain 2 is unchanged
    with open(test_json) as f:
        jdata_after = json.load(f)
    sd2 = jdata_after['solution']['subdomain data'][2]
    assert sd2['values'] == fake_volatile['values']
    assert sd2['description'] == fake_volatile['description']


@pytest.mark.unit
@pytest.mark.skipif(not os.path.isfile(SPIDER_JSON), reason='SPIDER output not available')
def test_remap_entropy_node_count_mismatch(spider_json_dir):
    """If the new mesh has different node count, remap should return False."""
    test_json = _copy_json(spider_json_dir)
    r_b_old, _, _ = _get_mesh_arrays_from_json(test_json)

    # Create mesh with wrong node count
    r_b_wrong = np.linspace(6.371e6, 3.504e6, 30)
    r_s_wrong = 0.5 * (r_b_wrong[:-1] + r_b_wrong[1:])
    mesh_path = os.path.join(spider_json_dir, 'mesh.dat')
    _make_mesh_file(mesh_path, r_b_wrong, r_s_wrong)

    result = remap_entropy_for_new_mesh(test_json, mesh_path, r_b_old[0])
    assert result is False


@pytest.mark.unit
@pytest.mark.skipif(not os.path.isfile(SPIDER_JSON), reason='SPIDER output not available')
def test_remap_entropy_on_blended_mesh(spider_json_dir):
    """Entropy remap on a 5%-clamped (blended) mesh produces correct S(r).

    Simulates the real workflow: Zalmoxis produces a 20% shifted mesh,
    blend_mesh_files clamps it to 5%, then remap_entropy_for_new_mesh
    adjusts the entropy. The result should be a valid interpolation.
    """
    test_json = _copy_json(spider_json_dir)
    r_b_old, r_s_old, S_s_old = _get_mesh_arrays_from_json(test_json)

    # Create old mesh (matching JSON)
    old_mesh = os.path.join(spider_json_dir, 'old.dat')
    _make_mesh_file(old_mesh, r_b_old, r_s_old)

    # Create 20% shifted new mesh
    new_mesh = os.path.join(spider_json_dir, 'new.dat')
    r_b_new = r_b_old * 1.20
    r_s_new = r_s_old * 1.20
    _make_mesh_file(new_mesh, r_b_new, r_s_new)

    # Blend to 5%
    actual_shift = blend_mesh_files(old_mesh, new_mesh, max_shift=0.05)
    assert actual_shift > 0.05  # confirms blending occurred

    # Read blended mesh
    r_b_blended = _read_mesh_file(new_mesh)[0]
    r_s_blended = _read_mesh_file(new_mesh)[4]

    # Remap entropy onto blended mesh
    result = remap_entropy_for_new_mesh(test_json, new_mesh, r_b_old[0])
    assert result is True

    # Verify reconstructed entropy is reasonable
    with open(test_json) as f:
        jdata = json.load(f)
    sd = jdata['solution']['subdomain data']
    S0_scaling = float(sd[1]['scaling'])
    dSdxi = np.array([float(v) for v in sd[0]['values']])
    S0 = float(sd[1]['values'][0])

    N_b = len(r_b_blended)
    N_s = N_b - 1
    R_surf = r_b_blended[0]
    R_cmb = r_b_blended[-1]
    coresize = R_cmb / R_surf
    radius_nondim = R_surf / RADIUS0
    dx_b = -radius_nondim * (1.0 - coresize) / (N_b - 1)
    xi_s = np.array(
        [radius_nondim * coresize - 0.5 * dx_b - (N_s - 1 - i) * dx_b for i in range(N_s)]
    )

    S_recon = np.zeros(N_s)
    for i in range(1, N_b - 1):
        S_recon[i] = dSdxi[i] * (xi_s[i] - xi_s[i - 1]) + S_recon[i - 1]
    S_recon += S0
    S_recon_phys = S_recon * S0_scaling

    # Expected: interpolation of old S(r) onto blended positions
    S_expected = np.interp(r_s_blended[::-1], r_s_old[::-1], S_s_old[::-1])[::-1]

    np.testing.assert_allclose(S_recon_phys, S_expected, rtol=1e-12)


# ============================================================================
# test _coresize_from_mesh
# ============================================================================


@pytest.mark.unit
def test_coresize_from_mesh_earth_like(spider_json_dir):
    """Earth-like mesh (coresize=0.55) should return correct ratio."""
    N_b = 50
    r_b = np.linspace(6.371e6, 6.371e6 * 0.55, N_b)
    r_s = 0.5 * (r_b[:-1] + r_b[1:])
    mesh_path = os.path.join(spider_json_dir, 'mesh.dat')
    _make_mesh_file(mesh_path, r_b, r_s)

    coresize = _coresize_from_mesh(mesh_path)
    assert pytest.approx(coresize, rel=1e-6) == 0.55


@pytest.mark.unit
def test_coresize_from_mesh_small_core(spider_json_dir):
    """CMF=0.1 (coresize~0.46) returns plausible ratio < 0.5."""
    N_b = 20
    r_b = np.linspace(8.0e6, 8.0e6 * 0.46, N_b)
    r_s = 0.5 * (r_b[:-1] + r_b[1:])
    mesh_path = os.path.join(spider_json_dir, 'mesh.dat')
    _make_mesh_file(mesh_path, r_b, r_s)

    coresize = _coresize_from_mesh(mesh_path)
    assert pytest.approx(coresize, rel=1e-6) == 0.46


@pytest.mark.unit
def test_coresize_from_mesh_minimal(spider_json_dir):
    """Minimal 3-node mesh still returns correct ratio."""
    r_b = np.array([1e7, 7e6, 5e6])
    r_s = np.array([8.5e6, 6e6])
    mesh_path = os.path.join(spider_json_dir, 'mesh.dat')
    _make_mesh_file(mesh_path, r_b, r_s)

    coresize = _coresize_from_mesh(mesh_path)
    assert pytest.approx(coresize, rel=1e-6) == 0.5


# ============================================================================
# test _write_mesh_file and round-trip with _read_mesh_file
# ============================================================================


@pytest.mark.unit
def test_write_mesh_file_roundtrip(spider_json_dir):
    """Write and read back a mesh file, verifying full-precision round-trip."""
    N_b = 50
    r_b = np.linspace(6.371e6, 3.504e6, N_b)
    r_s = 0.5 * (r_b[:-1] + r_b[1:])
    P_b = np.linspace(1e5, 130e9, N_b)
    rho_b = np.linspace(3500, 5500, N_b)
    g_b = np.linspace(-9.81, -10.5, N_b)
    P_s = 0.5 * (P_b[:-1] + P_b[1:])
    rho_s = 0.5 * (rho_b[:-1] + rho_b[1:])
    g_s = 0.5 * (g_b[:-1] + g_b[1:])

    mesh_path = os.path.join(spider_json_dir, 'mesh.dat')
    _write_mesh_file(mesh_path, r_b, P_b, rho_b, g_b, r_s, P_s, rho_s, g_s)

    result = _read_mesh_file(mesh_path)
    np.testing.assert_allclose(result[0], r_b, rtol=1e-14)
    np.testing.assert_allclose(result[1], P_b, rtol=1e-14)
    np.testing.assert_allclose(result[2], rho_b, rtol=1e-14)
    np.testing.assert_allclose(result[3], g_b, rtol=1e-14)
    np.testing.assert_allclose(result[4], r_s, rtol=1e-14)
    np.testing.assert_allclose(result[5], P_s, rtol=1e-14)
    np.testing.assert_allclose(result[6], rho_s, rtol=1e-14)
    np.testing.assert_allclose(result[7], g_s, rtol=1e-14)


@pytest.mark.unit
def test_write_mesh_file_format(spider_json_dir):
    """Verify the output file uses .15e format and correct header."""
    r_b = np.array([6.371e6, 3.504e6])
    r_s = np.array([4.937e6])
    P_b = np.array([0.0, 1.377e11])
    rho_b = np.array([3500.0, 5500.0])
    g_b = np.array([-9.81, -10.5])
    P_s = np.array([6.885e10])
    rho_s = np.array([4500.0])
    g_s = np.array([-10.15])

    mesh_path = os.path.join(spider_json_dir, 'mesh.dat')
    _write_mesh_file(mesh_path, r_b, P_b, rho_b, g_b, r_s, P_s, rho_s, g_s)

    with open(mesh_path) as f:
        header = f.readline()
        first_line = f.readline()

    assert header.strip() == '# 2 1'
    # .15e format produces strings like "6.371000000000000e+06"
    assert 'e+' in first_line or 'e-' in first_line


# ============================================================================
# test _rewrite_json_solution
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not os.path.isfile(SPIDER_JSON), reason='SPIDER output not available')
def test_rewrite_json_solution_updates_values(spider_json_dir):
    """Verify that subdomain 0 and 1 values are correctly overwritten."""
    test_json = _copy_json(spider_json_dir)

    with open(test_json) as f:
        orig = json.load(f)
    n_basic = int(orig['solution']['subdomain data'][0]['size'])

    new_dSdxi = np.linspace(-1.0, -0.5, n_basic)
    new_S0 = 0.42

    _rewrite_json_solution(test_json, new_dSdxi, new_S0)

    with open(test_json) as f:
        modified = json.load(f)
    sd = modified['solution']['subdomain data']

    # Check subdomain 0 updated
    for i, v in enumerate(sd[0]['values']):
        assert pytest.approx(float(v), rel=1e-15) == new_dSdxi[i]

    # Check subdomain 1 updated
    assert pytest.approx(float(sd[1]['values'][0]), rel=1e-15) == new_S0


@pytest.mark.unit
@pytest.mark.skipif(not os.path.isfile(SPIDER_JSON), reason='SPIDER output not available')
def test_rewrite_json_solution_preserves_structure(spider_json_dir):
    """JSON structure and non-solution fields remain intact after rewrite."""
    test_json = _copy_json(spider_json_dir)

    with open(test_json) as f:
        orig = json.load(f)

    n_basic = int(orig['solution']['subdomain data'][0]['size'])
    _rewrite_json_solution(test_json, np.zeros(n_basic), 1.0)

    with open(test_json) as f:
        modified = json.load(f)

    # Non-solution fields preserved
    assert modified['data']['radius_b']['scaling'] == orig['data']['radius_b']['scaling']
    assert modified['step'] == orig['step']


# ============================================================================
# test _check_eos_table_range
# ============================================================================


def _make_eos_table(path, head=5, nx=10, ny=10, x_scale=1e9, y_scale=1.0, y_max=3000.0):
    """Write a minimal fake P-S EOS table for testing."""
    with open(path, 'w') as f:
        f.write(f'# {head} {nx} {ny}\n')
        for _ in range(head - 2):
            f.write('# filler\n')
        f.write(f'# {x_scale} {y_scale} 1.0\n')
        y_min = 1000.0
        dy = (y_max - y_min) / max(ny - 1, 1)
        for j in range(ny):
            y_val = y_min + j * dy
            for i in range(nx):
                f.write(f'{float(i) / nx} {y_val / y_scale} 5000.0\n')


@pytest.mark.unit
def test_check_eos_range_entropy_mismatch(spider_json_dir, caplog):
    """Warn when solid entropy range is narrower than melt."""
    eos_dir = spider_json_dir
    _make_eos_table(os.path.join(eos_dir, 'density_solid.dat'), y_max=2400.0)
    _make_eos_table(os.path.join(eos_dir, 'density_melt.dat'), y_max=3200.0)

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.spider'):
        _check_eos_table_range(eos_dir, None, 100e9)
    assert any('narrower' in r.message for r in caplog.records)


@pytest.mark.unit
def test_check_eos_range_high_pressure(spider_json_dir, caplog):
    """High CMB pressure still triggers entropy range warning when solid range is narrow."""
    eos_dir = spider_json_dir
    _make_eos_table(os.path.join(eos_dir, 'density_solid.dat'), y_max=2400.0)
    _make_eos_table(os.path.join(eos_dir, 'density_melt.dat'), y_max=3200.0)

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.spider'):
        _check_eos_table_range(eos_dir, None, 500e9)
    assert any('narrower' in r.message for r in caplog.records)


@pytest.mark.unit
def test_check_eos_range_missing_files(spider_json_dir):
    """Missing EOS files should return silently (non-critical check)."""
    _check_eos_table_range(spider_json_dir, None, 500e9)  # No crash


@pytest.mark.unit
def test_check_eos_range_no_warnings(spider_json_dir, caplog):
    """Safe conditions (equal ranges, P < 400 GPa) produce no warnings."""
    eos_dir = spider_json_dir
    _make_eos_table(os.path.join(eos_dir, 'density_solid.dat'), y_max=3200.0)
    _make_eos_table(os.path.join(eos_dir, 'density_melt.dat'), y_max=3200.0)

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.spider'):
        _check_eos_table_range(eos_dir, None, 100e9)
    spider_warnings = [r for r in caplog.records if 'spider' in r.name]
    assert len(spider_warnings) == 0


@pytest.mark.unit
def test_check_eos_range_parse_failure(spider_json_dir):
    """Malformed EOS tables should return silently (parse failure path)."""
    eos_dir = spider_json_dir
    # Write garbage that will fail to parse
    with open(os.path.join(eos_dir, 'density_solid.dat'), 'w') as f:
        f.write('not a valid header\n')
    with open(os.path.join(eos_dir, 'density_melt.dat'), 'w') as f:
        f.write('also garbage\n')
    # Should not crash — hits the except (ValueError, IndexError, IOError) path
    _check_eos_table_range(eos_dir, None, 500e9)


# ============================================================================
# test _try_spider call-sequence building
# ============================================================================


_EOS_FILE_NAMES = [
    'density_melt.dat',
    'density_solid.dat',
    'thermal_exp_melt.dat',
    'thermal_exp_solid.dat',
    'heat_capacity_melt.dat',
    'heat_capacity_solid.dat',
    'adiabat_temp_grad_melt.dat',
    'adiabat_temp_grad_solid.dat',
    'temperature_melt.dat',
    'temperature_solid.dat',
]


def _setup_spider_env(tmp_path, *, with_mesh=False):
    """Create directories, fake files, and config for _try_spider tests.

    Returns (dirs, config, hf_row, eos_base, mc_base, mesh_path_or_None).
    eos_base and mc_base are the parents to patch EOS_DYNAMIC_DIR and
    MELTING_CURVES_DIR module-level constants.
    """
    spider_dir = tmp_path / 'spider'
    spider_dir.mkdir()
    # Fake spider executable
    spider_exec = spider_dir / 'spider'
    spider_exec.write_text('#!/bin/sh\nexit 0')
    spider_exec.chmod(0o755)

    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    data_dir = output_dir / 'data'
    data_dir.mkdir()

    # EOS directory matching config.interior_energetics.eos_dir
    eos_base = tmp_path / 'eos_dynamic'
    eos_dir = eos_base / 'WolfBower2018_MgSiO3' / 'P-S'
    eos_dir.mkdir(parents=True)
    for name in _EOS_FILE_NAMES:
        _make_eos_table(str(eos_dir / name))

    # Melting curves directory matching config.interior_energetics.melting_dir
    mc_base = tmp_path / 'melting_curves'
    mc_dir = mc_base / 'Wolf_Bower+2018'
    mc_dir.mkdir(parents=True)
    (mc_dir / 'liquidus_P-S.dat').write_text('dummy')
    (mc_dir / 'solidus_P-S.dat').write_text('dummy')

    config = MagicMock()
    config.interior_energetics.spider.tolerance_rel = 1e-4
    config.interior_energetics.tmagma_rtol = 0.02
    config.interior_energetics.tmagma_atol = 100.0
    config.interior_energetics.mixing_length = 'nearest'
    config.interior_energetics.spider.solver_type = 'cv_bdf'
    config.interior_energetics.trans_conduction = True
    config.interior_energetics.trans_convection = True
    config.interior_energetics.trans_mixing = True
    config.interior_energetics.trans_grav_sep = False
    config.interior_energetics.spider.matprop_smooth_width = 0.1
    config.interior_energetics.heat_tidal = False
    config.interior_energetics.heat_radiogenic = False
    config.interior_energetics.grain_size = 1e-3
    config.interior_energetics.rfront_loc = 0.4
    config.interior_energetics.rfront_wid = 0.15
    config.interior_energetics.num_levels = 50
    config.interior_energetics.num_tolerance = 1e-4
    config.planet.tsurf_init = 4000.0
    config.interior_energetics.kappah_floor = 0.0
    config.interior_energetics.flux_guess = -1
    config.interior_struct.eos_dir = 'WolfBower2018_MgSiO3'
    config.interior_struct.melting_dir = 'Wolf_Bower+2018'
    config.outgas.fO2_shift_IW = 0.0
    config.interior_struct.core_frac = 0.55
    config.interior_struct.core_density = 12500.0
    config.interior_struct.core_heatcap = 880.0

    dirs = {
        'spider': str(spider_dir),
        'output': str(output_dir) + '/',
        'output/data': str(data_dir),
        'proteus': str(tmp_path),
    }

    hf_row = {
        'Time': 0.0,
        'F_atm': 100.0,
        'T_eqm': 255.0,
        'R_int': 6.371e6,
        'gravity': 9.81,
        'T_magma': 3000.0,
    }

    mesh_path = None
    if with_mesh:
        N_b = 50
        r_b = np.linspace(6.371e6, 6.371e6 * 0.55, N_b)
        r_s = 0.5 * (r_b[:-1] + r_b[1:])
        mesh_path = str(tmp_path / 'mesh.dat')
        _make_mesh_file(mesh_path, r_b, r_s)

    return dirs, config, hf_row, str(eos_base), str(mc_base), mesh_path


@pytest.mark.unit
def test_try_spider_init_with_mesh(tmp_path):
    """_try_spider with IC_INTERIOR=1 and external mesh file.

    Verifies that the EOS path resolution, melting curve validation,
    and -MESH_SOURCE 1 arguments are correctly added to the call sequence.
    """
    from proteus.interior_energetics.spider import _try_spider

    dirs, config, hf_row, eos_base, mc_base, mesh_path = _setup_spider_env(
        tmp_path, with_mesh=True
    )

    with (
        patch('proteus.interior_energetics.spider.EOS_DYNAMIC_DIR', eos_base),
        patch('proteus.interior_energetics.spider.MELTING_CURVES_DIR', mc_base),
        patch('proteus.interior_energetics.spider.sp.run') as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        result = _try_spider(
            dirs,
            config,
            IC_INTERIOR=1,
            hf_all=None,
            hf_row=hf_row,
            step_sf=1.0,
            atol_sf=1.0,
            dT_max=1000.0,
            mesh_file=mesh_path,
        )

    assert result is True
    call_args = mock_run.call_args[0][0]

    # Verify mesh mode
    assert '-MESH_SOURCE' in call_args
    idx = call_args.index('-MESH_SOURCE')
    assert call_args[idx + 1] == '1'
    assert '-mesh_external_filename' in call_args

    # Verify EOS files
    assert '-melt_rho_filename' in call_args
    assert '-solid_rho_filename' in call_args
    assert '-melt_phase_boundary_filename' in call_args
    assert '-solid_phase_boundary_filename' in call_args

    # Verify coresize extracted from mesh (0.55)
    idx = call_args.index('-coresize')
    coresize_val = float(call_args[idx + 1])
    assert pytest.approx(coresize_val, rel=1e-3) == 0.55

    # Without M_core in hf_row, rho_core should fall back to config value
    idx = call_args.index('-rho_core')
    rho_val = float(call_args[idx + 1])
    assert pytest.approx(rho_val, rel=1e-3) == config.interior_struct.core_density


@pytest.mark.unit
def test_try_spider_rho_core_from_zalmoxis(tmp_path):
    """When hf_row contains M_core (set by Zalmoxis), SPIDER receives
    the effective average core density derived from M_core and R_cmb,
    not the static config.interior_struct.core_density value.

    This ensures the core thermal inertia in SPIDER's CMB boundary
    condition is consistent with Zalmoxis's self-consistent structure.
    """
    from proteus.interior_energetics.spider import _try_spider

    dirs, config, hf_row, eos_base, mc_base, mesh_path = _setup_spider_env(
        tmp_path, with_mesh=True
    )

    # Set M_core as Zalmoxis would (different from config.interior_struct.core_density
    # * 4/3 pi R_cmb^3 to demonstrate the fix)
    R_int = hf_row['R_int']  # 6.371e6 m
    coresize = 0.55  # matches mesh
    R_cmb = coresize * R_int
    M_core_zalmoxis = 1.94e24  # kg, realistic core mass from EOS
    hf_row['M_core'] = M_core_zalmoxis

    expected_rho = M_core_zalmoxis / (4.0 / 3.0 * np.pi * R_cmb**3)

    with (
        patch('proteus.interior_energetics.spider.EOS_DYNAMIC_DIR', eos_base),
        patch('proteus.interior_energetics.spider.MELTING_CURVES_DIR', mc_base),
        patch('proteus.interior_energetics.spider.sp.run') as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        result = _try_spider(
            dirs,
            config,
            IC_INTERIOR=1,
            hf_all=None,
            hf_row=hf_row,
            step_sf=1.0,
            atol_sf=1.0,
            dT_max=1000.0,
            mesh_file=mesh_path,
        )

    assert result is True
    call_args = mock_run.call_args[0][0]

    idx = call_args.index('-rho_core')
    rho_val = float(call_args[idx + 1])
    assert pytest.approx(rho_val, rel=1e-3) == expected_rho
    # Confirm it differs from the static config value
    assert rho_val != pytest.approx(config.interior_struct.core_density, rel=1e-2)


@pytest.mark.unit
def test_try_spider_init_aw(tmp_path):
    """_try_spider with IC_INTERIOR=1, no mesh file (Adams-Williamson mode).

    Verifies AW parameters are added instead of -MESH_SOURCE.
    """
    from proteus.interior_energetics.spider import _try_spider

    dirs, config, hf_row, eos_base, mc_base, _ = _setup_spider_env(tmp_path)

    with (
        patch('proteus.interior_energetics.spider.EOS_DYNAMIC_DIR', eos_base),
        patch('proteus.interior_energetics.spider.MELTING_CURVES_DIR', mc_base),
        patch('proteus.interior_energetics.spider.sp.run') as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        result = _try_spider(
            dirs,
            config,
            IC_INTERIOR=1,
            hf_all=None,
            hf_row=hf_row,
            step_sf=1.0,
            atol_sf=1.0,
            dT_max=1000.0,
        )

    assert result is True
    call_args = mock_run.call_args[0][0]

    # No MESH_SOURCE in AW mode
    assert '-MESH_SOURCE' not in call_args
    assert '-adams_williamson_rhos' in call_args
    assert '-adams_williamson_beta' in call_args


@pytest.mark.unit
def test_try_spider_missing_eos_dir(tmp_path):
    """Missing EOS directory raises FileNotFoundError."""
    from proteus.interior_energetics.spider import _try_spider

    dirs, config, hf_row, _, mc_base, _ = _setup_spider_env(tmp_path)

    # Both FWL_DATA EOS path and SPIDER-local fallback (lookup_data/) are absent
    with (
        patch('proteus.interior_energetics.spider.EOS_DYNAMIC_DIR', '/nonexistent/eos'),
        patch('proteus.interior_energetics.spider.MELTING_CURVES_DIR', mc_base),
    ):
        with pytest.raises(FileNotFoundError, match='SPIDER EOS directory not found'):
            _try_spider(
                dirs,
                config,
                IC_INTERIOR=1,
                hf_all=None,
                hf_row=hf_row,
                step_sf=1.0,
                atol_sf=1.0,
                dT_max=1000.0,
            )


@pytest.mark.unit
def test_try_spider_missing_melting_curves(tmp_path):
    """Missing melting curve files raise FileNotFoundError."""
    from proteus.interior_energetics.spider import _try_spider

    dirs, config, hf_row, eos_base, _, _ = _setup_spider_env(tmp_path)

    with (
        patch('proteus.interior_energetics.spider.EOS_DYNAMIC_DIR', eos_base),
        patch('proteus.interior_energetics.spider.MELTING_CURVES_DIR', '/nonexistent/mc'),
    ):
        with pytest.raises(FileNotFoundError, match='SPIDER phase boundary file'):
            _try_spider(
                dirs,
                config,
                IC_INTERIOR=1,
                hf_all=None,
                hf_row=hf_row,
                step_sf=1.0,
                atol_sf=1.0,
                dT_max=1000.0,
            )


@pytest.mark.unit
def test_try_spider_eos_fallback_to_local(tmp_path):
    """EOS dir resolves to SPIDER local fallback when FWL_DATA path missing."""
    from proteus.interior_energetics.spider import _try_spider

    dirs, config, hf_row, _, mc_base, _ = _setup_spider_env(tmp_path)

    # Create fallback EOS in SPIDER local dir
    local_eos = os.path.join(dirs['spider'], 'lookup_data', '1TPa-dK09-elec-free')
    os.makedirs(local_eos)
    for name in _EOS_FILE_NAMES:
        _make_eos_table(os.path.join(local_eos, name))

    with (
        patch('proteus.interior_energetics.spider.EOS_DYNAMIC_DIR', '/nonexistent/eos'),
        patch('proteus.interior_energetics.spider.MELTING_CURVES_DIR', mc_base),
        patch('proteus.interior_energetics.spider.sp.run') as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        result = _try_spider(
            dirs,
            config,
            IC_INTERIOR=1,
            hf_all=None,
            hf_row=hf_row,
            step_sf=1.0,
            atol_sf=1.0,
            dT_max=1000.0,
        )

    assert result is True
    call_args = mock_run.call_args[0][0]
    # EOS paths should reference the local fallback
    idx = call_args.index('-melt_rho_filename')
    assert '1TPa-dK09-elec-free' in call_args[idx + 1]


@pytest.mark.unit
def test_try_spider_subprocess_timeout(tmp_path):
    """SPIDER subprocess timeout returns False."""
    import subprocess as sub

    from proteus.interior_energetics.spider import _try_spider

    dirs, config, hf_row, eos_base, mc_base, _ = _setup_spider_env(tmp_path)

    with (
        patch('proteus.interior_energetics.spider.EOS_DYNAMIC_DIR', eos_base),
        patch('proteus.interior_energetics.spider.MELTING_CURVES_DIR', mc_base),
        patch(
            'proteus.interior_energetics.spider.sp.run',
            side_effect=sub.TimeoutExpired(cmd='spider', timeout=60),
        ),
    ):
        result = _try_spider(
            dirs,
            config,
            IC_INTERIOR=1,
            hf_all=None,
            hf_row=hf_row,
            step_sf=1.0,
            atol_sf=1.0,
            dT_max=1000.0,
        )

    assert result is False


@pytest.mark.unit
def test_blend_mesh_malformed_file(spider_json_dir):
    """Malformed mesh file in blend_mesh_files should return 0.0."""
    old_path = os.path.join(spider_json_dir, 'old.dat')
    new_path = os.path.join(spider_json_dir, 'new.dat')

    # Write malformed files
    with open(old_path, 'w') as f:
        f.write('garbage content\n')
    with open(new_path, 'w') as f:
        f.write('also garbage\n')

    result = blend_mesh_files(old_path, new_path, max_shift=0.05)
    assert result == 0.0


# ============================================================================
# test MyJSON, get_all_output_times, read_jsons, interp_rho_melt
# ============================================================================


def _make_spider_json(filepath, step=0, sim_time=0.0, num_stag=10, num_basic=11):
    """Write a minimal SPIDER-compatible JSON file for testing."""
    import json as json_mod

    n_s = num_stag
    n_b = num_basic

    data = {
        'step': step,
        'atmosphere': {
            'mass_liquid': {'scaling': 1, 'units': 'kg', 'values': [1e23]},
            'mass_solid': {'scaling': 1, 'units': 'kg', 'values': [3e24]},
            'mass_mantle': {'scaling': 1, 'units': 'kg', 'values': [4e24]},
            'mass_core': {'scaling': 1, 'units': 'kg', 'values': [2e24]},
            'temperature_surface': {'scaling': 1, 'units': 'K', 'values': [2500.0]},
            'Fatm': {'scaling': 1, 'units': 'W/m2', 'values': [1e5]},
        },
        'rheological_front_phi': {
            'phi_global': {'scaling': 1, 'units': 'None', 'values': [0.3]},
        },
        'rheological_front_dynamic': {
            'depth': {'scaling': 1, 'units': 'm', 'values': [3e6]},
        },
        'data': {
            'area_b': {'scaling': 1, 'units': 'm2', 'values': [5.1e14] * n_b},
            'Hradio_s': {'scaling': 1, 'units': 'W/kg', 'values': [1e-12] * n_s},
            'Htidal_s': {'scaling': 1, 'units': 'W/kg', 'values': [0.0] * n_s},
            'mass_s': {'scaling': 1, 'units': 'kg', 'values': [4e23] * n_s},
            'phi_s': {'scaling': 1, 'units': 'None', 'values': [0.5] * n_s},
            'phi_b': {'scaling': 1, 'units': 'None', 'values': [0.5] * n_b},
            'rho_s': {'scaling': 1, 'units': 'kg/m3', 'values': [4000.0] * n_s},
            'radius_b': {
                'scaling': 1,
                'units': 'm',
                'values': list(np.linspace(6.371e6, 3.48e6, n_b)),
            },
            'visc_b': {'scaling': 1, 'units': 'Pa.s', 'values': [1e21] * n_b},
            'temp_s': {'scaling': 1, 'units': 'K', 'values': [2500.0] * n_s},
            'pressure_s': {
                'scaling': 1,
                'units': 'Pa',
                'values': list(np.linspace(0, 135e9, n_s)),
            },
            'volume_s': {'scaling': 1, 'units': 'm3', 'values': [1e18] * n_s},
            'S_s': {'scaling': 1, 'units': 'J/(kg.K)', 'values': [2800.0] * n_s},
            'Jconv_b': {'scaling': 1, 'units': 'W/m2', 'values': [1e4] * n_b},
            'Jcond_b': {'scaling': 1, 'units': 'W/m2', 'values': [1e2] * n_b},
        },
    }

    with open(filepath, 'w') as f:
        json_mod.dump(data, f)


@pytest.mark.unit
def test_myjson_load_and_get(tmp_path):
    """MyJSON loads a JSON file and provides dict access."""
    from proteus.interior_energetics.spider import MyJSON

    fpath = str(tmp_path / '0.json')
    _make_spider_json(fpath, step=5)

    jobj = MyJSON(fpath)
    assert jobj.data_d is not None
    assert jobj.get_dict(['step']) == 5


@pytest.mark.unit
def test_myjson_missing_file(tmp_path):
    """MyJSON sets data_d to None when file doesn't exist."""
    from proteus.interior_energetics.spider import MyJSON

    jobj = MyJSON(str(tmp_path / 'missing.json'))
    assert jobj.data_d is None


@pytest.mark.unit
def test_myjson_get_dict_values(tmp_path):
    """MyJSON.get_dict_values returns correctly scaled values."""
    from proteus.interior_energetics.spider import MyJSON

    fpath = str(tmp_path / '0.json')
    _make_spider_json(fpath)

    jobj = MyJSON(fpath)
    # Scalar: temperature_surface (scaling=1, values=[2500.0])
    t_surf = jobj.get_dict_values(('atmosphere', 'temperature_surface'))
    assert t_surf == pytest.approx(2500.0)

    # Array: pressure_s
    p_s = jobj.get_dict_values(['data', 'pressure_s'])
    assert len(p_s) == 10
    assert p_s[0] == pytest.approx(0.0)


@pytest.mark.unit
def test_myjson_phase_boolean_arrays(tmp_path):
    """MyJSON phase boolean arrays work for staggered and basic nodes."""
    from proteus.interior_energetics.spider import MyJSON

    fpath = str(tmp_path / '0.json')
    _make_spider_json(fpath)

    jobj = MyJSON(fpath)
    # phi=0.5 everywhere → all mixed, no pure melt, no pure solid
    mix = jobj.get_mixed_phase_boolean_array(nodes='staggered')
    assert np.all(mix)
    melt = jobj.get_melt_phase_boolean_array(nodes='basic')
    assert not np.any(melt)
    solid = jobj.get_solid_phase_boolean_array(nodes='basic_internal')
    assert not np.any(solid)


@pytest.mark.unit
def test_get_all_output_times(tmp_path):
    """get_all_output_times returns sorted times from JSON filenames."""
    from proteus.interior_energetics.spider import get_all_output_times

    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    # Create fake JSON files
    for t in [100, 0, 500]:
        (data_dir / f'{t}.json').write_text('{}')

    times = get_all_output_times(str(tmp_path))
    np.testing.assert_array_equal(times, [0, 100, 500])


@pytest.mark.unit
def test_get_all_output_times_empty(tmp_path):
    """get_all_output_times raises on empty data directory."""
    from proteus.interior_energetics.spider import get_all_output_times

    data_dir = tmp_path / 'data'
    data_dir.mkdir()

    with pytest.raises(Exception, match='no files'):
        get_all_output_times(str(tmp_path))


@pytest.mark.unit
def test_read_jsons(tmp_path):
    """read_jsons loads MyJSON objects for given times."""
    from proteus.interior_energetics.spider import read_jsons

    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    _make_spider_json(str(data_dir / '0.json'), step=0)
    _make_spider_json(str(data_dir / '100.json'), step=1)

    jsons = read_jsons(str(tmp_path), [0, 100, 999])
    # 999 doesn't exist → skipped
    assert len(jsons) == 2
    assert jsons[0].data_d is not None


@pytest.mark.unit
def test_interp_rho_melt():
    """interp_rho_melt returns a density value from a synthetic lookup table."""
    from proteus.interior_energetics.spider import interp_rho_melt

    # Create a 4x3x3 lookup: [nS, nP, 3] with columns (P, S, rho)
    nP, nS = 3, 4
    P_vals = np.linspace(0, 135e9, nP)
    S_vals = np.linspace(2000, 3000, nS)
    lookup = np.zeros((nS, nP, 3))
    for j in range(nS):
        for i in range(nP):
            lookup[j, i, 0] = P_vals[i]
            lookup[j, i, 1] = S_vals[j]
            lookup[j, i, 2] = 3000.0 + 1e-9 * P_vals[i]

    rho = interp_rho_melt(S=2500.0, P=50e9, lookup=lookup)
    assert 3000 < rho < 3200
    assert isinstance(rho, float)


# ============================================================================
# test RunSPIDER (wrapper with retry logic)
# ============================================================================


@pytest.mark.unit
def test_run_spider_success_first_attempt():
    """RunSPIDER returns True when _try_spider succeeds on first attempt."""
    from proteus.interior_energetics.spider import RunSPIDER

    interior_o = MagicMock()
    interior_o.ic = 1
    interior_o.tides = np.zeros(10)

    config = MagicMock()
    config.interior_energetics.heat_tidal = False

    with patch('proteus.interior_energetics.spider._try_spider', return_value=True):
        result = RunSPIDER(
            dirs={'output': '/tmp', 'output/data': '/tmp/data', 'spider': '/tmp'},
            config=config,
            hf_all=None,
            hf_row={'F_atm': 100.0, 'T_eqm': 255.0},
            interior_o=interior_o,
        )

    assert result is True


@pytest.mark.unit
def test_run_spider_retry_then_success():
    """RunSPIDER retries with relaxed tolerances after first failure."""
    from proteus.interior_energetics.spider import RunSPIDER

    interior_o = MagicMock()
    interior_o.ic = 2
    interior_o.tides = np.zeros(10)

    config = MagicMock()
    config.interior_energetics.heat_tidal = False

    # Fail first attempt, succeed second
    with patch(
        'proteus.interior_energetics.spider._try_spider', side_effect=[False, True]
    ) as mock_try:
        result = RunSPIDER(
            dirs={'output': '/tmp', 'output/data': '/tmp/data', 'spider': '/tmp'},
            config=config,
            hf_all=None,
            hf_row={'F_atm': 100.0, 'T_eqm': 255.0},
            interior_o=interior_o,
        )

    assert result is True
    assert mock_try.call_count == 2


@pytest.mark.unit
def test_run_spider_all_attempts_fail():
    """RunSPIDER raises RuntimeError after max_attempts failures."""
    from proteus.interior_energetics.spider import RunSPIDER

    interior_o = MagicMock()
    interior_o.ic = 1
    interior_o.tides = np.zeros(10)

    config = MagicMock()
    config.interior_energetics.heat_tidal = False

    with (
        patch('proteus.interior_energetics.spider._try_spider', return_value=False),
        patch('proteus.interior_energetics.spider.UpdateStatusfile'),
        pytest.raises(RuntimeError, match='error occurred when executing SPIDER'),
    ):
        RunSPIDER(
            dirs={'output': '/tmp', 'output/data': '/tmp/data', 'spider': '/tmp'},
            config=config,
            hf_all=None,
            hf_row={'F_atm': 100.0, 'T_eqm': 255.0},
            interior_o=interior_o,
        )


@pytest.mark.unit
def test_run_spider_heat_tidal_active():
    """RunSPIDER limits dT_max when tidal heating is active."""
    from proteus.interior_energetics.spider import RunSPIDER

    interior_o = MagicMock()
    interior_o.ic = 1
    interior_o.tides = np.array([1e-5] * 10)  # > 1e-10

    config = MagicMock()
    config.interior_energetics.heat_tidal = True

    with patch('proteus.interior_energetics.spider._try_spider', return_value=True) as mock_try:
        RunSPIDER(
            dirs={'output': '/tmp', 'output/data': '/tmp/data', 'spider': '/tmp'},
            config=config,
            hf_all=None,
            hf_row={'F_atm': 100.0, 'T_eqm': 255.0},
            interior_o=interior_o,
        )

    # dT_max should be 4.0 (tidal heating limit)
    call_kwargs = mock_try.call_args
    assert call_kwargs[1].get('dT_max', call_kwargs[0][7]) == pytest.approx(4.0)


# ============================================================================
# test ReadSPIDER
# ============================================================================


@pytest.mark.unit
def test_read_spider_basic(tmp_path):
    """ReadSPIDER extracts scalars and arrays from a SPIDER JSON file."""
    from proteus.interior_energetics.common import Interior_t
    from proteus.interior_energetics.spider import ReadSPIDER

    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    _make_spider_json(str(data_dir / '0.json'), step=0, num_stag=10, num_basic=11)

    # Create a minimal lookup_rho_melt table
    nP, nS = 3, 4
    P_vals = np.linspace(0, 135e9, nP)
    S_vals = np.linspace(2000, 3200, nS)
    lookup = np.zeros((nS, nP, 3))
    for j in range(nS):
        for i in range(nP):
            lookup[j, i, 0] = P_vals[i]
            lookup[j, i, 1] = S_vals[j]
            lookup[j, i, 2] = 4000.0

    interior_o = Interior_t(11)
    interior_o.lookup_rho_melt = lookup

    config = MagicMock()
    config.planet.prevent_warming = False

    dirs = {
        'output': str(tmp_path),
        'output/data': str(data_dir),
    }

    sim_time, output = ReadSPIDER(dirs, config, R_int=6.371e6, interior_o=interior_o)

    assert sim_time == 0
    assert output['T_magma'] == pytest.approx(2500.0)
    assert output['Phi_global'] == pytest.approx(0.3)
    assert 0 <= output['Phi_global_vol'] <= 1.0
    assert len(interior_o.phi) == 10
    assert len(interior_o.radius) == 11


@pytest.mark.unit
def test_read_spider_prevent_warming(tmp_path):
    """ReadSPIDER clamps F_int to 1e-8 when prevent_warming is True."""
    from proteus.interior_energetics.common import Interior_t
    from proteus.interior_energetics.spider import ReadSPIDER

    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    _make_spider_json(str(data_dir / '0.json'), step=0, num_stag=10, num_basic=11)

    nP, nS = 3, 4
    P_vals = np.linspace(0, 135e9, nP)
    S_vals = np.linspace(2000, 3200, nS)
    lookup = np.zeros((nS, nP, 3))
    for j in range(nS):
        for i in range(nP):
            lookup[j, i, 0] = P_vals[i]
            lookup[j, i, 1] = S_vals[j]
            lookup[j, i, 2] = 4000.0

    interior_o = Interior_t(11)
    interior_o.lookup_rho_melt = lookup

    config = MagicMock()
    config.planet.prevent_warming = True

    dirs = {'output': str(tmp_path), 'output/data': str(data_dir)}

    sim_time, output = ReadSPIDER(dirs, config, R_int=6.371e6, interior_o=interior_o)

    # F_int clamped to at least 1e-8
    assert output['F_int'] >= 1e-8


@pytest.mark.unit
def test_read_spider_nan_temperature(tmp_path):
    """ReadSPIDER raises when T_magma is NaN."""
    from proteus.interior_energetics.common import Interior_t
    from proteus.interior_energetics.spider import ReadSPIDER

    data_dir = tmp_path / 'data'
    data_dir.mkdir()

    # Create a JSON with NaN temperature_surface
    fpath = str(data_dir / '0.json')
    _make_spider_json(fpath, step=0, num_stag=10, num_basic=11)

    # Patch the temperature to NaN
    with open(fpath) as f:
        jdata = json.load(f)
    jdata['atmosphere']['temperature_surface']['values'] = [float('nan')]
    with open(fpath, 'w') as f:
        json.dump(jdata, f)

    nP, nS = 3, 4
    lookup = np.zeros((nS, nP, 3))
    for j in range(nS):
        for i in range(nP):
            lookup[j, i, 0] = i * 45e9
            lookup[j, i, 1] = 2000 + j * 400
            lookup[j, i, 2] = 4000.0

    interior_o = Interior_t(11)
    interior_o.lookup_rho_melt = lookup

    config = MagicMock()
    config.planet.prevent_warming = False

    dirs = {'output': str(tmp_path), 'output/data': str(data_dir)}

    with pytest.raises(Exception, match='NaN'):
        ReadSPIDER(dirs, config, R_int=6.371e6, interior_o=interior_o)


# ============================================================================
# Additional MyJSON method coverage
# ============================================================================


@pytest.mark.unit
def test_myjson_get_dict_units(tmp_path):
    """MyJSON.get_dict_units returns units string or None for 'None'."""
    from proteus.interior_energetics.spider import MyJSON

    fpath = str(tmp_path / '0.json')
    _make_spider_json(fpath)

    jobj = MyJSON(fpath)
    # temperature_surface has units='K'
    units = jobj.get_dict_units(('atmosphere', 'temperature_surface'))
    assert units == 'K'

    # Fatm has units='W/m2'
    units = jobj.get_dict_units(('atmosphere', 'Fatm'))
    assert units == 'W/m2'

    # phi_global has units='None' → should return None
    units = jobj.get_dict_units(('rheological_front_phi', 'phi_global'))
    assert units is None


@pytest.mark.unit
def test_myjson_get_dict_values_internal(tmp_path):
    """MyJSON.get_dict_values_internal strips top and bottom nodes."""
    from proteus.interior_energetics.spider import MyJSON

    fpath = str(tmp_path / '0.json')
    _make_spider_json(fpath, num_basic=11)

    jobj = MyJSON(fpath)
    # radius_b has 11 values; internal should strip first and last → 9
    internal = jobj.get_dict_values_internal(['data', 'radius_b'])
    assert len(internal) == 9


@pytest.mark.unit
def test_myjson_phase_boolean_all_branches(tmp_path):
    """Phase boolean arrays work for basic, basic_internal, and staggered."""
    from proteus.interior_energetics.spider import MyJSON

    fpath = str(tmp_path / '0.json')
    _make_spider_json(fpath, num_stag=10, num_basic=11)

    jobj = MyJSON(fpath)

    # Test all node types for each phase method
    for method_name in (
        'get_mixed_phase_boolean_array',
        'get_melt_phase_boolean_array',
        'get_solid_phase_boolean_array',
    ):
        method = getattr(jobj, method_name)
        for nodes in ('basic', 'basic_internal', 'staggered'):
            result = method(nodes=nodes)
            assert isinstance(result, np.ndarray), f'{method_name}({nodes}) failed'
            if nodes == 'basic':
                assert len(result) == 11
            elif nodes == 'basic_internal':
                assert len(result) == 9
            elif nodes == 'staggered':
                assert len(result) == 10


# ============================================================================
# blend_mesh_files: node count mismatch branch
# ============================================================================


@pytest.mark.unit
def test_blend_mesh_node_count_mismatch(spider_json_dir, caplog):
    """blend_mesh_files returns 0.0 when old and new meshes have different node counts."""
    import logging

    # Create two mesh files with different node counts
    N_b_old = 50
    r_b_old = np.linspace(6.371e6, 3.48e6, N_b_old)
    r_s_old = 0.5 * (r_b_old[:-1] + r_b_old[1:])
    old_path = os.path.join(spider_json_dir, 'old_50.dat')
    _make_mesh_file(old_path, r_b_old, r_s_old)

    N_b_new = 30  # Different count
    r_b_new = np.linspace(6.371e6, 3.48e6, N_b_new)
    r_s_new = 0.5 * (r_b_new[:-1] + r_b_new[1:])
    new_path = os.path.join(spider_json_dir, 'new_30.dat')
    _make_mesh_file(new_path, r_b_new, r_s_new)

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_energetics.spider'):
        result = blend_mesh_files(old_path, new_path, max_shift=0.05)

    assert result == 0.0
    assert 'node count changed' in caplog.text.lower()


# ============================================================================
# _try_spider with IC_INTERIOR=2 (resume path)
# ============================================================================


@pytest.mark.unit
def test_try_spider_resume_ic2(tmp_path):
    """_try_spider with IC_INTERIOR=2 reads JSON to get step number."""
    from proteus.interior_energetics.spider import _try_spider

    dirs, config, hf_row, eos_base, mc_base, _ = _setup_spider_env(tmp_path)

    # IC_INTERIOR=2 needs a JSON file at the current time
    hf_row['Time'] = 100.0
    _make_spider_json(str(tmp_path / 'output' / 'data' / '100.json'), step=5)

    import pandas as pd

    hf_all = pd.DataFrame(
        {
            'Time': [0.0, 50.0, 100.0],
            'T_magma': [3000.0, 2800.0, 2600.0],
        }
    )

    with (
        patch('proteus.interior_energetics.spider.EOS_DYNAMIC_DIR', eos_base),
        patch('proteus.interior_energetics.spider.MELTING_CURVES_DIR', mc_base),
        patch('proteus.interior_energetics.spider.next_step', return_value=50.0),
        patch('proteus.interior_energetics.spider.sp.run') as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        result = _try_spider(
            dirs,
            config,
            IC_INTERIOR=2,
            hf_all=hf_all,
            hf_row=hf_row,
            step_sf=1.0,
            atol_sf=1.0,
            dT_max=1000.0,
        )

    assert result is True
    call_args = mock_run.call_args[0][0]
    # IC=2 sets nstepsmacro from step + nsteps (step=5, nsteps=1)
    idx = call_args.index('-nstepsmacro')
    assert call_args[idx + 1] == '6'


# ============================================================================
# _try_spider with heat_radiogen=True
# ============================================================================


@pytest.mark.unit
def test_try_spider_heat_radiogen(tmp_path):
    """_try_spider adds radionuclide flags when heat_radiogenic is True."""
    from proteus.interior_energetics.spider import _try_spider

    dirs, config, hf_row, eos_base, mc_base, _ = _setup_spider_env(tmp_path)

    # Enable radiogenic heat
    config.interior_energetics.heat_radiogenic = True
    config.interior_energetics.radio_tref = 4.5  # Gyr
    config.star.age_ini = 0.1  # Gyr
    config.interior_energetics.radio_Al = 0.0
    config.interior_energetics.radio_Fe = 0.0
    config.interior_energetics.radio_K = 240e-9  # ppm
    config.interior_energetics.radio_Th = 80e-9
    config.interior_energetics.radio_U = 20e-9

    with (
        patch('proteus.interior_energetics.spider.EOS_DYNAMIC_DIR', eos_base),
        patch('proteus.interior_energetics.spider.MELTING_CURVES_DIR', mc_base),
        patch('proteus.interior_energetics.spider.sp.run') as mock_run,
        patch(
            'proteus.interior_energetics.spider.radnuc_data',
            {
                'k40': {'abundance': 1.17e-4, 'heatprod': 2.92e-5, 'halflife': 1.25e9},
                'th232': {'abundance': 1.0, 'heatprod': 2.64e-5, 'halflife': 1.40e10},
                'u235': {'abundance': 7.2e-3, 'heatprod': 5.69e-4, 'halflife': 7.04e8},
                'u238': {'abundance': 0.9928, 'heatprod': 9.46e-5, 'halflife': 4.47e9},
            },
        ),
    ):
        mock_run.return_value = MagicMock(returncode=0)
        result = _try_spider(
            dirs,
            config,
            IC_INTERIOR=1,
            hf_all=None,
            hf_row=hf_row,
            step_sf=1.0,
            atol_sf=1.0,
            dT_max=1000.0,
        )

    assert result is True
    call_args = mock_run.call_args[0][0]

    # Radionuclide flags should be present
    assert '-radionuclide_names' in call_args
    assert '-k40_t0' in call_args
    assert '-th232_t0' in call_args
    assert '-u235_t0' in call_args
