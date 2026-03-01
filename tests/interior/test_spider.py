"""
Unit tests for proteus.interior.spider module — entropy remapping and mesh blending.

Tests the Phase 2 entropy remap logic that corrects SPIDER's solution
vector when the external mesh changes between timesteps (e.g. when
Zalmoxis updates the density profile from the evolved temperature),
and the mesh blending logic that clamps per-update radius shifts.

Testing standards and documentation:
- docs/test_infrastructure.md: Test infrastructure overview
- docs/test_categorization.md: Test marker definitions
- docs/test_building.md: Best practices for test construction

Functions tested:
- _read_mesh_file(): Parse SPIDER external mesh file (all 8 columns)
- _rewrite_json_solution(): Overwrite solution vector in SPIDER JSON
- blend_mesh_files(): Clamp mesh shift by linear blending
- remap_entropy_for_new_mesh(): Core remapping logic
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import numpy as np
import pytest

from proteus.interior.spider import (
    RADIUS0,
    _read_mesh_file,
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

    from proteus.interior.wrapper import update_structure_from_interior

    # Create minimal config mock
    config = MagicMock()
    config.struct.update_interval = 1000.0
    config.struct.update_min_interval = 100.0
    config.struct.update_dtmagma_frac = 0.03
    config.struct.update_dphi_abs = 0.05
    config.struct.mesh_max_shift = 0.05
    config.struct.mesh_convergence_interval = 10.0
    config.struct.zalmoxis.temperature_mode = 'isothermal'
    config.struct.zalmoxis.temperature_profile_file = None
    config.struct.zalmoxis.num_levels = 50
    config.interior.module = 'spider'
    config.interior.spider.num_levels = 50

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
        patch('proteus.interior.zalmoxis.zalmoxis_solver', return_value=(3.504e6, None)),
        patch('proteus.interior.wrapper.np.savetxt'),
        patch('proteus.interior.wrapper.shutil.copy2'),
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
