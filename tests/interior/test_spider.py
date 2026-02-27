"""
Unit tests for proteus.interior.spider module — entropy remapping.

Tests the Phase 2 entropy remap logic that corrects SPIDER's solution
vector when the external mesh changes between timesteps (e.g. when
Zalmoxis updates the density profile from the evolved temperature).

Testing standards and documentation:
- docs/test_infrastructure.md: Test infrastructure overview
- docs/test_categorization.md: Test marker definitions
- docs/test_building.md: Best practices for test construction

Functions tested:
- _read_mesh_file(): Parse SPIDER external mesh file
- _rewrite_json_solution(): Overwrite solution vector in SPIDER JSON
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


def _make_mesh_file(path: str, r_b: np.ndarray, r_s: np.ndarray) -> None:
    """Write a SPIDER-format mesh file with dummy P, rho, g values."""
    with open(path, 'w') as f:
        f.write(f'# {len(r_b)} {len(r_s)}\n')
        for r in r_b:
            f.write(f'{r:.6f} 1e9 4000.0 -9.81\n')
        for r in r_s:
            f.write(f'{r:.6f} 1e9 4000.0 -9.81\n')


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
# test _read_mesh_file
# ============================================================================


@pytest.mark.unit
def test_read_mesh_file_roundtrip(spider_json_dir):
    """Write a mesh file and verify _read_mesh_file parses it back correctly.

    Uses realistic Earth-like radii (6371 km surface, 3504 km CMB) with
    50 basic and 49 staggered nodes to match SPIDER's default grid.
    """
    N_b = 50
    N_s = N_b - 1
    r_b = np.linspace(6.371e6, 3.504e6, N_b)
    r_s = 0.5 * (r_b[:-1] + r_b[1:])

    mesh_path = os.path.join(spider_json_dir, 'mesh.dat')
    _make_mesh_file(mesh_path, r_b, r_s)

    r_b_read, r_s_read = _read_mesh_file(mesh_path)

    assert len(r_b_read) == N_b
    assert len(r_s_read) == N_s
    np.testing.assert_allclose(r_b_read, r_b, rtol=1e-6)
    np.testing.assert_allclose(r_s_read, r_s, rtol=1e-6)


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
