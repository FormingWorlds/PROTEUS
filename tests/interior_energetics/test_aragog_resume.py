"""Unit tests for proteus.interior_energetics.aragog_resume.

``pending_structure_update`` decides, from the interior snapshot of the row a
run resumes from, if the snapshot is on an earlier structure than the row, and
names the row and mesh file of that structure.

Invariants and contract clauses:

- Radial bounds: the chosen row and mesh file hold the bounds of the snapshot
  mesh to Aragog's mesh tolerance, ``max(1 m, 1e-9 * (R_int - R_core))``; a
  shift below the tolerance is the same mesh and a shift above it is not.
- No pending update when the snapshot mesh is the mesh of the row.
- The saved copy of the earlier row is preferred, then ``zalmoxis_output.dat.prev``.
- Without a row or a mesh file of the snapshot mesh: None and one warning.

Testing standards and documentation:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

import logging

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from proteus.interior_energetics.aragog_resume import (
    STRUCTURE_ROW_KEYS,
    pending_structure_update,
)
from proteus.utils.helper import format_subyear_time

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# A structure before and after a contraction of a 1 M_Earth planet [m].
OLD = {'R_core': 3.4113457e6, 'R_int': 6.8687907e6}
NEW = {'R_core': 3.3866558e6, 'R_int': 6.8190773e6}
T_OLD, T_NEW = 222.001, 322.004


def _mesh_file(path, r_core, r_int):
    """Write a 5-column Zalmoxis mesh file spanning the given radii [m]."""
    r = np.linspace(r_core, r_int, 6)
    cols = np.column_stack([r, np.linspace(1.3e11, 0.0, 6), np.full(6, 4500.0), r, r])
    np.savetxt(path, cols)


def _snapshot(data_dir, time, r_core, r_int, with_radii=True):
    """Write an interior snapshot whose basic-node radii [km] span the given radii [m]."""
    with nc.Dataset(data_dir / (format_subyear_time(time) + '_int.nc'), 'w') as ds:
        ds.createDimension('basic', 4)
        if with_radii:
            var = ds.createVariable('radius_b', 'f8', ('basic',))
            var[:] = np.linspace(r_core, r_int, 4) / 1.0e3


def _run_dir(tmp_path, *, snapshot_on=OLD, saved_copy=True, prev_copy=False):
    """A run directory stopped at T_NEW on the NEW structure; return (dir, hf_row, hf_all)."""
    data = tmp_path / 'data'
    data.mkdir()
    _snapshot(data, T_NEW, snapshot_on['R_core'], snapshot_on['R_int'])
    _mesh_file(data / 'zalmoxis_output.dat', NEW['R_core'], NEW['R_int'])
    if saved_copy:
        _mesh_file(data / (format_subyear_time(T_OLD) + '_zalmoxis.dat'), **_bounds(OLD))
    if prev_copy:
        _mesh_file(data / 'zalmoxis_output.dat.prev', **_bounds(OLD))
    rows = [
        {'Time': 122.0, **OLD, 'gravity': 8.29, 'M_core': 1.92e24, 'core_density': 11574.0},
        {'Time': T_OLD, **OLD, 'gravity': 8.29, 'M_core': 1.92e24, 'core_density': 11574.0},
        {'Time': T_NEW, **NEW, 'gravity': 8.56, 'M_core': 1.95e24, 'core_density': 12003.0},
    ]
    hf_all = pd.DataFrame(rows)
    return tmp_path, hf_all.iloc[-1].to_dict(), hf_all


def _bounds(row):
    return {'r_core': row['R_core'], 'r_int': row['R_int']}


@pytest.mark.physics_invariant
def test_the_earlier_structure_is_found_from_the_bounds_of_the_snapshot(tmp_path):
    """A snapshot on the pre-contraction radii names the latest row with those
    radii and the mesh file saved with it, and that file spans the snapshot
    bounds while the mesh of the resumed row is 25 km and 50 km away from them."""
    out, hf_row, hf_all = _run_dir(tmp_path)

    row, file = pending_structure_update(str(out), hf_row, hf_all)

    assert row['Time'] == pytest.approx(T_OLD)  # the latest of the 2 rows on OLD
    assert file.endswith(format_subyear_time(T_OLD) + '_zalmoxis.dat')
    r = np.loadtxt(file)[:, 0]
    assert r[0] == pytest.approx(OLD['R_core'], abs=1.0)
    assert r[-1] == pytest.approx(OLD['R_int'], abs=1.0)
    # Discrimination: the resumed row's own bounds are far outside the 1 m tolerance.
    assert abs(hf_row['R_core'] - r[0]) > 2.0e4
    assert abs(hf_row['R_int'] - r[-1]) > 4.0e4
    assert set(STRUCTURE_ROW_KEYS) <= set(row)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'shift_m, pending',
    [(0.0, False), (0.5, False), (2.0, True)],
    ids=['same_mesh', 'inside_the_1_m_tolerance', 'outside_the_1_m_tolerance'],
)
def test_a_snapshot_on_the_mesh_of_its_row_has_no_pending_update(
    tmp_path, caplog, shift_m, pending
):
    """The snapshot mesh is the mesh of the row when both bounds agree to
    ``max(1 m, 1e-9 * (R_int - R_core))``: 0.5 m is the same mesh, 2 m is not.
    With 2 m no earlier row holds the bounds either, so the result is still None."""
    shifted = {'R_core': NEW['R_core'], 'R_int': NEW['R_int'] + shift_m}
    out, hf_row, hf_all = _run_dir(tmp_path, snapshot_on=shifted)

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_energetics'):
        result = pending_structure_update(str(out), hf_row, hf_all)

    assert result is None
    # Only a snapshot off the mesh of its row reaches the search and its warning.
    assert len(caplog.records) == (1 if pending else 0)
    tol = max(1.0, 1.0e-9 * (NEW['R_int'] - NEW['R_core']))
    assert (shift_m > tol) is pending


def test_the_prev_copy_serves_when_the_saved_copy_is_gone(tmp_path):
    """Without the copy saved with the earlier row, ``zalmoxis_output.dat.prev``
    is taken if it spans that row's radii; a ``.prev`` on other radii is refused."""
    out, hf_row, hf_all = _run_dir(tmp_path, saved_copy=False, prev_copy=True)

    row, file = pending_structure_update(str(out), hf_row, hf_all)
    assert file.endswith('zalmoxis_output.dat.prev')
    assert row['R_int'] == pytest.approx(OLD['R_int'])

    # Edge: a .prev left by another structure does not hold the bounds.
    _mesh_file(out / 'data' / 'zalmoxis_output.dat.prev', NEW['R_core'], NEW['R_int'])
    assert pending_structure_update(str(out), hf_row, hf_all) is None


def test_no_mesh_file_of_the_earlier_structure_gives_none_and_one_warning(tmp_path, caplog):
    """With the snapshot on an earlier structure and no mesh file of it left, the
    result is None and exactly one warning names the two consequences."""
    out, hf_row, hf_all = _run_dir(tmp_path, saved_copy=False, prev_copy=False)

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_energetics'):
        result = pending_structure_update(str(out), hf_row, hf_all)

    assert result is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert 'compression work' in warnings[0] and 'mass parcel' in warnings[0]
    assert '%.6e' % OLD['R_int'] in warnings[0]


def test_a_missing_snapshot_or_one_without_radii_is_not_a_pending_update(tmp_path, caplog):
    """No snapshot file, or a snapshot without ``radius_b``, gives None without a
    warning: there is no mesh to compare, so the run continues on its row."""
    out, hf_row, hf_all = _run_dir(tmp_path)
    snapshot = out / 'data' / (format_subyear_time(T_NEW) + '_int.nc')

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_energetics'):
        snapshot.unlink()
        assert pending_structure_update(str(out), hf_row, hf_all) is None
        _snapshot(out / 'data', T_NEW, 0.0, 0.0, with_radii=False)
        assert pending_structure_update(str(out), hf_row, hf_all) is None

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
