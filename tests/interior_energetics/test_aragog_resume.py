"""Unit tests for proteus.interior_energetics.aragog_resume.

``pending_structure_update`` decides, from the interior snapshot of the row a
run resumes from, if the snapshot is on an earlier structure than the row, and
names the row and mesh file of that structure.

Invariants and contract clauses:

- Radial bounds: the chosen row and mesh file hold the bounds of the snapshot
  mesh to Aragog's mesh tolerance, ``max(1 m, 1e-9 * (R_int - R_core))``; a
  shift of either bound below the tolerance is the same mesh and a shift above
  it, in either direction, is not.
- No pending update when the snapshot mesh is the mesh of the row, or when the
  snapshot has another node count than the run.
- File order: the copy saved with the latest matching row, then
  ``zalmoxis_output.dat.prev``, then the copy of an older matching row.
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
T_OLDER, T_OLD, T_NEW = 122.0, 222.001, 322.004
N_BASIC = 4


def _mesh_file(path, bounds):
    """Write a 5-column Zalmoxis mesh file spanning the radii [m] of ``bounds``."""
    r = np.linspace(bounds['R_core'], bounds['R_int'], 6)
    cols = np.column_stack([r, np.linspace(1.3e11, 0.0, 6), np.full(6, 4500.0), r, r])
    np.savetxt(path, cols)


def _snapshot(data_dir, time, bounds, with_radii=True):
    """Write an interior snapshot whose basic-node radii [km] span ``bounds`` [m]."""
    with nc.Dataset(data_dir / (format_subyear_time(time) + '_int.nc'), 'w') as ds:
        ds.createDimension('basic', N_BASIC)
        if with_radii:
            var = ds.createVariable('radius_b', 'f8', ('basic',))
            var[:] = np.linspace(bounds['R_core'], bounds['R_int'], N_BASIC) / 1.0e3


def _saved(out, time):
    return out / 'data' / (format_subyear_time(time) + '_zalmoxis.dat')


def _run_dir(tmp_path, *, snapshot_on=OLD, saved_copy=True, prev_on=None):
    """A run directory stopped at T_NEW on the NEW structure; return (dir, hf_row, hf_all)."""
    data = tmp_path / 'data'
    data.mkdir()
    _snapshot(data, T_NEW, snapshot_on)
    _mesh_file(data / 'zalmoxis_output.dat', NEW)
    if saved_copy:
        _mesh_file(_saved(tmp_path, T_OLD), OLD)
    if prev_on is not None:
        _mesh_file(data / 'zalmoxis_output.dat.prev', prev_on)
    old = {**OLD, 'gravity': 8.29, 'M_core': 1.92e24, 'core_density': 11574.0}
    new = {**NEW, 'gravity': 8.56, 'M_core': 1.95e24, 'core_density': 12003.0}
    rows = [{'Time': T_OLDER, **old}, {'Time': T_OLD, **old}, {'Time': T_NEW, **new}]
    hf_all = pd.DataFrame(rows)
    return tmp_path, hf_all.iloc[-1].to_dict(), hf_all


@pytest.mark.physics_invariant
def test_the_earlier_structure_is_found_from_the_bounds_of_the_snapshot(tmp_path):
    """A snapshot on the pre-contraction radii names the latest row with those
    radii and the mesh file saved with it, and that file spans the snapshot
    bounds while the mesh of the resumed row is 25 km and 50 km away from them."""
    out, hf_row, hf_all = _run_dir(tmp_path, prev_on=OLD)  # both copies: the saved one wins

    row, file = pending_structure_update(str(out), hf_row, hf_all, N_BASIC)

    assert row['Time'] == pytest.approx(T_OLD)  # the latest of the 2 rows on OLD
    assert file == str(_saved(out, T_OLD))
    r = np.loadtxt(file)[:, 0]
    assert r[0] == pytest.approx(OLD['R_core'], abs=1.0)
    assert r[-1] == pytest.approx(OLD['R_int'], abs=1.0)
    # Discrimination: the resumed row's own bounds are far outside the 1 m tolerance.
    assert abs(hf_row['R_core'] - r[0]) > 2.0e4
    assert abs(hf_row['R_int'] - r[-1]) > 4.0e4
    assert set(STRUCTURE_ROW_KEYS) <= set(row)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('bound', ['R_core', 'R_int'])
@pytest.mark.parametrize(
    'shift_m, same_mesh',
    [(0.0, True), (0.5, True), (-0.5, True), (2.0, False), (-2.0, False)],
    ids=['equal', 'plus_half_m', 'minus_half_m', 'plus_2_m', 'minus_2_m'],
)
def test_a_snapshot_on_the_mesh_of_its_row_has_no_pending_update(
    tmp_path, caplog, bound, shift_m, same_mesh
):
    """The snapshot mesh is the mesh of the row when each bound agrees to
    ``max(1 m, 1e-9 * (R_int - R_core))``: half a metre on either bound is the
    same mesh, 2 m in either direction is not. With 2 m no earlier row holds
    the bounds either, so the search runs and ends in its warning."""
    out, hf_row, hf_all = _run_dir(tmp_path, snapshot_on={**NEW, bound: NEW[bound] + shift_m})

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_energetics'):
        result = pending_structure_update(str(out), hf_row, hf_all, N_BASIC)

    assert result is None
    # Only a snapshot off the mesh of its row reaches the search and its warning.
    assert len(caplog.records) == (0 if same_mesh else 1)


def test_the_prev_copy_and_then_an_older_saved_copy_serve(tmp_path, caplog):
    """Without the copy saved with the latest matching row,
    ``zalmoxis_output.dat.prev`` is taken if it spans that row's radii. A
    ``.prev`` on other radii is refused, and the copy saved with an older row on
    the same structure is taken with that row."""
    out, hf_row, hf_all = _run_dir(tmp_path, saved_copy=False, prev_on=OLD)

    row, file = pending_structure_update(str(out), hf_row, hf_all, N_BASIC)
    assert file.endswith('zalmoxis_output.dat.prev')
    assert row['Time'] == pytest.approx(T_OLD)

    # Edge: the next structure solve overwrote .prev, and an older row kept its copy.
    _mesh_file(out / 'data' / 'zalmoxis_output.dat.prev', NEW)
    _mesh_file(_saved(out, T_OLDER), OLD)
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_energetics'):
        row, file = pending_structure_update(str(out), hf_row, hf_all, N_BASIC)
    assert file == str(_saved(out, T_OLDER))
    assert row['Time'] == pytest.approx(T_OLDER)
    assert row['R_int'] == pytest.approx(OLD['R_int'])
    assert caplog.records == []


def test_no_mesh_file_of_the_earlier_structure_gives_none_and_one_warning(tmp_path, caplog):
    """With the snapshot on an earlier structure and no mesh file of it left
    (the ``.prev`` copy is on the new structure), the result is None and exactly
    one warning names the two consequences."""
    out, hf_row, hf_all = _run_dir(tmp_path, saved_copy=False, prev_on=NEW)

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_energetics'):
        result = pending_structure_update(str(out), hf_row, hf_all, N_BASIC)

    assert result is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert 'compression work' in warnings[0] and 'mass parcel' in warnings[0]
    assert '%.6e' % OLD['R_int'] in warnings[0]


def test_a_snapshot_that_cannot_be_compared_is_not_a_pending_update(tmp_path, caplog):
    """A snapshot on another node count than the run, a snapshot without
    ``radius_b`` and a missing snapshot each give None without a warning: the
    run continues on its row, as a run with no earlier structure to build on."""
    out, hf_row, hf_all = _run_dir(tmp_path)
    snapshot = out / 'data' / (format_subyear_time(T_NEW) + '_int.nc')

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_energetics'):
        # The same directory has a pending update for a run on 4 nodes, none for 5.
        assert pending_structure_update(str(out), hf_row, hf_all, N_BASIC) is not None
        assert pending_structure_update(str(out), hf_row, hf_all, N_BASIC + 1) is None
        snapshot.unlink()
        assert pending_structure_update(str(out), hf_row, hf_all, N_BASIC) is None
        _snapshot(out / 'data', T_NEW, NEW, with_radii=False)
        assert pending_structure_update(str(out), hf_row, hf_all, N_BASIC) is None

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
