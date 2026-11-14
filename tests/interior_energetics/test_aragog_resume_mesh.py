"""A resumed Aragog run builds the mesh of the interior snapshot it restores.

Module under test: ``proteus.interior_energetics.aragog_resume.refuse_another_mesh``.
Invariant: the snapshot entropy is put back node by node, so the basic radii that the
resume builds equal the ``radius_b`` of the snapshot to the radius tolerance of the
structure; another mesh or another node count stops the resume with the stored end
cells in the message. The stored radii come from a PROTEUS run on the 3000 m CMB cell
whose record holds no cell field, and from its resume on the uniform mesh.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import netCDF4 as nc
import numpy as np
import pytest

from proteus.interior_energetics.aragog_resume import refuse_another_mesh
from proteus.interior_energetics.resume_laws import ResumeSeparationLawError
from proteus.interior_struct.zalmoxis import mesh_radius_tolerance
from proteus.utils.helper import snapshot_path_for_time

pytestmark = pytest.mark.unit

RADII_KM = np.loadtxt(Path(__file__).parents[1] / 'data' / 'resume_mesh' / 'a6_radius_b_km.txt')
RUN, UNIFORM = RADII_KM[:, 0] * 1e3, RADII_KM[:, 1] * 1e3  # [m]
DEFAULTS = SimpleNamespace(cmb_cell_thickness=3000.0, surface_cell_thickness='uniform')


def _run(tmp_path, radii_m=RUN, time=5000.0):
    """A run directory with one interior snapshot at ``time`` holding ``radii_m``."""
    data = tmp_path / 'data'
    data.mkdir(parents=True)
    with nc.Dataset(snapshot_path_for_time(str(data), time, '_int.nc'), 'w') as ds:
        if radii_m is not None:
            ds.createDimension('basic', radii_m.size)
            ds.createVariable('radius_b', 'f8', ('basic',))[:] = radii_m / 1e3
    return str(tmp_path)


def _refuse(out, built, aragog=DEFAULTS, levels=80):
    refuse_another_mesh(out, 5000.0, built, aragog, levels, 'user.toml')


def test_a_resume_on_another_mesh_is_refused_with_the_stored_cells(tmp_path):
    """The run used the 3000 m CMB cell; a resume that builds the uniform mesh is refused,
    the message gives the stored and the built end cells and does not advise 0."""
    with pytest.raises(ResumeSeparationLawError) as err:
        _refuse(
            _run(tmp_path),
            UNIFORM,
            SimpleNamespace(cmb_cell_thickness=0.0, surface_cell_thickness=0.0),
        )
    text = str(err.value)
    assert 'bottom cell 3000.0 m' in text and 'bottom cell 35561.0 m' in text
    assert 'top cell 65798.8 m' in text and 'cmb_cell_thickness = 0.0' in text
    assert 'set it to 0.0' not in text and 'node by node' in text


def test_the_mesh_of_the_snapshot_resumes(tmp_path):
    """The same mesh, shifted by half the 1 m radius tolerance, goes on."""
    assert mesh_radius_tolerance(RUN[0], RUN[-1]) == pytest.approx(1.0)
    _refuse(_run(tmp_path), RUN + 0.5)
    assert np.max(np.abs(RUN + 0.5 - RUN)) < mesh_radius_tolerance(RUN[0], RUN[-1])


def _inner(shift):
    """RUN with every node but the 2 end nodes moved by ``shift`` [m]: the same bounds."""
    built = RUN.copy()
    built[1:-1] += shift
    return built


def _rescaled(radii, grow=2000.0):
    """``radii`` stretched to a structure ``grow`` m thicker, as a resume on its row builds."""
    return radii[0] + (radii - radii[0]) * (radii[-1] - radii[0] + grow) / (
        radii[-1] - radii[0]
    )


def test_a_mesh_shifted_beyond_the_radius_tolerance_is_refused(tmp_path):
    """Inner nodes moved by 1.5 m, above the 1 m tolerance and its 2 mm margin, stop the resume."""
    with pytest.raises(ResumeSeparationLawError, match='node by node') as err:
        _refuse(_run(tmp_path), _inner(1.5))
    assert '80 nodes' in str(err.value)


@pytest.mark.parametrize(('shift', 'refused'), [(1.0015, False), (1.0025, True)])
def test_the_tolerance_holds_a_2_mm_margin_above_1_m(tmp_path, shift, refused):
    """The radius tolerance is 1 m plus 2 mm for the node solve: 1.0015 m goes on, 1.0025 m stops."""
    out = _run(tmp_path)
    if refused:
        with pytest.raises(ResumeSeparationLawError, match='node by node'):
            _refuse(out, _inner(shift))
    else:
        _refuse(out, _inner(shift))
    assert (shift > 1.002) is refused


def test_a_resume_built_on_the_structure_of_its_row_goes_on(tmp_path, caplog):
    """When the solver is built on a row whose structure moved (no replay), the bounds differ
    but the end cells stay within 10 %: the resume goes on and logs the moved bounds."""
    built = _rescaled(RUN)
    with caplog.at_level('INFO'):
        _refuse(_run(tmp_path), built)
    assert built[-1] - RUN[-1] == pytest.approx(2000.0)
    assert any('structure of its row' in r.getMessage() for r in caplog.records)


def test_another_mesh_on_a_moved_structure_is_still_refused(tmp_path):
    """The uniform mesh on the moved structure has other end cells than the run: refused."""
    with pytest.raises(ResumeSeparationLawError, match='bottom cell 3000.0 m') as err:
        _refuse(_run(tmp_path), _rescaled(UNIFORM))
    assert 'node by node' in str(err.value)


def test_another_node_count_is_refused(tmp_path):
    """A resume with another num_levels builds another mesh: refused, naming both counts."""
    built = np.linspace(RUN[0], RUN[-1], 60)
    with pytest.raises(ResumeSeparationLawError, match='80 nodes') as err:
        _refuse(_run(tmp_path), built, levels=60)
    assert '60 nodes' in str(err.value) and 'num_levels = 60' in str(err.value)


def test_a_snapshot_without_radii_or_no_snapshot_is_not_checked(tmp_path):
    """An older snapshot without radius_b, or none at the row time, leaves the mesh unchecked."""
    out = _run(tmp_path / 'a', radii_m=None)
    with nc.Dataset(next(Path(out, 'data').glob('*_int.nc'))) as ds:
        assert 'radius_b' not in ds.variables
    _refuse(out, UNIFORM)
    (tmp_path / 'b').mkdir()
    assert not (tmp_path / 'b' / 'data').exists()
    _refuse(str(tmp_path / 'b'), UNIFORM)
