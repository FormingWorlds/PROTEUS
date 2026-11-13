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
    _refuse(_run(tmp_path), RUN + 0.5)


def test_another_node_count_is_refused(tmp_path):
    """A resume with another num_levels builds another mesh: refused, naming both counts."""
    built = np.linspace(RUN[0], RUN[-1], 60)
    with pytest.raises(ResumeSeparationLawError, match='80 nodes') as err:
        _refuse(_run(tmp_path), built, levels=60)
    assert '60 nodes' in str(err.value) and 'num_levels = 60' in str(err.value)


def test_a_snapshot_without_radii_or_no_snapshot_is_not_checked(tmp_path):
    """An older snapshot without radius_b, or none at the row time, leaves the mesh unchecked."""
    _refuse(_run(tmp_path / 'a', radii_m=None), UNIFORM)
    (tmp_path / 'b').mkdir()
    _refuse(str(tmp_path / 'b'), UNIFORM)
