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


def _aragog_mesh(mass, cmb=3000.0, top='uniform', r_in=3.48e6, r_out=6.371e6):
    """Basic radii [m] of the 80-node mesh that aragog builds for these settings."""
    import aragog
    from aragog.mesh import Mesh
    from aragog.parser import Parameters

    p = Parameters.from_file(Path(aragog.__file__).parent / 'cfg' / 'abe_solid.toml')
    p.mesh.mass_coordinates, p.mesh.number_of_nodes = mass, 80
    p.mesh.cmb_cell_thickness, p.mesh.surface_cell_thickness = cmb, top
    p.mesh.inner_radius, p.mesh.outer_radius = r_in, r_out
    return np.asarray(Mesh(p).basic.radii).ravel()


def test_a_mesh_shifted_beyond_the_radius_tolerance_is_refused(tmp_path):
    """Inner nodes moved by 1.5 m, above the 1 m tolerance and its 2 mm margin, stop the resume."""
    with pytest.raises(ResumeSeparationLawError, match='node by node') as err:
        _refuse(_run(tmp_path), _inner(1.5))
    assert '80 nodes' in str(err.value)


def test_the_tolerance_holds_a_2_mm_margin_above_1_m(tmp_path):
    """The radius tolerance is 1 m plus 2 mm for the node solve: 1.0015 m goes on, 1.0025 m stops."""
    out = _run(tmp_path)
    _refuse(out, _inner(1.0015))
    with pytest.raises(ResumeSeparationLawError, match='node by node'):
        _refuse(out, _inner(1.0025))


@pytest.mark.parametrize('mass', [True, False], ids=['mass', 'radius'])
@pytest.mark.parametrize(
    ('bounds', 'cells', 'goes_on'),
    [
        ({'r_out': 6.373e6}, (3000.0, 'uniform'), True),
        ({'r_in': 3.46e6}, (3000.0, 'uniform'), True),
        ({'r_in': 3.48e6 * 1.2, 'r_out': 6.371e6 * 1.2}, (3000.0, 'uniform'), True),
        ({'r_out': 6.373e6}, (3250.0, 'uniform'), False),
        ({'r_out': 6.391e6}, (2750.0, 'uniform'), False),
        ({'r_out': 6.373e6}, (0.0, 0.0), False),
        ({'r_out': 6.373e6}, (3000.0, 3.0e4), False),
    ],
    ids=[
        'top +2 km',
        'cmb -20 km',
        'planet +20 %',
        'cmb 3250 m',
        'cmb 2750 m',
        'uniform',
        'top 30 km',
    ],
)
def test_a_resume_on_a_moved_structure_keeps_the_cells_of_the_run(
    tmp_path, caplog, mass, bounds, cells, goes_on
):
    """On a structure that moved since the snapshot (no replay), aragog's own mesh of the run
    settings goes on, also after a 20 % larger planet; a set CMB or top cell that differs, or
    the uniform mesh, is refused. A set cell is a length; a uniform cell scales with the mantle."""
    out = _run(tmp_path, radii_m=_aragog_mesh(mass))
    built = _aragog_mesh(mass, *cells, **bounds)
    settings = SimpleNamespace(cmb_cell_thickness=cells[0], surface_cell_thickness=cells[1])
    if goes_on:
        with caplog.at_level('INFO'):
            _refuse(out, built, settings)
        assert any('structure of its row' in r.getMessage() for r in caplog.records)
    else:
        with pytest.raises(ResumeSeparationLawError, match='node by node'):
            _refuse(out, built, settings)
    run = _aragog_mesh(mass)
    assert max(abs(built[0] - run[0]), abs(built[-1] - run[-1])) > 1.0  # the structure moved


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
