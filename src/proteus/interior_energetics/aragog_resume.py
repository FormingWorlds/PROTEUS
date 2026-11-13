"""Find the structure update that an Aragog run had not applied when it stopped.

A loop integrates the interior on the structure it starts with and re-solves the
structure after that step; the solver takes the new mesh at the start of the
next loop. A run that stops between the two leaves an interior snapshot on the
earlier mesh and a helpfile row on the new one. The solver needs the earlier
mesh to record the compression work of the update and to carry the entropy to
the new mesh by mass parcel, so a resume has to build it first.
"""

from __future__ import annotations

import logging
import os

import netCDF4 as nc
import numpy as np
import pandas as pd

from proteus.utils.helper import snapshot_path_for_time

log = logging.getLogger('fwl.' + __name__)

# Helpfile columns that define the structure the Aragog mesh is built on.
STRUCTURE_ROW_KEYS = ('R_int', 'R_core', 'gravity', 'M_core', 'core_density')


def pending_structure_update(
    output_dir: str, hf_row: dict, hf_all: pd.DataFrame, n_basic: int
) -> tuple[dict, str] | None:
    """Row and mesh file of the structure the snapshot of ``hf_row`` was integrated on.

    Parameters
    ----------
    output_dir : str
        Run directory holding ``data/``.
    hf_row : dict
        Row the run resumes from; ``R_core`` and ``R_int`` in m, ``Time`` in yr.
    hf_all : pandas.DataFrame
        Helpfile table whose last row is ``hf_row``.
    n_basic : int
        Number of basic mesh nodes of the run that resumes.

    Returns
    -------
    tuple of (dict, str) or None
        An earlier row whose radii are the radial bounds of the snapshot mesh,
        and a mesh file that holds those bounds: the copy saved with the latest
        such row, else ``zalmoxis_output.dat.prev``, else the copy of an older
        such row. None when no update is pending: the snapshot mesh is the mesh
        of ``hf_row``, or the snapshot is absent, holds no ``radius_b``, or does
        not hold an entropy profile on ``n_basic`` nodes. None with a warning
        when no earlier row and file hold the snapshot mesh.

    Notes
    -----
    A mesh copy is saved only with a row that has a snapshot, and every structure
    solve overwrites ``zalmoxis_output.dat.prev``. With snapshots on every row the
    earlier structure is found unless its row lost its snapshot, as the landing
    row of a giant impact does; then only ``.prev`` can hold it. With sparser
    snapshots it is found only when the run stopped before the next structure solve.
    """
    from proteus.interior_struct.zalmoxis import (
        get_zalmoxis_output_filepath,
        mesh_radius_tolerance,
        zalmoxis_mesh_gaps,
    )

    def on_mesh(R_core, R_int):
        """True where the radii [m] are the bounds of the snapshot mesh."""
        tol = mesh_radius_tolerance(R_core, R_int)
        return (np.abs(R_core - r_inner) <= tol) & (np.abs(R_int - r_outer) <= tol)

    data_dir = os.path.join(output_dir, 'data')
    time = float(hf_row['Time'])
    snapshot = snapshot_path_for_time(data_dir, time, '_int.nc')
    if not os.path.isfile(snapshot):
        return None
    with nc.Dataset(snapshot) as ds:
        if 'radius_b' not in ds.variables or 'entropy_s' not in ds.variables:
            return None
        radii = np.asarray(ds['radius_b'][:], dtype=float) * 1.0e3  # km -> m
        n_entropy = ds['entropy_s'].size
    if radii.size != n_basic or n_entropy != n_basic - 1:
        return None
    r_inner, r_outer = float(radii[0]), float(radii[-1])
    if on_mesh(hf_row['R_core'], hf_row['R_int']):
        return None

    earlier = hf_all.iloc[:-1]
    held = np.flatnonzero(on_mesh(earlier['R_core'], earlier['R_int']).to_numpy())[::-1]
    prev = get_zalmoxis_output_filepath(output_dir) + '.prev'
    for k, i in enumerate(held):
        row = earlier.iloc[int(i)].to_dict()
        saved = snapshot_path_for_time(data_dir, float(row['Time']), '_zalmoxis.dat')
        for file in (saved, prev) if k == 0 else (saved,):
            gaps = zalmoxis_mesh_gaps(file, row)
            if gaps is not None and abs(gaps[0]) <= gaps[2] and abs(gaps[1]) <= gaps[2]:
                return row, file

    log.warning(
        'Resume: the interior snapshot at t = %.6e yr is on an earlier structure '
        '(R_core = %.6e m, R_int = %.6e m) and no row with a mesh file of that structure '
        'is left. The run continues on the structure of its row: the first step records no '
        'compression work, and the entropy is carried by node and not by mass parcel.',
        time,
        r_inner,
        r_outer,
    )
    return None


def refuse_another_mesh(
    output_dir: str, time: float, r_basic, aragog, num_levels: int, config_path: str
) -> None:
    """Refuse a resume whose solver mesh is not the mesh of the interior snapshot it restores.

    The snapshot entropy is put back node by node, so the basic radii the resume builds must
    equal the ``radius_b`` of the snapshot at ``time`` to the radius tolerance of the
    structure, ``max(1 m, 1e-9 (R_int - R_core))``, plus 2 mm for the mass-coordinate node
    solve. A snapshot that is absent or holds no ``radius_b`` is not checked.

    Parameters
    ----------
    output_dir : str
        Run directory holding ``data/``.
    time : float
        Time [yr] of the row the run resumes from.
    r_basic : array_like
        Basic node radii [m] of the solver the resume builds.
    aragog : object
        The ``interior_energetics.aragog`` config of the resume.
    num_levels : int
        ``interior_energetics.num_levels`` of the resume.
    config_path : str
        Config file of the resume, named in the refusal.

    Raises
    ------
    ResumeSeparationLawError
        If the node count or a radius differs beyond the tolerance.
    """
    from proteus.interior_energetics.resume_laws import ResumeSeparationLawError
    from proteus.interior_struct.zalmoxis import mesh_radius_tolerance

    snapshot = snapshot_path_for_time(os.path.join(output_dir, 'data'), time, '_int.nc')
    if not os.path.isfile(snapshot):
        return
    with nc.Dataset(snapshot) as ds:
        if 'radius_b' not in ds.variables:
            return
        stored = np.asarray(ds['radius_b'][:], dtype=float).ravel() * 1.0e3  # km -> m
    built = np.asarray(r_basic, dtype=float).ravel()
    tol = float(mesh_radius_tolerance(stored[0], stored[-1])) + 2.0e-3
    if stored.size == built.size and np.max(np.abs(stored - built)) <= tol:
        return

    def mesh(r):
        return (
            f'{r.size} nodes, bottom cell {r[1] - r[0]:.1f} m, top cell {r[-1] - r[-2]:.1f} m'
        )

    raise ResumeSeparationLawError(
        f'Resume refused: the interior snapshot {snapshot} holds a mesh of {mesh(stored)}, but '
        f'{config_path} builds {mesh(built)} (num_levels = {num_levels}, cmb_cell_thickness = '
        f'{aragog.cmb_cell_thickness!r}, surface_cell_thickness = '
        f'{aragog.surface_cell_thickness!r}); the snapshot entropy is put back node by node, so '
        'the resume must build the mesh of the run: set these fields to the values the run used'
    )
