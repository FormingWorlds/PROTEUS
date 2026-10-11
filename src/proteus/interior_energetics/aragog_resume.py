"""Find the structure update that an Aragog run had not applied when it stopped.

A loop integrates the interior on the structure it starts with and re-solves the
structure after that step; the solver takes the new mesh at the start of the
next loop. A run that stops between the two leaves an interior snapshot on the
earlier mesh and a helpfile row on the new one. The solver needs the earlier
mesh to book the compression work of the update and to carry the entropy to
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


def _on_mesh(R_core, R_int, r_inner: float, r_outer: float):
    """True where the radii [m] are the bounds of the mesh, to Aragog's mesh tolerance."""
    tol = np.maximum(1.0, 1.0e-9 * (R_int - R_core))
    return (np.abs(R_core - r_inner) <= tol) & (np.abs(R_int - r_outer) <= tol)


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
        of ``hf_row``, the snapshot is absent or holds no ``radius_b``, or its
        node count is not ``n_basic``. None with a warning when no earlier row
        and file hold the snapshot mesh.
    """
    from proteus.interior_struct.zalmoxis import (
        get_zalmoxis_output_filepath,
        zalmoxis_mesh_gaps,
    )

    data_dir = os.path.join(output_dir, 'data')
    time = float(hf_row['Time'])
    snapshot = snapshot_path_for_time(data_dir, time, '_int.nc')
    if not os.path.isfile(snapshot):
        return None
    with nc.Dataset(snapshot) as ds:
        if 'radius_b' not in ds.variables:
            return None
        radii = np.asarray(ds['radius_b'][:], dtype=float) * 1.0e3  # km -> m
    r_inner, r_outer = float(radii[0]), float(radii[-1])
    if radii.size != n_basic or _on_mesh(hf_row['R_core'], hf_row['R_int'], r_inner, r_outer):
        return None

    earlier = hf_all.iloc[:-1]
    held = _on_mesh(earlier['R_core'], earlier['R_int'], r_inner, r_outer).to_numpy()
    rows = [earlier.iloc[i].to_dict() for i in np.flatnonzero(held)[::-1]]
    files = [
        (row, snapshot_path_for_time(data_dir, float(row['Time']), '_zalmoxis.dat'))
        for row in rows
    ]
    files[1:1] = [(row, get_zalmoxis_output_filepath(output_dir) + '.prev') for row in rows[:1]]
    for row, file in files:
        gaps = zalmoxis_mesh_gaps(file, row)
        if gaps is not None and abs(gaps[0]) <= gaps[2] and abs(gaps[1]) <= gaps[2]:
            return row, file

    log.warning(
        'Resume: the interior snapshot at t = %.6e yr is on an earlier structure '
        '(R_core = %.6e m, R_int = %.6e m) and no row with a mesh file of that structure '
        'is left. The run continues on the structure of its row: the first step books no '
        'compression work, and the entropy is carried by node and not by mass parcel.',
        time,
        r_inner,
        r_outer,
    )
    return None
