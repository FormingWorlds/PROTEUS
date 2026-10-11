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


def _row_holds_mesh(row: dict, r_inner: float, r_outer: float) -> bool:
    """True when the radial bounds [m] are those of the row, to Aragog's mesh tolerance."""
    tol = max(1.0, 1.0e-9 * (row['R_int'] - row['R_core']))
    return abs(r_inner - row['R_core']) <= tol and abs(r_outer - row['R_int']) <= tol


def pending_structure_update(
    output_dir: str, hf_row: dict, hf_all: pd.DataFrame
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

    Returns
    -------
    tuple of (dict, str) or None
        The latest earlier row whose radii are the radial bounds of the
        snapshot mesh, and a mesh file that holds those bounds: the copy saved
        with that row, else ``zalmoxis_output.dat.prev``. None when the
        snapshot mesh is the mesh of ``hf_row``, so no update is pending, and
        None with a warning when no earlier row and file hold the snapshot mesh.
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
        r_basic = np.asarray(ds['radius_b'][:], dtype=float) * 1.0e3  # km -> m
    r_inner, r_outer = float(r_basic[0]), float(r_basic[-1])
    if _row_holds_mesh(hf_row, r_inner, r_outer):
        return None

    earlier = hf_all.iloc[:-1]
    for i in range(len(earlier) - 1, -1, -1):
        row = earlier.iloc[i].to_dict()
        if not _row_holds_mesh(row, r_inner, r_outer):
            continue
        saved = snapshot_path_for_time(data_dir, float(row['Time']), '_zalmoxis.dat')
        for file in (saved, get_zalmoxis_output_filepath(output_dir) + '.prev'):
            gaps = zalmoxis_mesh_gaps(file, row) if os.path.isfile(file) else None
            if gaps is not None and abs(gaps[0]) <= gaps[2] and abs(gaps[1]) <= gaps[2]:
                return row, file
        break

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
