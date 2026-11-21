"""The per-node record of the layer scheme in the interior snapshots.

The record of :mod:`proteus.outgas.node_record` is written beside the solver
fields of each Aragog ``<time>_int.nc`` snapshot, on its ``staggered``
dimension, so a resumed run reads it back from the snapshot it resumes from.
"""

from __future__ import annotations

import numpy as np

from proteus.outgas.node_record import RECORD_SPECIES, NodeRecord

# Per-node arrays written to the interior snapshot, with their units.
_RECORD_FIELDS = {
    'phase': '0 liquid, 1 mush, 2 solid',
    'psi': '',
    'psi_entry': '',
    't_entry': 'yr',
    'melt_fraction': '',
    't_darcy': 'yr',
    't_matrix': 'yr',
    't_held': 'yr',
}

# Per-species stores, by the prefix of their snapshot variables.
_STORES = ('crystal', 'pore')


def _snapshot_name(name: str) -> str:
    """Variable name of a per-node record array in the interior snapshot."""
    return f'trap_{name}_s'


def write_record(path: str, record: NodeRecord) -> None:
    """Add the per-node record to an interior snapshot, beside the solver fields.

    The snapshot must already hold the ``staggered`` dimension the solver
    writes, sized for this record. Variables present from an earlier write of
    the same file are overwritten.

    Raises
    ------
    ValueError
        If the snapshot has no ``staggered`` dimension of the record's size.
    """
    import netCDF4 as nc

    with nc.Dataset(path, mode='a') as ds:
        dim = ds.dimensions.get('staggered')
        if dim is None or len(dim) != record.size:
            raise ValueError(
                f'{path} has no staggered dimension of {record.size} nodes to hold '
                'the per-node trapping record'
            )
        columns = {name: getattr(record, name) for name in _RECORD_FIELDS}
        units = dict(_RECORD_FIELDS)
        for store in _STORES:
            for species, kg in getattr(record, store).items():
                columns[f'{store}_{species}'] = kg
                units[f'{store}_{species}'] = 'kg'
        for name, values in columns.items():
            key = _snapshot_name(name)
            var = ds.variables.get(key)
            if var is None:
                var = ds.createVariable(key, np.float64, ('staggered',))
            var[:] = np.asarray(values, dtype=float)
            var.units = units[name]


def read_record(path: str, n: int | None = None) -> NodeRecord | None:
    """The per-node record stored in an interior snapshot, or ``None``.

    ``n`` is the number of nodes the record must cover; ``None`` takes the
    snapshot's own ``staggered`` dimension. ``None`` is returned when the
    snapshot carries no record, as one written before the layer scheme ran
    does, or a record sized for another mesh.
    """
    import netCDF4 as nc

    with nc.Dataset(path) as ds:
        if n is None:
            dim = ds.dimensions.get('staggered')
            if dim is None:
                return None
            n = len(dim)
        names = [_snapshot_name(f) for f in _RECORD_FIELDS]
        if any(name not in ds.variables for name in names):
            return None

        def column(key, fill):
            return np.ma.filled(ds.variables[key][:].astype(float), fill)

        arrays = {f: column(_snapshot_name(f), np.nan) for f in _RECORD_FIELDS}
        stores = {store: {} for store in _STORES}
        for species in RECORD_SPECIES:
            for store in _STORES:
                key = _snapshot_name(f'{store}_{species}')
                if key in ds.variables:
                    stores[store][species] = column(key, 0.0)
    stored = list(arrays.values()) + [a for s in stores.values() for a in s.values()]
    if any(np.size(a) != n for a in stored):
        return None
    for store in _STORES:
        for species in RECORD_SPECIES:
            stores[store].setdefault(species, np.zeros(n))
    return NodeRecord(
        phase=np.rint(arrays['phase']).astype(int),
        psi=arrays['psi'],
        psi_entry=arrays['psi_entry'],
        t_entry=arrays['t_entry'],
        melt_fraction=arrays['melt_fraction'],
        t_darcy=arrays['t_darcy'],
        t_matrix=arrays['t_matrix'],
        t_held=arrays['t_held'],
        crystal=stores['crystal'],
        pore=stores['pore'],
    )
