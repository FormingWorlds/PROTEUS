"""Unit tests for ``proteus.outgas.node_record_io``.

The per-node record of the layer scheme is written beside the solver fields of
an interior snapshot and read back on a resume. These tests exercise:

* the round trip of every per-node array and both per-species stores, with
  the trapped mass conserved through it,
* a second write of the same step, which overwrites,
* the snapshots that read as no record: none stored, one sized for another
  mesh, one missing a single array,
* the refusal of a file without the solver's staggered dimension.

See ``docs/How-to/testing.md`` and ``docs/Explanations/test_framework.md``
for the test framework.
"""

from __future__ import annotations

import netCDF4 as nc
import numpy as np
import pytest

from proteus.outgas.node_record import RECORD_SPECIES, NodeRecord
from proteus.outgas.node_record_io import read_record, write_record

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_N = 5
# Every array of the record, each with values no other array shares.
_ARRAYS = (
    'phase',
    'psi',
    'psi_entry',
    't_entry',
    'melt_fraction',
    't_darcy',
    't_matrix',
    't_held',
)


def _snapshot(path: str, n: int = _N) -> str:
    """A bare snapshot with the staggered dimension the solver writes."""
    with nc.Dataset(path, mode='w') as ds:
        ds.createDimension('staggered', n)
        ds.createVariable('time', np.float64)
        ds['time'][0] = 100.0
    return path


def _record() -> NodeRecord:
    """A record of two solid, one mush and two liquid nodes with a distinct value everywhere."""
    record = NodeRecord.initial(
        np.array([2, 2, 1, 0, 0]), np.array([0.0, 0.0, 0.2, 1.0, 1.0]), np.zeros(_N), 50.0
    )
    record.psi_entry[2] = 0.31
    record.melt_fraction = np.array([0.0, 0.01, 0.27, 0.9, 1.0])
    record.t_darcy = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    record.t_matrix = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    record.t_held = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    record.crystal['H2O'] = np.array([1.0e18, 3.0e18, 0.0, 2.0e15, 0.0])
    record.pore['CO2'][2] = 4.0e17
    record.pore['He'][2] = 7.0e12
    return record


def test_every_array_of_the_record_round_trips_through_a_snapshot(tmp_path):
    """Every per-node array and both stores read back unchanged, NaN where the
    record holds NaN, and a second write of the same step overwrites."""
    path = _snapshot(str(tmp_path / '100p000_int.nc'))
    assert read_record(path, _N) is None
    record = _record()
    write_record(path, record)
    record.t_held[0] = 99.0
    write_record(path, record)
    back = read_record(path, _N)
    for name in _ARRAYS:
        np.testing.assert_array_equal(getattr(back, name), getattr(record, name), err_msg=name)
    assert back.t_held[0] == pytest.approx(99.0, rel=0.0)
    for sp in RECORD_SPECIES:
        np.testing.assert_array_equal(back.crystal[sp], record.crystal[sp], err_msg=sp)
        np.testing.assert_array_equal(back.pore[sp], record.pore[sp], err_msg=sp)
    assert back.phase.dtype.kind == 'i'


@pytest.mark.physics_invariant
def test_the_trapped_mass_is_conserved_through_a_snapshot(tmp_path):
    """The mass trapped per species, in crystals and in pore melt, is the same
    after the round trip; swapping the two stores would keep the totals but
    not the per-node split."""
    path = _snapshot(str(tmp_path / '100p000_int.nc'))
    record = _record()
    write_record(path, record)
    back = read_record(path, _N)
    assert back.trapped('CO2') == pytest.approx(4.0e17, rel=1e-15)
    assert back.trapped('H2O') == pytest.approx(4.002e18, rel=1e-15)
    assert back.trapped('He') == pytest.approx(7.0e12, rel=1e-15)
    assert back.pore['CO2'][2] == pytest.approx(4.0e17, rel=1e-15)
    assert back.crystal['CO2'][2] == pytest.approx(0.0, abs=0.0)


def test_snapshots_without_a_usable_record_read_as_none(tmp_path):
    """A snapshot sized for another mesh, or missing one array of the record,
    reads as no record; a file without the staggered dimension is refused."""
    path = _snapshot(str(tmp_path / '100p000_int.nc'))
    write_record(path, _record())
    assert read_record(path, _N + 1) is None
    with nc.Dataset(path, mode='a') as ds:
        ds.renameVariable('trap_t_held_s', 'other')
    assert read_record(path, _N) is None

    bare = str(tmp_path / 'bare.nc')
    with nc.Dataset(bare, mode='w') as ds:
        ds.createDimension('basic', _N + 1)
    with pytest.raises(ValueError, match='staggered'):
        write_record(bare, _record())
    wrong = _snapshot(str(tmp_path / 'wrong.nc'), n=_N + 2)
    with pytest.raises(ValueError, match='staggered'):
        write_record(wrong, _record())
