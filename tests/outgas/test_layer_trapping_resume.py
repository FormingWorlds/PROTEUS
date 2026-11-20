"""Unit tests for the resume path of ``proteus.outgas.layer_trapping``.

The per-node record of the layer scheme lives on the interior object and in
the interior snapshots. These tests exercise:

* the record written into the interior snapshot and read back on a resume,
* trapped mass from before a record assigned to the nodes that are not liquid,
* a record restarted on the melt fraction of the snapshot a run resumes from,
  also on a mesh that changed size, so nodes leaving the ocean still bury,
* the same trapped mass whether a run continues, resumes or changes its mesh,
* the end-of-run write and an unreadable snapshot.

The helpers come from ``test_layer_trapping.py``. See ``docs/How-to/testing.md``
and ``docs/Explanations/test_framework.md`` for the test framework.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from proteus.outgas.layer_drainage import PHASE_MUSH
from proteus.outgas.layer_trapping import (
    persist_node_record,
    restore_node_record,
)
from proteus.outgas.trapping import run_trapping
from proteus.utils.helper import snapshot_path_for_time

from .test_layer_trapping import (
    _C_H2O,
    _D_H2O,
    _H2O_LIQUID,
    _M_MANTLE,
    _M_NODE,
    _N,
    _X_FREEZE,
    _config,
    _hf_all,
    _hf_row,
    _interior,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _snapshot(output: str, time: float) -> str:
    """Bare interior snapshot with the staggered dimension the solver writes."""
    data = os.path.join(output, 'data')
    os.makedirs(data, exist_ok=True)
    path = snapshot_path_for_time(data, time, '_int.nc')
    with nc.Dataset(path, mode='w') as ds:
        ds.createDimension('staggered', _N)
        ds.createVariable('phi', 'f8', ('staggered',))[:] = np.ones(_N)
    return path


@pytest.mark.physics_invariant
def test_the_record_survives_a_resume_through_the_interior_snapshot(tmp_path):
    """The step writes the per-node record into its interior snapshot. A
    resumed run reads it back and continues exactly as the run that kept it in
    memory, so a node that remelts after the resume returns what it held."""
    output = str(tmp_path)
    dirs = {'output': output}
    hf_row = _hf_row(_X_FREEZE)
    path = _snapshot(output, hf_row['Time'])
    stiff = _config(melt_log10visc=30.0)
    run_trapping(stiff, hf_row, _hf_all(), _interior(_X_FREEZE), dirs=dirs)
    with nc.Dataset(path) as ds:
        stored = np.array(ds.variables['trap_pore_H2O_s'][:])
        stored += np.array(ds.variables['trap_crystal_H2O_s'][:])
    assert stored[6] > 0.0

    x = _X_FREEZE.copy()
    x[6] = 1.0
    hf_all = pd.DataFrame([hf_row])
    row = dict(hf_row, Time=3.0e4, Phi_global=float(np.mean(x)))
    kept = _interior(_X_FREEZE)
    run_trapping(stiff, _hf_row(_X_FREEZE), _hf_all(), kept)
    kept.phi = x
    in_memory = run_trapping(stiff, dict(row), hf_all, kept)

    resume = _config(resume=True, melt_log10visc=30.0)
    restarted = _interior(x)
    restore_node_record(resume, dirs, hf_row, restarted)
    assert restarted.trap_record.trapped('H2O') == pytest.approx(stored.sum(), rel=1e-12)
    resumed = run_trapping(resume, row, hf_all, restarted, dirs=dirs)
    assert resumed.trapped_kg['H2O'] == pytest.approx(in_memory.trapped_kg['H2O'], rel=1e-12)
    assert resumed.trapped_kg['H2O'] == pytest.approx(-stored[6], rel=1e-12)
    assert resumed.remelted


@pytest.mark.physics_invariant
def test_a_resume_without_a_record_assigns_the_trapped_mass_to_the_nodes(tmp_path, caplog):
    """A run resumed without a stored record holds trapped mass on the helpfile
    that no node holds. It is assigned to the nodes that are not liquid, so the
    record matches the helpfile and that mass remelts with its nodes; the mush
    the new record finds holds that mass and buries nothing new."""
    caplog.set_level(logging.WARNING, logger='fwl.proteus.outgas.layer_trapping')
    trapped = 4.0e19
    hf_row = _hf_row(
        _X_FREEZE,
        H2O_kg_liquid=_H2O_LIQUID - trapped,
        H2O_kg_solid=trapped,
        H2O_kg_trapped=trapped,
    )
    interior = _interior(_X_FREEZE)
    dirs = {'output': str(tmp_path)}
    restore_node_record(_config(resume=True), dirs, {'Time': 1.0e4}, interior)
    assert any('no per-node record' in r.getMessage() for r in caplog.records)
    assert interior.trap_record is None
    run_trapping(_config(resume=True), hf_row, _hf_all(), interior, dirs=dirs)
    assert interior.trap_record.trapped('H2O') == pytest.approx(trapped, rel=1e-9)
    # Discrimination: burying the mush it finds again would add C (0.2 + 0.4) m.
    assert hf_row['H2O_kg_trapped'] == pytest.approx(trapped, rel=1e-12)

    # Every node liquid again: everything trapped returns, the adopted mass included.
    hf_all = pd.DataFrame([hf_row])
    interior.phi = np.ones(_N)
    row = dict(hf_row, Time=3.0e4, Phi_global=1.0)
    run_trapping(_config(), row, hf_all, interior)
    assert row['H2O_kg_trapped'] == pytest.approx(0.0, abs=1e-6 * trapped)
    assert row['H2O_kg_liquid'] == pytest.approx(_H2O_LIQUID, rel=1e-12)


@pytest.mark.physics_invariant
def test_continuous_resumed_and_resized_runs_trap_the_same_mass():
    """After a first step that buries the mush, an unchanged step with a melt
    that cannot drain keeps the trapped mass, whether the run continues with
    its record, resumes without one, or finds one sized for another mesh."""
    config = _config(melt_log10visc=30.0)
    interior = _interior(_X_FREEZE)
    first = _hf_row(_X_FREEZE)
    run_trapping(config, first, _hf_all(), interior)
    held = first['H2O_kg_trapped']
    assert held == pytest.approx(_C_H2O * 0.6 * _M_NODE, rel=1e-10)
    hf_all = pd.DataFrame([first])
    from proteus.outgas.node_record import NodeRecord

    records = {
        'continuous': interior.trap_record,
        'resumed': None,
        'resized': NodeRecord.initial(np.zeros(5, dtype=int), np.ones(5), np.ones(5), 0.0),
    }
    for name, record in records.items():
        row = dict(first, Time=3.0e4)
        again = _interior(_X_FREEZE, trap_record=record)
        run_trapping(config, row, hf_all, again)
        assert row['H2O_kg_trapped'] == pytest.approx(held, rel=1e-12), name
        assert again.trap_record.size == _N, name
        assert again.trap_record.trapped('H2O') == pytest.approx(held, rel=1e-9), name


def test_persisting_without_a_record_or_a_snapshot_writes_nothing(tmp_path):
    """The end-of-run write skips a run that kept no record and a time that
    has no snapshot, and creates no file."""
    output = str(tmp_path)
    persist_node_record({'output': output}, {'Time': 2.0e4}, SimpleNamespace(trap_record=None))
    assert not os.path.exists(os.path.join(output, 'data'))

    interior = _interior(_X_FREEZE)
    run_trapping(_config(), _hf_row(_X_FREEZE), _hf_all(), interior)
    persist_node_record({'output': output}, {'Time': 2.0e4}, interior)
    assert not os.path.exists(os.path.join(output, 'data'))
    path = _snapshot(output, 2.0e4)
    persist_node_record({'output': output}, {'Time': 2.0e4}, interior)
    with nc.Dataset(path) as ds:
        assert 'trap_phase_s' in ds.variables
        assert int(ds.variables['trap_phase_s'][7]) == PHASE_MUSH


def test_an_unreadable_snapshot_restarts_the_record_with_a_warning(tmp_path, caplog):
    """A snapshot the record cannot be read from, or written into, is reported
    and the step goes on: on a resume the record restarts and adopts the
    trapped mass, and a failed write leaves the step's moves in place."""
    caplog.set_level(logging.WARNING, logger='fwl.proteus.outgas.layer_trapping')
    output = str(tmp_path)
    hf_row = _hf_row(_X_FREEZE)
    path = _snapshot(output, hf_row['Time'])
    with open(path, 'wb') as handle:
        handle.write(b'not a netCDF file')
    run_trapping(_config(), hf_row, _hf_all(), _interior(_X_FREEZE), dirs={'output': output})
    assert any('could not write' in r.getMessage() for r in caplog.records)
    assert hf_row['H2O_kg_trapped'] == pytest.approx(_C_H2O * 0.6 * _M_NODE, rel=1e-10)

    prev = pd.DataFrame([dict(hf_row)])
    row = dict(hf_row, Time=3.0e4)
    restarted = _interior(_X_FREEZE)
    restore_node_record(_config(resume=True), {'output': output}, dict(hf_row), restarted)
    assert any('could not read' in r.getMessage() for r in caplog.records)
    assert restarted.trap_record is None
    run_trapping(_config(resume=True), row, prev, restarted, dirs={'output': output})
    assert row['H2O_kg_trapped'] == pytest.approx(hf_row['H2O_kg_trapped'], rel=1e-6)


def _resume_snapshot(output: str, time: float, x: np.ndarray, n: int = _N) -> None:
    """Snapshot of an Aragog run with its melt fraction and radius but no record."""
    data = os.path.join(output, 'data')
    os.makedirs(data, exist_ok=True)
    r = np.linspace(3.5e6, 6.3e6, n + 1)
    centre = 0.5 * (r[:-1] + r[1:])
    with nc.Dataset(snapshot_path_for_time(data, time, '_int.nc'), mode='w') as ds:
        ds.createDimension('staggered', n)
        ds.createVariable('phi_s', 'f8', ('staggered',))[:] = x
        ds.createVariable('radius_s', 'f8', ('staggered',))[:] = centre / 1.0e3


@pytest.mark.physics_invariant
def test_a_resume_without_a_record_buries_the_nodes_that_leave_the_ocean(tmp_path):
    """Resumed from a snapshot without a record, the first step restarts the
    record on that snapshot's melt fraction: node 5, which leaves the ocean in
    that step, buries C F m as in the run that kept its record, while the mush
    found there holds the trapped mass. A snapshot on a finer mesh gives the
    same. Restarting on the current mesh instead buries nothing for node 5."""
    config = _config(melt_log10visc=30.0)
    first = _hf_row(_X_FREEZE)
    kept = _interior(_X_FREEZE)
    run_trapping(config, first, _hf_all(), kept)
    x = _X_FREEZE.copy()
    x[5] = 0.3
    hf_all = pd.DataFrame([first])
    row = dict(first, Time=3.0e4, Phi_global=float(np.mean(x)))
    kept.phi = x
    run_trapping(config, row, hf_all, kept)
    continuous = row['H2O_kg_trapped']
    # Node 5 buries C F m and its crystals D C m (1 - F), on the first step's ocean melt.
    c = first['H2O_kg_liquid'] / (first['trap_Phi_ocean'] * _M_MANTLE)
    entered = c * (0.5 + _D_H2O * 0.5) * _M_NODE
    assert continuous - first['H2O_kg_trapped'] == pytest.approx(entered, rel=1e-6)

    for n in (_N, 2 * _N):
        output = str(tmp_path / f'run{n}')
        fine = np.repeat(_X_FREEZE, n // _N)
        _resume_snapshot(output, first['Time'], fine, n)
        resumed = dict(first, Time=3.0e4, Phi_global=float(np.mean(x)))
        run_trapping(config, resumed, hf_all, _interior(x), dirs={'output': output})
        assert resumed['H2O_kg_trapped'] == pytest.approx(continuous, rel=1e-9), n
    blind = dict(first, Time=3.0e4, Phi_global=float(np.mean(x)))
    run_trapping(config, blind, hf_all, _interior(x))
    # Discrimination: without the earlier profile node 5 is found in the mush.
    assert continuous - blind['H2O_kg_trapped'] == pytest.approx(entered, rel=1e-6)
