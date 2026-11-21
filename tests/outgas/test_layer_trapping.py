"""Unit tests for ``proteus.outgas.layer_trapping``.

The layer trapping step runs the per-node drainage of ``outgas.trap_mode =
'layers'`` on the Aragog mesh and moves the net locked mass between the melt
and the solid reservoirs of the helpfile row. These tests exercise:

* the node profiles: each staggered node takes the mean of its two basic-node
  densities and gravities, with the surface gravity and the node volume times
  density as fallbacks, and no profiles without the phase-boundary densities,
* a first freezing step through ``run_trapping``: the mass it locks, pinned to
  the pore melt of the mush nodes plus the crystals' share, moved with the
  species and element totals unchanged, and its diagnostics recorded,
* remelting decided per node: a node that turns liquid returns what it held
  while a node that stays in the mush keeps it, and a net return marks the
  step as a remelting step for escape and desiccation,
* the fallback without profiles, for freezing, an unchanged and a remelting
  mantle, without the front scheme's burial ledger,
* node masses rescaled to the mantle mass, the ocean melt the chemistry
  dissolves into, the layer diagnostics, and a user-set mush viscosity,
* returns capped at the trapped reservoir,
* the per-node record written into the interior snapshot and read back on a
  resume, trapped mass from before the record assigned to the nodes, and the
  same trapped mass whether a run continues, resumes or changes its mesh.

See ``docs/How-to/testing.md`` and ``docs/Explanations/test_framework.md``
for the test framework.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from proteus.outgas.compaction import BRANCH_FALLBACK, BRANCH_REMELT
from proteus.outgas.layer_drainage import PHASE_LIQUID, PHASE_MUSH, node_geometry
from proteus.outgas.layer_trapping import (
    _capped_moves,
    layer_profiles,
    persist_node_record,
    restore_node_record,
)
from proteus.outgas.trapping import run_trapping, trapped_mass, trapping_returned_mass
from proteus.utils.constants import secs_per_year
from proteus.utils.helper import snapshot_path_for_time

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Eight nodes of 1e23 kg between a core at 3500 km and a surface at 6300 km.
_N = 8
_M_NODE = 1.0e23
_M_MANTLE = _N * _M_NODE
_D_H2O = 0.0017
# The previous step left 7.6e20 kg of water in 0.95 of the mantle, a melt
# concentration of exactly 1e-3.
_PHI_PREV = 0.95
_H2O_LIQUID = 7.6e20
_C_H2O = 1.0e-3
# Hydrogen and oxygen mass fractions of water.
_H_FRAC = 2.01588 / 18.01528
_O_FRAC = 1.0 - _H_FRAC


def _config(resume: bool = False, melt_log10visc: float = 2.0):
    """Config double with the fields the layer step reads, as SI and config units.

    A melt viscosity of 1e30 Pa s stops the pore melt from draining, which
    isolates what a step locks or returns from what the drainage moves.
    """
    outgas = SimpleNamespace(
        trap_mode='layers',
        trap_phi_min=0.01,
        trap_mush_log10visc=-1.0,
        D_const_H2O=_D_H2O,
        mass_thresh=1e16,
    )
    interior = SimpleNamespace(
        rfront_loc=0.5,
        grain_size=1.0e-3,
        melt_log10visc=melt_log10visc,
        aragog=SimpleNamespace(separation_buoyancy='mixture'),
    )
    return SimpleNamespace(
        outgas=outgas, interior_energetics=interior, params=SimpleNamespace(resume=resume)
    )


def _interior(x, **overrides):
    """Aragog-like interior: staggered melt fraction and masses, basic-node profiles."""
    r = np.linspace(3.5e6, 6.3e6, _N + 1)
    fields = dict(
        phi=np.asarray(x, dtype=float),
        radius=r,
        mass=np.full(_N, _M_NODE),
        density=np.full(_N, 4000.0),
        rho_solid_b=np.linspace(4200.0, 3800.0, _N + 1),
        rho_melt_b=np.linspace(3800.0, 3400.0, _N + 1),
        g_b=np.linspace(10.5, 9.8, _N + 1),
        trap_record=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _hf_row(x, time: float = 2.0e4, **overrides) -> dict:
    """Helpfile row whose global melt fraction is the mass-weighted mean of ``x``."""
    row = {
        'Time': time,
        'M_mantle': _M_MANTLE,
        'Phi_global': float(np.mean(x)),
        'gravity': 9.8,
        'H2O_kg_liquid': _H2O_LIQUID,
        'H2O_kg_atm': 0.0,
        'H2O_kg_solid': 0.0,
        'H2O_kg_total': _H2O_LIQUID,
        'H_kg_liquid': _H_FRAC * _H2O_LIQUID,
        'H_kg_solid': 0.0,
        'H_kg_total': _H_FRAC * _H2O_LIQUID,
        'O_kg_liquid': _O_FRAC * _H2O_LIQUID,
        'O_kg_solid': 0.0,
        'O_kg_total': _O_FRAC * _H2O_LIQUID,
    }
    row.update(overrides)
    return row


def _hf_all(time: float = 1.0e4, phi: float = _PHI_PREV) -> pd.DataFrame:
    return pd.DataFrame([{'Time': time, 'M_mantle': _M_MANTLE, 'Phi_global': phi}])


def _snapshot(output: str, time: float) -> str:
    """Bare interior snapshot with the staggered dimension the solver writes."""
    data = os.path.join(output, 'data')
    os.makedirs(data, exist_ok=True)
    path = snapshot_path_for_time(data, time, '_int.nc')
    with nc.Dataset(path, mode='w') as ds:
        ds.createDimension('staggered', _N)
        ds.createVariable('phi', 'f8', ('staggered',))[:] = np.ones(_N)
    return path


# The two upper nodes in the mush at melt fractions of 0.2 and 0.4, the rest
# liquid; the asymmetry keeps a per-node remelt apart from a global one.
_X_FREEZE = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.4])


@pytest.mark.physics_invariant
def test_node_profiles_are_the_mean_of_their_two_basic_nodes(caplog):
    """Each node takes the mean of the densities and gravities at its two
    boundaries, so a linear profile gives the value at the node midpoint; the
    surface gravity stands in for a missing gravity profile, the node volume
    times the density for missing masses, and without the phase-boundary
    densities there are no profiles."""
    interior = _interior(_X_FREEZE)
    profiles = layer_profiles({'gravity': 9.8}, interior)
    mid = np.linspace(4200.0, 3800.0, 2 * _N + 1)[1::2]
    np.testing.assert_allclose(profiles['rho_solid'], mid, rtol=1e-14)
    # Discrimination: the inner boundary value is 25 kg/m3 off the midpoint.
    assert abs(profiles['rho_solid'][0] - 4200.0) > 20.0
    np.testing.assert_allclose(profiles['gravity'], np.linspace(10.5, 9.8, 2 * _N + 1)[1::2])
    assert np.all(profiles['rho_melt'] < profiles['rho_solid'])

    caplog.set_level(logging.WARNING, logger='fwl.proteus.outgas.layer_trapping')
    no_gravity = layer_profiles({'gravity': 9.8}, _interior(_X_FREEZE, g_b=None))
    np.testing.assert_allclose(no_gravity['gravity'], 9.8)
    assert any('surface gravity' in r.getMessage() for r in caplog.records)

    no_mass = layer_profiles({'gravity': 9.8}, _interior(_X_FREEZE, mass=None))
    volume, _, _ = node_geometry(np.linspace(3.5e6, 6.3e6, _N + 1))
    np.testing.assert_allclose(no_mass['mass'], 4000.0 * volume, rtol=1e-14)

    # Error contract: a mesh whose density sits on the wrong nodes is refused.
    assert layer_profiles({}, _interior(_X_FREEZE, rho_melt_b=np.ones(_N))) is None
    assert layer_profiles({}, _interior(_X_FREEZE, rho_solid_b=None)) is None


@pytest.mark.physics_invariant
def test_a_first_freezing_step_buries_the_pore_melt_of_the_mush_it_finds():
    """The first layer step after initialisation finds two nodes in the mush
    with melt fractions of 0.2 and 0.4. Their pore melt buries C x m, all the
    water it held when it left the ocean, which moves from the melt to the
    solid with each species and element total unchanged. A new record knows no
    earlier melt fraction, so no crystals' share is taken at first sight."""
    config = _config()
    interior = _interior(_X_FREEZE)
    hf_row = _hf_row(_X_FREEZE)
    dm_rm = _M_MANTLE * (_PHI_PREV - float(np.mean(_X_FREEZE)))

    step = run_trapping(config, hf_row, _hf_all(), interior)

    expected = _C_H2O * (0.2 + 0.4) * _M_NODE
    assert step.mode == 'layers'
    assert step.trapped_kg['H2O'] == pytest.approx(expected, rel=1e-10)
    # Discrimination: the front formula D_eff C dM_RM at the mean fraction 0.3
    # traps 3.0e19 kg, and C taken on the current melt gives 6.9e19 kg.
    front = ((1.0 - 0.3) * _D_H2O + 0.3) * _C_H2O * dm_rm
    assert abs(step.trapped_kg['H2O'] - front) > 0.4 * expected
    c_now = _H2O_LIQUID / (_M_MANTLE * float(np.mean(_X_FREEZE)))
    current_melt = expected * c_now / _C_H2O
    assert abs(step.trapped_kg['H2O'] - current_melt) > 0.1 * expected
    assert hf_row['H2O_kg_solid'] == pytest.approx(expected, rel=1e-12)
    assert hf_row['H2O_kg_trapped'] == pytest.approx(expected, rel=1e-12)
    assert hf_row['H_kg_trapped'] == pytest.approx(_H_FRAC * expected, rel=1e-4)
    for name in ('H2O', 'H', 'O'):
        assert hf_row[f'{name}_kg_liquid'] + hf_row[f'{name}_kg_solid'] == pytest.approx(
            hf_row[f'{name}_kg_total'], rel=1e-12
        )
    assert int(hf_row['trap_n_front']) == 2
    assert int(hf_row['trap_n_exited']) == 0
    assert hf_row['trap_kg_step'] == pytest.approx(expected, rel=1e-12)
    assert np.isnan(hf_row['trap_frac_bound'])
    record = interior.trap_record
    assert list(np.flatnonzero(record.phase == PHASE_MUSH)) == [6, 7]
    assert record.trapped('H2O') == pytest.approx(hf_row['H2O_kg_trapped'], rel=1e-12)


@pytest.mark.physics_invariant
def test_remelting_is_decided_node_by_node():
    """After the first step, node 6 turns liquid again while node 7 stays in
    the mush, with a melt that cannot drain. The global melt fraction rises
    from 0.825 to 0.925, and only node 6 returns what it held, pore melt and
    crystals' share, 33% of the trapped mass, while node 7 keeps its own. A
    remelt by the share of the solid that melted would return 57% instead. The net return
    marks the step as a remelting step."""
    config = _config(melt_log10visc=30.0)
    interior = _interior(_X_FREEZE)
    hf_row = _hf_row(_X_FREEZE)
    run_trapping(config, hf_row, _hf_all(), interior)
    held = interior.trap_record.pore['H2O'] + interior.trap_record.crystal['H2O']
    trapped_before = hf_row['H2O_kg_trapped']
    hf_all = pd.DataFrame([hf_row])

    x = _X_FREEZE.copy()
    x[6] = 1.0
    interior.phi = x
    row = dict(hf_row, Time=3.0e4, Phi_global=float(np.mean(x)))
    step = run_trapping(config, row, hf_all, interior)

    assert step.remelted
    assert step.branch == BRANCH_REMELT
    assert trapping_returned_mass(row)
    returned = trapped_before - row['H2O_kg_trapped']
    assert returned == pytest.approx(held[6], rel=1e-12)
    assert row['H2O_kg_trapped'] == pytest.approx(held[7], rel=1e-12)
    assert held[7] > 0.0
    # Discrimination: the share of the solid that remelted, applied globally.
    share = (row['Phi_global'] - hf_row['Phi_global']) / (1.0 - hf_row['Phi_global'])
    assert abs(returned - share * trapped_before) > 0.05 * returned
    assert interior.trap_record.phase[6] == PHASE_LIQUID
    for name in ('H2O', 'H', 'O'):
        assert row[f'{name}_kg_liquid'] + row[f'{name}_kg_solid'] == pytest.approx(
            row[f'{name}_kg_total'], rel=1e-12
        )


@pytest.mark.physics_invariant
def test_without_profiles_the_step_traps_without_the_burial_ledger():
    """An interior without the phase-boundary densities cannot drain node by
    node, and neither the front integral nor its burial ledger is used in its
    place. Freezing traps D C dM_RM alone on the fallback branch; an unchanged
    melt fraction moves nothing; remelting returns the remelted share of the
    solid and takes the same share out of the record, so the next step with
    profiles returns the rest and nothing twice."""
    config = _config(melt_log10visc=30.0)
    dm_rm = _M_MANTLE * (_PHI_PREV - float(np.mean(_X_FREEZE)))
    hf_row = _hf_row(_X_FREEZE)
    blind = _interior(_X_FREEZE, rho_solid_b=None)
    step = run_trapping(config, hf_row, _hf_all(), blind)
    assert step.branch == BRANCH_FALLBACK
    assert hf_row['trap_branch'] == float(BRANCH_FALLBACK)
    assert hf_row['H2O_kg_solid'] == pytest.approx(_D_H2O * _C_H2O * dm_rm, rel=1e-12)
    assert blind.trap_record is None
    assert np.isnan(hf_row['trap_Phi_ocean'])

    flat = dict(hf_row, Time=2.5e4)
    still = run_trapping(config, flat, pd.DataFrame([hf_row]), blind)
    assert still.trapped_kg == {}
    assert flat['H2O_kg_trapped'] == pytest.approx(hf_row['H2O_kg_trapped'], rel=0.0)

    # A record from a step with profiles, then a remelt the interior cannot resolve.
    seen = _interior(_X_FREEZE)
    first = _hf_row(_X_FREEZE)
    run_trapping(config, first, _hf_all(), seen)
    trapped = first['H2O_kg_trapped']
    seen.rho_solid_b = None
    x = _X_FREEZE.copy()
    x[6] = 0.6
    up = dict(first, Time=3.0e4, Phi_global=float(np.mean(x)))
    run_trapping(config, up, pd.DataFrame([first]), seen)
    share = (up['Phi_global'] - first['Phi_global']) / (1.0 - first['Phi_global'])
    assert up['H2O_kg_trapped'] == pytest.approx((1.0 - share) * trapped, rel=1e-12)
    assert seen.trap_record.trapped('H2O') == pytest.approx(up['H2O_kg_trapped'], rel=1e-12)
    seen.rho_solid_b = np.linspace(4200.0, 3800.0, _N + 1)
    seen.phi = np.ones(_N)
    end = dict(up, Time=4.0e4, Phi_global=1.0)
    run_trapping(config, end, pd.DataFrame([up]), seen)
    assert end['H2O_kg_trapped'] == pytest.approx(0.0, abs=1e-9 * trapped)
    assert end['H2O_kg_liquid'] == pytest.approx(first['H2O_kg_liquid'] + trapped, rel=1e-12)


@pytest.mark.physics_invariant
def test_a_return_is_capped_at_the_trapped_reservoir():
    """A return larger than the trapped reservoir is limited to it, a smaller
    one and a burial are passed on whole, and a zero move is dropped."""
    moves = _capped_moves(
        {'H2O_kg_trapped': 1.0e18, 'CO2_kg_trapped': 0.0},
        {'H2O': -5.0e18, 'CO2': 2.0e17, 'N2': 0.0, 'Ar': -1.0},
    )
    assert moves == {'H2O': -1.0e18, 'CO2': 2.0e17}
    # A return within the reservoir passes unchanged.
    assert _capped_moves({'H2O_kg_trapped': 1.0e18}, {'H2O': -4.0e17}) == {'H2O': -4.0e17}


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


def _freeze_bottom_up(config, step_yr: float = 1.0e4):
    """Freeze the mantle one node per step from the base, at a fixed melt concentration.

    A first step finds the mantle molten. Each step after it the next node
    enters the mush at a melt fraction of 0.3 and the one below it freezes, so
    the mush drains up into the melt. Before each step the melt is reset to hold
    water at ``_C_H2O`` of the melt it was dissolved into, as a chemistry solve
    at a fixed concentration would leave it. Returns the crystallised fraction,
    the cumulative trapped water and the trapped water the front scheme's
    ``trapped_mass`` gives at the no-drainage bound for the same steps [kg].
    """
    interior = _interior(np.ones(_N))
    hf_row = _hf_row(np.ones(_N), time=0.0, H_kg_total=1e30, O_kg_total=1e30)
    hf_all = pd.DataFrame([hf_row])
    crystallised, layers, bound = [], [], []
    for k in range(-1, _N + 1):
        x = np.ones(_N)
        x[: max(k, 0)] = 0.0
        if 0 <= k < _N:
            x[k] = 0.3
        # A solve dissolves into the ocean melt the previous layer step reported.
        last = hf_all.iloc[-1]
        ocean = float(last.get('trap_Phi_ocean', float('nan')))
        melt = _M_MANTLE * (ocean if np.isfinite(ocean) else float(last['Phi_global']))
        row = dict(
            hf_all.iloc[-1].to_dict(), Time=(k + 2) * step_yr, Phi_global=float(np.mean(x))
        )
        row['H2O_kg_liquid'] = _C_H2O * melt
        row['H2O_kg_total'] = row['H2O_kg_liquid'] + row['H2O_kg_solid']
        interior.phi = x
        step = run_trapping(config, row, hf_all, interior)
        mass, _ = trapped_mass(_D_H2O, 0.5, _C_H2O, step.dm_rm, float('inf'))
        assert not step.supply_capped
        crystallised.append(1.0 - row['Phi_global'])
        layers.append(row.get('H2O_kg_trapped', 0.0))
        bound.append((bound[-1] if bound else 0.0) + mass)
        hf_all = pd.DataFrame([row])
    return np.array(crystallised), np.array(layers), np.array(bound)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_undrained_mush_traps_the_front_schemes_no_drainage_bound(validation_plot):
    """Cross-check against the front scheme: with a melt that cannot drain, a
    mantle frozen node by node at a fixed melt concentration traps what
    ``trapped_mass`` of the front scheme gives at its no-drainage bound,
    F_tl = phi_c, Eq. 6 of Sim, Hirschmann and Hier-Majumder (2024), summed
    over the same steps: (D + phi_c (1 - D)) C M_mantle in all. The layer
    scheme buries the pore melt as each node enters and the front scheme traps
    it as it crystallises, so the two agree once every node has frozen. With a
    melt of 0.1 Pa s the pore melt drains in the step each node freezes, whose
    solver porosity falls from that of x = 0.3 to zero and caps the flowing
    melt, and less is trapped, never below the crystals' share D C M_mantle.
    """
    x_stiff, stiff, bound = _freeze_bottom_up(_config(melt_log10visc=30.0))
    x_drain, drained, _ = _freeze_bottom_up(_config(melt_log10visc=-1.0))
    total = (_D_H2O + 0.5 * (1.0 - _D_H2O)) * _C_H2O * _M_MANTLE
    assert x_stiff[-1] == pytest.approx(1.0, rel=1e-12)
    assert stiff[-1] == pytest.approx(bound[-1], rel=1e-10)
    assert bound[-1] == pytest.approx(total, rel=1e-12)
    # Discrimination: the nodes' own entry melt fraction, 0.3, for phi_c, and
    # a scheme without the crystals' share.
    assert abs(stiff[-1] - (_D_H2O + 0.3 * (1.0 - _D_H2O)) * _C_H2O * _M_MANTLE) > 0.3 * total
    assert abs(stiff[-1] - 0.5 * (1.0 - _D_H2O) * _C_H2O * _M_MANTLE) > 1e-3 * total
    # Drainage lowers the burial, within the two bounds.
    assert _D_H2O * _C_H2O * _M_MANTLE < drained[-1] < 0.9 * stiff[-1]
    # The last node's pore melt drains out at the surface with nothing entering
    # the mush after it, so the final step is a net return.
    assert drained[-1] < drained[-2]

    def figure():
        import matplotlib.pyplot as plt

        norm = _C_H2O * _M_MANTLE
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.plot(x_stiff, bound / norm, '-', label='front scheme, F_tl = phi_c')
        ax.plot(x_stiff, stiff / norm, 'o', mfc='none', label='layers, melt 1e30 Pa s')
        ax.plot(x_drain, drained / norm, 's', mfc='none', label='layers, melt 0.1 Pa s')
        ax.plot([0.0, 1.0], [0.0, _D_H2O], ':', color='0.5', label='crystals alone, D')
        ax.set_xlabel('Crystallised fraction of the mantle')
        ax.set_ylabel('Trapped H2O / (C M)')
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    validation_plot(figure, 'layer_trapping_no_drainage_bound')


@pytest.mark.physics_invariant
def test_node_masses_are_rescaled_to_the_mantle_mass():
    """The solver's node masses, density times a fixed volume, grow past the
    mantle mass as it cools. Masses summing to 1.4 M_mantle are rescaled, so
    the mush buries C x m of the true node mass; the raw masses would bury
    40 percent more."""
    heavy = _interior(_X_FREEZE, mass=np.full(_N, 1.4 * _M_NODE))
    profiles = layer_profiles(_hf_row(_X_FREEZE), heavy)
    assert float(np.sum(profiles['mass'])) == pytest.approx(_M_MANTLE, rel=1e-14)
    hf_row = _hf_row(_X_FREEZE)
    run_trapping(_config(), hf_row, _hf_all(), heavy)
    expected = _C_H2O * 0.6 * _M_NODE
    assert hf_row['H2O_kg_trapped'] == pytest.approx(expected, rel=1e-10)
    assert abs(hf_row['H2O_kg_trapped'] - 1.4 * expected) > 0.3 * expected
    # Unequal raw masses keep their shares: a node twice as heavy buries twice as much.
    lopsided = np.full(_N, _M_NODE)
    lopsided[7] = 2.0 * _M_NODE
    profiles = layer_profiles(_hf_row(_X_FREEZE), _interior(_X_FREEZE, mass=lopsided))
    assert profiles['mass'][7] == pytest.approx(2.0 * profiles['mass'][6], rel=1e-14)


@pytest.mark.physics_invariant
def test_the_chemistry_dissolves_into_the_ocean_melt_alone():
    """With nodes 6 and 7 in the mush at 0.2 and 0.4, the liquid nodes hold 6
    of the 6.6 node masses of melt, so of Phi_global = 0.825 the ocean melt is
    0.75 of the mantle. The step reports it, the chemistry sees it for the
    length of its solve only, and the next step takes its concentration on it."""
    from proteus.outgas.layer_trapping import ocean_melt_for_chemistry, ocean_melt_fraction

    config = _config(melt_log10visc=30.0)
    interior = _interior(_X_FREEZE)
    hf_row = _hf_row(_X_FREEZE)
    run_trapping(config, hf_row, _hf_all(), interior)
    assert hf_row['trap_Phi_ocean'] == pytest.approx(0.825 * 6.0 / 6.6, rel=1e-12)
    assert hf_row['trap_Phi_ocean'] == pytest.approx(0.75, rel=1e-12)
    assert ocean_melt_fraction(0.5, np.zeros(3), np.ones(3), 0.5) == pytest.approx(0.0, abs=0.0)

    with ocean_melt_for_chemistry(config, hf_row):
        assert hf_row['Phi_global'] == pytest.approx(0.75, rel=1e-12)
    assert hf_row['Phi_global'] == pytest.approx(0.825, rel=1e-12)
    with pytest.raises(RuntimeError), ocean_melt_for_chemistry(config, hf_row):
        raise RuntimeError('solver failed')
    assert hf_row['Phi_global'] == pytest.approx(0.825, rel=1e-12)
    front = SimpleNamespace(outgas=SimpleNamespace(trap_mode='front'))
    with ocean_melt_for_chemistry(front, hf_row):
        assert hf_row['Phi_global'] == pytest.approx(0.825, rel=1e-12)
    unset = dict(hf_row, trap_Phi_ocean=float('nan'))
    with ocean_melt_for_chemistry(config, unset):
        assert unset['Phi_global'] == pytest.approx(0.825, rel=1e-12)

    # The next step's concentration is the dissolved mass over the ocean melt:
    # node 5 enters the mush and buries C' F m with C' = liquid / (0.75 M).
    x = _X_FREEZE.copy()
    x[5] = 0.3
    interior.phi = x
    liquid = 6.0e20
    row = dict(hf_row, Time=3.0e4, Phi_global=float(np.mean(x)), H2O_kg_liquid=liquid)
    before = row['H2O_kg_trapped']
    run_trapping(config, row, pd.DataFrame([hf_row]), interior)
    c_ocean = liquid / (0.75 * _M_MANTLE)
    assert interior.trap_record.pore['H2O'][5] == pytest.approx(
        c_ocean * 0.5 * _M_NODE, rel=1e-12
    )
    c_global = liquid / (0.825 * _M_MANTLE)

    assert abs(interior.trap_record.pore['H2O'][5] - c_global * 0.5 * _M_NODE) > 0.05 * (
        row['H2O_kg_trapped'] - before
    )

    # A previous row whose ocean melt the helpfile stored as 0, because no layer
    # step set it, takes the concentration on the global melt instead. A new
    # record buries its mush at first sight, node 5 at its own x = 0.3.
    unset_prev = dict(_hf_row(_X_FREEZE), trap_Phi_ocean=0.0)
    fresh = _interior(x)
    again = _hf_row(x, time=3.0e4, H2O_kg_liquid=liquid)
    run_trapping(config, again, pd.DataFrame([unset_prev]), fresh)
    expected = c_global * 0.3 * _M_NODE
    assert fresh.trap_record.pore['H2O'][5] == pytest.approx(expected, rel=1e-12)
    assert abs(fresh.trap_record.pore['H2O'][5] - c_ocean * 0.3 * _M_NODE) > 0.05 * expected


@pytest.mark.physics_invariant
def test_the_layer_diagnostics_describe_the_floor_exits_and_the_thickest_mush_layer():
    """Node 6 freezes after one step in the mush while node 7 stays: one node
    leaves through the floor after 1e4 yr, having drained, and node 7 alone is
    the thickest mush layer, whose base, thickness and solid density set the
    front speed and its Courant number. The node's trapped fraction is the
    water it locked over C m, and the return is the negative net move."""
    from proteus.outgas.trap_bookkeeping import front_speed

    config = _config()
    interior = _interior(_X_FREEZE)
    first = _hf_row(_X_FREEZE)
    run_trapping(config, first, _hf_all(), interior)
    x = _X_FREEZE.copy()
    x[6] = 0.0
    interior.phi = x
    row = dict(first, Time=3.0e4, Phi_global=float(np.mean(x)))
    run_trapping(config, row, pd.DataFrame([first]), interior)

    r = np.linspace(3.5e6, 6.3e6, _N + 1)
    rho_base = 0.5 * (
        np.linspace(4200.0, 3800.0, _N + 1)[7] + np.linspace(4200.0, 3800.0, _N + 1)[8]
    )
    dt_s = 1.0e4 * secs_per_year
    v = front_speed(_M_MANTLE * (first['Phi_global'] - row['Phi_global']), r[7], rho_base, dt_s)
    assert int(row['trap_n_exited']) == 1
    assert row['trap_n_mush'] == pytest.approx(1.0, rel=0.0)
    assert int(row['trap_n_front']) == 1
    assert row['trap_L_front'] == pytest.approx(r[8] - r[7], rel=1e-12)
    assert row['trap_t_res'] == pytest.approx(1.0e4, rel=1e-12)
    assert row['trap_frac_bound'] == pytest.approx(0.0, abs=0.0)
    assert row['trap_v_front'] == pytest.approx(v, rel=1e-3)
    assert row['trap_front_courant'] == pytest.approx(v * dt_s / (r[8] - r[7]), rel=1e-3)
    assert np.isnan(row['trap_w_matrix_over_vf'])
    assert row['trap_n_substeps'] >= 1.0
    locked = interior.trap_record.crystal['H2O'][6]
    assert row['trap_F_tl'] == pytest.approx(locked / (_C_H2O * _M_NODE), rel=1e-9)
    assert 0.0 < row['trap_F_tl'] < 0.2
    assert row['trap_kg_returned'] == pytest.approx(max(-row['trap_kg_step'], 0.0), rel=1e-12)
    assert row['trap_Phi_ocean'] == pytest.approx(0.8 * 6.0 / 6.4, rel=1e-12)


@pytest.mark.physics_invariant
def test_a_set_mush_viscosity_sets_the_matrix_time_of_the_mush_layer():
    """``trap_mush_log10visc = 21`` gives the matrix time of the thickest mush
    layer, nodes 6 and 7, as 1e21 Pa s over |drho| g L, with drho 400 kg m-3
    and g the mean of their gravities; the default of 1e19 would be 100 times
    shorter."""
    config = _config()
    config.outgas.trap_mush_log10visc = 21.0
    hf_row = _hf_row(_X_FREEZE)
    run_trapping(config, hf_row, _hf_all(), _interior(_X_FREEZE))
    g = np.linspace(10.5, 9.8, 2 * _N + 1)[1::2]
    length = 2.0 * 3.5e5
    expected = 1.0e21 / (400.0 * float(np.mean(g[6:])) * length) / secs_per_year
    assert hf_row['trap_tau_s'] == pytest.approx(expected, rel=1e-6)
    assert abs(hf_row['trap_tau_s'] - 0.01 * expected) > 0.5 * expected


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
