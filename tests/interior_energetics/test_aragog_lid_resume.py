"""Resume of the Aragog sub-grid lid: the snapshot stores the lid state and the resume restores it."""

from __future__ import annotations

import logging
import math
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import netCDF4 as nc
import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]
T_SNAP = 30039.3583587
LIDS = {
    'live': (0.182886055149, -321606520.677, float('nan')),
    'handed-over': (0.0, 0.0, 41230.5),
    'handed-over-time-unknown': (0.0, 0.0, float('nan')),
}


def _output(lid=None, n=6):
    """SolverOutput stand-in for the snapshot writer, with the lid fields of ``lid`` or none."""
    r_b = np.linspace(3.5e6, 6.371e6, n + 1)
    out = SimpleNamespace(
        S_final=np.linspace(3100.0, 3000.0, n),
        T_stag=np.linspace(4200.0, 1500.0, n),
        phi_stag=np.ones(n),
        r_stag=0.5 * (r_b[:-1] + r_b[1:]),
        P_stag=np.linspace(135e9, 1e5, n),
        r_basic=r_b,
        visc_stag=np.full(n, 0.1),
        rho_stag=np.linspace(5000.0, 3000.0, n),
        heat_flux=np.full(n + 1, 1.0e4),
        heating=np.zeros(n),
        mass_stag=np.full(n, 1.0e23),
        Phi_global=1.0,
    )
    if lid is not None:
        out.lid_delta, out.lid_E, out.lid_handover_time = lid
    return out


def _snapshot(tmp_path, lid=None, **raw):
    """Write the snapshot at T_SNAP with the real writer; ``raw`` sets lid variables directly."""
    from proteus.interior_energetics.aragog import AragogRunner
    from proteus.utils.helper import snapshot_path_for_time

    os.makedirs(tmp_path / 'data', exist_ok=True)
    AragogRunner._write_output_ncdf(str(tmp_path), T_SNAP, _output(lid))
    path = snapshot_path_for_time(str(tmp_path / 'data'), T_SNAP, '_int.nc')
    with nc.Dataset(path, mode='a') as ds:
        for name, value in raw.items():
            if name not in ds.variables:
                ds.createVariable(name, np.float64)
            ds[name][0] = value
    return str(tmp_path)


class _LidSolver:
    """Solver stand-in with the lid on or off whose ``set_initial_lid`` is aragog's own."""

    def __init__(self, n_lid=2):
        from aragog.solver.entropy_solver import EntropySolver

        self._n_lid, self._S0, self._n_stag = n_lid, None, 6
        self._lid_init = self._lid_handover_time = None
        self._real = EntropySolver.set_initial_lid
        self.seen = []

    def _n_core_state(self):
        return self._n_stag

    def set_initial_lid(self, delta, E_lid=None, handover_time=float('nan')):
        self._real(self, delta, E_lid, handover_time)
        self.seen.append((self._lid_init, self._lid_handover_time))

    # Resume calls of setup_or_update_solver that the lid does not need.
    def initialize(self):
        pass

    def set_initial_dSdr_cmb(self, value):
        pass

    def set_initial_entropy(self, S):
        pass


def _same(a, b):
    return all(x == y or (math.isnan(x) and math.isnan(y)) for x, y in zip(a, b))


@pytest.mark.parametrize('kind', sorted(LIDS))
def test_snapshot_stores_the_lid_state(tmp_path, kind):
    """The snapshot keeps the thickness, the energy and a known hand-over time of the lid."""
    from proteus.interior_energetics.aragog import _snapshot_scalar

    delta, E, t_ho = LIDS[kind]
    out = _snapshot(tmp_path, LIDS[kind])
    got = [
        _snapshot_scalar(out, T_SNAP, f'lid_{n}_state') for n in ('delta', 'E', 'handover_time')
    ]
    assert got[:2] == [(delta, 'ok'), (E, 'ok')]
    assert got[2] == ((None, 'absent') if math.isnan(t_ho) else (t_ho, 'ok'))


def test_snapshot_without_the_lid_stores_no_lid_and_warns_nothing(tmp_path, caplog):
    """With the lid off (NaN thickness) or a solver output without lid fields, no lid is stored."""
    from proteus.interior_energetics.aragog import _snapshot_scalar

    for lid in (None, (float('nan'), float('nan'), float('nan'))):
        with caplog.at_level('WARNING'):
            out = _snapshot(tmp_path, lid)
        assert _snapshot_scalar(out, T_SNAP, 'lid_delta_state') == (None, 'absent')
        assert _snapshot_scalar(out, T_SNAP, 'lid_E_state') == (None, 'absent')
    assert not [r for r in caplog.records if 'lid' in r.getMessage()]


@pytest.mark.parametrize('kind', sorted(LIDS))
def test_resume_hands_the_snapshot_lid_to_the_solver(tmp_path, kind, caplog):
    """A resume starts the lid from the snapshot, not as the phantom lid at lid_delta_lo."""
    from proteus.interior_energetics.aragog import AragogRunner

    out = _snapshot(tmp_path, LIDS[kind])
    interior_o = MagicMock()
    interior_o.aragog_solver = None
    solver = _LidSolver()

    def _setup(config, hf_row, interior_o, outdir, **kw):
        interior_o.aragog_solver = solver

    def _update(dt, hf_row, interior_o, output_dir=None, hf_all=None):
        interior_o._last_entropy = np.linspace(3100.0, 3000.0, 6)

    config = MagicMock()
    config.params.resume = True
    config.interior_energetics.aragog.core_bc = 'quasi_steady'
    with (
        patch.object(AragogRunner, 'setup_solver', side_effect=_setup),
        patch.object(AragogRunner, 'update_solver', side_effect=_update),
        patch.object(AragogRunner, '_pending_structure_update', return_value=None),
        patch.object(AragogRunner, '_maybe_install_jax_cvode_factory'),
        patch('proteus.interior_energetics.aragog._maybe_log_solver_environment'),
        patch('proteus.interior_energetics.aragog.check_solver_buoyancy'),
        caplog.at_level('WARNING'),
    ):
        AragogRunner.setup_or_update_solver(
            config, {'Time': T_SNAP}, interior_o, 100.0, {'output': out}
        )
    delta, E, t_ho = LIDS[kind]
    assert len(solver.seen) == 1
    assert solver.seen[0][0] == (delta, E)
    assert _same([solver.seen[0][1]], [t_ho])
    assert not [r for r in caplog.records if 'phantom lid' in r.getMessage()]


def _raw_fields(draw, st):
    """Lid variables of a snapshot: each absent, NaN, infinite, or a value in or out of range."""
    value = st.one_of(
        st.just(float('nan')),
        st.sampled_from([float('inf'), -float('inf'), 0.0, -1.0e-9, -0.5]),
        st.floats(min_value=-1.0e9, max_value=1.0e9, allow_nan=False),
    )
    raw = {}
    for name in ('lid_delta_state', 'lid_E_state', 'lid_handover_time_state'):
        if draw(st.booleans()):
            raw[name] = draw(value)
    return raw


def _expected(raw):
    """What the restore does with the raw variables: 'phantom', 'refuse', or the lid it sets."""
    finite = {k: v for k, v in raw.items() if math.isfinite(v)}
    if 'lid_delta_state' not in finite or 'lid_E_state' not in finite:
        return 'phantom'
    d, E = finite['lid_delta_state'], finite['lid_E_state']
    t_ho = finite.get('lid_handover_time_state', float('nan'))
    if not (math.isnan(t_ho) or t_ho >= 0.0):
        return 'refuse'
    if d == 0.0 and E == 0.0:
        return ((0.0, 0.0), t_ho)
    if d <= 0.0 or not math.isnan(t_ho):
        return 'refuse'
    return ((d, E), float('nan'))


def test_restore_of_missing_nan_and_out_of_range_lid_fields(tmp_path):
    """Property: a missing or non-finite thickness or energy starts the phantom lid with one
    warning; a value aragog refuses stops the resume with the snapshot named; any other set is
    restored as stored; a solver with the lid off takes nothing and logs nothing."""
    hypothesis = pytest.importorskip('hypothesis')
    st = hypothesis.strategies
    from proteus.interior_energetics.aragog import restore_lid_start

    counter = iter(range(10**6))

    @hypothesis.settings(derandomize=True, deadline=None, max_examples=200)
    @hypothesis.given(st.data())
    def check(data):
        raw = _raw_fields(data.draw, st)
        out = _snapshot(tmp_path / str(next(counter)), **raw)
        want = _expected(raw)
        log = logging.getLogger('fwl.proteus.interior_energetics.aragog')
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        log.addHandler(handler)
        try:
            off = _LidSolver(n_lid=0)
            restore_lid_start(off, out, T_SNAP)
            assert off.seen == [] and records == []
            solver = _LidSolver()
            if want == 'refuse':
                with pytest.raises(
                    ValueError, match=r'Resume: the lid of the snapshot at t = '
                ):
                    restore_lid_start(solver, out, T_SNAP)
            else:
                restore_lid_start(solver, out, T_SNAP)
            warned = [r for r in records if r.levelno == logging.WARNING]
            if want == 'phantom':
                assert solver.seen == []
                assert len(warned) == 1 and 'phantom lid' in warned[0].getMessage()
            elif want != 'refuse':
                assert len(solver.seen) == 1 and solver.seen[0][0] == want[0]
                assert _same([solver.seen[0][1]], [want[1]]) and warned == []
        finally:
            log.removeHandler(handler)

    check()
