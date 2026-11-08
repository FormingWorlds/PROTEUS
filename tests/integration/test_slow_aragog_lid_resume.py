"""Slow-tier integration test: a resume of a real Aragog run with the sub-grid lid starts from the stored lid.

Source: ``AragogRunner.setup_or_update_solver`` (resume branch) with ``restore_lid_start``
and the real Aragog ``EntropySolver`` with ``lid_mush_base`` on the EOS tables.

A resumed run is a new solver with no set lid and no previous solution. Its lid start is
the snapshot lid only when ``set_initial_lid`` runs before ``set_initial_entropy``; else the
start is the phantom lid, which needs the full floor gap ``S_sol + LID_FLOOR_GAP`` above the
solidus entropy of the top cell and refuses the two stored states below.

Design. One short run of the dummy planet on the real Aragog with the lid on writes its
snapshots. Three copies are resumed with the real ``Proteus.start``; a wrapper records the
solver after the first setup and stops the run there:

- the control: the last snapshot as stored (a live lid over a molten top cell);
- a lid handed over to the mesh, ``(0, 0)`` at 1900 yr, with the top cell 50 J/kg/K below the
  solidus entropy;
- a live lid with the top cell at ``S_sol + 0.85 J/kg/K``, inside the slack band
  ``S_sol + LID_FLOOR_SLACK .. S_sol + LID_FLOOR_GAP`` and above the heat-jump floor
  ``LID_DH_MIN``, both checked on the solver of the control.

Invariants asserted:

- Each resume hands the snapshot lid to the solver unchanged: the lid states of the start
  vector equal the stored thickness and energy, and the hand-over time is the stored one.
- The control resume reads the snapshot of its own copy (top-cell entropy as stored).

Wall time: about 90 s locally (macOS, M-series) for the run and the 3 resumes to the setup.

See also:
- docs/How-to/testing.md
- docs/Explanations/lid_coupling.md
"""

from __future__ import annotations

import math
import shutil

import netCDF4 as nc
import numpy as np
import pytest
from helpers import PROTEUS_ROOT

from proteus import Proteus
from proteus.config._interior import default_rtol

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]

CONFIG = PROTEUS_ROOT / 'input' / 'dummy.toml'
T_END = 2.0e3  # [yr] end of the short run; its last snapshot is the resume point
T_HANDOVER = 1.9e3  # [yr] hand-over time written into the handed-over copy
DS_SLACK = 0.85  # [J/kg/K] top cell above the solidus entropy in the slack-band copy


class _Stop(Exception):
    """Ends a resume after the first solver setup."""


def _make_runner(output_dir):
    """Real-Aragog runner with the sub-grid lid on the dummy planet, writing into ``output_dir``."""
    text = CONFIG.read_text()
    patched = text.replace('path = "auto"', f'path = "{output_dir}"', 1)
    assert patched != text, 'dummy config no longer carries the auto output path'
    cfg_path = output_dir.parent / f'{output_dir.name}_config.toml'
    cfg_path.write_text(patched)
    runner = Proteus(config_path=cfg_path)
    cfg = runner.config
    cfg.interior_energetics.module = 'aragog'
    cfg.interior_energetics.rtol = default_rtol('aragog')  # dummy.toml resolved it for 'dummy'
    cfg.interior_struct.melting_dir = 'Monteux-600'
    cfg.interior_energetics.kappah_floor = 10.0
    cfg.interior_energetics.aragog.lid_mush_base = True
    cfg.interior_energetics.aragog.rheology.enabled = True
    cfg.atmos_clim.surf_state = 'skin'
    cfg.params.dt.initial, cfg.params.dt.minimum, cfg.params.dt.maximum = 450.0, 45.0, 4500.0
    cfg.params.stop.time.minimum, cfg.params.stop.time.maximum = 1.0e2, T_END
    cfg.params.out.plot_mod, cfg.params.out.archive_mod = None, 'none'
    runner.init_directories()
    return runner


def _ensure_aragog_data(runner):
    """Download the Aragog lookup data if missing; offline, the run reports the missing file."""
    from proteus.utils.data import download_sufficient_data

    runner.config.params.offline = False
    try:
        download_sufficient_data(runner.config)
    except OSError:
        pass


def _last_snapshot(run_dir):
    """The ``_int.nc`` snapshot at the end of the run."""
    snaps = sorted(
        (run_dir / 'data').glob('*_int.nc'), key=lambda p: float(p.name[:-7].replace('p', '.'))
    )
    return snaps[-1]


def _edit(snapshot, S_top, delta, E_lid, t_ho):
    """Set the top-cell entropy and the stored lid of ``snapshot``."""
    with nc.Dataset(snapshot, 'a') as ds:
        ds['entropy_s'][-1] = S_top
        ds['lid_delta_state'][...] = delta
        ds['lid_E_state'][...] = E_lid
        if 'lid_handover_time_state' not in ds.variables:
            ds.createVariable('lid_handover_time_state', np.float64)
        ds['lid_handover_time_state'][...] = t_ho


def _resume_setup(run_dir):
    """Resume ``run_dir`` with the real ``Proteus.start`` and return the Aragog solver of its first
    setup together with the entropy profile the resume restored."""
    from proteus.interior_energetics.aragog import AragogRunner

    real = AragogRunner.setup_or_update_solver
    seen = {}

    def _record(config, hf_row, interior_o, dt, dirs, hf_all=None):
        real(config, hf_row, interior_o, dt, dirs, hf_all=hf_all)
        seen['solver'] = interior_o.aragog_solver
        seen['S'] = np.asarray(interior_o._last_entropy).copy()
        raise _Stop()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(AragogRunner, 'setup_or_update_solver', staticmethod(_record))
        runner = _make_runner(run_dir)
        with pytest.raises(_Stop):
            runner.start(resume=True, offline=True)
    return seen['solver'], seen['S']


@pytest.fixture(scope='module')
def lid_run(tmp_path_factory):
    """The short real-Aragog run with the lid, and the resume of its unchanged copy."""
    base = tmp_path_factory.mktemp('aragog_lid_resume')
    runner = _make_runner(base / 'run')
    _ensure_aragog_data(runner)
    runner.start(resume=False, offline=True)
    control = base / 'control'
    shutil.copytree(base / 'run', control)
    with nc.Dataset(_last_snapshot(control)) as ds:
        stored = (float(ds['lid_delta_state'][...]), float(ds['lid_E_state'][...]))
        S_top = float(ds['entropy_s'][-1])
    solver, S = _resume_setup(control)
    return base, solver, S, stored, S_top


def _copy(base, name, S_top, lid, t_ho):
    """A copy of the run with the edited last snapshot."""
    run = base / name
    shutil.copytree(base / 'run', run)
    _edit(_last_snapshot(run), S_top, *lid, t_ho)
    return run


@pytest.mark.slow
def test_the_control_resume_starts_from_the_stored_live_lid(lid_run):
    """The unchanged copy resumes its own snapshot and starts from the stored live lid."""
    _, solver, S, stored, S_top = lid_run
    assert S[-1] == S_top
    assert stored[0] > 0.0
    assert np.array_equal(solver._S0[-2:], stored)
    assert math.isnan(solver._lid_handover_time)


@pytest.mark.slow
def test_a_handed_over_lid_resumes_over_a_top_cell_below_the_solidus(lid_run):
    """A lid handed over to the mesh resumes as (0, 0) with its hand-over time while the top
    cell lies below the solidus entropy, where the phantom start refuses the state."""
    base, control, *_ = lid_run
    S_sol = control._lid_S_sol
    run = _copy(base, 'handed_over', S_sol - 50.0, (0.0, 0.0), T_HANDOVER)
    solver, S = _resume_setup(run)
    assert S[-1] == S_sol - 50.0
    assert np.array_equal(solver._S0[-2:], [0.0, 0.0])
    assert solver._lid_handover_time == T_HANDOVER


@pytest.mark.slow
def test_a_live_lid_resumes_in_the_slack_band_of_the_step_check(lid_run):
    """A live lid resumes unchanged with the top cell between the slack and the full floor gap,
    above the heat-jump floor: the step check accepts that state and the phantom start does not."""
    from aragog import surface_lid as sl

    base, control, _, stored, _ = lid_run
    S_sol, S_top = control._lid_S_sol, control._lid_S_sol + DS_SLACK
    assert sl.LID_FLOOR_SLACK <= DS_SLACK < sl.LID_FLOOR_GAP
    rho_b = control._lid_base_state(control._lid_T_sol, 0.0, S_sol)['rho']
    dh = float(control._lid_W(S_top) - control._lid_W(S_sol)) / rho_b
    assert dh >= sl.LID_DH_MIN
    run = _copy(base, 'slack_band', S_top, stored, float('nan'))
    solver, S = _resume_setup(run)
    assert S[-1] == S_top
    assert np.array_equal(solver._S0[-2:], stored)
