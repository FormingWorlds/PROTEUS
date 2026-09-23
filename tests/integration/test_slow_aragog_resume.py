"""Slow-tier integration test: a resumed real Aragog run continues the run.

Exercises the resume branch of ``Proteus.start()`` with the real Aragog
entropy solver on the ``energy_balance`` core boundary condition, where the
CMB entropy gradient is a state variable of the solve. Each ``_int.nc``
snapshot stores it (``dSdr_cmb_state``) together with the surface pressure of
the solver's Adams-Williamson mesh (``mesh_surface_pressure``), and the resume
restores both. Restarting the gradient from a finite difference of the
restored profile gives a first-step CMB flux away from the control and a
lasting offset in the cumulative energy-conservation residual; rebuilding the
mesh from the restored row's P_surf shifts every mesh pressure by the
atmospheric load and the run no longer follows the control.

Design. One uninterrupted control run writes a snapshot on every step. Two
copies of its output directory are cut back to the same earlier step: one is
resumed as it is, and one has ``dSdr_cmb_state`` deleted from the seam
snapshot, which is what a finite-difference restart sees. Three Aragog solver
setups in total; the second copy runs only two steps past the seam.

Invariants asserted:

- The seam snapshot holds a finite CMB gradient, and the resume hands exactly
  that value to the solver's ``set_initial_dSdr_cmb``; the copy without it
  hands None, so the solver starts from the finite difference.
- A fresh run's snapshot stores the setup mesh surface pressure, 0 Pa.
- The resumed run's energy-conservation residual and first-step ``F_cmb`` are
  closer to the control than the finite-difference restart's.
- Temperatures stay positive and melt fraction in [0, 1] after the seam.
- After the seam the resumed run reproduces the control row by row: the same
  step times, and CMB flux and temperatures within FLUX_RTOL and TEMP_RTOL.

Wall time: about 3 min locally (macOS, M-series) for the three runs.

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

import shutil

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest
from helpers import PROTEUS_ROOT

from proteus import Proteus

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]

CONFIG = PROTEUS_ROOT / 'input' / 'dummy.toml'

# Control length and seam, in main-loop rows. The seam sits after the init
# stage (3 loops) with several rows on each side.
N_ITERS = 12
SEAM_ITERS = 6

# Parity tolerances: above the resumed-run difference (5e-6 in F_cmb, 1e-8 in
# T_magma) and well below the 1e-2 and 5e-3 of a mesh built from P_surf.
FLUX_RTOL = 2.0e-3
TEMP_RTOL = 1.0e-5


def _config_with_output_path(output_dir):
    """Write a copy of the dummy config that runs into ``output_dir``."""
    text = CONFIG.read_text()
    patched = text.replace('path = "auto"', f'path = "{output_dir}"', 1)
    assert patched != text, 'dummy config no longer carries the auto output path'
    destination = output_dir.parent / f'{output_dir.name}_config.toml'
    destination.write_text(patched)
    return destination


def _make_runner(output_dir, iters_max):
    """Build a real-Aragog runner on the dummy config writing into ``output_dir``."""
    runner = Proteus(config_path=_config_with_output_path(output_dir))
    cfg = runner.config
    cfg.interior_energetics.module = 'aragog'
    cfg.interior_struct.melting_dir = 'Monteux-600'
    assert cfg.interior_energetics.aragog.core_bc == 'energy_balance'
    cfg.params.dt.initial = 1.0e2
    cfg.params.dt.minimum = 1.0e1
    cfg.params.dt.maximum = 1.0e3
    cfg.params.stop.time.minimum = 0.0
    cfg.params.stop.time.maximum = 1.0e9
    cfg.params.stop.solid.enabled = False
    cfg.params.stop.iters.minimum = 1
    cfg.params.stop.iters.maximum = iters_max
    cfg.params.out.write_mod = 1
    cfg.params.out.dt_write_rel = 0.0
    cfg.params.out.archive_mod = 'none'
    cfg.params.out.plot_mod = None
    runner.init_directories()
    return runner


def _ensure_aragog_data(runner):
    """Download the Aragog lookup data if missing, as the shared slow-tier
    fixture does; a network error leaves the later FileNotFoundError to report."""
    from proteus.utils.data import download_sufficient_data

    was_offline = runner.config.params.offline
    runner.config.params.offline = False
    try:
        download_sufficient_data(runner.config, clean=False)
    except OSError:
        pass
    finally:
        runner.config.params.offline = was_offline


def _snapshot_time(path):
    """Model time [yr] stored in an Aragog ``_int.nc`` snapshot."""
    with nc.Dataset(path) as ds:
        return float(np.asarray(ds['time'][:]).item())


def _cut_back(src, dst, n_rows):
    """Copy run ``src`` to ``dst`` keeping the first ``n_rows`` helpfile rows
    and the snapshots up to the last kept time; return that time and the seam
    snapshot path."""
    shutil.copytree(src, dst)
    hf_path = dst / 'runtime_helpfile.csv'
    t_seam = float(pd.read_csv(hf_path, sep=r'\s+')['Time'].iloc[n_rows - 1])
    with open(hf_path) as f:
        lines = f.readlines()
    with open(hf_path, 'w') as f:
        f.writelines(lines[: n_rows + 1])
    seam = None
    for snap in sorted((dst / 'data').glob('*_int.nc')):
        t_snap = _snapshot_time(snap)
        if t_snap > t_seam * (1 + 1e-12):
            snap.unlink()
        elif abs(t_snap - t_seam) <= 1e-9 * max(t_seam, 1.0):
            seam = snap
    assert seam is not None, f'no snapshot at the seam time {t_seam} yr'
    return t_seam, seam


def _drop_gradient(snapshot):
    """Rewrite ``snapshot`` without ``dSdr_cmb_state`` (a finite-difference
    restart), copying every other variable unchanged."""
    tmp = snapshot.with_suffix('.tmp.nc')
    with nc.Dataset(snapshot) as src, nc.Dataset(tmp, 'w') as dst:
        for name, dim in src.dimensions.items():
            dst.createDimension(name, len(dim))
        for name, var in src.variables.items():
            if name == 'dSdr_cmb_state':
                continue
            out = dst.createVariable(name, var.dtype, var.dimensions)
            out[:] = var[:]
            if 'units' in var.ncattrs():
                out.units = var.units
    tmp.replace(snapshot)


def _resume(run_dir, iters_max):
    """Resume ``run_dir``; return the helpfile and the first CMB gradient
    PROTEUS hands to the solver's ``set_initial_dSdr_cmb`` (the restore call,
    made before any solve), recorded by a wrapper around the real method."""
    from aragog.solver import EntropySolver

    real = EntropySolver.set_initial_dSdr_cmb
    passed = []

    def _spy(self, value):
        passed.append(value)
        return real(self, value)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(EntropySolver, 'set_initial_dSdr_cmb', _spy)
        runner = _make_runner(run_dir, iters_max)
        runner.start(resume=True, offline=True)
    assert passed, 'the resume never set the CMB gradient override'
    return runner.hf_all.reset_index(drop=True), passed[0]


@pytest.fixture(scope='module')
def resume_runs(tmp_path_factory):
    """Control run plus the restored and the finite-difference resumes."""
    base = tmp_path_factory.mktemp('aragog_resume')
    control = _make_runner(base / 'control', N_ITERS)
    _ensure_aragog_data(control)
    control.start(resume=False, offline=True)
    ctrl = control.hf_all.reset_index(drop=True)
    # The end-of-run write replaces the last in-loop snapshot of the control.
    last_snap = max((base / 'control' / 'data').glob('*_int.nc'), key=_snapshot_time)
    with nc.Dataset(last_snap) as ds:
        final = (
            float(np.asarray(ds['dSdr_cmb_state'][:]).item())
            if 'dSdr_cmb_state' in ds.variables
            else None
        )

    t_seam, seam = _cut_back(base / 'control', base / 'restored', SEAM_ITERS + 1)
    with nc.Dataset(seam) as ds:
        stored = float(np.asarray(ds['dSdr_cmb_state'][:]).item())
        mesh_P = float(np.asarray(ds['mesh_surface_pressure'][:]).item())
    _, seam_fd = _cut_back(base / 'control', base / 'fd_restart', SEAM_ITERS + 1)
    _drop_gradient(seam_fd)

    restored, start_restored = _resume(base / 'restored', N_ITERS)
    fd, start_fd = _resume(base / 'fd_restart', SEAM_ITERS + 2)
    return {
        'ctrl': ctrl,
        'restored': restored,
        'fd': fd,
        't_seam': t_seam,
        'stored': stored,
        'start_restored': start_restored,
        'start_fd': start_fd,
        'final': final,
        'mesh_P': mesh_P,
    }


@pytest.mark.slow
@pytest.mark.physics_invariant
def test_resume_restores_the_cmb_entropy_gradient(resume_runs):
    """A resumed Aragog run hands the stored CMB gradient to the solver, and
    its energy-conservation residual and first-step CMB flux are closer to the
    uninterrupted control than a finite-difference restart.

    Physical scenario: a molten 1 M_Earth mantle on the real Aragog solver
    (energy_balance core boundary) with dummy structure, atmosphere, outgas and
    escape; the resume starts after main-loop row six.
    """
    r = resume_runs
    ctrl, res, fd = r['ctrl'], r['restored'], r['fd']
    # The stored gradient is finite and is exactly what the resume hands over.
    assert np.isfinite(r['stored'])
    # A run that ends normally leaves the gradient in its final snapshot.
    assert r['final'] is not None and np.isfinite(r['final'])
    assert r['start_restored'] == r['stored']
    # Discrimination: without the stored value the override is cleared.
    assert r['start_fd'] is None

    first = int(np.argmax(ctrl['Time'].to_numpy() > r['t_seam']))
    for D in (res, fd):
        assert float(D['Time'].iloc[first]) == pytest.approx(float(ctrl['Time'].iloc[first]))
    d_flux = {
        k: abs(D['F_cmb'].iloc[first] / ctrl['F_cmb'].iloc[first] - 1)
        for k, D in (('res', res), ('fd', fd))
    }
    d_res = {
        k: abs(D['E_residual_cons_frac'].iloc[first] - ctrl['E_residual_cons_frac'].iloc[first])
        for k, D in (('res', res), ('fd', fd))
    }
    assert d_flux['res'] < 0.5 * d_flux['fd'], d_flux
    assert d_res['res'] < 0.5 * d_res['fd'], d_res

    after = (res['Time'] > r['t_seam']).to_numpy()
    assert (res['T_magma'][after] > 0).all() and (res['T_cmb'][after] > 0).all()
    assert res['Phi_global'][after].between(0.0, 1.0).all()


@pytest.mark.slow
@pytest.mark.physics_invariant
def test_resume_matches_the_uninterrupted_control(resume_runs):
    """After the seam the resumed run reproduces the control row by row: same
    times, CMB flux within FLUX_RTOL and temperatures within TEMP_RTOL.

    Physical scenario: the same molten mantle and seam as the gradient test;
    the resumed Aragog mesh must be the one the control built at setup, not
    one rebuilt from the restored P_surf.
    """
    r = resume_runs
    ctrl, res = r['ctrl'], r['restored']
    # A fresh run builds its mesh at the setup surface pressure, 0 Pa.
    assert r['mesh_P'] == pytest.approx(0.0, abs=1.0)
    # Discrimination: the seam row's P_surf (about 8.1e3 bar at t = 202 yr on
    # the main-branch tables) is far from it, so a mesh rebuilt from the row
    # would differ.
    seam_row = int(np.argmin(np.abs(ctrl['Time'].to_numpy() - r['t_seam'])))
    assert abs(float(ctrl['P_surf'].iloc[seam_row]) * 1e5 - r['mesh_P']) > 1e8
    assert len(res) == len(ctrl)
    after = (ctrl['Time'] > r['t_seam']).to_numpy()
    np.testing.assert_allclose(res['Time'][after], ctrl['Time'][after], rtol=1e-9)
    np.testing.assert_allclose(res['F_cmb'][after], ctrl['F_cmb'][after], rtol=FLUX_RTOL)
    for key in ('T_cmb', 'T_magma'):
        np.testing.assert_allclose(res[key][after], ctrl[key][after], rtol=TEMP_RTOL)
