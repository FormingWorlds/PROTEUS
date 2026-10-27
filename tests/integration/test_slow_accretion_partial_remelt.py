"""Slow test: a partial giant-impact re-melt on a real Aragog solver.

Source: ``proteus.interior_energetics.wrapper._remelt_aragog`` at f < 1, with
the rule in ``proteus.interior_energetics.remelt``. A short run cools the
mantle from its initial condition; the test then re-melts f = 0.3 of it on the
live solver, for the energy_balance core and for the core_module core. The unit
tests use a fake solver; only a real one has the staggered mesh, the EOS
density and the restart path these clauses depend on.

Contract clauses checked:
  - the solver restarts from the partial profile, not from the molten initial
    condition that ``_set_entropy_ic`` writes, so the post-impact T_cmb follows
    the base cell the impact did not melt
  - the melt fills from the largest staggered radius down: the outermost cell
    takes the initial condition, the base keeps its end-of-step entropy
  - the melted mass closes against f times the structural mass of the pre-impact
    mesh (M_mantle), and not against the volume or the EOS mass rho(P, S_end) V
  - no cell cools
  - a core_module core keeps its temperature when the base is not re-melted
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from helpers import PROTEUS_ROOT

from proteus import Proteus
from proteus.config._interior import default_rtol
from proteus.interior_energetics.aragog import AragogRunner
from proteus.interior_energetics.wrapper import remelt_mantle
from proteus.utils.data import download_sufficient_data

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]

CONFIG = PROTEUS_ROOT / 'input' / 'dummy.toml'

# Melted mass fraction: several cells in full, one in part, the base untouched.
F_MELT = 0.3

# A cooled leg of a few Aragog steps [yr]; the impact is applied after it.
STOP_TIME = 3.0e2
DT = 1.0e2

# As in test_slow_accretion_resume.py: a molten start and a coarse EOS table.
MELTING_DIR = 'Monteux-600'
TSURF_INIT = 4000.0
LOOKUP_NP = 150
LOOKUP_NS = 60


def _cooled_runner(output_dir, core_bc):
    """Run a short cooled leg with no accretion and return the runner with its live solver."""
    text = CONFIG.read_text()
    patched = text.replace('path = "auto"', f'path = "{output_dir}"', 1)
    assert patched != text, 'dummy config no longer carries the auto output path'
    config_path = output_dir.parent / f'{output_dir.name}_config.toml'
    config_path.write_text(patched)

    runner = Proteus(config_path=config_path)
    cfg = runner.config
    cfg.interior_energetics.module = 'aragog'
    cfg.interior_energetics.rtol = default_rtol('aragog')
    cfg.interior_energetics.aragog.core_bc = core_bc
    cfg.planet.temperature_mode = 'isothermal'
    cfg.planet.tsurf_init = TSURF_INIT
    cfg.interior_struct.melting_dir = MELTING_DIR
    cfg.interior_struct.zalmoxis.lookup_nP = LOOKUP_NP
    cfg.interior_struct.zalmoxis.lookup_nS = LOOKUP_NS
    if core_bc == 'core_module':
        # The core profile needs the Zalmoxis structure, solved once.
        cfg.interior_struct.module = 'zalmoxis'
        cfg.interior_struct.zalmoxis.equilibrate_init = False
        cfg.interior_struct.zalmoxis.update_interval = 0
    cfg.params.stop.solid.enabled = False
    cfg.params.stop.time.minimum = 0.0
    cfg.params.stop.time.maximum = STOP_TIME
    cfg.params.stop.iters.minimum = 1
    cfg.params.stop.iters.maximum = 50
    cfg.params.dt.initial = DT
    cfg.params.dt.minimum = DT / 10.0
    cfg.params.dt.maximum = DT
    cfg.params.out.plot_mod = None
    cfg.params.out.archive_mod = 'none'
    # Fetch the lookup tables here; the run itself starts offline.
    cfg.params.offline = False
    try:
        download_sufficient_data(cfg)
    finally:
        cfg.params.offline = True
    try:
        runner.start(resume=False, offline=True)
    finally:
        shutil.rmtree(runner.directories['temp'], ignore_errors=True)
    return runner


@pytest.fixture(scope='module', params=['energy_balance', 'core_module'])
def partial_remelt(request, tmp_path_factory):
    """Re-melt F_MELT of a cooled real Aragog mantle; return the states the clauses compare."""
    core_bc = request.param
    runner = _cooled_runner(tmp_path_factory.mktemp(f'remelt_{core_bc}') / 'run', core_bc)
    solver = runner.interior_o.aragog_solver
    S_end = np.asarray(solver.entropy_staggered[:, -1], dtype=float).copy()
    T_core_pre = solver.get_current_core_temperature() if core_bc == 'core_module' else None

    captured = {}
    real = AragogRunner._set_entropy_ic

    def _spy(*args, **kwargs):
        out = real(*args, **kwargs)
        captured['S_molten'] = np.asarray(out, dtype=float).ravel().copy()
        return out

    hf_row = dict(runner.hf_row)
    mp = pytest.MonkeyPatch()
    mp.setattr(AragogRunner, '_set_entropy_ic', _spy)
    try:
        remelt_mantle(runner.directories, runner.config, hf_row, runner.interior_o, f=F_MELT)
    finally:
        mp.undo()

    P = np.asarray(solver._P_stag_flat, dtype=float).ravel()
    V = np.asarray(solver._volume_flat, dtype=float).ravel()
    eos = solver.entropy_eos
    return {
        'core_bc': core_bc,
        'solver': solver,
        'interior_o': runner.interior_o,
        'hf_row': hf_row,
        'S_end': S_end,
        'S_molten': captured['S_molten'],
        'S_new': np.asarray(runner.interior_o._last_entropy, dtype=float),
        'P': P,
        'V': V,
        'r': np.asarray(solver._r_stag_flat, dtype=float).ravel(),
        'mass': np.asarray(solver.evaluator.mesh.staggered_effective_density).ravel() * V,
        'mass_eos': np.asarray(eos.density(P, S_end), dtype=float).ravel() * V,
        'T_of': lambda p, s: float(
            np.asarray(eos.temperature(np.array([p]), np.array([s])))[0]
        ),
        'T_core_pre': T_core_pre,
    }


@pytest.mark.slow
@pytest.mark.physics_invariant
def test_the_solver_restarts_from_the_partial_profile(partial_remelt):
    """The solver's start entropy is the partial profile, and T_cmb follows what the impact left.

    On energy_balance T_cmb follows the unmelted base cell, not the molten one. On core_module
    T_cmb is the temperature of the core top, which keeps its pre-impact value.
    """
    d = partial_remelt
    n = d['S_new'].size
    np.testing.assert_array_equal(np.asarray(d['solver']._S0, dtype=float)[:n], d['S_new'])
    t_cmb = float(d['hf_row']['T_cmb'])
    if d['core_bc'] == 'core_module':
        assert t_cmb == pytest.approx(float(d['T_core_pre']), rel=1e-9)
        assert float(d['hf_row']['core_T_top']) == pytest.approx(t_cmb, rel=1e-12)
        return
    base = int(np.argmin(d['r']))
    t_new = d['T_of'](d['P'][base], d['S_new'][base])
    t_molten = d['T_of'](d['P'][base], d['S_molten'][base])
    assert abs(t_molten - t_new) > 20.0, 'the cooled base must differ from the molten one'
    assert abs(t_cmb - t_new) < 0.5 * abs(t_cmb - t_molten)


@pytest.mark.slow
@pytest.mark.physics_invariant
def test_the_melt_fills_from_the_largest_radius_down(partial_remelt):
    """The outermost cell takes the initial condition, the base keeps S_end, and no cell cools."""
    d = partial_remelt
    S_end, S_molten, S_new = d['S_end'], d['S_molten'], d['S_new']
    assert np.all(S_end < S_molten), 'the cooled mantle must lie below the initial condition'
    top, base = int(np.argmax(d['r'])), int(np.argmin(d['r']))
    np.testing.assert_equal(S_new[top], S_molten[top])
    np.testing.assert_equal(S_new[base], S_end[base])
    assert np.all(S_new >= S_end)
    w = (S_new - S_end) / (S_molten - S_end)
    assert np.all(np.diff(w[np.argsort(-d['r'], kind='stable')]) <= 1e-12)


@pytest.mark.slow
@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_melted_mass_closes_on_the_pre_impact_cell_masses(partial_remelt, validation_plot):
    """sum(w m) = f sum(m) for the structural cell mass m of the pre-impact mesh, the mass
    M_mantle reports; the cell volumes and the EOS masses rho(P, S_end) V do not close."""
    d = partial_remelt
    w = (d['S_new'] - d['S_end']) / (d['S_molten'] - d['S_end'])
    m, V = d['mass'], d['V']
    m_eos = d['mass_eos']
    shares = [float(np.sum(w * x)) / float(np.sum(x)) for x in (m, V, m_eos)]
    assert shares[0] == pytest.approx(F_MELT, rel=1e-9)
    # A volume basis puts the melt in other cells: the density rises with depth.
    assert abs(shares[1] - F_MELT) > 1e-3
    assert abs(shares[2] - F_MELT) > 1e-3
    assert float(np.sum(m)) == pytest.approx(float(d['hf_row']['M_mantle']), rel=1e-9)
    assert 0.0 < float(d['hf_row']['f_remelt']) == F_MELT

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            ['Structural mass', 'Cell volume', r'EOS mass $\rho(P, S_\mathrm{end})V$'],
            shares,
            width=0.5,
        )
        ax.axhline(F_MELT, ls='--', label=f'$f$ = {F_MELT}')
        ax.set_ylim(0.0, 0.45)
        ax.set_ylabel('Melted share of the mantle')
        ax.set_title(f'Core boundary: {d["core_bc"]}', loc='left', fontsize=9)
        ax.legend()
        fig.tight_layout()
        return fig

    validation_plot(figure, f'interior_energetics_remelt_melted_mass_{d["core_bc"]}')


@pytest.mark.slow
@pytest.mark.physics_invariant
def test_a_core_module_core_keeps_its_temperature_under_an_unmelted_base(partial_remelt):
    """With the base left at S_end the core_module core starts the next solve at its own temperature."""
    d = partial_remelt
    if d['core_bc'] != 'core_module':
        pytest.skip('energy_balance has no core state of its own')
    T_core_pre = float(d['T_core_pre'])
    assert np.isfinite(T_core_pre) and T_core_pre > 0.0
    assert float(d['solver']._T_core_init) == pytest.approx(T_core_pre, rel=1e-12)
