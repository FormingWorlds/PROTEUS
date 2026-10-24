"""Slow-tier test: the separation buoyancy of the config reaches the aragog solver PROTEUS builds.

Source: ``interior_energetics.aragog.separation_buoyancy``, through
``AragogRunner.setup_or_update_solver`` (numpy evaluator and JAX factory parameters).
A short PROTEUS run with the real Aragog writes the mesh and tables; the solver is then
built from that run in each mode, set on one mush entropy profile, and its gravitational
separation flux compared.

Invariants asserted:

- The profile has basic nodes in the Blake-Kozeny-Carman, Rumpf-Gupte and Stokes regimes.
- jgrav_b('solid') / jgrav_b('mixture') equals aragog's mobility ratio
  F_solid / F_mixture at every node with a flux: 1 / (1 - phi) on the matrix regimes
  (McKenzie 1984, Eq. 3.7, against Abe 1995, Eq. 40), towards 1 on the Stokes regime.
- Without aragog's ratio: 1 / (1 - phi) on the Blake-Kozeny-Carman nodes and on the
  Rumpf-Gupte nodes with 0.3 < phi < 0.5, and 1 within 1e-2 on the Stokes nodes above 0.91.
- The JAX separation velocity from the parameters PROTEUS builds for the factory gives
  the same ratio.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import pytest
from helpers import PROTEUS_ROOT

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]

_ZETA_1, _ZETA_2 = 0.0769452, 0.771462  # regime crossings of the aragog mobility


def _short_run(tmp_path):
    """Two steps of the dummy planet on the real Aragog, output kept under tmp_path."""
    from proteus import Proteus
    from proteus.utils.data import download_sufficient_data

    runner = Proteus(config_path=PROTEUS_ROOT / 'input' / 'dummy.toml')
    cfg = runner.config
    cfg.params.out.path = str(tmp_path / 'run')
    cfg.interior_energetics.module = 'aragog'
    cfg.interior_struct.melting_dir = 'Monteux-600'
    cfg.params.dt.initial, cfg.params.dt.minimum, cfg.params.dt.maximum = 450.0, 45.0, 4500.0
    cfg.params.stop.time.minimum, cfg.params.stop.time.maximum = 1.0e2, 1.0e3
    cfg.params.out.plot_mod, cfg.params.out.archive_mod = None, 'none'
    runner.init_directories()
    cfg.params.offline = False
    try:
        download_sufficient_data(cfg)
    except OSError:
        pass  # without network a missing file fails the run below
    runner.start(resume=False, offline=True)
    return runner


def _mush_state(runner, mode):
    """The solver PROTEUS builds in ``mode``, on an entropy from 1 % to 97 % melt from the CMB up."""
    from proteus.interior_energetics.aragog import AragogRunner
    from proteus.interior_energetics.common import Interior_t

    cfg = copy.deepcopy(runner.config)
    cfg.interior_energetics.aragog.separation_buoyancy = mode
    cfg.params.resume = False
    interior_o = Interior_t(cfg.interior_energetics.num_levels)
    interior_o._spider_eos_dir = runner.interior_o._spider_eos_dir
    interior_o.aragog_solver = None
    hf_row = runner.hf_all.iloc[-1].to_dict()
    AragogRunner.setup_or_update_solver(cfg, hf_row, interior_o, 1.0, runner.directories)
    solver = interior_o.aragog_solver
    P, eos = np.asarray(solver._P_stag_flat), solver.entropy_eos
    S_sol, S_liq = eos.solidus_entropy(P), eos.liquidus_entropy(P)
    solver.set_initial_entropy(S_sol + np.linspace(0.01, 0.97, P.size) * (S_liq - S_sol))
    # A solve of no length gives get_state() a solution at the set entropy.
    solver.parameters.solver.end_time = solver.parameters.solver.start_time + 1.0e-9
    solver.solve()
    return cfg, solver, solver.get_state()


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_the_configured_buoyancy_reaches_the_built_solver(tmp_path):
    """The aragog solver built by PROTEUS in 'solid' and in 'mixture' carries a gravitational
    separation flux whose ratio is aragog's mobility ratio at each node (McKenzie 1984, Eq. 3.7;
    Abe 1995, Eq. 40): 1 / (1 - phi) on the matrix regimes. A field the runner drops or swaps
    gives the ratio 1 or its inverse, which fails by up to the factor 2.8 of this profile."""
    from aragog.eos.entropy_phase import mobility_function

    runner = _short_run(tmp_path)
    (cfg, solver, mix), (_, solver_sol, sol) = (
        _mush_state(runner, m) for m in ('mixture', 'solid')
    )
    por = mix.porosity_b
    # The solve of no length moves each state by round-off only.
    np.testing.assert_allclose(sol.porosity_b, por, rtol=1e-12, atol=1e-15)
    flux = mix.jgrav_b != 0.0
    assert np.count_nonzero(flux & (por > 0.0) & (por < _ZETA_1)) >= 2, 'no BKC node'
    assert np.count_nonzero(flux & (por > _ZETA_1 + 0.05) & (por < _ZETA_2 - 0.1)) >= 5, (
        'no RG node'
    )
    assert np.count_nonzero(flux & (por > _ZETA_2 + 0.05)) >= 2, 'no Stokes node'

    a = cfg.interior_energetics.grain_size
    want = mobility_function(por, a, 'solid') / mobility_function(por, a, 'mixture')
    ratio = sol.jgrav_b[flux] / mix.jgrav_b[flux]
    np.testing.assert_allclose(ratio, want[flux], rtol=1e-9)
    rg = flux & (por > 0.3) & (por < 0.5)  # the Stokes term moves the ratio by < 1.4e-4 here
    np.testing.assert_allclose(
        sol.jgrav_b[rg] / mix.jgrav_b[rg], 1.0 / (1.0 - por[rg]), rtol=1e-3
    )
    bkc = flux & (por > 0.0) & (por < _ZETA_1)  # aragog's law gives 1 / (1 - phi) to 1e-7
    np.testing.assert_allclose(
        sol.jgrav_b[bkc] / mix.jgrav_b[bkc], 1.0 / (1.0 - por[bkc]), rtol=1e-6
    )
    stokes = flux & (por > 0.91)  # aragog's law gives 1 to 1e-2 above phi 0.9015
    assert np.count_nonzero(stokes) >= 2, 'no Stokes node above phi 0.91'
    np.testing.assert_allclose(sol.jgrav_b[stokes] / mix.jgrav_b[stokes], 1.0, rtol=1e-2)
    assert ratio.max() > 1.5  # an ignored field gives 1 on every node

    # The front drainage moves melt with the built solver's own mobility law: its real
    # evaluator on the run's tables, at the evaluator's own porosity, in each mode, compared
    # at unit drag (the drainage takes the melt viscosity, the solver the configured drag).
    from proteus.outgas.compaction import darcy_velocity

    drag = cfg.interior_energetics.aragog.separation_viscosity
    for mode, built in (('mixture', solver), ('solid', solver_sol)):
        phase = built.state.phase_basic
        rho_s, rho_l = phase.phase_boundary_densities()
        p, v = phase.porosity(), phase.relative_velocity()
        on = (rho_s - rho_l > 1.0) & (p > 1.0e-3) & (p < 0.999)
        assert np.count_nonzero(on) >= 10, mode
        eta = (
            phase._viscosity_val if drag == 'mixture' else np.full(p.shape, phase._visc_liquid)
        )
        g = np.broadcast_to(phase._g, p.shape)[on]
        w = darcy_velocity(p[on], a, (rho_s - rho_l)[on], g, 1.0, buoyancy=mode)
        np.testing.assert_allclose(w, v[on] * eta[on], rtol=1e-12)

    # The JAX path: the separation velocity from the factory parameters PROTEUS builds.
    jnp = pytest.importorskip('jax.numpy')
    from aragog.jax.phase import relative_velocity

    from proteus.interior_energetics.aragog import _cached_entropy_eos_jax
    from proteus.interior_energetics.aragog_phase import build_jax_phase_params

    assert getattr(solver, '_jax_cvode_factory', None) is not None
    eos_dir = runner.interior_o._spider_eos_dir
    if not (eos_dir and os.path.isdir(eos_dir)):  # the fallback of setup_solver
        eos_dir = os.path.join(runner.directories['output'], 'data', 'spider_eos')
    eos_jax = _cached_entropy_eos_jax(str(eos_dir))
    P_b = jnp.asarray(solver._P_basic_flat)
    args = [jnp.asarray(x) for x in (mix.rho_basic, mix.phi_basic, mix.g_b)]

    def velocity(mode):
        cfg.interior_energetics.aragog.separation_buoyancy = mode
        params = build_jax_phase_params(cfg)
        return np.asarray(relative_velocity(eos_jax, params, P_b, *args, jnp.ones_like(P_b)))

    inner = flux & (por > 0.01) & (por < 0.99)
    ratio_jax = velocity('solid')[inner] / velocity('mixture')[inner]
    np.testing.assert_allclose(ratio_jax, sol.jgrav_b[inner] / mix.jgrav_b[inner], rtol=1e-9)
