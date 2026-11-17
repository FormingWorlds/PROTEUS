"""Slow-tier integration test: structure re-solves reuse Aragog's compiled JAX programs.

Real Zalmoxis re-solves the structure every 100 yr and real Aragog steps the mantle with the
JAX right-hand side and Jacobian (option Z) on each new structure. The node count, the EOS
table shapes and the static settings stay the same across the re-solves, so the programs that
Aragog compiled for the first structure must serve every later one: after the second accepted
re-solve the process compiles no JAX program to the end of the run.

The compile count comes from JAX's backend-compile event; a log handler on the structure
wrapper marks each accepted re-solve ("Zalmoxis re-solve wall"). Atmosphere, star, orbit,
escape and chemistry are dummy modules; CALLIOPE stays from the shared config.

Runtime: about 9 min on an Apple M1 Ultra with one XLA thread.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.timeout(10800)]

_COMPILE = '/jax/core/compile/backend_compile_duration'
_CONFIG = Path(__file__).parent / 'zalmoxis_aragog_calliope.toml'


@pytest.mark.slow
def test_structure_resolves_reuse_the_compiled_programs(tmp_path):
    """Five or more accepted re-solves that move the radius, with no JAX compile after the
    second one; the JAX factory ran, and the run compiled before the first re-solve."""
    jax = pytest.importorskip('jax')
    from proteus import Proteus

    compiles, marks = [], []
    jax.monitoring.register_event_duration_secs_listener(
        lambda name, _d, **_kw: compiles.append(name) if name == _COMPILE else None
    )

    class _Resolve(logging.Handler):
        def emit(self, record):
            if record.getMessage().startswith('Zalmoxis re-solve wall'):
                marks.append(len(compiles))

    wrapper_log = logging.getLogger('fwl.proteus.interior_energetics.wrapper')
    handler = _Resolve(level=logging.INFO)
    wrapper_log.addHandler(handler)
    try:
        runner = Proteus(config_path=_CONFIG)
        runner.config.params.out.path = str(tmp_path / 'resolve_compiles')
        zal = runner.config.interior_struct.zalmoxis
        zal.update_interval, zal.update_min_interval = 100.0, 0.0
        runner.init_directories()
        dt = runner.config.params.dt
        dt.initial, dt.minimum, dt.maximum = 50.0, 5.0, 50.0
        stop = runner.config.params.stop.time
        stop.minimum, stop.maximum = 1e2, 650.0
        runner.config.params.out.plot_mod, runner.config.params.out.write_mod = None, 0
        runner.config.params.out.archive_mod = 'none'
        runner.start(resume=False, offline=True)
    finally:
        wrapper_log.removeHandler(handler)

    solver = runner.interior_o.aragog_solver
    assert getattr(solver, '_jax_factory_call_count', 0) > 0, 'option Z did not run'
    assert len(marks) >= 5, f'{len(marks)} accepted re-solves, need 5'
    assert marks[0] > 0, 'no JAX compile seen before the first re-solve'
    radii = runner.hf_all['R_int'].round(0).unique()
    assert radii.size >= 5, f'the radius took only {radii.size} values'
    n_end = len(compiles)
    assert n_end == marks[1], (
        f'{n_end - marks[1]} JAX compiles after the second re-solve; '
        f'count at each re-solve {marks}, at the end {n_end}'
    )
