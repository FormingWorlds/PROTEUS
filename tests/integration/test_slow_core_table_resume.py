"""The core temperature table across a resume.

A run of ``tests/integration/zalmoxis_aragog_calliope.toml`` with ``core_bc = 'core_module'``,
``temperature_mode = 'liquidus_super'`` and a structure update every 100 yr stops at 250 yr and
resumes. PROTEUS keeps no state of the core table: each solve builds it from the row and the
live core. A table built from the restored row and core of the resumed run must therefore
equal, bit for bit, the table built from the stopped run at its last row.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

pytest.importorskip('zalmoxis')

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]

CONFIG = pathlib.Path(__file__).parent / 'zalmoxis_aragog_calliope.toml'


def _runner(outdir, tmax):
    from proteus import Proteus

    runner = Proteus(config_path=str(CONFIG))
    runner.config.params.out.path = str(outdir)
    runner.config.planet.temperature_mode = 'liquidus_super'
    runner.config.interior_energetics.aragog.core_bc = 'core_module'
    z = runner.config.interior_struct.zalmoxis
    z.update_interval, z.update_min_interval = 100.0, 0.0
    runner.init_directories()
    dt = runner.config.params.dt
    dt.initial, dt.minimum, dt.maximum = 50.0, 5.0, 50.0
    runner.config.params.stop.time.minimum, runner.config.params.stop.time.maximum = 100.0, tmax
    out = runner.config.params.out
    out.plot_mod, out.write_mod, out.dt_write_rel, out.archive_mod = None, 1, 0.0, 'none'
    return runner


def _tables(config, row, t_core, profiles):
    """The live-core table and the row-fit table (the energy_balance input) of one state."""
    from proteus.interior_struct.core_temperature import structure_core_table

    return (
        structure_core_table(config, row, t_core, profiles),
        structure_core_table(config, row, float(row['T_cmb'])),
    )


def _equal(a, b):
    return np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


@pytest.mark.physics_invariant
def test_the_resumed_run_builds_the_core_table_of_the_stopped_run(tmp_path, monkeypatch):
    """On the stopped run at its last row (250 yr), the table of the production dispatch
    _structure_core_table is the live-core table. At the first Aragog solve after the resume,
    before it integrates, the live-core table (restored start T_core, restored core budget) and
    the row-fit table (restored row T_cmb and structure) equal those of the stopped run bit for
    bit. A 1-ulp change of the restored T_core or T_cmb changes its table, so the comparison can
    see a restore that is off by round-off."""
    import aragog.solver as aragog_solver

    from proteus.interior_energetics.aragog_core import start_core_temperature
    from proteus.interior_energetics.wrapper import _structure_core_table

    outdir = tmp_path / 'run'
    first = _runner(outdir, 250.0)
    first.start(resume=False, offline=True)
    solver = first.interior_o.aragog_solver
    t_end = solver.get_current_core_temperature()
    end = _tables(first.config, dict(first.hf_row), t_end, solver._core_module_budget.profiles)
    assert _equal(
        _structure_core_table(first.config, first.interior_o, dict(first.hf_row)), end[0]
    )

    second = _runner(outdir, 260.0)
    seen = {}
    solve = aragog_solver.EntropySolver.solve

    def first_solve(self, *a, **k):
        if 'tables' not in seen:
            t0 = start_core_temperature(self, 'core_module')
            row = dict(second.hf_row)
            profiles = self._core_module_budget.profiles
            seen['time'], seen['t_core'] = float(row['Time']), t0
            seen['tables'] = _tables(second.config, row, t0, profiles)
            up = {**row, 'T_cmb': np.nextafter(float(row['T_cmb']), np.inf)}
            seen['canary'] = _tables(second.config, up, np.nextafter(t0, np.inf), profiles)
        return solve(self, *a, **k)

    monkeypatch.setattr(aragog_solver.EntropySolver, 'solve', first_solve)
    second.start(resume=True, offline=True)

    assert seen['time'] == pytest.approx(250.0, rel=1e-12)
    assert seen['t_core'] == t_end
    for k in (0, 1):
        assert _equal(seen['tables'][k], end[k])
        assert not _equal(seen['canary'][k], end[k])
