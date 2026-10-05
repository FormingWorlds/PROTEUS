"""Integration test: escape debits the volatile ledger the Zalmoxis target reads.

The main loop runs with the dummy modules and an escape rate, with and without
an accretion module, while each structure re-solve is a stub that takes its dry
mass target from the real ``load_zalmoxis_configuration``. That call reads the
PALEOS EOS tables, so the test needs the integration data set.

Invariants tested:
  - each escape step lowers M_volatile_change by the escaped mass
  - mass_tot stays the rock anchor
  - every structure re-solve targets mass_tot + V less the volatiles

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import numpy as np
import pytest

from proteus import Proteus
from proteus.utils.constants import M_earth
from tests.integration.test_smoke_accretion import _escape_runner

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


@pytest.mark.physics_invariant
@pytest.mark.parametrize('accretion', [True, False], ids=['accretion', 'escape-only'])
def test_main_loop_escape_debit_reaches_the_zalmoxis_target(tmp_path, monkeypatch, accretion):
    """With the Zalmoxis structure, with or without accretion, each escape step
    lowers the M_volatile_change ledger by the escaped mass, mass_tot stays the
    rock anchor, and every structure re-solve targets mass_tot + V less the
    volatiles.

    The re-solve is a stub that runs each step after the init stage and takes
    its dry target from the real load_zalmoxis_configuration; no impact lands,
    so only escape moves the ledger. A sign flip or a missing debit call
    changes the ledger and the re-solved interior mass.
    """
    import proteus.interior_energetics.wrapper as interior_wrapper
    from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration

    calls = []

    def target_solve(config, hf_row):
        hf_row['M_int'] = load_zalmoxis_configuration(config, hf_row)['planet_mass']
        interior_wrapper.update_planet_mass(hf_row)
        calls.append(hf_row['Time'])

    def initial_solve(dirs, config, hf_all, hf_row, outdir, **kwargs):
        interior_wrapper.determine_interior_radius_with_dummy(
            dirs, config, hf_all, hf_row, outdir
        )
        target_solve(config, hf_row)

    def resolve(dirs, config, hf_row, interior_o, t, T, phi, force=False):
        target_solve(config, hf_row)
        return hf_row['Time'], hf_row['T_magma'], hf_row['Phi_global']

    monkeypatch.setattr(interior_wrapper, 'solve_structure', initial_solve)
    monkeypatch.setattr(interior_wrapper, 'update_structure_from_interior', resolve)
    monkeypatch.setattr(Proteus, '_solve_structure_baseline_if_needed', lambda self: None)
    monkeypatch.setattr(Proteus, '_save_zalmoxis_output', lambda self: None)
    runner = _escape_runner(tmp_path / 'esc_zal', accretion=accretion)
    runner.config.interior_struct.module = 'zalmoxis'
    runner.config.interior_struct.zalmoxis.update_interval = 1.0
    runner.config.interior_struct.zalmoxis.equilibrate_init = False
    if accretion:
        runner.config.accretion.dummy.time_last = 1.0e9  # the impact never lands
    mass_before = runner.config.planet.mass_tot
    runner.start(resume=False, offline=True)

    hf = runner.hf_all[runner.hf_all['Time'] > 0.0]
    escaped = float(hf['esc_kg_cumulative'].iloc[-1])
    net = hf['M_volatile_change'].to_numpy()
    assert escaped > 0.0
    assert len(calls) > len(hf) // 2, 'the stub must re-solve inside the loop'
    assert -net[-1] == pytest.approx(escaped + float(hf['M_desiccated'].iloc[-1]), rel=1e-9)
    assert runner.config.planet.mass_tot == pytest.approx(mass_before, rel=1e-15)
    # Each row: interior = rock anchor + V (here the ledger, no rock) - volatiles.
    # The first post-init row still takes the init-stage budgets after its solve.
    expected_int = mass_before * M_earth + net - hf['M_ele'].to_numpy()
    np.testing.assert_allclose(hf['M_int'].to_numpy()[1:], expected_int[1:], rtol=1e-12)
