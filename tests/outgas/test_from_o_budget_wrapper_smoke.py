"""Round-trip of the from_O_budget CALLIOPE wrapper through the real solver.

Companion to ``test_from_o_budget_wrapper.py``: that file mocks CALLIOPE and
pins the wrapper's logic at unit tier, while these cases drive the real solver
and cost smoke-tier time. They live in their own file because a file carries
one tier, so that the tier filters select every test exactly once.

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

import numpy as np
import pytest

from proteus.outgas.calliope import calc_surface_pressures
from proteus.utils.constants import gas_list

from ._from_o_budget_helpers import _earth_hf_row, _make_from_o_budget_config

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60)]


@pytest.mark.parametrize(
    'dIW',
    [
        -2.0,
        0.0,
        2.0,
        4.0,
    ],
)
def test_smoke_from_o_budget_round_trip_through_wrapper(dIW):
    """Round-trip: run the wrapper in legacy mode at ``fO2_shift_IW = dIW``
    to derive the implied O budget, then re-run in from_O_budget mode with that
    budget and verify ``fO2_shift_IW_derived ≈ dIW``.

    Mirrors the CALLIOPE Stage 2 ``TestRoundTrip`` pattern but exercises
    the *PROTEUS wrapper* (calc_surface_pressures + construct_options +
    target assembly), so a regression in the wrapper-level glue (wrong
    target dict, wrong fO2_hint plumbing, key-name mismatch in the
    output filter) is caught here, not just in CALLIOPE.
    """
    dirs = {'output': '/tmp/test'}

    # Legacy leg
    config_legacy = _make_from_o_budget_config(fO2_shift_IW=dIW)
    config_legacy.planet.fO2_source = 'user_constant'
    hf_row_legacy = _earth_hf_row()
    calc_surface_pressures(dirs, config_legacy, hf_row_legacy)

    target_O = hf_row_legacy['O_kg_total']
    assert target_O > 0, 'legacy run must produce a positive O budget'
    assert hf_row_legacy['fO2_shift_IW_derived'] == pytest.approx(dIW)

    # from_O_budget leg, feeding the implied O budget back in
    config_path_c = _make_from_o_budget_config(fO2_shift_IW=dIW)
    hf_row_path_c = _earth_hf_row(O_kg_total=target_O)
    calc_surface_pressures(dirs, config_path_c, hf_row_path_c)

    derived = hf_row_path_c['fO2_shift_IW_derived']
    delta = abs(derived - dIW)

    # 0.05 dex is the same tolerance used by CALLIOPE Stage 2
    # TestRoundTrip; a wrapper-level glue bug (wrong fO2_hint, wrong
    # target['O'] source) would push delta well past 0.1 dex.
    assert delta < 0.05, f'wrapper round-trip failed at dIW={dIW}: derived={derived:.4f}'

    # from_O_budget must preserve the authoritative O budget across the call.
    assert hf_row_path_c['O_kg_total'] == pytest.approx(target_O, rel=1e-12)


def test_smoke_from_o_budget_residual_bounded_by_tolerance():
    """Under from_O_budget the wrapper writes the 5th residual into ``O_res``.
    With a converged solve the absolute residual must be well below the
    target; otherwise the chemistry would silently mis-conserve O across
    iterations. Discriminating: a value above ``target_O * 1e-3`` (0.1%)
    would indicate the per-element tolerance gate is broken.
    """
    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config(fO2_shift_IW=2.0)

    # Bootstrap the O budget from a legacy run so it's by-construction reachable.
    cfg_seed = _make_from_o_budget_config(fO2_shift_IW=2.0)
    cfg_seed.planet.fO2_source = 'user_constant'
    seed_hf = _earth_hf_row()
    calc_surface_pressures(dirs, cfg_seed, seed_hf)
    target_O = seed_hf['O_kg_total']

    hf_row = _earth_hf_row(O_kg_total=target_O)
    calc_surface_pressures(dirs, config, hf_row)

    assert np.isfinite(hf_row['O_res'])
    assert abs(hf_row['O_res']) <= max(target_O * 1e-3, 1e9)


def test_smoke_from_o_budget_mass_conservation_invariant():
    """End-to-end check that issue #677's mass-conservation invariant
    holds under from_O_budget: M_atm <= M_planet (which here is approximated
    by M_int + sum of element budgets, since the wrapper does not run
    update_planet_mass).

    The atmosphere mass aggregated from CALLIOPE's per-species
    ``s_kg_atm`` outputs must not exceed the planet's total tracked
    mass. A bug that re-introduced the O-skipping asymmetry would let
    M_atm exceed this bound at high H budgets.
    """
    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config(fO2_shift_IW=4.0)

    cfg_seed = _make_from_o_budget_config(fO2_shift_IW=4.0)
    cfg_seed.planet.fO2_source = 'user_constant'
    seed_hf = _earth_hf_row()
    calc_surface_pressures(dirs, cfg_seed, seed_hf)
    target_O = seed_hf['O_kg_total']

    hf_row = _earth_hf_row(O_kg_total=target_O)
    calc_surface_pressures(dirs, config, hf_row)

    M_atm = sum(float(hf_row.get(s + '_kg_atm', 0.0)) for s in gas_list)
    M_planet_lb = hf_row['M_int'] + (
        hf_row['H_kg_total']
        + hf_row['C_kg_total']
        + hf_row['N_kg_total']
        + hf_row['S_kg_total']
        + hf_row['O_kg_total']
    )

    assert M_atm > 0, 'sanity: outgassing produced a non-empty atmosphere'
    assert M_atm <= M_planet_lb, (
        f'from_O_budget atmosphere ({M_atm:.3e} kg) exceeds tracked planet '
        f'mass lower bound ({M_planet_lb:.3e} kg), issue #677 regression?'
    )
