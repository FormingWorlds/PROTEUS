"""Unit tests for the outgassing temperature floor of ``proteus.outgas.calliope``.

``construct_options`` clips ``T_magma`` to ``outgas.T_floor`` for the CALLIOPE input, so
the gas_prs target masses (``calc_target_masses``) and the solve (``calc_surface_pressures``)
use one temperature, and ``hf_row['T_magma']`` keeps its value.

Invariants tested:
  - below T_floor the gas_prs targets equal those at T_floor; above it they are CALLIOPE's at T_magma
  - both solve entry points receive the floored temperature
  - the clip warning logs where the floor changes a result: the solve, and gas_prs targets

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip('calliope')
pytest.importorskip('hypothesis')

from hypothesis import example, given, settings
from hypothesis import strategies as st

from proteus.outgas.calliope import (
    calc_surface_pressures,
    calc_target_masses,
    construct_options,
)
from tests.outgas._calliope_helpers import (
    _HF_ROW,
    _element_mode_config,
    _surface_pressure_hf_row,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

ELEMENTS = ('H', 'C', 'N', 'S')
CLIP = 'Outgassing temperature clipped'


@lru_cache(maxsize=2)
def _dummy_calliope_config(volatile_mode):
    """input/dummy.toml with CALLIOPE outgassing; gas_prs from 20/30/1/1 bar H2O/CO2/N2/S2."""
    from proteus.config import read_config_object

    config = read_config_object(str(Path(__file__).parents[2] / 'input' / 'dummy.toml'))
    config.outgas.module = 'calliope'
    config.planet.volatile_mode = volatile_mode
    for gas, bar in (('H2O', 20.0), ('CO2', 30.0), ('N2', 1.0), ('S2', 1.0)):
        setattr(config.planet.gas_prs, gas, bar)
    return config


def _targets(config, T_magma):
    """H, C, N, S target masses [kg] from calc_target_masses at ``T_magma``."""
    hf_row = dict(_HF_ROW, T_magma=T_magma)
    calc_target_masses({}, config, hf_row)
    # The clip applies to the CALLIOPE input only; the planet keeps its temperature.
    assert hf_row['T_magma'] == T_magma
    return [hf_row[f'{e}_kg_total'] for e in ELEMENTS]


@pytest.mark.physics_invariant
def test_gas_prs_targets_below_the_floor_are_the_targets_at_the_floor():
    """At 100 K below T_floor the targets equal those at T_floor."""
    config = _dummy_calliope_config('gas_prs')
    T_floor = config.outgas.T_floor
    floor = _targets(config, T_floor)
    assert _targets(config, T_floor - 100) == floor
    # The targets depend on T_magma above the floor, so the comparison can fail.
    assert _targets(config, 2 * T_floor) != floor


@pytest.mark.physics_invariant
@settings(max_examples=40, deadline=None)
@example(frac=0.5, above=4300.0)
@given(
    frac=st.floats(min_value=1e-6, max_value=1.0, exclude_max=True),
    above=st.floats(0.0, 4300.0),
)
def test_the_calliope_temperature_is_floored_over_its_domain(frac, above):
    """Below T_floor the CALLIOPE input is T_floor and the targets are those at T_floor; at
    and above it the input is T_magma and the targets are CALLIOPE's at T_magma."""
    from calliope.solve import get_target_from_pressures

    config = _dummy_calliope_config('gas_prs')
    T_floor = config.outgas.T_floor
    T = frac * T_floor
    assert construct_options({}, config, dict(_HF_ROW, T_magma=T))['T_magma'] == T_floor
    assert _targets(config, T) == _targets(config, T_floor)
    T = T_floor + above
    opts = construct_options({}, config, dict(_HF_ROW, T_magma=T))
    assert opts['T_magma'] == T
    direct = get_target_from_pressures(dict(opts, T_magma=T))
    assert _targets(config, T) == [direct[e] for e in ELEMENTS]


@pytest.mark.parametrize(
    'source, solver',
    [
        ('user_constant', 'equilibrium_atmosphere'),
        ('from_O_budget', 'equilibrium_atmosphere_authoritative_O'),
    ],
)
def test_both_solves_take_the_floored_temperature(caplog, source, solver):
    """Both CALLIOPE entry points get T_floor for a T_magma below it, with one clip warning."""
    config = _element_mode_config({})
    config.planet.fO2_source = source
    T_floor = config.outgas.T_floor
    hf_row = dict(_surface_pressure_hf_row(), T_magma=T_floor - 100)
    result = {'fO2_shift_derived': 0.0, 'O_res': 0.0}
    with (
        patch(f'proteus.outgas.calliope.{solver}', return_value=result) as solve,
        caplog.at_level('WARNING', logger='fwl.proteus.outgas.calliope'),
    ):
        calc_surface_pressures({'output': '/tmp/test'}, config, hf_row)
    assert solve.call_args.args[1]['T_magma'] == T_floor
    assert hf_row['T_magma'] == T_floor - 100
    assert caplog.text.count(CLIP) == 1


@pytest.mark.parametrize('mode, warnings', [('gas_prs', 1), ('elements', 0)])
def test_a_target_call_warns_only_where_the_floor_changes_the_targets(caplog, mode, warnings):
    """Below T_floor a gas_prs target call logs one clip warning; an elements target call,
    whose targets do not depend on T_magma, logs none; at T_floor neither logs."""
    config = _dummy_calliope_config(mode)
    T_floor = config.outgas.T_floor
    for T, n in ((T_floor - 100, warnings), (T_floor, 0)):
        caplog.clear()
        with caplog.at_level('WARNING', logger='fwl.proteus.outgas.calliope'):
            _targets(config, T)
        assert caplog.text.count(CLIP) == n
    if mode == 'gas_prs':
        assert _targets(config, T_floor - 100) == _targets(config, T_floor)
    else:  # the premise of no warning: elements targets do not depend on T_magma
        from calliope.solve import get_target_from_params

        opts = construct_options({}, config, dict(_HF_ROW, T_magma=T_floor))
        below = get_target_from_params(dict(opts, T_magma=T_floor - 100))
        assert below == get_target_from_params(opts)
        assert _targets(config, 2 * T_floor) == _targets(config, T_floor)
