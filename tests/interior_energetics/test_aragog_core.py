"""Unit tests for the core_module core in the PROTEUS Aragog wrapper (aragog_core.py)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_write_core_diagnostics_wiring_and_cache():
    """The diagnostics writer converts the helpfile F_cmb into a CMB heat
    flow through the budget's own area (q = F * 4 pi r_cmb^2), forwards
    the configured q_radio, writes all seven core_* columns, and rebuilds
    the wrapper-side entropy budget only when the solver's budget object
    changes identity (a structure re-solve), not on every call.

    The mocked budget returns physically plausible Earth-core values
    (r_icb ~ 1.2e6 m, C_eff ~ 1.8e27 J/K) so a units slip (missing area
    factor, W vs W/m2) shifts q_cmb by 14 orders of magnitude and fails
    the call-argument assertion rather than passing on round numbers.
    """
    from proteus.config._interior import AragogCoreModule
    from proteus.interior_energetics.aragog_core import write_core_diagnostics

    config = MagicMock()
    config.interior_energetics.aragog.core_module = AragogCoreModule(
        q_radio=2.0e12, k_core=90.0
    )
    runner = SimpleNamespace(_config=config)

    r_cmb = 3.48e6
    budget = MagicMock()
    budget.profiles.r_cmb = r_cmb
    budget.r_icb.return_value = 1.22e6
    budget.effective_capacity.return_value = 1.77e27
    solver = MagicMock()
    solver._core_module_budget = budget
    runner.aragog_solver = solver

    entropy = MagicMock()
    entropy.entropy_margin.return_value = 9.6e8
    entropy.b_rms_core.return_value = 1.1e-3
    output = {'T_cmb': 4864.0, 'F_cmb': 1.5e5}
    with (
        patch('aragog.core.CoreEntropyBudget', return_value=entropy) as mock_ent_cls,
        patch('aragog.core.crystallization_regime', return_value=1) as mock_regime,
    ):
        write_core_diagnostics(runner, output)

        # Entropy budget built once from the config's diagnostics fields.
        assert mock_ent_cls.call_args.kwargs['k_core'] == pytest.approx(90.0)

        # No elapsed time (the init call): the fallback carries the
        # area factor, q = F * 4 pi r_cmb^2.
        q_expected = 1.5e5 * 4.0 * np.pi * r_cmb**2
        args = entropy.entropy_margin.call_args
        assert args.args[0] == pytest.approx(4864.0)
        assert args.args[1] == pytest.approx(q_expected, rel=1e-12)
        assert args.kwargs['q_radio'] == pytest.approx(2.0e12)
        assert args.kwargs['t_shell'] is None
        assert budget.effective_capacity.call_args.args == (pytest.approx(4864.0),)
        assert budget.effective_capacity.call_args.kwargs == {}
        _ = mock_regime  # regime asserted through the output below

        # With elapsed time the diagnostics take the CMB power from the call's heat, not from
        # F_cmb times the area; the two inputs differ here, so a regression fails the pin.
        from proteus.utils.constants import secs_per_year

        output_dt = {'T_cmb': 4864.0, 'F_cmb': 1.5e5, 'step_dE_F_cmb_J': 3.0e28}
        write_core_diagnostics(runner, output_dt, dt_actual_yr=100.0)
        q_avg = 3.0e28 / (100.0 * secs_per_year)
        assert abs(q_avg - q_expected) > 0.1 * q_expected  # paths distinguishable
        assert entropy.entropy_margin.call_args.args[1] == pytest.approx(q_avg, rel=1e-12)

        assert output['core_r_icb'] == pytest.approx(1.22e6)
        assert output['core_C_eff'] == pytest.approx(1.77e27)
        assert output['core_dynamo_margin'] == pytest.approx(9.6e8)
        assert output['core_B_rms'] == pytest.approx(1.1e-3)
        assert output['core_regime'] == pytest.approx(1.0)
        assert output['core_strat_depth'] == pytest.approx(0.0)
        assert output['core_T_top'] == pytest.approx(4864.0)

        # A stratified core: the layer base from the solve bounds the capacity, the shell
        # reaches the entropy margin, and its top cell is the top of the core.
        t_shell = np.array([4870.0, 4910.0, 4955.0])
        strat = SimpleNamespace(core_T_shell=t_shell, core_layer_base=r_cmb - 250.0e3)
        write_core_diagnostics(runner, output, out=strat)
        assert output['core_strat_depth'] == pytest.approx(250.0e3)
        assert output['core_T_top'] == pytest.approx(4955.0)
        kwargs = budget.effective_capacity.call_args.kwargs
        assert kwargs['gravitational_upper'] == pytest.approx(r_cmb - 250.0e3)
        assert entropy.entropy_margin.call_args.kwargs['t_shell'] is t_shell

        # Second call, same budget object: the entropy budget is reused.
        write_core_diagnostics(runner, output)
        assert mock_ent_cls.call_count == 1

        # New budget object (structure re-solve): rebuilt once.
        solver._core_module_budget = MagicMock()
        solver._core_module_budget.profiles.r_cmb = r_cmb
        solver._core_module_budget.r_icb.return_value = 1.22e6
        solver._core_module_budget.effective_capacity.return_value = 1.77e27
        write_core_diagnostics(runner, output)
        assert mock_ent_cls.call_count == 2

    # Guard return: a solver without the budget (mode mismatch after a
    # config edit mid-run) writes nothing rather than crashing.
    bare = MagicMock(spec=[])
    runner.aragog_solver = bare
    before = dict(output)
    write_core_diagnostics(runner, output)
    assert output == before
