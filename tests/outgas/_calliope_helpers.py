"""
Shared fixtures-as-functions for the CALLIOPE wrapper tests.

The unit tier (``test_calliope.py``) mocks CALLIOPE around this config and
helpfile row, and the smoke tier (``test_calliope_cold_start_smoke.py``) drives
the real solver with them, so the two tiers cannot drift onto different inputs.

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from proteus.utils.constants import element_list, noble_gases


def _element_mode_config(noble_included, He_mode='kg', He_budget=0.0, reservoir='mantle'):
    """Minimal element-mode config for construct_options with noble control.

    `noble_included` maps a noble gas symbol to its include flag. Only the
    attributes construct_options reads in element mode are populated.
    """
    config = MagicMock()
    config.outgas.fO2_shift_IW = 4.0
    config.outgas.calliope.solubility = True
    config.planet.volatile_mode = 'elements'
    config.planet.volatile_reservoir = reservoir
    config.planet.gas_prs.get_pressure = lambda s: 0.0

    def _is_included(s):
        if s in noble_gases:
            return noble_included.get(s, False)
        return True

    config.outgas.calliope.is_included = _is_included

    elem = config.planet.elements
    elem.use_metallicity = False
    elem.H_mode, elem.H_budget = 'kg', 1.5e20
    elem.C_mode, elem.C_budget = 'kg', 1.0e20
    elem.N_mode, elem.N_budget = 'kg', 2.0e18
    elem.S_mode, elem.S_budget = 'kg', 5.0e19
    for gas in noble_gases:
        setattr(elem, f'{gas}_mode', 'kg')
        setattr(elem, f'{gas}_budget', 0.0)
    elem.He_mode, elem.He_budget = He_mode, He_budget
    return config


_HF_ROW = {
    'M_mantle': 4.0e24,
    'M_int': 4.5e24,
    'gravity': 9.81,
    'R_int': 6.37e6,
    'Phi_global': 1.0,
    'T_magma': 1800.0,
}


def _surface_pressure_hf_row():
    """Helpfile row with a whole-planet inventory for every tracked element.

    Each element carries a distinct mass so a target built from the wrong
    element set is visible in the values, not only in the key set. The noble
    inventories are trace relative to the volatiles, which is the realistic
    regime.

    Every total is deliberately DIFFERENT from the matching config budget in
    `_element_mode_config`, so a target populated from the static config instead
    of the running whole-planet totals is visible in the values. Reading the
    config would silently discard every escape debit accumulated so far.
    """
    hf_row = dict(_HF_ROW)
    hf_row['Time'] = 0.0
    for e in element_list:
        hf_row[f'{e}_kg_total'] = 1.0e16 if e in noble_gases else 1.0e20
    # H budget in the config is 1.5e20; He budget is 3.0e16.
    hf_row['H_kg_total'] = 1.2e20
    hf_row['He_kg_total'] = 2.0e16
    return hf_row


@pytest.fixture(autouse=True)
def _restore_global_rng():
    """Give the global NumPy RNG back to later tests in the state it had."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _cold_start_config(noble_included=None, He_budget=0.0):
    """Element-mode config on the user_constant fO2 path, no noble gas active by default."""
    config = _element_mode_config(noble_included or {}, He_budget=He_budget)
    config.outgas.T_floor = 1200.0
    config.outgas.mass_thresh = 1.0e16
    config.outgas.solver_atol = 1e-6
    config.outgas.solver_rtol = 1e-4
    config.outgas.calliope.nguess = 100
    config.outgas.calliope.nsolve = 500
    config.outgas.calliope.p_guess_max = 1.0e5
    config.planet.fO2_source = 'user_constant'
    return config
