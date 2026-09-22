"""
Shared fixtures-as-functions for the from_O_budget wrapper tests.

These build the config and helpfile row that both tiers need: the unit tier
mocks CALLIOPE around them (``test_from_o_budget_wrapper.py``) and the smoke
tier drives the real solver with them
(``test_from_o_budget_wrapper_smoke.py``). Keeping them in one place means the
two tiers cannot drift onto different inputs.

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

from proteus.utils.constants import element_list, vol_list


def _make_from_o_budget_config(fO2_shift_IW: float = 4.0):
    """Build a MagicMock Config consistent with from_O_budget dispatch.

    Only the fields the wrapper reaches into are populated; everything
    else inherits the MagicMock default (auto-attribute) and is irrelevant
    to the wrapper's control flow because solvevol_inp/target construction
    is the only consumer.
    """
    config = MagicMock()
    config.outgas.module = 'calliope'
    config.outgas.fO2_shift_IW = fO2_shift_IW
    config.outgas.T_floor = 1200.0
    config.outgas.mass_thresh = 1e8
    config.outgas.solver_atol = 1e-8
    config.outgas.solver_rtol = 1e-5
    config.outgas.calliope.nguess = 100
    config.outgas.calliope.nsolve = 500
    config.outgas.calliope.p_guess_max = 5.0e6  # distinctive (non-default) for forwarding test
    config.outgas.calliope.solubility = True
    for s in vol_list:
        setattr(config.outgas.calliope, f'include_{s}', True)
    # Noble gases are opt-in and off in this fixture, so is_included is true
    # only for the reaction-network volatiles it configures.
    for gas in ('He', 'Ne', 'Ar', 'Kr', 'Xe'):
        setattr(config.outgas.calliope, f'include_{gas}', False)
    config.outgas.calliope.is_included = lambda s: s in vol_list
    config.planet.fO2_source = 'from_O_budget'
    config.planet.volatile_mode = 'elements'
    config.planet.volatile_reservoir = 'mantle'
    config.planet.elements.use_metallicity = False
    config.planet.elements.H_mode = 'kg'
    config.planet.elements.H_budget = 1.5e20
    config.planet.elements.C_mode = 'kg'
    config.planet.elements.C_budget = 1.5e19
    config.planet.elements.N_mode = 'kg'
    config.planet.elements.N_budget = 8.0e18
    config.planet.elements.S_mode = 'kg'
    config.planet.elements.S_budget = 8.0e20
    config.planet.elements.O_mode = 'kg'
    config.planet.elements.O_budget = 1.0e22
    return config


def _earth_hf_row(O_kg_total: float = 1.0e22) -> dict:
    """Helpfile row populated up to the point ``calc_surface_pressures``
    is called: structural quantities and per-element targets are set, but
    the gas-phase quantities are not."""
    hf: dict = {
        'M_mantle': 4.03e24,
        'M_int': 5.97e24,
        'gravity': 9.81,
        'R_int': 6.371e6,
        'Phi_global': 1.0,
        'T_magma': 1800.0,
        'Time': 0.0,
    }
    # element_list = [H, O, C, N, S, Si, Mg, Fe, Na]; CALLIOPE only
    # consumes H/C/N/S(/O). Pre-populate every element so the target
    # construction (loop over element_list) can read the slot uniformly.
    for e in element_list:
        hf[f'{e}_kg_total'] = 0.0
    hf['H_kg_total'] = 1.5e20
    hf['C_kg_total'] = 1.5e19
    hf['N_kg_total'] = 8.0e18
    hf['S_kg_total'] = 8.0e20
    hf['O_kg_total'] = O_kg_total
    return hf
