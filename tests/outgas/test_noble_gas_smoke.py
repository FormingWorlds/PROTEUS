"""End-to-end smoke test for noble gas outgassing through CALLIOPE.

Exercises the real PROTEUS-to-CALLIOPE contract that the mocked unit tests
in ``test_calliope.py`` cannot: a noble gas budget expressed the way
``construct_options`` builds it (an inclusion flag plus a mantle-relative
ppmw value) is turned into a target inventory by CALLIOPE's real target
builder and closes mass through CALLIOPE's real equilibrium solver.

This test needs a CALLIOPE that supports noble gases. On the pinned release
that predates that support it skips, so the fast PR gate stays green; it
activates automatically once the CALLIOPE version pin is bumped.

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('calliope')

import calliope.constants as _cal_constants
from calliope.solve import equilibrium_atmosphere, get_target_from_params

from proteus.utils.constants import noble_gases

# Skip the whole module on a CALLIOPE that predates noble gas support, so the
# PR gate (which installs the released CALLIOPE) stays green. Detect the
# capability with getattr rather than a direct import, because importing a name
# that does not exist yet on the released CALLIOPE crashes collection instead
# of skipping the module.
if 'He' not in getattr(_cal_constants, 'noble_gases', ()):
    pytest.skip('installed CALLIOPE has no noble gas support', allow_module_level=True)

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60)]


def _base_ddict():
    """CALLIOPE options dict as PROTEUS's construct_options assembles it."""
    d = {
        'M_mantle': 4.0e24,
        'Phi_global': 1.0,
        'T_magma': 1800.0,
        'gravity': 9.81,
        'radius': 6.37e6,
        'fO2_shift_IW': 0.0,
        'hydrogen_earth_oceans': 1.0,
        'CH_ratio': 0.1,
        'nitrogen_ppmw': 2.0,
        'sulfur_ppmw': 50.0,
    }
    for sp in ('H2O', 'CO2', 'O2', 'H2', 'CH4', 'CO', 'N2', 'S2', 'SO2', 'H2S', 'NH3'):
        d[f'{sp}_included'] = 1 if sp in ('H2O', 'CO2', 'N2', 'S2') else 0
    for gas in noble_gases:
        d[f'{gas}_included'] = 0
        d[f'{gas}_ppmw'] = 0.0
    return d


@pytest.mark.physics_invariant
def test_ppmw_budget_becomes_target_and_closes_mass_through_solver():
    """A He ppmw budget passed the way construct_options builds it becomes a
    kg target through CALLIOPE's target builder, and that target closes mass
    through the real equilibrium solver.
    """
    ddict = _base_ddict()
    ddict['He_included'] = 1
    ddict['He_ppmw'] = 5.0  # ppmw relative to the mantle mass

    target = get_target_from_params(ddict)
    # The target builder converts ppmw to kg relative to the mantle mass.
    expected_he_kg = 5.0 * 1e-6 * ddict['M_mantle']
    assert target['He'] == pytest.approx(expected_he_kg, rel=1e-9)
    # Discrimination: a dropped 1e-6 factor would make the He target 1e6x
    # larger than the whole mantle, which is unphysical.
    assert target['He'] < ddict['M_mantle']

    np.random.seed(0)
    out = equilibrium_atmosphere(
        target, ddict, print_result=False, opt_solver=False, nguess=3000
    )
    reservoir = out['He_kg_atm'] + out['He_kg_liquid']
    assert reservoir == pytest.approx(expected_he_kg, rel=1e-5)
    # The He atmospheric and dissolved reservoir keys PROTEUS reads are present
    # and non-negative.
    assert out['He_kg_atm'] >= 0.0
    assert out['He_kg_liquid'] >= 0.0
    assert out['He_bar'] > 0.0


def test_disabled_noble_gas_produces_no_target():
    """A noble gas left off (the default) gets no target from the builder, so
    a run with no noble budget carries nothing extra into the solve.
    """
    ddict = _base_ddict()  # all noble gases off
    target = get_target_from_params(ddict)
    for gas in noble_gases:
        assert gas not in target
    # The C-H-O-N-S targets are still built.
    for elem in ('H', 'C', 'N', 'S'):
        assert target[elem] > 0.0
