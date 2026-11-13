"""Unit tests for ``proteus.outgas.fractionated_debit``, the per-element debit of a frozen atmosphere.

Invariants checked: each element of the atmosphere, gases and graphite
together, holds its mass less the requested loss; no gas mass is negative;
a loss larger than an element holds removes all of it and no more. Reference:
the analytic speciation of a water atmosphere that loses only hydrogen.
"""

from __future__ import annotations

import numpy as np
import pytest

from proteus.outgas.common import element_masses_from_species
from proteus.outgas.fractionated_debit import STOICH, plan_debit
from proteus.utils.constants import element_mmw
from proteus.utils.helper import eval_gas_mmw

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _elements(species_kg: dict) -> dict:
    """Element masses of a set of gases [kg]."""
    return element_masses_from_species({g: m for g, m in species_kg.items() if m > 0.0})


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_water_losing_hydrogen_leaves_its_oxygen_as_o2(validation_plot):
    """Hydrogen escaping from water leaves water and O2, in analytic proportions.

    Physical scenario: 1e20 kg of water loses a share f of its hydrogen and
    no oxygen. Each removed H2O gives up two H and leaves one O, so the water
    keeps (1 - f) of its moles and the freed oxygen forms f n / 2 moles of O2,
    with n the moles of water. Discrimination: scaling the whole atmosphere
    by one factor, as an unfractionated debit does, would remove oxygen too
    and form no O2.
    """
    n_h2o = 1.0e20 / eval_gas_mmw('H2O')
    m_h = 2.0 * n_h2o * element_mmw['H']
    shares = np.array([0.05, 0.25, 0.5, 0.9])
    water, o2, ref_water, ref_o2 = [], [], [], []
    for f in shares:
        new, applied, graphite = plan_debit({'H2O': 1.0e20}, {'H': f * m_h})
        ref_w = (1.0 - f) * n_h2o * eval_gas_mmw('H2O')
        ref_o = 0.5 * f * n_h2o * eval_gas_mmw('O2')
        assert new['H2O'] == pytest.approx(ref_w, rel=1e-12)
        assert new['O2'] == pytest.approx(ref_o, rel=1e-12)
        assert applied['H'] == pytest.approx(f * m_h, rel=1e-12)
        assert applied['O'] == pytest.approx(0.0, abs=1e-6 * m_h)
        assert graphite == pytest.approx(0.0, abs=1e-30)
        # The uniform debit of the same mass keeps no O2 and removes oxygen.
        assert new['O2'] > 0.01 * 1.0e20 * f
        water.append(new['H2O'])
        o2.append(new['O2'])
        ref_water.append(ref_w)
        ref_o2.append(ref_o)

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(shares, np.array(ref_water) / 1e20, '-', label='H$_2$O, analytic')
        ax.plot(shares, np.array(ref_o2) / 1e20, '--', label='O$_2$, analytic')
        ax.plot(shares, 1.0 - shares * 2 * element_mmw['H'] / eval_gas_mmw('H2O'), ':',
                label='Uniform debit of the same mass')  # fmt: skip
        ax.plot(shares, np.array(water) / 1e20, 'o', label='H$_2$O, debit')
        ax.plot(shares, np.array(o2) / 1e20, 's', mfc='none', label='O$_2$, debit')
        ax.set_xlabel('Share of the hydrogen removed [1]')
        ax.set_ylabel('Gas mass over the initial water mass [1]')
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    validation_plot(figure, 'outgas_fractionated_debit_water')


@pytest.mark.physics_invariant
def test_every_element_balances_for_random_atmospheres_and_losses():
    """Gases plus graphite hold each element's mass less its loss, for any mixture.

    Physical scenario: 200 random mixtures of the volatile and noble gases,
    each losing a random share of every element, from none to all of it.
    """
    rng = np.random.default_rng(20261113)
    gases = sorted(STOICH)
    for _ in range(200):
        species = {g: float(rng.uniform(0.0, 1.0e19)) * (rng.random() < 0.7) for g in gases}
        held = _elements(species)
        losses = {e: float(rng.uniform(0.0, 1.0)) * m for e, m in held.items()}
        new, applied, graphite = plan_debit(species, losses)
        after = _elements(new)
        assert all(m >= 0.0 for m in new.values())
        for e, m in held.items():
            assert applied[e] == pytest.approx(losses[e], rel=1e-9, abs=1e-6)
            kept = after.get(e, 0.0) + (graphite if e == 'C' else 0.0)
            assert kept == pytest.approx(m - losses[e], rel=1e-9, abs=1e3)
        assert graphite >= 0.0


@pytest.mark.physics_invariant
def test_methane_losing_hydrogen_condenses_its_carbon_as_graphite():
    """Carbon whose hydrogen escaped, with no oxygen to bind it, condenses as graphite.

    Physical scenario: a pure 1e18 kg methane atmosphere loses 60 % of its
    hydrogen and no carbon. The methane keeps 40 % of its moles; the carbon of
    the other 60 % has no partner gas left and condenses. Keeping it in the gas
    instead would need hydrogen the step removed.
    """
    n = 1.0e18 / eval_gas_mmw('CH4')
    m_h = 4.0 * n * element_mmw['H']
    new, applied, graphite = plan_debit({'CH4': 1.0e18}, {'H': 0.6 * m_h})
    assert new['CH4'] == pytest.approx(0.4 * 1.0e18, rel=1e-12)
    assert graphite == pytest.approx(0.6 * n * element_mmw['C'], rel=1e-12)
    assert applied['H'] == pytest.approx(0.6 * m_h, rel=1e-12)
    assert new['H2'] == pytest.approx(0.0, abs=1e-6)

    # With oxygen to spare, the carbon forms CO and none condenses.
    mixed, _, graphite_o = plan_debit({'CH4': 1.0e18, 'O2': 1.0e18}, {'H': 0.6 * m_h})
    assert graphite_o == pytest.approx(0.0, abs=1e-6)
    assert mixed['CO'] == pytest.approx(0.6 * n * eval_gas_mmw('CO'), rel=1e-9)


@pytest.mark.physics_invariant
def test_a_loss_beyond_the_holding_empties_the_element_only():
    """Asking for more of an element than the atmosphere holds removes all of it.

    Physical scenario: an N2 and CO2 atmosphere asked to lose twice its
    nitrogen. The nitrogen goes, the CO2 stays whole, and the applied loss is
    the nitrogen held, not the amount asked.
    """
    new, applied, _ = plan_debit({'N2': 5.0e17, 'CO2': 1.0e19}, {'N': 1.0e18, 'Xx': 3.0})
    assert new['N2'] == pytest.approx(0.0, abs=1e-3)
    assert new['CO2'] == pytest.approx(1.0e19, rel=1e-12)
    assert applied['N'] == pytest.approx(5.0e17, rel=1e-12)
    assert 'Xx' not in applied

    unchanged, none, _ = plan_debit({'Ar': 1.0e16, 'H2O': 1.0e18}, {'H': -5.0})
    assert unchanged['H2O'] == pytest.approx(1.0e18, rel=1e-12)
    assert unchanged['Ar'] == pytest.approx(1.0e16, rel=1e-12)
    assert none['H'] == pytest.approx(0.0, abs=1e-6)
