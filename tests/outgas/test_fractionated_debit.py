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
from proteus.outgas.fractionated_debit import (
    FROZEN_DEBIT_KEY,
    STOICH,
    apply_frozen_debit,
    plan_debit,
    plan_frozen_debit,
)
from proteus.utils.constants import element_list, element_mmw, gas_list
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
        # A uniform debit of the same mass would also take oxygen; here the
        # atmosphere keeps all of it.
        assert _elements(new)['O'] == pytest.approx(_elements({'H2O': 1.0e20})['O'], rel=1e-12)
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

    m_h = 1.0e18 * 2 * element_mmw['H'] / eval_gas_mmw('H2O')
    unchanged, none, _ = plan_debit({'Ar': 1.0e16, 'H2O': 1.0e18}, {'H': -0.3 * m_h})
    assert unchanged['H2O'] == pytest.approx(1.0e18, rel=1e-12)
    assert unchanged['Ar'] == pytest.approx(1.0e16, rel=1e-12)
    assert none['H'] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.physics_invariant
def test_leftover_carbon_takes_oxygen_before_hydrogen():
    """Unpaired carbon forms CO with the oxygen freed this step before CH4 with free H2.

    Physical scenario: 1e20 mol each of CH4, H2O and H2 lose a quarter of
    their hydrogen. Every gas keeps three quarters of its moles, which frees
    2.5e19 mol each of carbon and oxygen. They pair into 2.5e19 mol of CO, so
    no graphite forms and the H2 keeps its three quarters. Taking hydrogen
    first would have built CH4 out of the free H2 and left the oxygen as O2.
    """
    n = 1.0e20
    species = {g: n * eval_gas_mmw(g) for g in ('CH4', 'H2O', 'H2')}
    m_h = 8.0 * n * element_mmw['H']
    new, applied, graphite = plan_debit(species, {'H': 0.25 * m_h})
    assert new['CO'] == pytest.approx(0.25 * n * eval_gas_mmw('CO'), rel=1e-12)
    assert new['CH4'] == pytest.approx(0.75 * n * eval_gas_mmw('CH4'), rel=1e-12)
    assert new['H2'] == pytest.approx(0.75 * n * eval_gas_mmw('H2'), rel=1e-12)
    assert new['O2'] == pytest.approx(0.0, abs=1e-6 * n)
    assert graphite == pytest.approx(0.0, abs=1e-6)
    assert applied['H'] == pytest.approx(0.25 * m_h, rel=1e-12)


def test_a_balance_that_fails_to_close_raises(monkeypatch):
    """The element balance is checked, and a failure stops the step."""
    import proteus.outgas.fractionated_debit as fd

    new, _applied, _g = fd.plan_debit({'H2O': 1.0e18}, {'H': 1.0e16})
    assert new['O2'] > 0.0  # the same debit closes at the real tolerance
    monkeypatch.setattr(fd, 'CLOSURE_RTOL', -1.0)
    with pytest.raises(RuntimeError, match='element balance'):
        fd.plan_debit({'H2O': 1.0e18}, {'H': 1.0e16})


def _frozen_row(species_kg, time=5.0e8, p_vol=10.0, p_vap=0.0):
    """Helpfile row of a frozen atmosphere whose totals equal what the gases hold."""
    row = {f'{g}_kg_atm': 0.0 for g in gas_list}
    row.update({f'{e}_kg_total': 0.0 for e in element_list})
    row.update({f'{g}_kg_atm': m for g, m in species_kg.items()})
    vol = {g: m for g, m in species_kg.items() if g in STOICH}
    for e, m in _elements(vol).items():
        row[f'{e}_kg_atm'] = m
        row[f'{e}_kg_total'] = m
    m_vol = sum(vol.values())
    row.update(
        Time=time,
        M_vol_atm=m_vol,
        M_atm=sum(species_kg.values()),
        M_planet=6.0e24,
        P_vol=p_vol,
        P_vap=p_vap,
        P_surf=p_vol + p_vap,
    )
    return row


@pytest.mark.physics_invariant
def test_an_element_escape_strips_completely_leaves_no_trace_in_the_atmosphere():
    """A frozen atmosphere that loses all its sulfur records zero sulfur, and closes.

    Physical scenario: an N2 atmosphere with a little S2 over a frozen mantle
    loses all of its sulfur. The S2 is gone, the atmospheric sulfur column
    reads zero, and the element closure that PROTEUS checks with trapping on
    holds for every element.
    """
    from proteus.utils.coupler import assert_mass_conservation

    row = _frozen_row({'N2': 1.0e19, 'S2': 1.0e16})
    target = {e: row[f'{e}_kg_total'] for e in element_list}
    target['S'] = 0.0
    out = plan_frozen_debit(row, target)
    for e, m in out.items():
        row[f'{e}_kg_total'] = m
    apply_frozen_debit(row)
    assert row['S2_kg_atm'] == pytest.approx(0.0, abs=1e-6)
    assert row['S_kg_atm'] == pytest.approx(0.0, abs=1e-6)
    assert row['S_kg_total'] == pytest.approx(0.0, abs=1e-6)
    assert row['N_kg_atm'] == pytest.approx(1.0e19, rel=1e-12)
    assert_mass_conservation(row, check_element_closure=True, closure_rtol=1e-9)


@pytest.mark.physics_invariant
def test_vapour_pressure_and_rock_elements_are_left_alone():
    """Rock vapour keeps its mass and pressure, and rock elements keep their totals.

    Physical scenario: a water and SiO atmosphere over a frozen mantle, with a
    volatile pressure of 10 bar and a vapour pressure of 2 bar, loses a fifth
    of its hydrogen; the escaping share of silicon is not taken. The volatile
    pressure scales with the volatile column mass, the surface pressure adds
    the unchanged vapour pressure, and each partial pressure is its mole
    fraction of that surface pressure.
    """
    row = _frozen_row({'H2O': 1.0e19, 'SiO': 1.0e17}, p_vol=10.0, p_vap=2.0)
    row['Si_kg_total'] = si_total = 3.0e21
    target = {e: row[f'{e}_kg_total'] for e in element_list}
    target['H'] *= 0.8
    target['Si'] = 0.5 * si_total
    out = plan_frozen_debit(row, target)
    assert out['Si'] == pytest.approx(si_total, rel=1e-15)
    for e, m in out.items():
        row[f'{e}_kg_total'] = m
    m_vol_old = row['M_vol_atm']
    apply_frozen_debit(row)
    m_vol = sum(row[f'{g}_kg_atm'] for g in gas_list if g in STOICH)
    assert row['P_vol'] == pytest.approx(10.0 * m_vol / m_vol_old, rel=1e-12)
    assert row['P_surf'] == pytest.approx(row['P_vol'] + 2.0, rel=1e-12)
    assert row['SiO_kg_atm'] == pytest.approx(1.0e17, rel=1e-15)
    n = {g: row[f'{g}_kg_atm'] / eval_gas_mmw(g) for g in gas_list}
    n_tot = sum(n.values())
    assert row['H2O_mol_atm'] == pytest.approx(n['H2O'], rel=1e-12)
    assert row['H2O_bar'] == pytest.approx(n['H2O'] / n_tot * row['P_surf'], rel=1e-12)
    assert row['atm_kg_per_mol'] == pytest.approx(row['M_atm'] / n_tot, rel=1e-12)
    assert row['M_vol_atm'] == pytest.approx(m_vol, rel=1e-12)


@pytest.mark.physics_invariant
def test_graphite_reaches_the_row_through_the_escape_step():
    """Hydrogen escaping from a frozen methane atmosphere leaves graphite on the row.

    Physical scenario: 1e18 kg of CH4 over a frozen mantle, with escape
    carrying hydrogen only at 1e3 kg/s for 1e3 yr. The carbon of the methane
    that lost its hydrogen condenses: the atmosphere's carbon falls by the
    graphite mass, the planet's carbon total does not, and the graphite is
    recorded in C_kg_condensed and esc_graphite_kg alike.
    """
    from types import SimpleNamespace
    from unittest.mock import patch

    from proteus.escape.wrapper import run_escape
    from proteus.utils.constants import secs_per_year

    row = _frozen_row({'CH4': 1.0e18})
    row.update({f'esc_rate_{e}': 0.0 for e in element_list})
    c_total, c_atm = row['C_kg_total'], row['C_kg_atm']

    def fake(config, hf_row, atmos_o=None, dt=0.0):
        hf_row['esc_rate_total'] = hf_row['esc_rate_H'] = 1.0e3
        return 1.0e3

    config = SimpleNamespace(
        escape=SimpleNamespace(
            module='zephyrus', uses_dispatcher=True, reservoir='pxuv', step_max_frac=0.25
        ),
        outgas=SimpleNamespace(mass_thresh=1e10, prevent_desaturation=False),
    )
    with patch('proteus.escape.dispatcher.run_dispatcher', side_effect=fake):
        run_escape(config, row, dt=1.0e3, atmosphere_only=True)
    assert FROZEN_DEBIT_KEY in row
    apply_frozen_debit(row)
    lost_h = 1.0e3 * secs_per_year * 1.0e3
    graphite = row['esc_graphite_kg']
    assert graphite == pytest.approx(
        lost_h / (4 * element_mmw['H']) * element_mmw['C'], rel=1e-9
    )
    assert row['C_kg_condensed'] == pytest.approx(graphite, rel=1e-12)
    assert row['C_kg_total'] == pytest.approx(c_total, rel=1e-12)
    assert row['C_kg_atm'] == pytest.approx(c_atm - graphite, rel=1e-9)


def test_a_debit_planned_on_another_step_is_discarded_with_a_warning(caplog):
    """A plan left from an earlier step is not applied to this one."""
    import logging

    row = _frozen_row({'N2': 1.0e19}, time=2.0)
    row[FROZEN_DEBIT_KEY] = {'Time': 1.0, 'kg': {'N2': 5.0e18}, 'graphite_kg': 0.0}
    with caplog.at_level(logging.WARNING):
        apply_frozen_debit(row)
    assert 'Discarded' in caplog.text
    assert row['N2_kg_atm'] == pytest.approx(1.0e19, rel=1e-15)
    assert FROZEN_DEBIT_KEY not in row
