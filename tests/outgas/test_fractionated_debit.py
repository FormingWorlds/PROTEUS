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
    # Leftover oxygen booked as one atom per O2 molecule doubles it: off balance.
    monkeypatch.setitem(fd.OWN_GAS, 'O', ('O2', 1))
    with pytest.raises(RuntimeError, match='debit fails for O'):
        fd.plan_debit({'H2O': 1.0e18}, {'H': 1.0e16})


def test_the_row_key_literals_name_the_published_key():
    """The row key is written as a literal for the output reference; each copy matches."""
    import inspect
    import re

    import proteus.outgas.fractionated_debit as fd

    found = re.findall(r"'(_esc_frozen[^']*)'", inspect.getsource(fd))
    assert len(found) >= 3
    assert set(found) == {FROZEN_DEBIT_KEY}


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


@pytest.mark.physics_invariant
def test_hydrogen_held_in_rock_vapour_stays_counted_in_the_atmosphere():
    """Each volatile element's atmospheric mass drops by its loss, so vapour atoms stay.

    Physical scenario: a water atmosphere of 1e19 kg with 1e17 kg of NaOH
    vapour over a frozen mantle. LavAtmos counts the hydrogen of the NaOH in
    the atmospheric H column and keeps the O column to the volatiles. Escape
    takes a fifth of the hydrogen the water holds. Afterwards the H column is
    the hydrogen of the gases the debit handles plus that of the NaOH, which
    kept its mass, and the O column is that of the volatile gases; recounting
    the volatile gases alone would drop the NaOH hydrogen.
    """
    row = _frozen_row({'H2O': 1.0e19, 'NaOH': 1.0e17})
    naoh = {'H': 1.0e17 * element_mmw['H'] / eval_gas_mmw('NaOH'), 'O': 0.0}
    row['H_kg_atm'] += naoh['H']
    h_water = row['H_kg_total']
    target = {e: row[f'{e}_kg_total'] for e in element_list}
    target['H'] -= 0.2 * h_water
    out = plan_frozen_debit(row, target)
    for e, m in out.items():
        row[f'{e}_kg_total'] = m
    apply_frozen_debit(row)
    gases = _elements({g: row[f'{g}_kg_atm'] for g in gas_list if g in STOICH})
    assert row['NaOH_kg_atm'] == pytest.approx(1.0e17, rel=1e-15)
    assert gases['H'] == pytest.approx(0.8 * h_water, rel=1e-9)
    for e in ('H', 'O'):
        assert row[f'{e}_kg_atm'] == pytest.approx(gases[e] + naoh[e], rel=1e-9)


@pytest.mark.physics_invariant
def test_noble_gases_stripped_completely_leave_no_negative_total():
    """A noble gas held only in the atmosphere and stripped ends at zero, not below.

    Physical scenario: frozen atmospheres of N2 with each noble gas at masses
    spread over six decades, every noble gas lost in full. The planned whole-
    planet totals are zero or more for every noble gas and every draw.
    """
    rng = np.random.default_rng(7)
    nobles = ['He', 'Ne', 'Ar', 'Kr', 'Xe']
    for _ in range(200):
        gases = {'N2': 1.0e19, **{g: 10.0 ** rng.uniform(12.0, 18.0) for g in nobles}}
        row = _frozen_row(gases)
        for g in nobles:
            row[f'{g}_kg_total'] = gases[g]
        target = {e: row[f'{e}_kg_total'] for e in element_list}
        target.update({g: 0.0 for g in nobles})
        out = plan_frozen_debit(row, target)
        assert min(out[g] for g in nobles) >= 0.0
        assert max(out[g] for g in nobles) <= 1e-12 * max(gases.values())


def _frozen_pxuv_config():
    from types import SimpleNamespace

    return SimpleNamespace(
        escape=SimpleNamespace(
            module='zephyrus', uses_dispatcher=True, reservoir='pxuv', step_max_frac=0.25
        ),
        outgas=SimpleNamespace(mass_thresh=1e10, prevent_desaturation=False),
    )


@pytest.mark.physics_invariant
def test_rock_vapour_is_out_of_the_frozen_step_cap_and_shares():
    """On a frozen mantle the step cap and the escaping shares count only what can leave.

    Physical scenario: 1e16 kg of water over a frozen mantle beside 1e19 kg of
    SiO and 1e18 kg each of Mg and Fe vapour, with dispatched rates of 900 kg/s
    of silicon, 300 of magnesium, 100 of iron, 600 of oxygen, and 1 of
    hydrogen, plus a negative nitrogen rate that must count as none. Rock
    vapour is not eroded, so over 1e3 yr the water loses 1 and 600 kg/s worth
    of H and O, and the requested 601 kg/s is measured against the 1e16 kg the
    water holds. Over 2e5 yr the request, 0.38 of the water, is capped at a
    quarter of it, shared 1:600.
    """
    from unittest.mock import patch

    from proteus.escape.wrapper import run_escape
    from proteus.utils.constants import secs_per_year

    def fake(config, hf_row, atmos_o=None, dt=0.0):
        hf_row.update(esc_rate_Si=900.0, esc_rate_Mg=300.0, esc_rate_Fe=100.0)
        hf_row.update(esc_rate_O=600.0, esc_rate_H=1.0, esc_rate_N=-50.0)
        hf_row['esc_rate_total'] = 1901.0
        return 1901.0

    for dt, capped in ((1.0e3, False), (2.0e5, True)):
        row = _frozen_row({'H2O': 1.0e16, 'SiO': 1.0e19})
        row.update({f'esc_rate_{e}': 0.0 for e in element_list})
        si = 1.0e19 * element_mmw['Si'] / eval_gas_mmw('SiO')
        row['Si_kg_atm'] = row['Si_kg_total'] = si
        for e in ('Mg', 'Fe'):
            row[f'{e}_kg_atm'] = row[f'{e}_kg_total'] = 1.0e18
        before = {e: row[f'{e}_kg_total'] for e in ('H', 'O')}
        with patch('proteus.escape.dispatcher.run_dispatcher', side_effect=fake):
            run_escape(_frozen_pxuv_config(), row, dt=dt, atmosphere_only=True)
        requested = 601.0 * secs_per_year * dt
        assert row['esc_clamp_frac'] == pytest.approx(requested / 1.0e16, rel=1e-9)
        lost = 0.25 * 1.0e16 if capped else requested
        assert before['H'] - row['H_kg_total'] == pytest.approx(lost / 601.0, rel=1e-9)
        assert before['O'] - row['O_kg_total'] == pytest.approx(lost * 600.0 / 601.0, rel=1e-9)
        assert row['Si_kg_total'] == pytest.approx(si, rel=1e-15)
        assert (row['Mg_kg_total'], row['Fe_kg_total']) == pytest.approx(
            (1.0e18, 1.0e18), rel=1e-15
        )
        assert row['esc_step_kg'] == pytest.approx(lost, rel=1e-9)


def test_a_molten_step_drops_a_frozen_plan_and_debits_rock_vapour():
    """A molten pxuv step clears a stale frozen plan and takes rock vapour's share.

    Physical scenario: 1e14 kg of water with 1e14 kg of silicon vapour over a
    molten mantle, escaping 1 kg/s of hydrogen and 2 kg/s of silicon for one
    year, with a plan left on the row by an earlier frozen step. The plan is
    dropped, and with the mantle molten the silicon is debited too: the totals
    fall by one and two years' worth of the rates.
    """
    from unittest.mock import patch

    from proteus.escape.wrapper import run_escape
    from proteus.utils.constants import secs_per_year

    def fake(config, hf_row, atmos_o=None, dt=0.0):
        hf_row.update(esc_rate_H=1.0, esc_rate_Si=2.0, esc_rate_total=3.0)
        return 3.0

    row = _frozen_row({'H2O': 1.0e14})
    row.update({f'esc_rate_{e}': 0.0 for e in element_list})
    row['Si_kg_atm'] = row['Si_kg_total'] = 1.0e14
    h_before = row['H_kg_total']
    row[FROZEN_DEBIT_KEY] = {'Time': row['Time'], 'kg': {'H2O': 0.0}, 'graphite_kg': 0.0}
    with patch('proteus.escape.dispatcher.run_dispatcher', side_effect=fake):
        run_escape(_frozen_pxuv_config(), row, dt=1.0, atmosphere_only=False)
    assert FROZEN_DEBIT_KEY not in row
    assert h_before - row['H_kg_total'] == pytest.approx(secs_per_year, rel=1e-9)
    assert 1.0e14 - row['Si_kg_total'] == pytest.approx(2.0 * secs_per_year, rel=1e-9)


def test_a_declined_debit_returns_the_elements_it_was_asked_about():
    """With an element subset, a declined debit returns the totals of that subset alone.

    Physical scenario: a frozen atmosphere of 1e9 kg of water, under the 1e10 kg
    threshold, and the same atmosphere with an unreadable hydrogen column. No
    loss is taken either way, and rock elements outside the subset do not
    appear, so their totals are not rewritten.
    """
    from proteus.escape.wrapper import calc_new_elements

    subset = ['H', 'O', 'C', 'N', 'S']
    small = _frozen_row({'H2O': 1.0e9})
    out = calc_new_elements(small, 1.0, 'pxuv', min_thresh=1e10, esc_mass=1.0, elements=subset)
    assert sorted(out) == sorted(subset)
    assert out['H'] == pytest.approx(small['H_kg_total'], rel=1e-15)
    broken = dict(_frozen_row({'H2O': 1.0e19}), H_kg_atm=float('nan'))
    out = calc_new_elements(broken, 1.0, 'pxuv', min_thresh=1e10, esc_mass=1.0, elements=subset)
    assert sorted(out) == sorted(subset)


def test_an_element_column_never_goes_below_zero():
    """An applied loss larger than an element's atmospheric column leaves it at zero.

    Physical scenario: a debit that removes 1e18 kg of hydrogen from a column
    holding 1.1e18 kg, after an upstream step left that column at half its
    value. The column reads zero, not -4.5e17 kg.
    """
    row = _frozen_row({'H2O': 1.0e19})
    row['H_kg_atm'] *= 0.5
    row[FROZEN_DEBIT_KEY] = {
        'Time': row['Time'],
        'kg': {'H2O': 0.0, 'O2': 1.0e19 * 15.999 / 18.015},
        'applied_kg': {'H': 1.0e18},
        'graphite_kg': 0.0,
    }
    apply_frozen_debit(row)
    assert row['H_kg_atm'] == pytest.approx(0.0, abs=0.0)
    assert row['H2O_kg_atm'] == pytest.approx(0.0, abs=0.0)
