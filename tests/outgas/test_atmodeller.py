"""Tests for the atmodeller outgassing wrapper's per-element reservoir accounting.

Module under test: ``proteus.outgas.atmodeller``.

The atmodeller solver reports per-species masses but not the per-element
reservoir totals that CALLIOPE writes natively. ``_populate_volatile_element_
reservoirs`` reconstructs the per-element ``{e}_kg_atm`` / ``_kg_liquid`` /
``_kg_solid`` / ``_kg_total`` fields from the per-species inventory by
stoichiometry. These tests pin that reconstruction and guard the coupling
invariant that makes it necessary: with the per-element atmospheric reservoir
populated, the escape step (``calc_new_elements`` under the default
``reservoir = "outgas"``) keeps a live element budget instead of collapsing it
to zero (which would otherwise make atmodeller skip and freeze the atmosphere).

Invariants exercised:

* Mass conservation: the species mass split among constituent elements sums
  back to the total species mass (no mass created or lost).
* Stoichiometric fidelity: each element's mass fraction in a species equals the
  closed-form ``n * m_element / m_species`` (pinned for H/O in water).
* Positivity / boundedness: an empty inventory yields exactly zero per element.
* Coupling regression: a populated element budget survives the outgas-reservoir
  escape debit (does not collapse to zero).

See ``docs/How-to/test_infrastructure.md``, ``docs/How-to/test_building.md``,
``docs/How-to/test_categorization.md``.
"""

from __future__ import annotations

import pytest

from proteus.escape.wrapper import calc_new_elements
from proteus.outgas.atmodeller import (
    _VOLATILE_ELEMENT_STOICH,
    _populate_volatile_element_reservoirs,
)
from proteus.utils.constants import element_mmw

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.physics_invariant
def test_water_split_matches_closed_form_mass_fractions():
    """A pure-water inventory splits into H and O by the closed-form fractions.

    For H2O the atomic-O mass fraction is m_O / (2 m_H + m_O) = 0.88809 and the
    H fraction is the complement, 0.11191, fixed numbers independent of the
    implementation's atomic-mass table. The discrimination guard rules out a
    naive equal (0.5/0.5) split, which would put H and O each at half the water
    mass and miss the pinned fractions by ~0.39 (far above tolerance).
    """
    m_h2o = 4.0e21  # kg, both reservoirs populated to exercise atm+liquid
    hf_row = {
        'H2O_kg_atm': 2.5e21,
        'H2O_kg_liquid': 1.5e21,
        'H2O_kg_solid': 0.0,
    }
    _populate_volatile_element_reservoirs(hf_row, element_mmw)

    # Pinned closed-form fractions (hand-computed from IUPAC atomic masses).
    assert hf_row['O_kg_total'] == pytest.approx(m_h2o * 0.88809, rel=1e-4)
    assert hf_row['H_kg_total'] == pytest.approx(m_h2o * 0.11191, rel=1e-4)

    # Discrimination guard: a wrong equal-split implementation (0.5 each) would
    # place H at 2.0e21 kg, differing from the correct 4.47e20 kg by >> tol.
    equal_split_H = m_h2o * 0.5
    assert abs(hf_row['H_kg_total'] - equal_split_H) > 0.3 * m_h2o

    # Mass conservation: the split conserves the water mass exactly.
    assert hf_row['H_kg_total'] + hf_row['O_kg_total'] == pytest.approx(m_h2o, rel=1e-12)
    # Reservoir partition sums to the total for each element.
    assert hf_row['H_kg_atm'] + hf_row['H_kg_liquid'] + hf_row['H_kg_solid'] == pytest.approx(
        hf_row['H_kg_total'], rel=1e-12
    )


@pytest.mark.physics_invariant
def test_mixed_inventory_conserves_total_mass_across_elements():
    """A multi-species inventory conserves total mass when split to elements.

    The sum of all per-element totals must equal the sum of all per-species
    totals: splitting molecules into atoms moves mass between bookkeeping slots
    but creates and destroys none. Uses water, carbon dioxide, and nitrogen so
    H, O, C and N are all exercised with distinct, non-degenerate stoichiometry.
    """
    hf_row = {
        'H2O_kg_atm': 3.0e20,
        'H2O_kg_liquid': 1.0e20,
        'H2O_kg_solid': 0.0,
        'CO2_kg_atm': 2.0e20,
        'CO2_kg_liquid': 0.0,
        'CO2_kg_solid': 0.0,
        'N2_kg_atm': 5.0e19,
        'N2_kg_liquid': 0.0,
        'N2_kg_solid': 0.0,
    }
    species_total = (
        4.0e20 + 2.0e20 + 5.0e19  # H2O + CO2 + N2
    )
    _populate_volatile_element_reservoirs(hf_row, element_mmw)

    element_total = sum(hf_row[f'{e}_kg_total'] for e in ('H', 'O', 'C', 'N', 'S', 'Si'))
    assert element_total == pytest.approx(species_total, rel=1e-12)

    # Nitrogen is elemental in N2, so its whole mass maps to N (a non-trivial
    # check distinct from the fractional H/O split), and C is strictly positive.
    assert hf_row['N_kg_total'] == pytest.approx(5.0e19, rel=1e-12)
    assert hf_row['C_kg_total'] > 0.0
    assert hf_row['S_kg_total'] == 0.0  # no S-bearing species present


@pytest.mark.physics_invariant
def test_empty_inventory_yields_zero_per_element():
    """An inventory with no volatile species mass leaves every element at zero.

    Boundary case: the desiccated / pre-outgas limit. Every per-element total
    must be exactly zero (not a stale or negative value), so a genuinely empty
    atmosphere is reported as empty rather than carrying phantom mass.
    """
    hf_row = {
        f'{sp}_kg_{r}': 0.0
        for sp in _VOLATILE_ELEMENT_STOICH
        for r in ('atm', 'liquid', 'solid')
    }
    _populate_volatile_element_reservoirs(hf_row, element_mmw)

    for e in ('H', 'O', 'C', 'N', 'S', 'Si'):
        assert hf_row[f'{e}_kg_total'] == 0.0
    # And the atmospheric reservoir specifically is zero (the field escape reads).
    assert hf_row['H_kg_atm'] == 0.0


@pytest.mark.physics_invariant
def test_element_budget_survives_outgas_reservoir_escape():
    """The populated element budget is NOT collapsed by the outgas-escape debit.

    Regression for the frozen-atmosphere defect: escape's ``calc_new_elements``
    under ``reservoir = "outgas"`` reads ``{e}_kg_atm``. When atmodeller leaves
    those at zero, the routine sees an empty reservoir and returns a zeroed
    inventory, after which atmodeller is skipped for want of a mass constraint
    and the atmosphere freezes. With the per-element atmospheric reservoir
    populated from species, a small escape debit leaves the budget intact.
    """
    hf_row = {
        'H2O_kg_atm': 2.0e21,
        'H2O_kg_liquid': 2.0e21,
        'H2O_kg_solid': 0.0,
        'CO2_kg_atm': 1.0e20,
        'CO2_kg_liquid': 0.0,
        'CO2_kg_solid': 0.0,
        'esc_rate_total': 1.0e3,  # kg/s, a small but non-zero escape rate
    }
    _populate_volatile_element_reservoirs(hf_row, element_mmw)
    h_before = hf_row['H_kg_total']
    assert h_before > 1.0e20  # sanity: the budget is well above the escape floor

    # One year of escape at this rate removes ~3.2e10 kg, far below the budget.
    tgt = calc_new_elements(hf_row, dt=1.0, reservoir='outgas')

    # The element budget survives: H stays close to its pre-escape value and is
    # nowhere near collapsed to zero.
    assert tgt['H'] == pytest.approx(h_before, rel=1e-6)
    assert tgt['H'] > 0.9 * h_before

    # Discrimination: with the atmospheric reservoir zeroed (the bug state), the
    # same call collapses every element to zero.
    bug_row = dict(hf_row)
    for e in element_mmw:
        bug_row[f'{e}_kg_atm'] = 0.0
    tgt_bug = calc_new_elements(bug_row, dt=1.0, reservoir='outgas')
    assert tgt_bug['H'] == 0.0
    assert all(v == 0.0 for v in tgt_bug.values())
