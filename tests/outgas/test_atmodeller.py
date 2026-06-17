"""Tests for the atmodeller outgassing wrapper's per-element reservoir accounting.

Module under test: ``proteus.outgas.atmodeller``.

The atmodeller solver reports per-species masses but not the per-element
reservoir masses that CALLIOPE writes natively. ``_populate_volatile_element_
reservoirs`` reconstructs the per-element ``{e}_kg_atm`` / ``_kg_liquid`` /
``_kg_solid`` fields from the per-species inventory by stoichiometry. It does
NOT write the per-element ``{e}_kg_total`` slot, which is owned by escape (the
running-budget owner) and, for oxygen, by the authoritative ``O_kg_total``
write. These tests pin that reconstruction and guard the coupling invariant
that motivates it: with the per-element atmospheric reservoir populated, the
escape step (``calc_new_elements`` under the default ``reservoir = "outgas"``)
keeps a live element budget instead of collapsing it to zero (which would
otherwise make atmodeller skip and freeze the atmosphere).

Invariants exercised:

* Stoichiometric fidelity / analytic limit: each element's mass fraction in a
  species equals the closed-form ``n * m_element / m_species`` (pinned for H/O
  in water against IUPAC atomic masses; reference_pinned).
* Mass conservation: the species mass split among constituent elements sums
  back to the total species mass.
* Positivity / boundedness: an empty inventory yields exactly zero per element.
* Coupling regression: a populated element budget survives a non-negligible
  outgas-reservoir escape debit, whereas the zeroed-reservoir bug state
  collapses it.

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
from proteus.utils.constants import element_mmw, secs_per_year, vol_list

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _element_total(hf_row: dict, e: str) -> float:
    """Sum the per-reservoir element masses the helper writes."""
    return sum(float(hf_row.get(f'{e}_kg_{r}', 0.0)) for r in ('atm', 'liquid', 'solid'))


def test_stoich_table_covers_exactly_the_volatile_species():
    """The composition table must match constants.vol_list exactly.

    A species in ``vol_list`` absent from the table would have its element mass
    silently dropped (a mass leak); an extra species would track an element the
    rest of the pipeline ignores. The import-time assertion enforces this; this
    test pins the contract and documents the deliberate exclusion of the SiO
    vapour species, which atmodeller does not outgas.
    """
    assert set(_VOLATILE_ELEMENT_STOICH) == set(vol_list)
    assert 'SiO' not in _VOLATILE_ELEMENT_STOICH  # vapour, not outgassed here
    # Every element referenced has a molar mass available (no KeyError path).
    for comp in _VOLATILE_ELEMENT_STOICH.values():
        assert all(element_mmw[el] > 0.0 for el in comp)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_water_split_matches_closed_form_mass_fractions():
    """A pure-water inventory splits into H and O by the closed-form fractions.

    For H2O the atomic-O mass fraction is m_O / (2 m_H + m_O) = 0.88809 and the
    H fraction is the complement, 0.11191 (analytic limit from IUPAC atomic
    masses, https://iupac.qmul.ac.uk/AtWt/). The discrimination guard rules out
    a naive equal (0.5/0.5) split, which would put H and O each at half the
    water mass and miss the pinned fractions by ~0.39 (far above tolerance).
    The helper writes the reservoir split only; the element total is its sum.
    """
    m_h2o = 4.0e21  # kg, both reservoirs populated to exercise atm+liquid
    hf_row = {
        'H2O_kg_atm': 2.5e21,
        'H2O_kg_liquid': 1.5e21,
        'H2O_kg_solid': 0.0,
    }
    _populate_volatile_element_reservoirs(hf_row, element_mmw)

    o_total = _element_total(hf_row, 'O')
    h_total = _element_total(hf_row, 'H')

    # Pinned closed-form fractions (hand-computed from IUPAC atomic masses).
    assert o_total == pytest.approx(m_h2o * 0.88809, rel=1e-4)
    assert h_total == pytest.approx(m_h2o * 0.11191, rel=1e-4)

    # Discrimination guard: an equal-split (0.5 each) would place H at 2.0e21 kg,
    # differing from the correct 4.47e20 kg by >> tol.
    assert abs(h_total - m_h2o * 0.5) > 0.3 * m_h2o

    # Mass conservation: the split conserves the water mass exactly.
    assert h_total + o_total == pytest.approx(m_h2o, rel=1e-12)
    # The atmospheric reservoir (the field escape reads) carries its share.
    assert hf_row['H_kg_atm'] == pytest.approx(2.5e21 * 0.11191, rel=1e-4)
    # Crucially the total slot is NOT written by the helper (escape owns it).
    assert 'H_kg_total' not in hf_row


@pytest.mark.physics_invariant
def test_mixed_inventory_conserves_total_mass_across_elements():
    """A multi-species inventory conserves total mass when split to elements.

    The sum of all per-element reservoir masses must equal the sum of all
    per-species masses: splitting molecules into atoms moves mass between
    bookkeeping slots but creates and destroys none. Water, carbon dioxide and
    nitrogen exercise H, O, C and N with distinct, non-degenerate stoichiometry.
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
    species_total = 4.0e20 + 2.0e20 + 5.0e19  # H2O + CO2 + N2
    _populate_volatile_element_reservoirs(hf_row, element_mmw)

    element_total = sum(_element_total(hf_row, e) for e in ('H', 'O', 'C', 'N', 'S', 'Si'))
    assert element_total == pytest.approx(species_total, rel=1e-12)

    # Nitrogen is elemental in N2, so its whole mass maps to N (a non-trivial
    # check distinct from the fractional H/O split), and C is strictly positive.
    assert _element_total(hf_row, 'N') == pytest.approx(5.0e19, rel=1e-12)
    assert _element_total(hf_row, 'C') > 0.0
    assert _element_total(hf_row, 'S') == 0.0  # no S-bearing species present


@pytest.mark.physics_invariant
def test_empty_inventory_yields_zero_per_element():
    """An inventory with no volatile species mass leaves every element at zero.

    Boundary case: the desiccated / pre-outgas limit. Every per-element
    reservoir must be exactly zero (not a stale or negative value), so a
    genuinely empty atmosphere is reported as empty rather than carrying phantom
    mass. A NaN species mass must also not leak into the reservoirs.
    """
    hf_row = {
        f'{sp}_kg_{r}': 0.0
        for sp in _VOLATILE_ELEMENT_STOICH
        for r in ('atm', 'liquid', 'solid')
    }
    hf_row['H2O_kg_atm'] = float('nan')  # a non-finite input must be rejected
    _populate_volatile_element_reservoirs(hf_row, element_mmw)

    for e in ('H', 'O', 'C', 'N', 'S', 'Si'):
        assert _element_total(hf_row, e) == 0.0
    # The atmospheric reservoir specifically is a hard zero (the field escape
    # reads), and the NaN did not propagate.
    assert hf_row['H_kg_atm'] == 0.0


@pytest.mark.physics_invariant
def test_element_budget_survives_outgas_reservoir_escape():
    """A populated element budget survives a real outgas-escape debit.

    Regression for the frozen-atmosphere defect: escape's ``calc_new_elements``
    under ``reservoir = "outgas"`` reads ``{e}_kg_atm``. When atmodeller leaves
    those at zero, the routine sees an empty reservoir and returns a zeroed
    inventory, after which atmodeller is skipped and the atmosphere freezes.
    With the per-element atmospheric reservoir populated from species, the debit
    is applied (not collapsed) and matches the closed-form escape attribution.
    """
    hf_row = {
        'H2O_kg_atm': 2.0e21,
        'H2O_kg_liquid': 2.0e21,
        'H2O_kg_solid': 0.0,
        'CO2_kg_atm': 1.0e20,
        'CO2_kg_liquid': 0.0,
        'CO2_kg_solid': 0.0,
        # A deliberately large rate so the debit is a few percent of the budget,
        # not a negligible perturbation the tolerance would hide.
        'esc_rate_total': 3.0e12,  # kg/s
    }
    _populate_volatile_element_reservoirs(hf_row, element_mmw)
    # Whole-planet element budget (escape's old_total) = the reservoir sum.
    for e in ('H', 'O', 'C'):
        hf_row[f'{e}_kg_total'] = _element_total(hf_row, e)
    h_budget = hf_row['H_kg_total']
    assert h_budget > 1.0e20  # sanity

    dt = 1.0  # yr
    esc_mass = 3.0e12 * secs_per_year * dt
    m_vols_atm = sum(float(hf_row.get(f'{e}_kg_atm', 0.0)) for e in ('H', 'O', 'C'))
    expected_h = h_budget - esc_mass * (hf_row['H_kg_atm'] / m_vols_atm)

    tgt = calc_new_elements(hf_row, dt=dt, reservoir='outgas')

    # The debit is non-negligible AND matches the closed-form attribution.
    assert (h_budget - tgt['H']) > 0.01 * h_budget  # a real, resolvable debit
    assert tgt['H'] == pytest.approx(expected_h, rel=1e-9)

    # Discrimination: with the atmospheric reservoir zeroed (the bug state), the
    # same call collapses every element to zero instead of debiting it.
    bug_row = dict(hf_row)
    for e in element_mmw:
        bug_row[f'{e}_kg_atm'] = 0.0
    tgt_bug = calc_new_elements(bug_row, dt=dt, reservoir='outgas')
    assert tgt_bug['H'] == 0.0
    assert all(v == 0.0 for v in tgt_bug.values())


@pytest.mark.physics_invariant
def test_o2_endpoint_keeps_live_budget_no_false_desiccation():
    """A desiccated-to-O2 atmosphere keeps a live budget (no false refusal).

    The T1 desiccation endpoint: H/C/N/S deplete and the atmosphere becomes
    nearly pure O2. O2 is pure oxygen, so the reservoir reconstruction maps its
    whole mass to O_kg_atm. Escape's outgas reservoir then sees a non-empty O
    reservoir and debits it normally; the per-element budget does NOT collapse to
    zero, which is what previously zeroed the whole inventory and made the
    desiccation gate read a spurious 'unexplained loss' on the O2 endpoint.
    """
    hf_row = {
        'O2_kg_atm': 9.0e21,
        'O2_kg_liquid': 0.0,
        'O2_kg_solid': 0.0,
        # H/C/N/S species depleted to trace amounts (below mass_thresh).
        'H2O_kg_atm': 1.0e2,
        'H2O_kg_liquid': 0.0,
        'H2O_kg_solid': 0.0,
        'esc_rate_total': 1.0e3,  # kg/s, modest escape
    }
    _populate_volatile_element_reservoirs(hf_row, element_mmw)

    # O2 is pure oxygen, so its whole mass maps to the O atmospheric reservoir.
    assert hf_row['O_kg_atm'] == pytest.approx(9.0e21, rel=1e-12)
    # H is only the trace water, far below the O inventory (asymmetric endpoint).
    assert hf_row['H_kg_atm'] < 1.0e2

    # Whole-planet budget = reservoir sum; escape then debits without collapse.
    for e in ('O', 'H'):
        hf_row[f'{e}_kg_total'] = _element_total(hf_row, e)
    tgt = calc_new_elements(hf_row, dt=1.0, reservoir='outgas')

    # The O budget survives the escape call (does NOT collapse to zero); this is
    # the mechanism that keeps the desiccation gate from a spurious refusal at
    # the O2 endpoint.
    assert tgt['O'] > 0.99 * 9.0e21

    # Discrimination: the pre-fix state (no O reservoir) collapses the budget,
    # which is the false-refusal / livelock the fix removes.
    bug_row = dict(hf_row)
    for e in element_mmw:
        bug_row[f'{e}_kg_atm'] = 0.0
    tgt_bug = calc_new_elements(bug_row, dt=1.0, reservoir='outgas')
    assert tgt_bug['O'] == 0.0


def test_model_cache_builds_once_per_species_network():
    """The solver model is built once per species network and then reused.

    atmodeller compiles its JIT solver on first build and caches it on the model
    instance; constructing a fresh model every solve forces a recompile whose
    compilation memory accumulates to an out-of-memory kill after a few dozen
    steps. The cache must return the identical instance for a repeated species
    signature (built once) and build a fresh instance only when the active
    network changes, while keeping earlier networks cached (not evicted).
    """
    from proteus.outgas.atmodeller import _MODEL_CACHE, _cached_model

    _MODEL_CACHE.clear()
    builds = []

    def build():
        builds.append(1)
        return object()

    sig_full = ('H2O', 'CO2', 'O2')
    m1 = _cached_model(sig_full, build)
    m2 = _cached_model(sig_full, build)  # repeat signature -> cache hit
    assert m1 is m2  # same instance reused (solver not rebuilt)
    assert len(builds) == 1  # built exactly once for the repeated network

    # A reduced network (an element dropped below threshold) builds a distinct
    # model; the build runs again only for the new signature.
    sig_reduced = ('H2O', 'O2')
    m3 = _cached_model(sig_reduced, build)
    assert m3 is not m1
    assert len(builds) == 2

    # The original network is still cached (not evicted by the new one).
    assert _cached_model(sig_full, build) is m1
    assert len(builds) == 2
    _MODEL_CACHE.clear()
