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

See ``docs/How-to/testing.md``,
``docs/Explanations/test_framework.md``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from proteus.escape.wrapper import calc_new_elements
from proteus.outgas.atmodeller import (
    _VOLATILE_ELEMENT_STOICH,
    _populate_volatile_element_reservoirs,
    calc_surface_pressures_atmodeller,
)
from proteus.utils.constants import element_mmw, gas_list, secs_per_year, vol_list

# NOTE: do NOT importorskip('atmodeller') at module scope. The helper tests below
# exercise only pure-Python reconstruction (`_populate_volatile_element_reservoirs`,
# `_cached_model`, the stoichiometry table) and must stay collectable on CI
# runners that omit atmodeller. The end-to-end tests that patch
# `atmodeller.EquilibriumModel` call `importorskip` locally instead.

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

    # Boundary: with the atmospheric reservoir zeroed there is nothing to
    # partition the loss over, so the call is a no-op that PRESERVES the
    # whole-planet totals. The historical failure mode returned the zeroed
    # reservoir dict here, collapsing every total to zero once written back.
    empty_row = dict(hf_row)
    for e in element_mmw:
        empty_row[f'{e}_kg_atm'] = 0.0
    tgt_empty = calc_new_elements(empty_row, dt=dt, reservoir='outgas')
    assert tgt_empty['H'] == pytest.approx(h_budget, rel=1e-12)
    assert tgt_empty['H'] > 1.0e20  # discriminates against the zero collapse


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

    # Boundary: with no atmospheric reservoir at all the call is a no-op that
    # preserves the O budget; a collapse to zero here is the false-refusal /
    # livelock failure mode, on either side of the reservoir reconstruction.
    empty_row = dict(hf_row)
    for e in element_mmw:
        empty_row[f'{e}_kg_atm'] = 0.0
    tgt_empty = calc_new_elements(empty_row, dt=1.0, reservoir='outgas')
    assert tgt_empty['O'] == pytest.approx(hf_row['O_kg_total'], rel=1e-12)
    assert tgt_empty['O'] > 8.0e21  # discriminates against the zero collapse


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

    # Boundary: the empty species network (no active species) is a legal,
    # distinct key and is cached like any other.
    empty = _cached_model((), build)
    assert empty is not m1
    assert _cached_model((), build) is empty
    assert len(builds) == 3
    _MODEL_CACHE.clear()


# ---------------------------------------------------------------------------
# Atmospheric-mass sourcing + species-key mapping regression
# (calc_surface_pressures_atmodeller end-to-end with atmodeller mocked)
# ---------------------------------------------------------------------------


def _species_mass_frac(stoich: dict, el: str) -> float:
    """Mass fraction of element ``el`` in a species, from ``element_mmw``."""
    sp_mmw = sum(n * element_mmw[e] for e, n in stoich.items())
    return stoich.get(el, 0) * element_mmw[el] / sp_mmw


# proteus name -> (atmodeller output key, stoichiometry). Sulfur dioxide is
# keyed 'O2S_g' by atmodeller (Hill notation), the case the wrapper must map.
_SP_INFO = {
    'H2O': ('H2O_g', {'H': 2, 'O': 1}),
    'H2': ('H2_g', {'H': 2}),
    'S2': ('S2_g', {'S': 2}),
    'SO2': ('O2S_g', {'S': 1, 'O': 2}),
    'O2': ('O2_g', {'O': 2}),
}

# Single partial pressure fed for every species, deliberately inconsistent with
# the gas masses so the p*A/g reconstruction cannot reproduce the conserving
# split (see _build_conserving_output).
_FIXED_PRESSURE_BAR = 5.0

# Each scenario hand-builds the species distribution that atmodeller's solver
# WOULD produce at the named redox regime (the solver itself is mocked, so the
# IW label documents which speciation the case represents, it is not computed
# from fO2 here; the real solver-driven fO2 sweep is covered by the integration
# test test_atmodeller_dummy_two_timesteps). Reducing keeps S in S2 and H in H2;
# oxidising moves H into H2O and S into the Hill-keyed SO2 ('O2S_g'); the middle
# case mixes both. Each species' assigned element mass and atmosphere/melt split
# set its atmodeller ``gas_mass`` / ``dissolved_mass`` so the per-element split
# must sum back to the budget.
_REGRESSION_SCENARIOS = {
    'reducing_IWm2': {
        'fO2': -2.0,
        # proteus_name: (primary_element, element_mass_assigned, atm_fraction)
        'assign': {
            'H2': ('H', 1.2e22, 0.6),
            'S2': ('S', 1.92e21, 0.6),
        },
    },
    'mid_IW0': {
        'fO2': 0.0,
        'assign': {
            'H2O': ('H', 0.5 * 1.2e22, 0.6),
            'H2': ('H', 0.5 * 1.2e22, 0.6),
            'S2': ('S', 0.4 * 1.92e21, 0.6),
            'SO2': ('S', 0.6 * 1.92e21, 0.6),
        },
    },
    'oxidising_IWp4': {
        'fO2': 4.0,
        'assign': {
            'H2O': ('H', 1.2e22, 0.6),
            'SO2': ('S', 1.92e21, 0.6),
        },
    },
}

_H_BUDGET = 1.2e22
_S_BUDGET = 1.92e21


def _make_user_constant_config(fO2_shift_IW: float):
    """MagicMock Config for the atmodeller backend under user_constant fO2."""
    config = MagicMock()
    config.outgas.module = 'atmodeller'
    config.outgas.fO2_shift_IW = fO2_shift_IW
    config.outgas.T_floor = 1200.0
    config.outgas.mass_thresh = 1e8
    config.outgas.solver_atol = 1e-8
    config.outgas.solver_rtol = 1e-5
    config.outgas.atmodeller.solver_max_steps = 100
    config.outgas.atmodeller.solver_multistart = 1
    config.outgas.atmodeller.solver_mode = 'robust'
    config.outgas.atmodeller.include_condensates = False
    for name in ('H2O', 'CO2', 'H2', 'N2', 'S2', 'CO', 'CH4'):
        setattr(config.outgas.atmodeller, f'solubility_{name}', None)
    for name in ('H2O', 'CO2', 'H2', 'CH4', 'CO'):
        setattr(config.outgas.atmodeller, f'eos_{name}', None)
    config.planet.fO2_source = 'user_constant'
    config.interior_struct.core_frac = 0.3
    return config


def _hf_row_with_HS_budget() -> dict:
    """Helpfile row seeded up to the outgas call, only H and S above threshold."""
    hf: dict = {
        'M_planet': 5.97e24,
        'R_int': 6.371e6,
        'gravity': 9.81,
        'T_magma': 2200.0,
        'Phi_global': 1.0,
    }
    for e in element_mmw:
        hf[f'{e}_kg_total'] = 0.0
    hf['H_kg_total'] = _H_BUDGET
    hf['S_kg_total'] = _S_BUDGET
    for s in gas_list:
        for r in ('atm', 'liquid', 'solid'):
            hf[f'{s}_kg_{r}'] = 0.0
        hf[f'{s}_bar'] = 0.0
        hf[f'{s}_vmr'] = 0.0
    return hf


def _build_conserving_output(assign: dict, fixed_pressure: float = _FIXED_PRESSURE_BAR):
    """Build a fake atmodeller output that conserves the assigned element budgets.

    For each species the assigned element mass divided by that element's mass
    fraction gives the species total mass; the atmosphere/melt split sets
    ``gas_mass`` / ``dissolved_mass``. Partial pressures are deliberately a
    single fixed value, INDEPENDENT of the masses, so a regression that reverts
    to the p*A/g reconstruction (which would assign every species the same
    column mass) cannot reproduce the conserving split.
    """
    asdict: dict = {'O2_g': {'log10dIW_1_bar': np.array(-0.1), 'gas_mass': np.array(1.0e10)}}
    quick_look: dict = {'O2_g': np.array(1.0e-6)}
    # Record the intended gas_mass per proteus species for the test's asserts.
    gas_by_species: dict = {}
    for proteus_name, (element, el_mass, atm_frac) in assign.items():
        atmod_key, stoich = _SP_INFO[proteus_name]
        sp_total = el_mass / _species_mass_frac(stoich, element)
        gas = atm_frac * sp_total
        dissolved = (1.0 - atm_frac) * sp_total
        asdict[atmod_key] = {
            'gas_mass': np.array(gas),
            'dissolved_mass': np.array(dissolved),
        }
        quick_look[atmod_key] = np.array(fixed_pressure)
        gas_by_species[proteus_name] = gas
    out = MagicMock()
    out.quick_look.return_value = quick_look
    out.total_pressure.return_value = np.array(fixed_pressure * len(quick_look))
    out.asdict.return_value = asdict
    return out, gas_by_species


@pytest.mark.physics_invariant
@pytest.mark.parametrize('scenario', sorted(_REGRESSION_SCENARIOS), ids=lambda s: s)
def test_element_split_closes_against_budget_across_redox(scenario):
    """The per-element reservoir split closes to the conserved budget across fO2.

    Regression for two atmospheric-mass bugs in the atmodeller wrapper:

    1. ``{sp}_kg_atm`` is sourced from atmodeller's per-species ``gas_mass``,
       not the thin-atmosphere reconstruction ``p*1e5*A/g``. That
       reconstruction recovers the species' MOLE-fraction share of the column
       (``x_i * M_atm``), mis-weighting each species by ``mu_mean / mu_i``, so
       the per-element split would not close against the conserved element
       budget. The reducing case (H carried by light H2) is where the
       mis-weighting is largest.
    2. Sulfur dioxide is keyed ``'O2S_g'`` by atmodeller (Hill notation), so the
       wrapper must map ``SO2 -> 'O2S'``. With the stale ``'SO2'`` key the
       species' atmospheric mass is silently dropped and S under-counts in an
       SO2-dominated oxidising atmosphere (the IW+4 case here).

    Verifies for H and S (the constrained elements; their ``{e}_kg_total`` is
    the input budget, NOT recomputed by the wrapper under user_constant):
    - ``{e}_kg_atm + {e}_kg_liquid + {e}_kg_solid == {e}_kg_total`` to 1e-9.
    - O closure as a guard (its total IS recomputed, so closure is consistency).
    - Discrimination guard 1: the p*A/g reconstruction for an H-carrying species
      differs from the sourced ``gas_mass`` by far more than the closure
      tolerance, so a revert to p*A/g would break closure.
    - Discrimination guard 2 (oxidising/mid): ``SO2_kg_atm`` is non-zero, i.e.
      the Hill-keyed species was captured rather than dropped.
    - Every assigned species' ``{sp}_kg_atm`` equals its ``gas_mass`` (pins the
      gas/dissolved partition per species, not just the per-element total, so a
      gas<->dissolved swap on any single species is caught).
    """
    pytest.importorskip('atmodeller')  # patches atmodeller.EquilibriumModel below
    spec = _REGRESSION_SCENARIOS[scenario]
    config = _make_user_constant_config(spec['fO2'])
    hf_row = _hf_row_with_HS_budget()
    fake_out, gas_by_species = _build_conserving_output(spec['assign'])

    fake_model = MagicMock()
    fake_model.output = fake_out

    from proteus.outgas.atmodeller import _MODEL_CACHE

    # A static-network mock would otherwise be served from a prior scenario's
    # cached model (the active element set {H, S} is identical across scenarios).
    _MODEL_CACHE.clear()
    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller({'output': '/tmp/test'}, config, hf_row)
    _MODEL_CACHE.clear()

    # Per-element closure on the constrained elements. The equality form (tight
    # rel) discriminates a factor / exponent error in the split.
    for el, budget in (('H', _H_BUDGET), ('S', _S_BUDGET)):
        split = sum(float(hf_row[f'{el}_kg_{r}']) for r in ('atm', 'liquid', 'solid'))
        assert split == pytest.approx(budget, rel=1e-9), (
            f'{scenario}: {el} split {split:.6e} != budget {budget:.6e}'
        )

    # O closure: O_kg_total is recomputed by the wrapper from the same species
    # masses, so the split must agree with it (internal consistency guard). O is
    # present at least as the buffer O2 floor in every scenario.
    o_split = sum(float(hf_row[f'O_kg_{r}']) for r in ('atm', 'liquid', 'solid'))
    assert o_split == pytest.approx(float(hf_row['O_kg_total']), rel=1e-9)
    assert float(hf_row['O_kg_total']) > 0.0  # O-bearing species present

    # Per-species pin: every assigned species' atmospheric mass equals its
    # gas_mass, so a gas<->dissolved swap on any single species (which preserves
    # the species total and would slip past the per-element closure above) is
    # caught. This covers S2 in the reducing case, which the closure alone leaves
    # unpinned.
    for sp_name, gas_mass in gas_by_species.items():
        assert hf_row[f'{sp_name}_kg_atm'] == pytest.approx(gas_mass, rel=1e-9), (
            f'{scenario}: {sp_name}_kg_atm != sourced gas_mass'
        )

    # Discrimination 1: the sourced atmospheric mass is gas_mass, and the p*A/g
    # reconstruction would differ by >> the closure tolerance. Pick an
    # H-carrying species present in this scenario.
    h_species = next(s for s in spec['assign'] if 'H' in _SP_INFO[s][1])
    area = 4.0 * np.pi * hf_row['R_int'] ** 2
    recon = _FIXED_PRESSURE_BAR * 1e5 * area / hf_row['gravity']
    gas_mass = gas_by_species[h_species]
    assert abs(gas_mass - recon) > 0.1 * gas_mass, (
        f'{scenario}: p*A/g recon {recon:.3e} too close to gas_mass {gas_mass:.3e} '
        'to discriminate the fix'
    )

    # Discrimination 2: when S sits in SO2, its Hill-keyed atmospheric mass must
    # be captured (non-zero), not dropped by a stale 'SO2_g' lookup.
    if 'SO2' in spec['assign']:
        assert hf_row['SO2_kg_atm'] > 0.0


@pytest.mark.physics_invariant
def test_atm_mass_falls_back_to_thin_atm_recon_when_gas_mass_absent(caplog):
    """When atmodeller omits gas_mass, the wrapper falls back to p*A/g and warns.

    The primary path sources ``{sp}_kg_atm`` from ``gas_mass``; the fallback
    (kept for an atmodeller output-schema change that drops the field) must
    reconstruct the column mass as ``p*1e5*A/g`` and emit a warning so the
    degraded, mis-weighted accounting is not silent. Exercises the branch that
    the conserving-output tests never hit (they always supply gas_mass).
    """
    import logging

    pytest.importorskip('atmodeller')
    config = _make_user_constant_config(0.0)
    hf_row = _hf_row_with_HS_budget()

    # H2O present with a partial pressure but NO gas_mass field -> fallback.
    out = MagicMock()
    out.quick_look.return_value = {'H2O_g': np.array(40.0), 'O2_g': np.array(1.0e-6)}
    out.total_pressure.return_value = np.array(40.0)
    out.asdict.return_value = {
        'H2O_g': {'dissolved_mass': np.array(1.0e20)},  # no 'gas_mass'
        'O2_g': {'log10dIW_1_bar': np.array(-0.1), 'gas_mass': np.array(1.0e10)},
    }
    fake_model = MagicMock()
    fake_model.output = out

    from proteus.outgas.atmodeller import _MODEL_CACHE

    _MODEL_CACHE.clear()
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.outgas.atmodeller'):
        with patch('atmodeller.EquilibriumModel', return_value=fake_model):
            calc_surface_pressures_atmodeller({'output': '/tmp/test'}, config, hf_row)
    _MODEL_CACHE.clear()

    area = 4.0 * np.pi * hf_row['R_int'] ** 2
    expected = 40.0 * 1e5 * area / hf_row['gravity']  # p*1e5*A/g
    assert hf_row['H2O_kg_atm'] == pytest.approx(expected, rel=1e-9)
    # The fallback must be loud, not silent.
    assert any(
        "missing 'gas_mass'" in r.getMessage() and 'p*A/g' in r.getMessage()
        for r in caplog.records
    ), 'fallback to p*A/g did not emit the expected warning'
    # Discrimination: the reconstruction is NOT a no-op zero and NOT the dissolved
    # mass; it is the column mass from the partial pressure.
    assert hf_row['H2O_kg_atm'] > 0.0
    assert hf_row['H2O_kg_atm'] != pytest.approx(1.0e20, rel=1e-3)


@pytest.mark.physics_invariant
def test_condensed_mass_enters_solid_reservoir_and_closes_carbon():
    """Condensed (graphite) carbon is routed to C_kg_solid so C still closes.

    atmodeller reports condensed-phase mass separately (``element_C.
    condensed_mass``); the gas-species inventory does not carry it. Without
    folding it into the solid reservoir, ``C_kg_atm + C_kg_liquid + C_kg_solid``
    would under-count the carbon budget exactly like the SO2 atmospheric drop.
    Here graphite holds part of the carbon budget; the split must still close.
    """
    pytest.importorskip('atmodeller')
    mC, mO = element_mmw['C'], element_mmw['O']
    c_frac_co2 = mC / (mC + 2 * mO)
    C_BUDGET = 5.0e21
    # Split the carbon budget: 30% in gas/dissolved CO2, 70% condensed graphite.
    co2_total = 0.3 * C_BUDGET / c_frac_co2
    co2_gas = 0.6 * co2_total
    co2_dissolved = 0.4 * co2_total
    condensed_C = 0.7 * C_BUDGET

    config = _make_user_constant_config(0.0)
    config.outgas.atmodeller.include_condensates = True
    hf_row = _hf_row_with_HS_budget()
    hf_row['H_kg_total'] = 0.0  # carbon-only so the active network is C-bearing
    hf_row['S_kg_total'] = 0.0
    hf_row['C_kg_total'] = C_BUDGET

    out = MagicMock()
    out.quick_look.return_value = {'CO2_g': np.array(20.0), 'O2_g': np.array(1.0e-6)}
    out.total_pressure.return_value = np.array(20.0)
    out.asdict.return_value = {
        'CO2_g': {'gas_mass': np.array(co2_gas), 'dissolved_mass': np.array(co2_dissolved)},
        'O2_g': {'log10dIW_1_bar': np.array(-0.1), 'gas_mass': np.array(1.0e10)},
        'element_C': {'condensed_mass': np.array(condensed_C)},
    }
    fake_model = MagicMock()
    fake_model.output = out

    from proteus.outgas.atmodeller import _MODEL_CACHE

    _MODEL_CACHE.clear()
    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller({'output': '/tmp/test'}, config, hf_row)
    _MODEL_CACHE.clear()

    # The condensed carbon lands in the solid reservoir.
    assert hf_row['C_kg_solid'] == pytest.approx(condensed_C, rel=1e-9)
    # Full closure: atmosphere + melt + solid == the carbon budget.
    c_split = sum(float(hf_row[f'C_kg_{r}']) for r in ('atm', 'liquid', 'solid'))
    assert c_split == pytest.approx(C_BUDGET, rel=1e-9), (
        f'C split {c_split:.6e} != budget {C_BUDGET:.6e}'
    )
    # Discrimination: dropping the condensed term (the pre-fix state) under-counts
    # by the graphite mass, i.e. the gas+liquid part alone misses 70% of carbon.
    c_gas_liquid = float(hf_row['C_kg_atm']) + float(hf_row['C_kg_liquid'])
    assert c_gas_liquid == pytest.approx(0.3 * C_BUDGET, rel=1e-6)
    assert abs(c_gas_liquid - C_BUDGET) > 0.5 * C_BUDGET
