"""
Redox-budget (RB) computations for the unified redox module (#57).

Implements the Evans (2012) / Hirschmann (2023) redox-budget formalism
as applied to PROTEUS reservoirs.

Conventions
-----------

Reference states (Evans mantle-like, globally; see plan v6 §2.1):

    Fe : Fe²⁺    C : C⁰     S : S²⁻    H : H¹⁺
    O  : O²⁻     N : N⁰     Si: Si⁴⁺   Mg : Mg²⁺
    Cr : Cr³⁺

Sign convention (Hirschmann 2023; see plan v6 §2.2):

    v_i = z_sample − z_ref                   [electrons per atom]
    RB_species = Σ_{atoms} v_i               [per molecule]
    R_reservoir = Σ_species n_species · RB   [mol electrons]

Positive RB → reservoir is more oxidised than the reference state.

References
----------
- Evans 2012, Earth-Sci. Rev. 113, 11-32 (formalism).
- Hirschmann 2023, EPSL 619, 118311 (Earth inventories).
- Schaefer et al. 2024, JGR Planets 129, e2023JE008262 (Eq 13).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    pass

log = logging.getLogger('fwl.' + __name__)


# ------------------------------------------------------------------
# Reference-state declaration — locked for the whole module
# ------------------------------------------------------------------
REFERENCE_STATES: dict[str, int] = {
    'Fe': 2,
    'C': 0,
    'S': -2,
    'H': 1,
    'O': -2,
    'N': 0,
    'Si': 4,
    'Mg': 2,
    'Cr': 3,
}


# ------------------------------------------------------------------
# Per-species RB coefficients — derived mechanically from
# REFERENCE_STATES and per-species per-atom oxidation-state
# assignments. See test_budget.py for the derivation lock.
# ------------------------------------------------------------------

# Composition table: species → list of (element, oxidation_state,
# multiplicity). Used both to derive RB_COEF and to self-document the
# assignments for future reviewers.
SPECIES_COMPOSITION: dict[str, tuple[tuple[str, int, int], ...]] = {
    # --- Atmospheric species (emitted by CALLIOPE / atmodeller today)
    'H2O':  (('H',  1, 2), ('O', -2, 1)),
    'H2':   (('H',  0, 2),),
    'O2':   (('O',  0, 2),),
    'CO2':  (('C',  4, 1), ('O', -2, 2)),
    'CO':   (('C',  2, 1), ('O', -2, 1)),
    'CH4':  (('C', -4, 1), ('H',  1, 4)),
    'N2':   (('N',  0, 2),),
    'NH3':  (('N', -3, 1), ('H',  1, 3)),
    'S2':   (('S',  0, 2),),
    'SO2':  (('S',  4, 1), ('O', -2, 2)),
    'H2S':  (('S', -2, 1), ('H',  1, 2)),
    # --- Vapor species (vap_list in constants.py)
    'SiO':  (('Si', 2, 1), ('O', -2, 1)),
    'SiO2': (('Si', 4, 1), ('O', -2, 2)),
    'MgO':  (('Mg', 2, 1), ('O', -2, 1)),
    'FeO2': (('Fe', 4, 1), ('O', -2, 2)),
    # --- Mineral / melt species for R_mantle bookkeeping
    'FeO':    (('Fe', 2, 1), ('O', -2, 1)),
    'Fe2O3':  (('Fe', 3, 2), ('O', -2, 3)),
    'Fe0':    (('Fe', 0, 1),),
    'FeS2':   (('Fe', 2, 1), ('S', -1, 2)),
}

# Species that CALLIOPE / atmodeller / future upstream atm chemistries
# may emit but that this module refuses to quantify because their
# oxidation-state assignments are ambiguous under our convention.
# `redox_budget_atm` logs a warning and skips these rather than
# invent a value.
DEFERRED_SPECIES: frozenset[str] = frozenset({
    'SO3', 'H2SO4',  # S+6; O assignment in peroxide contexts debatable
    'OCS',           # S-C electronegativity marginal (χ 2.58 vs 2.55)
    'SO',
    'HCN',           # C assignment ambiguous (N electronegativity > C)
    'N2O',           # N at mixed oxidation states
    'NO', 'NO2',
})


def _compute_rb_coef(species: str) -> int:
    """RB coefficient for a species from SPECIES_COMPOSITION."""
    total = 0
    for element, z_sample, mult in SPECIES_COMPOSITION[species]:
        z_ref = REFERENCE_STATES[element]
        total += mult * (z_sample - z_ref)
    return total


RB_COEF: dict[str, int] = {
    species: _compute_rb_coef(species)
    for species in SPECIES_COMPOSITION
}


# ------------------------------------------------------------------
# Public budget functions
# ------------------------------------------------------------------

def redox_budget_species(species: str, n_moles: float) -> float:
    """
    RB contribution of `n_moles` of `species` in mol of electrons.

    Returns zero (with a warning) if the species is in
    `DEFERRED_SPECIES`; raises KeyError for completely unknown
    species.
    """
    if species in DEFERRED_SPECIES:
        log.warning(
            'redox_budget_species: %s is deferred (see DEFERRED_SPECIES); '
            'RB contribution skipped', species,
        )
        return 0.0
    return n_moles * RB_COEF[species]


def redox_budget_atm(hf_row: dict) -> float:
    """
    Atmospheric redox budget R_atm in mol electrons.

    Sums RB contributions over every supported species for which
    `hf_row` contains a `{species}_mol_atm` key. Deferred species and
    species unknown to this module are skipped with a warning.
    """
    total = 0.0
    for species in RB_COEF:
        key = f'{species}_mol_atm'
        if key in hf_row:
            total += hf_row[key] * RB_COEF[species]
    return total


def redox_budget_mantle(hf_row: dict, mantle_comp=None) -> float:
    """
    Mantle redox budget R_mantle in mol electrons (bulk formulation).

    Components (plan v6 §2.5):
      * Fe³⁺ in melt: n_Fe3_melt · (+1)
      * Fe⁰ in mantle metal (stoichiometric disproportionation): n_Fe0_solid_total · (−2)
      * Per-cell solid Fe³⁺: aggregated in n_Fe3_solid_total · (+1)
      * Dissolved volatiles (H2O, H2, CO2, …) in melt: Σ n_{species}_mol_liquid · RB_{species}

    Ferric/ferrous in the SOLID MANTLE (non-metal) is computed through
    `n_Fe3_solid_total` only. The stub `advance_fe_reservoirs` in
    `partitioning.py` populates this in `composition` / `fO2_init`
    modes; static mode leaves it at zero.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row. Consumed keys: `n_Fe3_melt`,
        `n_Fe3_solid_total`, `n_Fe0_solid_total`, and
        `{species}_mol_liquid` for every supported species.
    mantle_comp : optional
        Kept in the signature for forward compatibility with
        Commit C; currently unused (all state flows through hf_row).
    """
    fe3_melt = float(hf_row.get('n_Fe3_melt', 0.0))
    fe3_solid = float(hf_row.get('n_Fe3_solid_total', 0.0))
    fe0_solid = float(hf_row.get('n_Fe0_solid_total', 0.0))

    rb_fe = (fe3_melt + fe3_solid) * (+1) + fe0_solid * (-2)

    rb_dissolved = 0.0
    for species in RB_COEF:
        key = f'{species}_mol_liquid'
        if key in hf_row:
            rb_dissolved += hf_row[key] * RB_COEF[species]

    return rb_fe + rb_dissolved


def redox_budget_core(hf_row: dict, core_comp=None) -> float:
    """
    Core redox budget R_core in mol electrons.

    Under the global Fe²⁺ reference, pure Fe⁰ in the core contributes
    v = −2 per atom. For Earth's ~1.8e24 kg core the absolute value is
    ≈ −6.4 × 10²⁵ mol e⁻ (~280 × BSE RB). Conservation tracks the
    time-derivative of this quantity; see plan v6 §2.4.

    Parameters
    ----------
    hf_row : dict
        Consumed keys: per-element `{element}_kg_core` populated by
        the (future) #526 metal-silicate partitioning, plus a
        `dm_Fe0_to_core_cum` accumulator for Mariana's
        disproportionation drain.
    core_comp : optional
        Kept in the signature for forward compatibility.
    """
    from proteus.utils.constants import element_mmw

    # Fe⁰ in core: v = -2 per atom.
    fe_kg = float(hf_row.get('Fe_kg_core', 0.0))
    mw_fe = element_mmw['Fe']
    rb_fe_core = (fe_kg / mw_fe) * (-2)

    # Other light elements dissolved in core (#526 hook).
    # H: v = -1 per atom (H⁰ vs H¹⁺ ref).
    h_kg = float(hf_row.get('H_kg_core', 0.0))
    rb_h = (h_kg / element_mmw['H']) * (-1)

    # O: v = +2 per atom (O⁰ vs O²⁻ ref).
    o_kg = float(hf_row.get('O_kg_core', 0.0))
    rb_o = (o_kg / element_mmw['O']) * (+2)

    # Si: v = 0 per atom (Si⁰ vs Si⁴⁺ ref → v = -4).
    # We pick v = -4 to make Si⁰ reduced relative to Si⁴⁺ reference.
    # Using Si mass fraction from CoreComp.
    si_kg = float(hf_row.get('Si_kg_core', 0.0))
    rb_si = (si_kg / element_mmw['Si']) * (-4)

    # C, S, N: not yet wired.
    # (Would follow same pattern: n · (z_sample − z_ref).)
    return rb_fe_core + rb_h + rb_o + rb_si


def redox_budget_total(hf_row: dict) -> float:
    """R_total = R_atm + R_mantle + R_core."""
    return (
        redox_budget_atm(hf_row)
        + redox_budget_mantle(hf_row)
        + redox_budget_core(hf_row)
    )
