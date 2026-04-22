"""
Redox-budget conservation: escape debit + per-step conservation check
(#57).

Used by the main loop in `proteus.py` (Commit C) to:

1. Debit `R_budget_atm` by the species-resolved escape flux each step
   via :func:`debit_escape`.
2. Assert that ΔR_total = ΔR_escape + ΔR_dispro to within the
   user-configurable `config.redox.soft_conservation_tol` (fraction of
   |R_total|) via :func:`assert_redox_conserved`.

Plan v6 §2.6 (debit rule), §3.1 step 7 (conservation check).
"""
from __future__ import annotations

import logging

log = logging.getLogger('fwl.' + __name__)


def debit_escape(
    hf_row: dict,
    escape_fluxes_species_kg: dict[str, float],
) -> float:
    """
    Debit `hf_row['R_budget_atm']` by the species-resolved escape flux
    over this step.

    The per-species rule (plan v6 §2.6):

        ΔR_atm_escape = − Σ_species (n_species_escaped · RB_species)

    where `n_species_escaped` is `escape_fluxes_species_kg[species] /
    mw_species`. Returns the ΔR for diagnostics.

    Sign convention: escaping H₂ debits a negative RB (−2 per mol);
    the atmosphere's R_atm therefore RISES (becomes more oxidised).
    Consistent with Hirschmann 2023's H₂O-disproportionation story.

    Parameters
    ----------
    hf_row : dict
        Live helpfile row. Updated in place: `R_budget_atm`,
        `R_escaped_cum`.
    escape_fluxes_species_kg : dict[str, float]
        Mapping from supported species name (e.g. 'H2', 'H2O', 'CO2',
        …) to kg lost this step. Missing species treated as zero.
    """
    from proteus.redox.budget import DEFERRED_SPECIES, RB_COEF
    from proteus.utils.constants import element_mmw

    delta_r = 0.0
    for species, kg_escaped in escape_fluxes_species_kg.items():
        if kg_escaped == 0.0:
            continue
        if species in DEFERRED_SPECIES:
            log.warning(
                'debit_escape: %s is deferred; contribution to '
                'R_atm skipped', species,
            )
            continue
        if species not in RB_COEF:
            log.warning(
                'debit_escape: unknown species %s; contribution to '
                'R_atm skipped', species,
            )
            continue
        # Estimate molecular mass from species name.
        mw = _species_molar_mass(species, element_mmw)
        n_escaped = kg_escaped / mw
        delta_r -= n_escaped * RB_COEF[species]

    hf_row['R_budget_atm'] = (
        float(hf_row.get('R_budget_atm', 0.0)) + delta_r
    )
    hf_row['R_escaped_cum'] = (
        float(hf_row.get('R_escaped_cum', 0.0)) + delta_r
    )
    return delta_r


def _species_molar_mass(species: str, element_mmw: dict) -> float:
    """
    Parse a simple molecular formula like 'H2O', 'SO2', 'NH3' into a
    molar mass in kg/mol. Keeps the budget module independent of
    external libraries like `periodictable`.
    """
    import re

    # Tokenise element-count pairs: e.g. 'SiO2' → [('Si', 2), ('O', 1·implicit)]
    pattern = re.compile(r'([A-Z][a-z]?)(\d*)')
    total = 0.0
    for elem, count_str in pattern.findall(species):
        if not elem:
            continue
        count = int(count_str) if count_str else 1
        total += element_mmw[elem] * count
    return total


def assert_redox_conserved(
    hf_row: dict,
    hf_row_prev: dict,
    *,
    soft_tol: float = 1e-3,
    include_core: bool = True,
    include_mantle: bool = True,
    include_atm: bool = True,
) -> float:
    """
    Soft check on per-step redox-budget conservation.

    Computes residual = |ΔR_total − ΔR_escape − ΔR_dispro| / |R_total|
    where:
      ΔR_total = R_total(hf_row) − R_total(hf_row_prev)
      ΔR_escape = `hf_row['R_escaped_cum']` −
                  `hf_row_prev['R_escaped_cum']`
      ΔR_dispro = reserved for future disproportionation flux
                  (Mariana's metal-saturation path; zero in stub).

    If `residual > soft_tol`, logs a warning and writes the value to
    `hf_row['redox_conservation_residual']`. Does NOT raise (per
    plan v6 §3.1 step 7: "soft-fail"). Callers can inspect the
    stored residual to take action.

    Returns the residual for programmatic use.
    """
    from proteus.redox.budget import (
        redox_budget_atm,
        redox_budget_core,
        redox_budget_mantle,
    )

    def total(row):
        acc = 0.0
        if include_atm:
            acc += redox_budget_atm(row)
        if include_mantle:
            acc += redox_budget_mantle(row)
        if include_core:
            acc += redox_budget_core(row)
        return acc

    r_now = total(hf_row)
    r_prev = total(hf_row_prev)
    delta_total = r_now - r_prev
    delta_escape = (
        float(hf_row.get('R_escaped_cum', 0.0))
        - float(hf_row_prev.get('R_escaped_cum', 0.0))
    )
    delta_dispro = 0.0  # Mariana / Schaefer §2.7 hook (Commit C+)

    expected_delta = delta_escape + delta_dispro
    abs_r = max(abs(r_now), 1.0)  # avoid /0 at t=0
    residual = abs(delta_total - expected_delta) / abs_r
    hf_row['redox_conservation_residual'] = residual

    if residual > soft_tol:
        log.warning(
            'redox conservation soft-fail: residual=%.3e > tol=%.3e '
            '(ΔR_total=%.3e, ΔR_escape=%.3e, ΔR_dispro=%.3e)',
            residual, soft_tol, delta_total, delta_escape, delta_dispro,
        )
    return residual
