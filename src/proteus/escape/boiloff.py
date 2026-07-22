# Initial-condition correction for envelopes that are not gravitationally bound
# Authors: Tim Lichtenberg
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.utils.constants import element_list

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def apply_boiloff_ic(config: Config, hf_row: dict) -> None:
    """Reduce the volatile inventory to the fraction that survives boil-off.

    A planet whose envelope extends towards its Bondi radius sheds mass through
    a wind driven by the envelope's own thermal energy. That phase runs to
    completion well before the epoch this model integrates, so an envelope
    still inflated at the initial condition describes a state that would
    already have been shed. This scales the whole-planet volatile inventory
    down to the surviving fraction so the evolution starts from a bound
    envelope.

    All elements are scaled by the same factor, leaving the configured
    elemental ratios unchanged. The reduced inventory is re-partitioned between
    atmosphere and melt by the next outgassing call, and the initialisation
    loop re-enters this function, so the correction converges over the
    initialisation passes rather than relying on a single analytic step.

    Parameters
    ----------
    config : Config
        Configuration options. No effect unless ``escape.boiloff_ic`` is set.
    hf_row : dict
        Helpfile variables for this iteration. Reads ``M_planet``, ``R_obs``
        and ``T_eqm``; scales every ``{element}_kg_total`` and accumulates the
        removed mass into ``M_boiloff_kg``.

    Notes
    -----
    The surviving fraction is Owen & Wu (2016), Equation 16, evaluated in the
    radius ratio ``R_obs / R_Bondi``; it is the sole criterion, because it is
    the variable that result is derived in. The restricted Jeans parameter of
    Fossati et al. (2017) is reported alongside for context but does not gate
    the correction: the two are quoted as equivalent at a threshold of 20 only
    when both are evaluated for atomic hydrogen, so pairing a Jeans threshold
    with a Bondi radius taken for a hydrogen-helium envelope would place
    ordinary planets between them. Both quantities come from
    :mod:`zephyrus.boiloff`.
    """
    if not getattr(config.escape, 'boiloff_ic', False):
        return

    from zephyrus.boiloff import boiloff_mass_factor, bondi_radius, restricted_jeans

    mass = float(hf_row.get('M_planet', 0.0))
    radius = float(hf_row.get('R_obs', 0.0))
    t_eq = float(hf_row.get('T_eqm', 0.0))

    # Before the first structure solve these are zero or absent, and the
    # correction has nothing to act on.
    if not all(np.isfinite(v) and v > 0.0 for v in (mass, radius, t_eq)):
        return

    r_bondi = bondi_radius(mass, t_eq)
    factor = boiloff_mass_factor(radius, r_bondi)

    if factor >= 1.0:
        return

    lam = restricted_jeans(mass, radius, t_eq)

    removed = 0.0
    for element in element_list:
        key = f'{element}_kg_total'
        before = float(hf_row.get(key, 0.0))
        if before <= 0.0:
            continue
        hf_row[key] = before * factor
        removed += before - hf_row[key]

    hf_row['M_boiloff_kg'] = float(hf_row.get('M_boiloff_kg', 0.0)) + removed

    log.info(
        'Boil-off IC correction: Lambda = %.3g, R/R_Bondi = %.3g, '
        'keeping %.3g of the volatile inventory (%.3e kg removed)',
        lam,
        radius / r_bondi,
        factor,
        removed,
    )
