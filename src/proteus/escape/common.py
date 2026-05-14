# Shared functions used by escape calculation
from __future__ import annotations

import logging

from proteus.utils.constants import element_list

log = logging.getLogger('fwl.' + __name__)


def calc_unfract_fluxes(hf_row: dict, reservoir: str, min_thresh: float):
    """Calculate elemental escape rates, without fractionating.

    Updates the elemental escape fluxes in hf_row.

    Under the issue #677 fix (whole-planet O accounting), O is now part
    of the partitioning denominator and gets its own ``esc_rate_O``.
    Pre-fix code skipped O on the grounds that its mass was buffered
    from an "infinite" mantle FeO reservoir; the consequence was that
    ``esc_rate_total`` (Zephyrus's bulk MLR, which physically includes
    the O atoms leaving in H2O / CO2 / SO2) got attributed 100 percent
    to H+C+N+S, over-debiting those elements by a factor of up to ~8x
    at high H_ppmw with oxidising fO2.

    The fix is conceptually simple: distribute ``esc_rate_total`` over
    ALL element mass fractions (including O) so the per-element rates
    sum back to the bulk MLR.

    Parameters
    ----------
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
        reservoir: str
            Element reservoir representing the escaping composition (bulk, outgas)
        min_thresh: float
            Minimum threshold for element mass [kg]. Inventories below this are set to zero.
    """

    # which composition sets bulk escape?
    match reservoir:
        case 'bulk':  # bulk planet
            key = '_kg_total'
        case 'outgas':  # atmosphere
            key = '_kg_atm'
        case _:
            raise ValueError(f"Invalid escape reservoir '{reservoir}'")

    # Calculate mass of elements in reservoir. Issue #677: O is now in
    # the denominator so sum(esc_rate_e for e in element_list) ==
    # esc_rate_total to within rounding, instead of equalling
    # esc_rate_total only after excluding O.
    res = {}
    for e in element_list:
        res[e] = float(hf_row.get(e + key, 0.0))
    M_vols = sum(list(res.values()))

    # check if we just desiccated the planet...
    if M_vols < min_thresh:
        log.debug('    Total mass of volatiles below threshold in escape calculation')
        return

    # calculate the current mass mixing ratio for each element
    #     if escape is unfractionating, this should be conserved
    for e in res.keys():
        emr = res[e] / M_vols
        log.debug('    %2s (%s) mass ratio = %.2e ' % (e, reservoir, emr))
        hf_row['esc_rate_' + e] = hf_row['esc_rate_total'] * emr
