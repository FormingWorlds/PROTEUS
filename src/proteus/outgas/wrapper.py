# Generic outgassing wrapper
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.outgas.common import expected_keys
from proteus.utils.constants import element_list, gas_list

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def calc_target_elemental_inventories(dirs: dict, config: Config, hf_row: dict):
    """
    Calculate total amount of volatile elements in the planet
    """

    # zero by default, in case not included
    for e in element_list:
        hf_row[e + '_kg_total'] = 0.0

    # Calculate target elemental inventories from delivery config.
    # CALLIOPE's calc_target_masses computes element budgets (H, C, N, S)
    # from the delivery parameters. This is needed by ALL outgas modules,
    # not just CALLIOPE, because the element budgets drive the mass balance.
    from proteus.outgas.calliope import calc_target_masses

    calc_target_masses(dirs, config, hf_row)

    # Update total mass of tracked elements.
    # #57 Commit D: O is now a first-class element. At this point
    # (during init before any outgas call) `O_kg_total` is zero
    # because the helpfile row was freshly zero-initialised and no
    # outgas step has run yet. Downstream `run_outgassing` calls
    # `populate_O_kg` which fills O_kg_total from the species
    # inventory, and `update_planet_mass` then re-sums M_ele with
    # the correct O contribution. Code that needs the pre-#57
    # semantics (non-O totals) should call
    # `proteus.utils.coupler.M_ele_excl_O(hf_row)`.
    hf_row['M_ele'] = 0.0
    for e in element_list:
        hf_row['M_ele'] += hf_row[e + '_kg_total']


def check_desiccation(config: Config, hf_row: dict) -> bool:
    """
    Check if the planet has desiccated. This is done by checking if all volatile masses
    are below a threshold AND verifying that the loss is consistent with cumulative
    escape.

    Parameters
    ----------
        config : Config
            Configuration object
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only

    Returns
    -------
        bool
            True if desiccation occurred, False otherwise

    Notes
    -----
    The escape-balance gate guards against the failure mode where an
    upstream AGNI or outgas failure wipes the atmosphere as a side
    effect (e.g. NaN volume mixing ratios that get zeroed downstream),
    causing all `*_kg_total` to collapse below `mass_thresh` in a single
    iteration without `esc_kg_cumulative` having moved. The CHILI sweep
    R7/R21 cascades fired this exact pattern: AGNI failed, the
    atmosphere was wiped, and the old check_desiccation reported success
    instead of letting the deadlock detector catch the upstream failure.

    The gate compares (M_vol_initial - current M_ele) against
    `1.5 * esc_kg_cumulative + 1e3 kg`. The 1.5x slack absorbs rounding
    and threshold-truncation noise; the 1e3 kg floor prevents the gate
    from firing on pathologically tiny inventories. If the gate is
    inactive (no baseline tracked yet, e.g. resuming an old CSV without
    `M_vol_initial`), the function falls back to the old threshold-only
    behaviour.
    """

    # Threshold check: refuse desiccation while any non-oxygen element
    # is still above the per-element mass threshold. O is deliberately
    # excluded from the desiccation criterion because a planet with a
    # dry atmosphere (all H / C / N / S escaped) but an oxidised silicate
    # mantle still contains ~O(1e24 kg) of O in silicate bonds — that
    # is NOT "desiccation" in the volatile-inventory sense.
    for e in element_list:
        if e == 'O':
            continue   # see rationale above; O is first-class but not a volatile
        if hf_row[e + '_kg_total'] > config.outgas.mass_thresh:
            log.info('Not desiccated, %s = %.2e kg' % (e, hf_row[e + '_kg_total']))
            return False  # return, and allow run_outgassing to proceed

    # Escape-balance gate. Only enforced when a baseline has been
    # snapshotted by `escape.wrapper.run_escape` (first escape call).
    m_init_raw = hf_row.get('M_vol_initial', None)
    try:
        m_init = float(m_init_raw) if m_init_raw is not None else 0.0
    except (TypeError, ValueError):
        m_init = 0.0
    if not np.isfinite(m_init) or m_init <= 0.0:
        # No baseline -> fall back to old threshold-only behaviour.
        return True

    cur_m_ele = sum(
        float(hf_row.get(f'{e}_kg_total', 0.0))
        for e in element_list
        if e != 'O'
    )
    lost = m_init - cur_m_ele
    esc_cum = float(hf_row.get('esc_kg_cumulative', 0.0))

    # Allow 1.5x scaling slack plus a 1 t absolute floor for noise.
    if lost > 1.5 * esc_cum + 1.0e3:
        log.error(
            'Desiccation check refused: %.2e kg of volatile mass loss is '
            'unexplained by cumulative escape (%.2e kg). Initial=%.2e kg, '
            'current=%.2e kg. This usually indicates an AGNI or outgas-side '
            'failure that wiped the atmosphere as a side effect, not real '
            'escape. The deadlock detector should fire on the next iteration.',
            lost - esc_cum,
            esc_cum,
            m_init,
            cur_m_ele,
        )
        return False

    return True


def run_outgassing(dirs: dict, config: Config, hf_row: dict):
    """
    Run outgassing model to get new volatile surface pressures

    Parameters
    ----------
        dirs : dict
            Dictionary of directory paths
        config : Config
            Configuration object
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
    """

    log.info('Solving outgassing...')

    # Run outgassing calculation
    if config.outgas.module == 'calliope':
        from proteus.outgas.calliope import calc_surface_pressures

        calc_surface_pressures(dirs, config, hf_row)
    elif config.outgas.module == 'atmodeller':
        from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

        calc_surface_pressures_atmodeller(dirs, config, hf_row)
    elif config.outgas.module == 'dummy':
        from proteus.outgas.dummy import calc_surface_pressures_dummy

        calc_surface_pressures_dummy(dirs, config, hf_row)

    # Apply binodal-controlled H2 partitioning.
    # When global_miscibility is enabled, the binodal is handled radially
    # by Zalmoxis (solve_miscible_interior), and the H2 partition was
    # already set during the structure update. Skip the bulk binodal here.
    # When global_miscibility is disabled but h2_binodal is on, use the
    # original bulk binodal override from Rogers+2025.
    if config.interior_struct.zalmoxis.global_miscibility:
        log.debug('Skipping apply_binodal_h2: handled by Zalmoxis (global_miscibility)')
    elif config.outgas.h2_binodal:
        from proteus.outgas.binodal import apply_binodal_h2

        apply_binodal_h2(hf_row, config)

    # calculate total atmosphere mass (from sum of volatile masses)
    hf_row['M_atm'] = 0.0
    for s in gas_list:
        hf_row['M_atm'] += hf_row[s + '_kg_atm']

    # Surface the implicit oxygen inventory computed by CALLIOPE /
    # atmodeller via the fO2 buffer. Pre-#57 PROTEUS never wrote
    # O_kg_{atm,liquid,solid,total} to hf_row, so downstream code had
    # to treat O as a skipped element. After Commit D, O is
    # first-class: this helper aggregates O mass from the
    # {species}_mol_* columns and writes the O_kg_* columns,
    # enabling the `if e == 'O': continue` branches elsewhere to be
    # removed. Static-mode runs preserve physics because the sum
    # follows from CALLIOPE's own internally-consistent partitioning.
    from proteus.utils.coupler import populate_O_kg
    populate_O_kg(hf_row)

    # print outgassed partial pressures (in order of descending abundance)
    mask = [hf_row[s + '_vmr'] for s in gas_list]
    for i in np.argsort(mask)[::-1]:
        s = gas_list[i]
        _p = hf_row[s + '_bar']
        _x = hf_row[s + '_vmr']
        _s = '    %-6s     = %-9.2f bar (%.2e VMR)' % (s, _p, _x)
        if _p > 0.01:
            log.info(_s)
        else:
            # don't spam log with species of negligible abundance
            log.debug(_s)

    # print total pressure and mmw
    log.info('    total      = %-9.2f bar' % hf_row['P_surf'])
    log.info('    mmw        = %-9.5f g mol-1' % (hf_row['atm_kg_per_mol'] * 1e3))


def run_crystallized(config: Config, hf_row: dict):
    """Handle crystallized mantle: no volatile exchange but preserve reservoirs.

    After the mantle solidifies (Phi_global < phi_crit), volatiles can no
    longer exchange between the mantle and atmosphere. Dissolved volatiles
    are trapped in the solid mantle. The atmosphere retains its current
    composition but does not gain or lose volatiles.

    Unlike run_desiccated() which zeros all reservoirs, this function
    preserves them so they continue to influence the structure calculation
    and can be tracked in the helpfile.

    Parameters
    ----------
    config : Config
        Configuration object
    hf_row : dict
        Dictionary of helpfile variables, at this iteration only
    """
    log.info('Crystallized mantle: volatile exchange frozen, reservoirs preserved')
    # No changes to species reservoirs (atm, liquid, solid) — they stay
    # as-is from the last outgas step. The atmosphere module will still
    # compute radiative transfer with the existing composition.
    # Commit D.1 (#57): refresh O_kg_* so that if any external hook
    # mutated species inventories between outgas steps, the O
    # bookkeeping stays consistent.
    from proteus.utils.coupler import populate_O_kg
    populate_O_kg(hf_row)


def run_desiccated(config: Config, hf_row: dict):
    """
    Handle desiccation of the planet. This substitutes for run_outgassing when the planet
    has lost its entire volatile inventory.

    Parameters
    ----------
        config : Config
            Configuration object
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
    """

    # if desiccated, set all gas masses to zero
    log.info('Desiccation has occurred - no volatiles remaining')

    # Do not set these to zero - avoid divide by zero elsewhere in the code
    excepted_keys = ['atm_kg_per_mol']
    for g in gas_list:
        excepted_keys.append(f'{g}_vmr')

    # Set most values to zero
    for k in expected_keys():
        if k not in excepted_keys:
            hf_row[k] = 0.0

    # Commit D.1 (#57): `expected_keys()` excludes `*_kg_total` columns
    # so the zero-loop above leaves `O_kg_total` stale at its last
    # outgas value. Re-populate all O_kg_* explicitly from the now-
    # zeroed species inventory so the reservoir-sum invariant
    # (O_kg_total == O_kg_atm + O_kg_liquid + O_kg_solid == 0) holds.
    from proteus.utils.coupler import populate_O_kg
    populate_O_kg(hf_row)
