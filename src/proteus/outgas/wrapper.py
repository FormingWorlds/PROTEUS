# Generic outgassing wrapper
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.outgas.common import expected_keys
from proteus.outgas.lavatmos_v2 import compute_silicate_outgassing

#from proteus.outgas.lavatmos import compute_silicate_outgassing
from proteus.utils.constants import element_list, element_mmw, secs_per_year, vap_list, vol_list

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

# Molar mass of FeO [kg/mol] for the FeO_mantle_wt_pct conversion. Atomic
# masses come from utils.constants.element_mmw (IUPAC); FeO = Fe + O.
_M_FeO = element_mmw['Fe'] + element_mmw['O']
# Mass fraction of O in FeO. Approximately 0.2227.
_O_FRAC_OF_FeO = element_mmw['O'] / _M_FeO


def _resolve_oxygen_budget(config: Config, hf_row: dict) -> float | None:
    """Resolve user-supplied IC oxygen budget [kg] under the issue #677 fix.

    Returns the atmospheric+dissolved O mass implied by the user's
    ``planet.elements.O_mode`` and ``O_budget`` settings. This value
    becomes ``hf_row['O_kg_total']`` at IC, before the first outgas call
    runs and equilibrates the gas-phase O against the fO2 buffer.

    Returns ``None`` when ``O_mode == 'ic_chemistry'``, signalling that
    PROTEUS should NOT pre-populate O_kg_total and instead let CALLIOPE
    (or atmodeller) supply the equilibrium-derived value on the first
    outgas call. This is the default and defers the IC O budget to the
    fO2-buffered equilibrium.

    Modes:
        'ppmw' : O_kg = O_budget * 1e-6 * M_reservoir, where M_reservoir
            is ``hf_row['M_mantle']`` or ``hf_row['M_int']`` per
            ``config.planet.volatile_reservoir`` (mirrors H/C/N/S ppmw).
        'kg' : O_kg = O_budget.
        'FeO_mantle_wt_pct' : O_kg = M_mantle * (O_budget / 100) *
            (M_O / M_FeO). The user-facing number is interpreted as an
            alternative unit for the volatile O budget, NOT as a change
            to the mantle composition (PALEOS density tables still assume
            their built-in FeO content). This lets petrologists set the
            O inventory in familiar mantle-FeO wt% terms while keeping
            the EOS untouched.
        'ic_chemistry' : returns None (defer to first outgas call).
    """
    elem = config.planet.elements
    M_mantle = float(hf_row.get('M_mantle', 0.0))
    if config.planet.volatile_reservoir == 'mantle+core':
        M_reservoir = float(hf_row.get('M_int', 0.0))
    else:
        M_reservoir = M_mantle

    match elem.O_mode:
        case 'ppmw':
            return float(elem.O_budget) * 1e-6 * M_reservoir
        case 'kg':
            return float(elem.O_budget)
        case 'FeO_mantle_wt_pct':
            return M_mantle * (float(elem.O_budget) / 100.0) * _O_FRAC_OF_FeO
        case 'ic_chemistry':
            return None
        case _:
            # Unreachable; the in_() validator on Elements.O_mode rejects
            # anything else at config-load time. Guard kept so future
            # additions to the mode set raise a clear error if a wrapper
            # path is forgotten.
            raise ValueError(f"Unknown O_mode '{elem.O_mode}'")



def calc_target_elemental_inventories(dirs: dict, config: Config, hf_row: dict):
    """
    Calculate total amount of volatile elements in the planet.

    Under the issue #677 fix (whole-planet O accounting), this function
    also pre-populates ``hf_row['O_kg_total']`` from the user's O_mode/
    O_budget settings (unless O_mode == 'ic_chemistry', in which case the
    first outgas call sets it). The atmosphere+dissolved O thus enters
    the M_ele aggregation and the Zalmoxis dry-mass subtraction, closing
    the bookkeeping asymmetry that can let M_atm exceed M_planet.
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

    # Pre-populate O_kg_total from user budget (whole-planet O accounting,
    # issue #677). Stashes the user-supplied value as O_kg_user so the IC
    # consistency check can compare it against CALLIOPE's first-call
    # equilibrium output and flag a >50% divergence (see check_ic_oxygen_budget).
    o_kg_user = _resolve_oxygen_budget(config, hf_row)
    if o_kg_user is not None:
        hf_row['O_kg_total'] = o_kg_user
        # Stash the user budget as the one-shot IC-check baseline, but do not
        # re-arm the check once it has fired. check_ic_oxygen_budget resets
        # this to the -1.0 sentinel after it runs; re-stashing on every init
        # iteration would otherwise make the check fire repeatedly.
        if float(hf_row.get('O_kg_user_ic', 0.0)) >= 0.0:
            hf_row['O_kg_user_ic'] = o_kg_user
    else:
        # ic_chemistry mode: leave O_kg_total at 0; first CALLIOPE call writes it.
        hf_row['O_kg_user_ic'] = -1.0  # sentinel: no user budget supplied

    # Update total mass of tracked elements. Issue #677 fix: O is
    # included in M_ele. M_planet = M_int + M_ele
    # therefore reflects the atmospheric+dissolved O mass that CALLIOPE
    # produces from the fO2 buffer, closing the asymmetry. Defensive
    # .get() default lets the sum survive pre-IC hf_row states.
    hf_row['M_ele'] = 0.0
    for e in element_list:
        hf_row['M_ele'] += float(hf_row.get(e + '_kg_total', 0.0))


def check_ic_oxygen_budget(
    config: Config,
    hf_row: dict,
    threshold_frac: float = 0.5,
) -> None:
    """Verify the user-supplied IC oxygen budget against CALLIOPE's chemistry.

    After the first ``run_outgassing`` call at IC, CALLIOPE (or
    atmodeller) has written ``hf_row['O_kg_total']`` from the fO2-
    buffered equilibrium. We compare that against the user-supplied
    O_budget that ``calc_target_elemental_inventories`` stashed earlier
    in ``hf_row['O_kg_user_ic']``. A large divergence usually means
    either:

    1. The user under-specified O_budget so the planet cannot actually
       support the equilibrium atmosphere CALLIOPE wants. The implied
       mantle FeO reservoir is being over-drawn.
    2. The user over-specified O_budget so M_planet now carries phantom
       oxygen mass that has no physical home (not in the atmosphere,
       not in the mantle FeO that PALEOS density accounts for).

    Both cases are best caught loudly at IC rather than silently
    corrupting the trajectory.

    Skipped when:
      - ``planet.fO2_source != "user_constant"``. When
        fO2_source = 'from_O_budget' the user O budget is the
        *authoritative* input that drives the chemistry,
        so any "divergence" between it and the derived O inventory is
        zero by construction. The check is meaningful only when fO2 is
        the input and O is the output.
      - ``O_kg_user_ic`` is the sentinel (-1.0), which marks
        ``O_mode == 'ic_chemistry'`` (the user opted into chemistry-
        derived O so any discrepancy is, by definition, accepted).

    Parameters
    ----------
    config : Config
        Configuration object. Reads ``config.planet.fO2_source`` to
        decide whether the check applies.
    hf_row : dict
        Helpfile row after the first outgas call at IC.
    threshold_frac : float
        Maximum allowed relative divergence between user O_budget and
        CALLIOPE's IC equilibrium value. Default 0.5 (50 percent).

    Raises
    ------
    ValueError
        If divergence exceeds ``threshold_frac``.
    """
    # Gate on fO2_source. The 'from_O_budget' source makes user O
    # authoritative so there is no divergence to flag.
    if config.planet.fO2_source != 'user_constant':
        return

    user_O = float(hf_row.get('O_kg_user_ic', -1.0))
    if user_O < 0.0:
        # Sentinel: O_mode == 'ic_chemistry' or check already fired.
        return

    chem_O = float(hf_row.get('O_kg_total', 0.0))
    if chem_O <= 0.0:
        # Degenerate state; nothing to compare against. Skip.
        return

    rel_div = abs(user_O - chem_O) / chem_O
    log.info(
        'IC oxygen budget check: user=%.3e kg, CALLIOPE-equilibrium=%.3e kg, '
        'relative divergence=%.1f%% (threshold %.0f%%)',
        user_O,
        chem_O,
        rel_div * 100,
        threshold_frac * 100,
    )

    if rel_div > threshold_frac:
        raise ValueError(
            f'IC oxygen budget mismatch (issue #677 consistency check): '
            f'user O_budget implies {user_O:.3e} kg, CALLIOPE equilibrium at IC '
            f'gives {chem_O:.3e} kg (relative divergence {rel_div * 100:.1f}%, '
            f'threshold {threshold_frac * 100:.0f}%). Either: '
            f'(a) adjust planet.elements.O_budget closer to the chemistry value; '
            f'(b) switch to O_mode="ic_chemistry" to defer to CALLIOPE; '
            f'(c) change outgas.fO2_shift_IW to bring chemistry into line with '
            f'the budget. Persistent disagreement usually indicates the implied '
            f'mantle FeO reservoir is being over- or under-drawn.'
        )

    # Mark check as done so subsequent outgas calls in init_stage don't re-fire.
    hf_row['O_kg_user_ic'] = -1.0


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

    # Threshold check: refuse desiccation while ANY element is still above
    # the per-element mass threshold. Issue #677 fix: O is included now.
    # Under D1A the user's choice was to require O_kg_total below threshold
    # as well, on the grounds that an atmosphere with substantial O is not
    # meaningfully "desiccated" even if H/C/N/S are depleted. In practice
    # CALLIOPE drives O_kg_total to near-zero once H/C/N/S vanish, so this
    # change rarely affects the desiccation timing, but it keeps the
    # semantics honest under whole-planet O accounting.
    for e in element_list:
        if float(hf_row.get(e + '_kg_total', 0.0)) > config.outgas.mass_thresh:
            log.info(
                'Not desiccated, %s = %.2e kg' % (e, float(hf_row.get(e + '_kg_total', 0.0)))
            )
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

    # Issue #677 fix: include O in cur_m_ele to match the M_vol_initial
    # baseline (which is now also summed over the full element_list).
    cur_m_ele = sum(float(hf_row.get(f'{e}_kg_total', 0.0)) for e in element_list)
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

    # planet.fO2_source dispatch. Two runtime paths are wired today:
    # 'user_constant' (fO2 buffered to the configured IW offset) for
    # every backend, and 'from_O_budget' (authoritative-O chemistry)
    # for the CALLIOPE
    # and atmodeller backends. The Config-level validator
    # (planet_fO2_source_compat) rejects 'from_mantle_redox' and the
    # from_O_budget + dummy combo at config-load, so this guard is
    # unreachable under a normally-loaded Config. It remains as defence
    # in depth for programmatic Config construction (tests, scripted
    # runs) that bypasses the validators.
    fO2_source = config.planet.fO2_source
    if fO2_source not in ('user_constant', 'from_O_budget'):
        raise NotImplementedError(
            f'planet.fO2_source = "{fO2_source}" is recognised by the '
            'config schema but its runtime path is not yet wired into '
            'run_outgassing.'
        )

    if fO2_source == 'from_O_budget' and config.outgas.module == 'dummy':
        raise NotImplementedError(
            'planet.fO2_source = "from_O_budget" requires chemistry to '
            'invert against; outgas.module = "dummy" has none.'
        )

    log.info('Solving outgassing...')

    if config.outgas.silicates:
        gas_list = vol_list + config.outgas.vaplist
    else:
        gas_list = vol_list + vap_list
    # Default the derived-fO2 helpfile columns to the user-configured
    # buffer offset before the backend dispatch. Each backend overrides
    # the default as appropriate for its chemistry. Under fO2_source =
    # "user_constant" the pre-seeded value is the offset the chemistry
    # equilibrated to (because the fugacity constraint set it); leaving
    # it untouched is the right behaviour. Under fO2_source =
    # "from_O_budget" CALLIOPE and atmodeller overwrite the pre-seed
    # with their solver-derived value. The pre-seed exists primarily so
    # the dummy backend and any solve-skipped path land at the
    # physically meaningful configured value rather than the
    # ZeroHelpfileRow default 0.0, which downstream consumers would
    # otherwise mis-interpret as "the chemistry ran at exactly IW".
    hf_row['fO2_shift_IW_derived'] = float(config.outgas.fO2_shift_IW)
    hf_row['O_res'] = 0.0

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

    log.debug('Outgassing complete, calculating atmospheric composition...')
    log.debug('comparison to H2S output by iterating over gas list')
    log.debug('    %-6s     = %-9.2f bar (%.2e VMR)' % ('H2S', hf_row['H2S_bar'], hf_row['H2S_vmr']))

    # calculate total atmosphere mass from sum of gas species
    hf_row['M_atm'] = 0.0
    for s in gas_list:
        #log.info('species %s'%s)
        #log.info('the mass of this species - if silicate should be zero: %s'%hf_row[s + '_kg_atm'])
    #for s in vol_list:
        hf_row['M_atm'] += hf_row[s + '_kg_atm']

    # Derive element mass ratios in atmosphere
    for e1 in element_list:
        for e2 in element_list:
            key = f'{e2}/{e1}_atm'  # key to be set in helpfile
            if key not in hf_row:
                continue
            if min(hf_row[f'{e1}_kg_atm'], hf_row[f'{e2}_kg_atm']) < 1e-30:
                hf_row[key] = 0.0
            else:
                hf_row[key] = hf_row[f'{e2}_kg_atm'] / hf_row[f'{e1}_kg_atm']

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
        #log.info('mass of this species: %s %4e'%s % hf_row[s + '_kg_atm'])

    # print total pressure and mmw
    log.info('    total      = %-9.2f bar' % hf_row['P_surf'])
    log.info('    mmw        = %-9.5f g mol-1' % (hf_row['atm_kg_per_mol'] * 1e3))


def run_crystallized(config: Config, hf_row: dict, dt: float):
    """Handle a crystallized mantle: volatile exchange frozen, but escape
    still erodes the atmosphere.

    After the mantle solidifies (Phi_global < phi_crit), volatiles can no
    longer exchange between mantle and atmosphere: dissolved volatiles are
    trapped in the solid and outgassing no longer replenishes the atmosphere.

    Escape, however, continues. ``run_escape`` runs earlier in the main loop
    with ``atmosphere_only=True`` in this regime, so it debits the whole-planet
    element totals (``*_kg_total``) proportional to atmospheric abundance. The
    same escaped mass is removed from the atmospheric reservoirs here by scaling
    them with the retained fraction. Because the chemistry is frozen and the
    escape is unfractionated, the scaling is composition-preserving: partial-
    pressure ratios, VMRs, and the mean molecular weight are unchanged. Sizing
    the loss from the atmosphere in both places keeps the per-element
    ``*_kg_total`` and the atmospheric reservoirs mutually consistent for every
    ``escape.reservoir`` setting.

    Parameters
    ----------
    config : Config
        Configuration object. Used to confirm the escape model is unfractionated.
    hf_row : dict
        Dictionary of helpfile variables, at this iteration only.
    dt : float
        Length of the current step [yr], used to size the escaped mass.
    """
    # Uniform scaling is valid only while escape removes atmospheric species in
    # proportion to their abundance. A fractionating model would change the
    # composition and require per-species debiting, so refuse it explicitly at
    # the point that relies on the assumption.

    if config.outgas.silicates:
        gas_list = vol_list + config.outgas.vaplist
        log.info('lavatmos should be running')
    else:
        log.info('lavatmos should not be running')
        gas_list = vol_list + vap_list

    reservoir = getattr(config.escape, 'reservoir', 'outgas')
    if reservoir not in ('outgas', 'bulk'):
        raise NotImplementedError(
            f'run_crystallized assumes unfractionated escape; escape.reservoir='
            f"'{reservoir}' is not supported once the mantle has solidified."
        )

    m_atm = float(hf_row.get('M_atm', 0.0))
    esc_rate = float(hf_row.get('esc_rate_total', 0.0))

    if dt <= 0.0:
        # A non-positive step is a coupling error, not a benign no-op: surface it
        # so a regression that stalls the clock here does not look like "escape
        # silently stopped removing atmosphere".
        log.warning(
            'Crystallized mantle: non-positive dt=%.3e yr; skipping atmospheric '
            'escape debit this step.',
            dt,
        )
        return

    if m_atm <= 0.0 or esc_rate <= 0.0:
        # No atmosphere or no active escape: reservoirs stay as-is.
        log.info('Crystallized mantle: volatile exchange frozen, reservoirs preserved')
        return

    # Mass lost to escape over this step (esc_rate is kg s-1, dt is yr). With
    # the mantle frozen it comes entirely out of the atmosphere.
    esc_step_kg = esc_rate * secs_per_year * dt
    retained = max(0.0, (m_atm - esc_step_kg) / m_atm)

    # Scale the atmospheric reservoirs by the retained fraction. Uniform
    # scaling preserves composition, so `*_vmr` and `atm_kg_per_mol` (mmw)
    # are left unchanged.
    for s in gas_list:
        hf_row[f'{s}_kg_atm'] = hf_row.get(f'{s}_kg_atm', 0.0) * retained
        hf_row[f'{s}_bar'] = hf_row.get(f'{s}_bar', 0.0) * retained
    hf_row['P_surf'] = hf_row.get('P_surf', 0.0) * retained
    hf_row['M_atm'] = m_atm * retained

    log.info(
        'Crystallized mantle: volatile exchange frozen; escape removed '
        '%.3e kg from the atmosphere (retained fraction %.4f)',
        m_atm - hf_row['M_atm'],
        retained,
    )


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

    if config.outgas.silicates:
        gas_list = vol_list + config.outgas.vaplist
        log.info('lavatmos should be running')
    else:
        log.info('lavatmos should not be running')
        gas_list = vol_list + vap_list

    # Do not set these to zero - avoid divide by zero elsewhere in the code
    excepted_keys = ['atm_kg_per_mol']
    for g in gas_list:
        excepted_keys.append(f'{g}_vmr')

    # Set most values to zero
    for k in expected_keys(config):
        if k not in excepted_keys:
            hf_row[k] = 0.0

    if config.outgas.silicates:
        compute_silicate_outgassing(config,hf_row)




def lavatmos_calliope_run(dirs: dict, config: Config, hf_row: dict):
    """function which runs lavatmos and calliope in a loop until they have converged.
    This allows for a consistentt computation of melt outgassing and dissolution
    Parameters
    ----------
        dirs : dict
            Dictionary of directory paths
        config : Config
            Configuration object
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
    """

    gas_list = vol_list + config.outgas.vaplist

    #reset all silicate masses to zero:
    for s in gas_list:
        if s in vol_list:
            continue
        else:
            hf_row[s + '_bar'] =0.0
            hf_row[s + '_vmr']=0.0
            hf_row[s + '_kg_atm']=0.0
            hf_row[s+ '_kg_tot']=0.0

    for e in element_list:
        if e in ['H','C','N','O','S','P']:
            continue
        else:
            hf_row[e + '_kg_atm']=0.0
            hf_row[e+ '_kg_tot']=0.0


    run_outgassing(dirs, config, hf_row)

    if config.outgas.silicates:

        #this needs to be commented out for runninglavatmos with the installation from github
        #lavadir = os.environ.get("LAVATMOS_DIR")
        #if lavadir:
            #log.info('Lavatmos directory found: %s' % lavadir)
        #else:
            #log.warning('Lavatmos directory not found, did you set the LAVATMOS_DIR environment variable?')
        if hf_row['Phi_global'] > 0.00:
            compute_silicate_outgassing(dirs, config, hf_row)
        else:
            log.info('planet has solidified, no silicate outgassing occurs')
