"""Escape through the ZEPHYRUS regime dispatcher.

The dispatcher selects the escape regime of each state (boil-off, a
hydrodynamic wind limited by energy, recombination or photons, hydrostatic
Jeans escape, or Roche-lobe overflow) and returns a rate for each element.
This module builds its inputs from the helpfile row and the atmosphere column,
and writes its verdict back to the row.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from proteus.utils.constants import M_sun, element_list, gas_list, secs_per_year

if TYPE_CHECKING:
    from proteus.atmos_clim.common import Atmos_t
    from proteus.config import Config
    from proteus.config._escape import Zephyrus

log = logging.getLogger('fwl.' + __name__)

# Helpfile code of each regime label. 0 is a row the dispatcher did not set.
REGIME_CODES = {
    'boiloff': 1,
    'hydrodynamic:EL': 2,
    'hydrodynamic:RR': 3,
    'hydrodynamic:PL': 4,
    'hydrostatic': 5,
    'roche_overflow': 6,
}
REGIME_LABELS = {code: label for label, code in REGIME_CODES.items()}

# Helpfile code of where the atmosphere column came from; 0 is a row the dispatcher did not set.
PROFILE_SOLVE = 1  # the most recent converged climate solve
PROFILE_KEPT = 2  # an earlier converged solve, the most recent one being rejected
PROFILE_ISOTHERMAL = 3  # an isothermal column built from the helpfile row

# Smallest interior and bolometric flux passed on [W m-2]. They only cap the
# bolometric rate past the boil-off threshold and feed the overflow diagnostic.
F_FLOOR = 1.0e-10

# Value written for a diagnostic the dispatcher did not evaluate.
NOT_EVALUATED = -1.0


def dispatch_settings(zcfg: Zephyrus):
    """Return the ZEPHYRUS dispatch settings for a ``escape.zephyrus`` block.

    Pressures are converted from bar to Pa. An unset ``tidal`` takes the
    dispatcher default, which is on.
    """
    from zephyrus import DispatchSettings

    return DispatchSettings(
        base_method=zcfg.base_method,
        base_out_of_range=zcfg.base_out_of_range,
        P_photo=float(zcfg.P_photo) * 1e5,
        P_base_fixed=float(zcfg.Pxuv) * 1e5,
        kn_crit=float(zcfg.kn_crit),
        kn_hysteresis=float(zcfg.kn_hysteresis),
        gate=zcfg.gate,
        efficiency=float(zcfg.efficiency),
        efficiency_mode=zcfg.efficiency_mode,
        T_exo_mode=zcfg.T_exo_mode,
        T_exo_value=float(zcfg.T_exo),
        cool_atomic=bool(zcfg.cool_atomic),
        cool_co2_band=bool(zcfg.cool_co2_band),
        cool_o_finestructure=bool(zcfg.cool_o_finestructure),
        cool_recombination=bool(zcfg.cool_recombination),
        fractionate=bool(zcfg.fractionate),
        tidal=True if zcfg.tidal is None else bool(zcfg.tidal),
        photon_limit=bool(zcfg.photon_limit),
        recombination_limit=bool(zcfg.recombination_limit),
        nozzle_temperature=zcfg.nozzle_temperature,
        residual_mode=zcfg.residual_mode,
        lambda_crit=float(zcfg.lambda_crit),
        kzz=float(zcfg.kzz),
        T_anchor_decades=float(zcfg.T_anchor_decades),
        hydrostatic_levels_min=int(zcfg.hydrostatic_levels_min),
        hydrostatic_levels_max=int(zcfg.hydrostatic_levels_max),
        hydrostatic_rtol=float(zcfg.hydrostatic_rtol),
    )


def _mean_particle_mass(vmr: dict, n_levels: int):
    """Mean particle mass per level [kg] from the gas mixing ratios."""
    from zephyrus.composition import species_mass_amu
    from zephyrus.constants import amu

    total = np.zeros(n_levels)
    weighted = np.zeros(n_levels)
    for gas, x in vmr.items():
        total += x
        weighted += x * species_mass_amu(gas) * amu
    return weighted / total


def _unbound_column(M_p, R_p, T, composition, p_surf, p_top, n_levels=20):
    """Isothermal column of an envelope unbound at its surface, out to twice the surface radius.

    Integrates the same hydrostatic balance as the bound stand-in, in closed
    form, ``1/r = 1/R_p - (k T / G M mu) ln(p_surf / p)``, and stops where the
    Jeans parameter has halved (or at ``p_top``). The dispatcher reads the
    surface Jeans parameter, well below the boil-off threshold, and gives a
    boil-off verdict whose rate does not depend on where the column stops.
    """
    from zephyrus.composition import species_mass_amu
    from zephyrus.constants import G, amu, kb

    tot = sum(composition.values())
    mu = sum(x * species_mass_amu(g) for g, x in composition.items()) / tot * amu
    lam = G * M_p * mu / (kb * T * R_p)
    depth = np.linspace(0.0, min(0.5 * lam, math.log(p_surf / p_top)), n_levels)
    r = 1.0 / (1.0 / R_p - kb * T / (G * M_p * mu) * depth)
    vmr = {g: np.full(n_levels, x / tot) for g, x in composition.items()}
    return {'p': p_surf * np.exp(-depth), 'r': r, 'T': np.full(n_levels, float(T)), 'vmr': vmr}


def _isothermal_column(config: Config, hf_row: dict) -> dict | None:
    """Isothermal column at the equilibrium temperature, from the surface to ``p_top``.

    The stand-in for an atmosphere backend with no column of its own, for a
    climate solve that left none (AGNI's transparent mode), and for the first
    escape step of a resumed run. Its temperature is ``T_eqm``, the planet's
    radiative equilibrium, rather than a skin temperature: the stand-in is the
    whole column, not its top. An envelope already unbound at the surface gets
    the short column of :func:`_unbound_column`. None when the surface pressure
    does not exceed ``p_top`` or the atmosphere holds no gas.
    """
    from zephyrus.profiles import isothermal_profile

    composition = {
        g: float(hf_row.get(f'{g}_vmr', 0.0))
        for g in gas_list
        if float(hf_row.get(f'{g}_vmr', 0.0)) > 0.0
    }
    p_surf, p_top = float(hf_row.get('P_surf', 0.0)), float(config.atmos_clim.p_top)
    if not composition or not np.isfinite(p_surf) or p_surf <= p_top:
        return None
    args = (
        float(hf_row['M_planet']),
        float(hf_row['R_int']),
        float(hf_row['T_eqm']),
        composition,
        p_surf * 1e5,
        p_top * 1e5,
    )
    try:
        prof = isothermal_profile(*args)
    except ValueError as exc:
        if 'unbound' not in str(exc):
            raise
        log.warning('Isothermal stand-in unbound at the surface: short column to 2 R_int')
        return _unbound_column(*args)
    return {'p': prof.p, 'r': prof.r, 'T': prof.T, 'vmr': prof.vmr}


def escape_profile(config: Config, hf_row: dict, atmos_o: Atmos_t | None):
    """Return the dispatcher profile and the code of its source.

    Parameters
    ----------
    config : Config
        Run configuration.
    hf_row : dict
        Helpfile row of this iteration.
    atmos_o : Atmos_t or None
        Atmosphere state holding the last converged column, if any.

    Returns
    -------
    profile : zephyrus.profiles.Profile or None
        The column, base to top, in SI, or None when there is no atmosphere
        to build one from.
    source : int
        ``PROFILE_SOLVE``, ``PROFILE_KEPT`` or ``PROFILE_ISOTHERMAL``.
    """
    from zephyrus.profiles import Profile

    column = getattr(atmos_o, 'escape_profile', None) if atmos_o is not None else None
    if column is None:
        column, source = _isothermal_column(config, hf_row), PROFILE_ISOTHERMAL
        if column is None:
            return None, source
    elif atmos_o.escape_profile_current:
        source = PROFILE_SOLVE
    else:
        source = PROFILE_KEPT
    mmw = _mean_particle_mass(column['vmr'], len(column['p']))
    profile = Profile(p=column['p'], r=column['r'], T=column['T'], vmr=column['vmr'], mmw=mmw)
    profile.validate()
    return profile, source


def _finite_or(value, default: float = NOT_EVALUATED) -> float:
    """Return a finite float, or ``default`` for a missing or non-finite value."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def escape_inputs(config: Config, hf_row: dict, profile, dt: float):
    """Build the dispatcher inputs from the helpfile row.

    Parameters
    ----------
    config : Config
        Run configuration.
    hf_row : dict
        Helpfile row of this iteration.
    profile : zephyrus.profiles.Profile
        Atmosphere column, base to top.
    dt : float
        Step length [yr].

    Returns
    -------
    inputs : zephyrus.EscapeInputs
    """
    from zephyrus import EscapeInputs

    code = int(_finite_or(hf_row.get('esc_rate_branch', 0.0), 0.0))
    reservoirs = {e: float(hf_row.get(f'{e}_kg_atm', 0.0)) for e in element_list}
    return EscapeInputs(
        M_p=float(hf_row['M_planet']),
        R_p=float(hf_row['R_int']),
        M_star=float(config.star.mass) * M_sun,
        a=float(hf_row['semimajorax']),
        e=float(hf_row['eccentricity']),
        T_eq=float(hf_row['T_eqm']),
        F_xuv=max(0.0, float(hf_row['F_xuv'])),
        F_bol=max(F_FLOOR, float(hf_row['F_ins'])),
        F_int=max(F_FLOOR, float(hf_row['F_int'])),
        kappa_photo=float(config.escape.zephyrus.kappa_photo),
        profile=profile,
        settings=dispatch_settings(config.escape.zephyrus),
        prev_regime=REGIME_LABELS.get(code),
        atm_converged=None,
        age=float(hf_row['age_star']) * secs_per_year,
        reservoirs={e: m for e, m in reservoirs.items() if m > 0.0},
        dt=float(dt) * secs_per_year,
    )


def store_result(hf_row: dict, result, source: int) -> None:
    """Write a dispatch result to the helpfile row.

    Raises
    ------
    ValueError
        If the dispatcher returns a rate for an element PROTEUS does not track.
    """
    unknown = set(result.per_species) - set(element_list)
    if unknown:
        raise ValueError(f'dispatcher returned rates for untracked elements: {sorted(unknown)}')
    diag = result.diagnostics
    hydrostatic = diag.get('hydrostatic', {})
    base = diag.get('base_level', {})

    hf_row['esc_rate_total'] = float(result.mdot)
    for e in element_list:
        hf_row[f'esc_rate_{e}'] = float(result.per_species.get(e, 0.0))
    hf_row['esc_regime'] = float(REGIME_CODES[result.regime])
    hf_row['esc_rate_branch'] = float(REGIME_CODES[result.rate_branch])
    hf_row['esc_kn_sonic'] = _finite_or(diag.get('knudsen', {}).get('kn_sc'))
    hf_row['esc_lambda'] = _finite_or(diag.get('lambda_gate'))
    hf_row['esc_T_exo'] = _finite_or(hydrostatic.get('T_exo'))
    hf_row['esc_T_esc_neutral'] = _finite_or(hydrostatic.get('T_esc_neutral'))
    hf_row['esc_T_esc_plasma'] = _finite_or(hydrostatic.get('T_esc_plasma'))
    hf_row['esc_f_ion_exo'] = _finite_or(hydrostatic.get('f_plus_exo'))
    hf_row['esc_base_clamp_dex'] = _finite_or(base.get('clamp_decades'), 0.0)
    hf_row['esc_profile_source'] = float(source)
    p_base = _finite_or(base.get('p_Pa'))
    if p_base > 0.0:
        hf_row['p_xuv'] = p_base * 1e-5
        hf_row['R_xuv'] = _finite_or(base.get('r_m'), hf_row.get('R_xuv', 0.0))


def _no_escape(hf_row: dict) -> float:
    """Record a step without escape: zero rates and no dispatcher verdict.

    The diagnostics read as not evaluated and the wind base sits at the
    surface, as under the dummy module, so nothing of an earlier step is left
    beside the zero rate. The regime memory is cleared with them.
    """
    hf_row['esc_rate_total'] = 0.0
    for e in element_list:
        hf_row[f'esc_rate_{e}'] = 0.0
    for key in ('esc_regime', 'esc_rate_branch', 'esc_profile_source', 'esc_base_clamp_dex'):
        hf_row[key] = 0.0
    for key in (
        'esc_kn_sonic',
        'esc_lambda',
        'esc_T_exo',
        'esc_T_esc_neutral',
        'esc_T_esc_plasma',
        'esc_f_ion_exo',
    ):
        hf_row[key] = NOT_EVALUATED
    hf_row['p_xuv'] = float(hf_row.get('P_surf', 0.0))
    hf_row['R_xuv'] = float(hf_row.get('R_int', 0.0))
    return 0.0


def run_dispatcher(
    config: Config, hf_row: dict, atmos_o: Atmos_t | None = None, dt: float = 0.0
) -> float:
    """Run the ZEPHYRUS dispatcher on this iteration's state.

    Parameters
    ----------
    config : Config
        Run configuration.
    hf_row : dict
        Helpfile row of this iteration, updated in place with the bulk and
        per-element rates and the regime diagnostics.
    atmos_o : Atmos_t or None
        Atmosphere state holding the last converged column. Without one, an
        isothermal column stands in.
    dt : float
        Step length [yr].

    Returns
    -------
    mdot : float
        Bulk escape rate [kg s-1].
    """
    from zephyrus import dispatch

    fluxes = {k: float(hf_row.get(k, np.nan)) for k in ('F_xuv', 'F_ins', 'F_int')}
    bad = sorted(k for k, v in fluxes.items() if not np.isfinite(v))
    if bad:
        log.warning('Escape skipped this step: non-finite %s', ', '.join(bad))
        return _no_escape(hf_row)
    profile, source = escape_profile(config, hf_row, atmos_o)
    if profile is None:
        log.info('Escape skipped this step: no atmosphere above p_top to escape from')
        return _no_escape(hf_row)
    inputs = escape_inputs(config, hf_row, profile, dt)
    if source != PROFILE_ISOTHERMAL:
        inputs.atm_converged = source == PROFILE_SOLVE
    result = dispatch(inputs)
    store_result(hf_row, result, source)

    raised = sorted(
        k if v is True else f'{k}={v}'
        for k, v in result.flags.items()
        if v is not False and v is not None
    )
    log.info(
        'Escape regime %s (rate from %s), Kn at the sonic point %.2e',
        result.regime,
        result.rate_branch,
        hf_row['esc_kn_sonic'],
    )
    if raised:
        log.info('Escape flags: %s', ', '.join(raised))
    if source == PROFILE_KEPT:
        log.warning('Escape read the atmosphere column of an earlier converged solve')
    elif source == PROFILE_ISOTHERMAL and config.atmos_clim.module != 'dummy':
        log.warning('Escape read an isothermal stand-in: the climate solve left no column')
    return float(result.mdot)
