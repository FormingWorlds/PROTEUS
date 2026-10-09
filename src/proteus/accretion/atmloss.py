# Atmospheric loss calculations and logging for giant impact accretion
from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

from proteus.utils.helper import element_list

if TYPE_CHECKING:
    from proteus.accretion.common import ImpactEvent
    from proteus.config import Config

log = logging.getLogger('fwl.proteus.accretion.wrapper')

# Atmospheric mass threshold (fraction of planet mass) above which a warning is logged
# for the thin-atmosphere Kegerreis et al. (2020) scaling law.
_ATMLOSS_THIN_ATM_WARN = 0.03


def _as_float(val: object) -> float:
    """Convert value to float, returning NaN on ValueError or TypeError."""
    try:
        return float(val)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return float('nan')


def _format_roche_flag(
    name: str,
    diagnostics: dict,
    fitted_range: Mapping[str, tuple[float, float]],
    has_range_flag: bool = True,
) -> str:
    """Format one out-of-range flag, with the clamp value when one is active.

    Parameters
    ----------
    name : str
        Flag name ('f_atm', 'M_t_earth', 'gamma', 'b', 'R_ratio', 'v_ratio',
        'v_sub_escape', or 'X_FF_zero_energy').
    diagnostics : dict
        Diagnostics dictionary from ``ImpactLossResult`` containing physical
        parameter values and any clamped bounds under ``'clamped'``.
    fitted_range : Mapping[str, tuple[float, float]]
        Mapping of parameter names to fitted (lo, hi) range tuples.
    has_range_flag : bool, default=True
        Whether an out-of-range parameter flag, a clamp, or an f_atm flag is present.
        When False (inside the fitted range), the zero-energy far-field flag omits
        the extrapolation note.

    Returns
    -------
    str
        Formatted string describing parameter value, fitted range, and any clamp.
    """
    if name == 'v_sub_escape':
        val = float(diagnostics.get('v_ratio', 0.0))
        return f'v_sub_escape: near-field loss held at v_c = v_esc (v_ratio = {val:.3g})'
    if name == 'X_FF_zero_energy':
        val = float(diagnostics.get('X_FF_zero_energy', 0.0))
        extrap_text = ' (log10 f_atm extrapolation)' if has_range_flag else ''
        return f'far-field loss of {val:.3g} without impact energy{extrap_text}'
    if name not in fitted_range:
        return name
    lo, hi = fitted_range[name]
    val = float(diagnostics[name])
    msg = f'{name} = {val:.3g} (fitted {lo:g} to {hi:g})'
    clamped = diagnostics.get('clamped', {})
    if name in clamped:
        msg += f', evaluated at {clamped[name]:g}'
    return msg


def _zephyrus_loss_fraction(config: Config, hf_row: dict, event: ImpactEvent) -> float:
    """Evaluate giant-impact atmosphere loss using the zephyrus scaling laws.

    For ``roche2026``, the target mass passed to ZEPHYRUS is the refractory
    mass ``event.M_target_before * (1 - f_atm)``; ``kegerreis2020`` receives
    the total event mass directly. Kegerreis et al. (2020) take the radii at
    the base of the atmosphere, the bulk densities without it, and name their
    scenarios by atmosphere-free masses; the total event mass changes X by a
    relative amount of order f_atm.

    Parameters
    ----------
    config : Config
        Model configuration; reads ``accretion.atmloss_law``.
    hf_row : dict
        Current helpfile row, supplying planet mass and volatile budgets.
    event : ImpactEvent
        The impact being applied, supplying collision parameters.

    Returns
    -------
    float
        Eroded atmosphere loss fraction.

    Raises
    ------
    ImportError
        If ``zephyrus.collision.impact_loss`` is not available.
    ValueError
        If ``accretion.atmloss_law`` is unknown, or if ``roche2026`` is
        selected and ``M_planet`` in ``hf_row`` is missing, non-finite,
        or non-positive, or if atmosphere masses yield negative, non-finite,
        or ``f_atm >= 1``.
    """
    try:
        from zephyrus.collision import ROCHE2026_FITTED_RANGE, impact_loss
    except ImportError as exc:
        raise ImportError(
            "accretion.atmloss_module = 'zephyrus' needs a fwl-zephyrus "
            'installation that provides zephyrus.collision.impact_loss; '
            'upgrade fwl-zephyrus'
        ) from exc

    law = config.accretion.atmloss_law
    m_planet = _as_float(hf_row.get('M_planet'))
    m_atm = sum(_as_float(hf_row.get(f'{e}_kg_atm', 0.0)) for e in element_list)
    has_valid_m_planet = 0.0 < m_planet < math.inf
    f_atm = m_atm / m_planet if has_valid_m_planet else 0.0

    match law:
        case 'roche2026':
            if not has_valid_m_planet:
                raise ValueError(
                    "accretion.atmloss_law = 'roche2026' requires a finite, positive "
                    f'M_planet in hf_row, got {hf_row.get("M_planet")!r}'
                )
            if not 0.0 <= m_atm < math.inf:
                raise ValueError(
                    "accretion.atmloss_law = 'roche2026' requires finite, non-negative "
                    f"atmosphere mass from '<e>_kg_atm' in hf_row, got {m_atm!r}"
                )
            if f_atm >= 1.0:
                raise ValueError(
                    "accretion.atmloss_law = 'roche2026' requires atmosphere mass "
                    f"fraction f_atm < 1 from '<e>_kg_atm' and 'M_planet', got {f_atm!r}"
                )
            m_target = event.M_target_before * (1.0 - f_atm)
        case 'kegerreis2020':
            m_target = event.M_target_before
        case _:
            raise ValueError(f"Unknown accretion.atmloss_law: '{law}'")

    result = impact_loss(
        law=law,
        v_c=event.v_impact,
        M_i=event.M_impactor,
        M_t=m_target,
        R_i=event.R_impactor,
        R_t=event.R_target_before,
        b=event.impact_parameter,
        rho_i=event.rho_impactor,
        rho_t=event.rho_target,
        f_atm=f_atm,
    )

    f_loss = float(result.fraction)
    _log_zephyrus_loss(law, event, result, f_atm, ROCHE2026_FITTED_RANGE)
    return f_loss


def _log_zephyrus_loss(
    law: str,
    event: ImpactEvent,
    result,
    f_atm: float,
    fitted_range: Mapping[str, tuple[float, float]],
) -> None:
    """Log warnings and diagnostics for zephyrus impact erosion laws.

    Parameters
    ----------
    law : str
        Selected erosion scaling law name.
    event : ImpactEvent
        The impact event being applied.
    result : ImpactLossResult
        Loss result object containing fraction, diagnostics, and flags.
    f_atm : float
        Target atmospheric mass fraction m_atm / M_planet.
    fitted_range : Mapping[str, tuple[float, float]]
        Parameter bounds for Roche et al. (2026) out-of-range checks.
    """
    f_loss = float(result.fraction)
    diagnostics = result.diagnostics
    v_ratio = float(diagnostics['v_ratio'])
    gamma = float(diagnostics['gamma'])

    if law == 'kegerreis2020':
        if f_atm > _ATMLOSS_THIN_ATM_WARN:
            log.warning(
                '    the atmosphere is %.1f%% of the planet mass, beyond the '
                'thin-atmosphere regime (about 1%%) the impact erosion law is '
                'fitted for; the eroded fraction is extrapolated',
                100.0 * f_atm,
            )
        log.info(
            '    impact erosion law: %s, v_ratio=%.3f, gamma=%.3f, loss fraction=%.3f',
            law,
            v_ratio,
            gamma,
            f_loss,
        )
    elif law == 'roche2026':
        clamped = diagnostics.get('clamped') or {}
        names = list(result.flags)
        for k in clamped:
            if k not in names:
                names.append(k)
        if names:
            has_range_flag = any(
                f in fitted_range or f == 'v_sub_escape' for f in names
            ) or bool(clamped)
            flag_msgs = [
                _format_roche_flag(f, diagnostics, fitted_range, has_range_flag=has_range_flag)
                for f in names
            ]
            has_clamp_in_msg = any('evaluated at' in m for m in flag_msgs)
            clamp_tail = (
                "; only the fit terms use the value marked 'evaluated at', "
                "and v_esc, Q'_R, and the mass ratio use the raw collision state"
                if has_clamp_in_msg
                else ''
            )
            has_extrapolated = any(f in fitted_range or f == 'v_sub_escape' for f in names)
            extrap_tail = '; the loss fraction is extrapolated' if has_extrapolated else ''
            header = (
                'Roche et al. (2026) law outside its fitted range: '
                if has_range_flag
                else 'Roche et al. (2026) law: '
            )
            log.warning(
                '    impact at t = %.4e yr: %s%s%s%s',
                event.time,
                header,
                '; '.join(flag_msgs),
                extrap_tail,
                clamp_tail,
            )
        x_nf = float(diagnostics['X_NF'])
        x_ff = float(diagnostics['X_FF'])
        log.info(
            '    impact erosion law: %s, f_atm=%.4e, v_ratio=%.3f, gamma=%.3f, '
            'loss fraction=%.3f (X_NF=%.3f, X_FF=%.3f)',
            law,
            f_atm,
            v_ratio,
            gamma,
            f_loss,
            x_nf,
            x_ff,
        )
