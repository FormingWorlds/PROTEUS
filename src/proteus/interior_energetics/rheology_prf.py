"""Elastic-moduli profiles of the partially molten mantle for the tidal interior profile.

Selected by ``interior_energetics.rheology_elastic_prf``
(:meth:`proteus.interior_energetics.common.Interior_t.update_rheology`):

``'kervazo'``
    The fits of Kervazo et al. (2021, A&A 650, A72, eqs. 1-3 and Table 3) to the melt
    fraction ``phi`` alone: shear and bulk modulus, and the viscosity for the dummy
    interior. Bulk viscosity ``eta / phi``.

``'nur'``
    The critical-porosity model of Nur et al. (1998, The Leading Edge 17, 357, p. 361): a
    modified Voigt interpolation that loses its framework rigidity at the configured melt
    *volume* fraction ``nur_critical_melt_volume_fraction``, which is the one parameter of
    the elastic profile. Drained-frame moduli, with the shear modulus of the suspension set
    by the unrelaxed melt (:func:`nur_framework_moduli`); the tidal code combines the
    frame, grain and liquid bulk moduli, and receives the undrained modulus at the melt
    volume fraction (:func:`undrained_bulk_modulus`). Bulk viscosity ``zeta = (4/3) eta (1 - v) / v`` (:func:`bulk_viscosity`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import erf

from proteus.interior_energetics import rheology_endmembers as rem
from proteus.utils.constants import B_ein

if TYPE_CHECKING:
    from proteus.config import Config
    from proteus.interior_energetics.common import Interior_t

log = logging.getLogger('fwl.' + __name__)


V_FLOOR = 1e-15  # melt volume fraction floor of the bulk viscosity (finite at phi = 0)


# ============================================================================
# Kervazo et al. (2021) fits
# ============================================================================


@dataclass
class rheo_t:
    dotl: float
    delta: float
    xi: float
    gamma: float
    phist: float


# Lookup parameters for rheological properties
#     Taken from Kervazo+21 (https://doi.org/10.1051/0004-6361/202039433), Table 3.
#     Their phi_* is the critical crystal fraction of Costa et al. (2009); the law below
#     takes the melt fraction, so phist is the critical melt fraction 1 - phi_*.
par_visc = rheo_t(
    1.0, 25.7, 1.17e-9, 5.0, 1.0 - 0.569
)  # Note liquid viscosity is overwritten to 1 Pa s here, no metter what is specified in the config.
par_shear = rheo_t(
    10.0, 2.10, 7.08e-7, 5.0, 1.0 - 0.597
)  # Note 10 Pa is much lower than 1e10 Pa commenly adopted for the unrelaxed liquid shear modulus.
par_bulk = rheo_t(
    1e9, 2.62, 0.102, 5.0, 1.0 - 0.712
)  # Note the profiles collapse at different critical melt fractions.


# Evaluate big Phi at a given layer
def _bigphi(phi: float, par: rheo_t):
    # Kervazo+21 Eq. 2
    return (1.0 - phi) / (1.0 - par.phist)


# Evaluate big F at a given layer
def _bigf(phi: float, par: rheo_t):
    # Kervazo+21 Eq. 3
    numer = np.pi**0.5 * _bigphi(phi, par) * (1.0 + _bigphi(phi, par) ** par.gamma)
    denom = 2.0 * (1.0 - par.xi)
    return (1.0 - par.xi) * erf(numer / denom)


# Evaluate rheological parameter at a given layer
def eval_rheoparam(phi: float, which: str):
    # Kervazo+21 Eq. 1
    match which:
        case 'visc':
            par = par_visc
        case 'shear':
            par = par_shear
        case 'bulk':
            par = par_bulk
        case _:
            raise ValueError(f"Invalid rheological parameter 'f{which}'")
    # Evaluate parameter
    numer = 1.0 + _bigphi(phi, par) ** par.delta
    denom = (1.0 - _bigf(phi, par)) ** (B_ein * (1 - par.phist))
    return par.dotl * numer / denom


def update_viscosity_kervazo(interior_o: Interior_t) -> None:
    """Set ``interior_o.visc`` from the melt fraction (the dummy interior's viscosity).
    Note: this overwrites the viscosity parameters in the config, see table above.
    """
    for i, p in enumerate(interior_o.phi):
        interior_o.visc[i] = eval_rheoparam(p, 'visc')


def update_rheology_kervazo(interior_o: Interior_t) -> None:
    """Set ``interior_o.shear``, ``bulk`` and ``bulk_visc`` from the melt fraction.

    The fits give no drained or melt bulk modulus, so ``bulkd``, ``bulk_liq`` and
    ``bulk_und`` are None. The bulk viscosity is Kervazo+21 Eq. 4, ``eta / phi`` on the
    current ``interior_o.visc``.
    """
    interior_o.bulkd = None
    interior_o.bulk_liq = None
    interior_o.bulk_und = None
    for i, p in enumerate(interior_o.phi):
        interior_o.shear[i] = eval_rheoparam(p, 'shear')
        interior_o.bulk[i] = eval_rheoparam(p, 'bulk')

    # Kervazo+21 Eq. 4
    interior_o.bulk_visc = np.asarray(interior_o.visc, float) / (
        np.maximum(np.asarray(interior_o.phi, float), V_FLOOR)
    )


# ============================================================================
# common functions of the mush moduli
# ============================================================================


def volume_fraction(phi, rho_sol, rho_liq):
    """Melt volume fraction from the melt mass fraction and the phase densities."""
    phi = np.clip(np.asarray(phi, float), 0.0, 1.0)
    return phi * rho_sol / (phi * rho_sol + (1.0 - phi) * rho_liq)


def undrained_bulk_modulus(bulkd, kappa_sol, kappa_liq, v):
    """Undrained (Biot-Gassmann) bulk modulus [Pa] of a melt-saturated frame.

    ``K_u = K_d + alpha^2 / S`` with ``alpha = 1 - K_d / kappa_sol`` and the storativity
    ``S = v / kappa_liq + (alpha - v) / kappa_sol``, for the melt *volume* fraction ``v``
    (Gassmann 1951).

    Parameters
    ----------
    bulkd, kappa_sol, kappa_liq : array_like
        Drained frame, grain and melt bulk moduli [Pa].
    v : array_like
        Melt volume fraction, in [0, 1].
    """
    kd = np.asarray(bulkd, float)
    ks = np.asarray(kappa_sol, float)
    kl = np.asarray(kappa_liq, float)
    v = np.clip(np.asarray(v, float), 0.0, 1.0)
    alpha = 1.0 - kd / ks
    storage = v / kl + (alpha - v) / ks
    # a frame as stiff as the grain (alpha = 0) has no undrained stiffening: K_u = K_d = kappa_sol
    with np.errstate(divide='ignore', invalid='ignore'):
        ku = kd + np.where(alpha > 0.0, alpha**2 / storage, 0.0)
    return np.asarray(ku, float)


def bulk_viscosity(eta, v):
    """Bulk (compaction) viscosity [Pa s], ``zeta = (4/3) eta (1 - v) / v``.

    Prud'homme and Bird (1978, J. Non-Newtonian Fluid Mech. 3, 261), as given by Simpson,
    Spiegelman and Weinstein (2010, eq. 11b).
    """
    v = np.maximum(np.asarray(v, float), V_FLOOR)
    return 4.0 / 3.0 * np.asarray(eta, float) * (1.0 - v) / v


def hs_shear_bounds(v, kappa_sol, mu_sol, kappa_liq, mu_liq):
    """Hashin-Shtrikman (1963) shear bounds [Pa] for solid with melt fraction ``v``."""
    lo = mu_liq + (1.0 - v) / (
        1.0 / (mu_sol - mu_liq)
        + 6.0
        * (kappa_liq + 2.0 * mu_liq)
        * v
        / (5.0 * mu_liq * (3.0 * kappa_liq + 4.0 * mu_liq))
    )
    hi = mu_sol + v / (
        1.0 / (mu_liq - mu_sol)
        + 6.0
        * (kappa_sol + 2.0 * mu_sol)
        * (1.0 - v)
        / (5.0 * mu_sol * (3.0 * kappa_sol + 4.0 * mu_sol))
    )
    return lo, hi


def hs_void_upper(v, kappa_sol, mu_sol):
    """Hashin-Shtrikman upper bounds [Pa] on the bulk and shear moduli with empty pores."""
    zeta = mu_sol / 6.0 * (9.0 * kappa_sol + 8.0 * mu_sol) / (kappa_sol + 2.0 * mu_sol)
    kappa = 4.0 * kappa_sol * mu_sol * (1.0 - v) / (4.0 * mu_sol + 3.0 * v * kappa_sol)
    return kappa, mu_sol * (1.0 - v) * zeta / (zeta + v * mu_sol)


def layer_endmembers(endmembers: rem.PaleosEndMembers, pres, temp) -> dict:
    """End members per layer: the solid at the layer's (P, T), the melt at the melting curve.

    The crystals of a mush layer sit at the layer temperature, below the PALEOS melting
    temperature; above it the solid side of the melting curve is used. PALEOS has no
    supercooled melt, so the melt properties are those of the melt side of the curve.
    """
    melt = endmembers.melting(pres)
    sub = endmembers.solid(pres, temp)
    return {
        'rho_sol': sub['rho'],
        'kappa_sol': sub['kappa'],
        'mu_sol': sub['mu'],
        'rho_liq': melt['rho_liq'],
        'kappa_liq': melt['kappa_liq'],
        'mu_liq': melt['mu_liq'],
    }


# ============================================================================
# Modified Voigt (nur) for the crystal-melt mush
# ============================================================================


def nur_critical_volume_fraction(config: Config) -> float:
    """Configured critical melt volume fraction for the Nur model.

    The value is deliberately required when ``rheology_elastic_prf='nur'``: a sandstone
    value should not silently be assumed to apply to a crystal-melt mush.
    """
    rh = config.interior_energetics.rheology
    value = getattr(rh, 'nur_critical_melt_volume_fraction', None)
    if value is None:
        raise ValueError(
            'Set interior_energetics.rheology.nur_critical_melt_volume_fraction '
            "(melt volume fraction in (0, 1)) when rheology_elastic_prf='nur'"
        )
    value = float(value)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError('nur_critical_melt_volume_fraction must lie strictly between 0 and 1')
    return value


def _check_nur_inputs(em: dict, v_crit: float) -> float:
    """Validate the critical fraction and the end members of the Nur model."""
    v_crit = float(v_crit)
    if not np.isfinite(v_crit) or not 0.0 < v_crit < 1.0:
        raise ValueError('v_crit must lie strictly between 0 and 1')
    for name in ('rho_sol', 'rho_liq', 'kappa_sol', 'mu_sol', 'kappa_liq', 'mu_liq'):
        if not np.all(np.isfinite(em[name]) & (np.asarray(em[name]) > 0.0)):
            raise ValueError(f'Nur rheology: end member {name} missing or not positive')
    return v_crit


def _nur_shear(v, v_crit: float, em: dict):
    """Shear modulus [Pa] of the critical-porosity model with a finite melt shear modulus."""
    k_s, mu_s, k_l, mu_l = em['kappa_sol'], em['mu_sol'], em['kappa_liq'], em['mu_liq']
    lo, hi = hs_shear_bounds(v, k_s, mu_s, k_l, mu_l)
    mu_c, _ = hs_shear_bounds(v_crit, k_s, mu_s, k_l, mu_l)
    x = np.clip(v / v_crit, 0.0, 1.0)
    below = (1.0 - x) * mu_s + x * mu_c
    return np.clip(np.where(v < v_crit, below, lo), lo, hi)


def nur_effective_moduli(phi, em: dict, v_crit: float) -> dict:
    """Saturated effective moduli of the modified Voigt bound of Nur et al. (1998, p. 361)."""
    v_crit = _check_nur_inputs(em, v_crit)
    k_s, k_l = em['kappa_sol'], em['kappa_liq']
    v = volume_fraction(phi, em['rho_sol'], em['rho_liq'])
    k_at_crit = 1.0 / ((1.0 - v_crit) / k_s + v_crit / k_l)
    x = np.clip(v / v_crit, 0.0, 1.0)
    k_below = (1.0 - x) * k_s + x * k_at_crit
    k_reuss = 1.0 / ((1.0 - v) / k_s + v / k_l)
    k_eff = np.where(v <= v_crit, k_below, k_reuss)
    return {
        'bulk': np.asarray(k_eff, float),
        'shear': np.asarray(_nur_shear(v, v_crit, em), float),
    }


def nur_framework_moduli(phi, em: dict, v_crit: float) -> dict:
    """Nur critical-porosity model as drained-frame inputs for the tidal code."""
    v_crit = _check_nur_inputs(em, v_crit)
    v = volume_fraction(phi, em['rho_sol'], em['rho_liq'])
    frame = np.maximum(1.0 - v / v_crit, 0.0)
    # Avoid a tiny positive frame modulus when round-off leaves v infinitesimally below v_c.
    frame = np.where(v >= v_crit * (1.0 - 1e-12), 0.0, frame)
    k_hi, _ = hs_void_upper(v, em['kappa_sol'], em['mu_sol'])
    return {
        'shear': np.asarray(_nur_shear(v, v_crit, em), float),
        'bulkd': np.asarray(np.minimum(em['kappa_sol'] * frame, k_hi), float),
        'bulk': np.asarray(em['kappa_sol'], float),
    }


def update_rheology_nur(interior_o: Interior_t, config: Config) -> None:
    """Set moduli with the Nur critical-porosity framework model.

    The viscosity itself remains supplied by the selected interior viscosity law. The
    resulting ``bulk_visc`` is computed from that viscosity
    and the local melt volume fraction.
    """
    endmembers = rem.load_endmembers(
        rem.paleos_mgsio3_path(), config.interior_energetics.rheology.melt_shear_modulus
    )
    em = layer_endmembers(endmembers, interior_o.pres, interior_o.temp)
    out = nur_framework_moduli(interior_o.phi, em, nur_critical_volume_fraction(config))
    interior_o.shear = out['shear']
    interior_o.bulk = out['bulk']
    interior_o.bulkd = out['bulkd']
    interior_o.bulk_liq = np.asarray(em['kappa_liq'], float)
    v = volume_fraction(interior_o.phi, em['rho_sol'], em['rho_liq'])
    interior_o.bulk_und = undrained_bulk_modulus(out['bulkd'], out['bulk'], em['kappa_liq'], v)
    interior_o.bulk_visc = bulk_viscosity(interior_o.visc, v)
