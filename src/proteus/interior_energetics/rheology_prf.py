"""Elastic moduli of the crystal-melt mush for the tidal interior profile.

Selected for the Aragog interior (:func:`proteus.interior_energetics.common.rheology_model`);
the other interiors keep the Kervazo et al. (2021) fits. Per layer, from the melt mass
fraction ``phi``, pressure and temperature:

- end members from the PALEOS MgSiO3 unified table (:mod:`rheology_endmembers`): the
  solid at the layer's (P, T) and the melt on the melt side of the melting curve;
- melt volume fraction ``v = phi rho_sol / (phi rho_sol + (1 - phi) rho_liq)``;
- framework collapse at ``v_crit = 1 - y phi_m``, with ``phi_m`` from Mader et al. (2013,
  eq. 49) and ``y`` the framework fraction of the config;
- unrelaxed shear modulus ``mu = mu_HS- + (mu_SCS - mu_HS-) f_net``, clipped to the
  Hashin-Shtrikman bounds with the melt at (``kappa_liq``, ``G_inf``), where ``mu_SCS`` is
  the self-consistent scheme with unrelaxed melt inclusions and ``f_net`` the drained SCS
  shear modulus with its collapse moved to ``v_crit`` (no tail), normalised by ``mu_sol``;
- drained bulk modulus: the same stretched drained SCS, at most the HS upper bound for
  empty pores, and 0 above ``v_crit``;
- grain bulk modulus (Obliqua ``bulk``): ``kappa_sol``;
- bulk viscosity ``zeta = (4/3) eta (1 - v) / v`` (:func:`bulk_viscosity`).

The viscosity is left to the interior module.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.interior_energetics import rheology_endmembers as rem
from proteus.interior_energetics import rheology_tables as rt

if TYPE_CHECKING:
    from proteus.config import Config
    from proteus.interior_energetics.common import Interior_t

log = logging.getLogger('fwl.' + __name__)


def volume_fraction(phi, rho_sol, rho_liq):
    """Melt volume fraction from the melt mass fraction and the phase densities."""
    phi = np.clip(np.asarray(phi, float), 0.0, 1.0)
    return phi * rho_sol / (phi * rho_sol + (1.0 - phi) * rho_liq)


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


V_FLOOR = 1e-15  # melt volume fraction floor of the bulk viscosity (finite at phi = 0)


def bulk_viscosity(eta, v):
    """Bulk (compaction) viscosity [Pa s], ``zeta = (4/3) eta (1 - v) / v``.

    Prud'homme and Bird (1978, J. Non-Newtonian Fluid Mech. 3, 261), as given by Simpson,
    Spiegelman and Weinstein (2010, eq. 11b): a spherical shell of incompressible viscous
    matrix around a spherical cavity, with ``eta`` the matrix shear viscosity and ``v`` the
    cavity (melt) volume fraction. It equals the Hashin-Shtrikman upper bound for empty
    pores in an incompressible matrix through the elastic-viscous correspondence. The melt
    fraction is floored at ``V_FLOOR`` so the value stays finite in solid layers.
    """
    v = np.maximum(np.asarray(v, float), V_FLOOR)
    return 4.0 / 3.0 * np.asarray(eta, float) * (1.0 - v) / v


def critical_volume_fraction(config: Config) -> float:
    """Melt volume fraction at which the crystal framework loses its rigidity."""
    rh = config.interior_energetics.rheology
    phi_m = rt.max_packing_fraction(rh.packing_fraction, rh.packing_width, rh.aspect_ratio)
    return 1.0 - rh.framework_fraction * phi_m


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
    }


def mush_moduli(phi, em: dict, v_crit: float, g_inf: float, tables: rt.ScsTables) -> dict:
    """Unrelaxed shear, drained bulk and grain bulk moduli [Pa] per layer.

    Parameters
    ----------
    phi : array_like
        Melt mass fraction per layer.
    em : dict
        End members per layer (:func:`layer_endmembers`), densities in kg/m3, moduli in Pa.
    v_crit : float
        Melt volume fraction of the framework collapse.
    g_inf : float
        Unrelaxed melt shear modulus [Pa].
    tables : ScsTables
        SCS tables.

    Raises
    ------
    ValueError
        If an end member is not finite or the solid Poisson ratio or modulus ratios fall
        outside the SCS tables.
    """
    k_s, mu_s, k_l = em['kappa_sol'], em['mu_sol'], em['kappa_liq']
    for name in ('rho_sol', 'rho_liq', 'kappa_sol', 'mu_sol', 'kappa_liq'):
        if not np.all(np.isfinite(em[name]) & (em[name] > 0)):
            raise ValueError(f'mush rheology: end member {name} missing or not positive')
    v = volume_fraction(phi, em['rho_sol'], em['rho_liq'])
    nu = rt.poisson_ratio(k_s, mu_s)
    k_frame, mu_frame = tables.framework(v, v_crit, nu)
    kd_hi, mud_hi = hs_void_upper(v, k_s, mu_s)
    f_net = np.clip(np.minimum(mu_frame * k_s, mud_hi) / mu_s, 0.0, 1.0)
    mu_scs = k_s * tables.unrelaxed_mu(v, nu, k_l / k_s, g_inf / k_s)
    lo, hi = hs_shear_bounds(v, k_s, mu_s, k_l, g_inf)
    shear = np.clip(lo + (mu_scs - lo) * f_net, lo, hi)
    return {
        'shear': shear,
        'bulkd': np.minimum(k_frame * k_s, kd_hi),
        'bulk': np.asarray(k_s, float),
    }


def update_mush_rheology(interior_o: Interior_t, config: Config) -> None:
    """Set ``interior_o.shear``, ``bulk`` (grain modulus), ``bulkd``, ``bulk_visc`` and ``bulk_liq``.

    The bulk viscosity uses the interior module's shear viscosity ``interior_o.visc``.
    """
    endmembers = rem.load_endmembers(rem.paleos_mgsio3_path())
    em = layer_endmembers(endmembers, interior_o.pres, interior_o.temp)
    out = mush_moduli(
        interior_o.phi,
        em,
        critical_volume_fraction(config),
        config.interior_energetics.rheology.melt_shear_modulus,
        rt.load_tables(),
    )
    interior_o.shear = out['shear']
    interior_o.bulk = out['bulk']
    interior_o.bulkd = out['bulkd']
    interior_o.bulk_liq = np.asarray(em['kappa_liq'], float)
    v = volume_fraction(interior_o.phi, em['rho_sol'], em['rho_liq'])
    interior_o.bulk_visc = bulk_viscosity(interior_o.visc, v)
