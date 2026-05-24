"""Analytical structure-estimate helpers for IC fallback paths.

Wraps the Noack & Lasbleis (2020) scaling laws for rocky planets so the
zero-iteration / first-call paths in ``proteus.interior_struct.zalmoxis``
and ``proteus.interior_energetics.common`` can produce mass-aware
estimates of P_cmb without running the full Zalmoxis structure solver.

Reference: Noack & Lasbleis (2020), A&A 638, A129.
    "Parameterisations of interior properties of rocky planets"
    https://doi.org/10.1051/0004-6361/202037723

Only the P_cmb estimate is exposed here; the full radial profiles still
live in ``proteus.interior_struct.dummy`` because they're only consumed
by the dummy structure module.
"""

from __future__ import annotations

import logging

import numpy as np

from proteus.utils.constants import M_earth, const_G

log = logging.getLogger('fwl.' + __name__)


def iron_fractions(
    core_frac: float,
    core_frac_mode: str,
    fe_mantle: float = 0.1,
) -> tuple[float, float, float]:
    """Compute iron fractions from core fraction and mantle iron number.

    Single source of truth used by ``interior_struct.dummy.solve_dummy_structure``
    and the iter-0 P_cmb fallback path.

    Parameters
    ----------
    core_frac : float
        Core fraction (mass or radius, per ``core_frac_mode``). Must be
        in (0, 1); validated by attrs in ``config/_struct.py``.
    core_frac_mode : str
        Either ``'mass'`` or ``'radius'``.
    fe_mantle : float
        Mantle iron number ``#Fe_M`` (fraction of iron-bearing minerals).
        Default 0.1.

    Returns
    -------
    x_cmf : float
        Core mass fraction.
    x_fe : float
        Total planet iron weight fraction.
    x_fem : float
        Iron mass fraction in the mantle.

    Raises
    ------
    ValueError
        If ``core_frac`` is not in (0, 1) or ``core_frac_mode`` is unknown.
    """
    if not 0.0 < core_frac < 1.0:
        raise ValueError(f'core_frac must be in (0, 1); got {core_frac!r}')
    if core_frac_mode not in ('mass', 'radius'):
        raise ValueError(f"core_frac_mode must be 'mass' or 'radius'; got {core_frac_mode!r}")

    m_fe = 55.845
    m_mg = 24.305
    m_si = 28.0855
    m_o = 15.999

    x_fem = (2 * fe_mantle * m_fe) / (
        2 * ((1 - fe_mantle) * m_mg + fe_mantle * m_fe) + m_si + 4 * m_o
    )

    if core_frac_mode == 'mass':
        x_cmf = core_frac
    else:
        x_cmf = core_frac**2.5
        x_cmf = max(0.01, min(x_cmf, 0.80))

    x_fe = x_fem + x_cmf * (1 - x_fem)
    return x_cmf, x_fe, x_fem


def estimate_P_cmb_NL20(
    mass_tot_M_earth: float,
    core_frac: float,
    core_frac_mode: str,
    fe_mantle: float = 0.1,
) -> float:
    """Estimate the core-mantle boundary pressure for a rocky planet.

    Used as the iter-0 fallback for the ``liquidus_super`` IC mode and
    for the structure-side T_cmb anchor when ``hf_row['P_cmb']`` is not
    yet populated. Scales correctly with mass; replaces the older
    hardcoded 135 GPa Earth-only fallback that produced ~1500-2000 K
    iter-0 T_cmb anchor offsets for 3-10 M_Earth planets.

    Implements Noack & Lasbleis (2020), A&A 638, A129, Eqs. 5, 9, 12,
    13, 14, 15, 16. The 1 M_Earth/CMF=0.325 default returns ~125 GPa,
    consistent with PREM (Dziewonski & Anderson 1981).

    Parameters
    ----------
    mass_tot_M_earth : float
        Total planet mass in Earth masses. Must be > 0.
    core_frac : float
        Core fraction (mass or radius, per ``core_frac_mode``).
    core_frac_mode : str
        Either ``'mass'`` or ``'radius'``.
    fe_mantle : float
        Mantle iron number. Default 0.1.

    Returns
    -------
    P_cmb : float
        Pressure at the core-mantle boundary [Pa].

    Raises
    ------
    ValueError
        On non-positive mass or invalid ``core_frac`` / ``core_frac_mode``.
    """
    if mass_tot_M_earth <= 0:
        raise ValueError(f'mass_tot_M_earth must be > 0; got {mass_tot_M_earth!r}')

    x_cmf, x_fe, _ = iron_fractions(core_frac, core_frac_mode, fe_mantle)
    M_p = mass_tot_M_earth * M_earth
    m_ratio = mass_tot_M_earth

    R_p = (7030.0 - 1840.0 * x_fe) * m_ratio**0.282 * 1e3
    R_c = 4850.0 * x_cmf**0.328 * m_ratio**0.266 * 1e3
    if R_c >= R_p:
        R_c = 0.9 * R_p

    rho_m = (1.0 - x_cmf) * M_p / (4.0 / 3.0 * np.pi * (R_p**3 - R_c**3))
    g_surf = const_G * M_p / R_p**2
    g_cmb = const_G * x_cmf * M_p / R_c**2
    g_m_av = 0.5 * (g_surf + g_cmb)
    P_cmb = g_m_av * rho_m * (R_p - R_c)
    return float(P_cmb)
