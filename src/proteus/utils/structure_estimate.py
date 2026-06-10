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

from proteus.utils.constants import M_earth, const_G, element_mmw

log = logging.getLogger('fwl.' + __name__)


def nl20_planet_radius_km(x_fe: float, m_ratio: float) -> float:
    """NL20 (2020) Eq. 5 planet radius [km] from iron weight fraction and mass.

    Shared with ``interior_struct.dummy`` so the radius-mode inversion and the
    dummy structure use one definition of the scaling laws.
    """
    return (7030.0 - 1840.0 * x_fe) * m_ratio**0.282


def nl20_core_radius_km(x_cmf: float, m_ratio: float) -> float:
    """NL20 (2020) Eq. 9 core radius (hot profile) [km] from core mass fraction.

    Shared with ``interior_struct.dummy`` (see ``nl20_planet_radius_km``).
    """
    return 4850.0 * x_cmf**0.328 * m_ratio**0.266


def _nl20_radius_fraction(x_cmf: float, m_ratio: float, x_fem: float) -> float:
    """Realized core radius fraction R_c/R_p under NL20 Eqs. 5 and 9.

    The m_ratio exponents do not fully cancel (R_c scales as m^0.266 and R_p as
    m^0.282), so the ratio retains a weak mass dependence; that is why the
    inversion below needs the planet mass.
    """
    x_fe = x_fem + x_cmf * (1.0 - x_fem)
    return nl20_core_radius_km(x_cmf, m_ratio) / nl20_planet_radius_km(x_fe, m_ratio)


def _x_cmf_for_radius_fraction(radius_frac: float, m_ratio: float, x_fem: float) -> float:
    """Invert NL20 for the core mass fraction whose realized R_c/R_p equals the
    requested core radius fraction.

    R_c/R_p increases monotonically with x_cmf (R_c grows via Eq. 9 while R_p
    shrinks because the iron weight fraction rises in Eq. 5), so a bracketed
    bisection converges. A requested fraction outside the achievable NL20 range
    is clamped to the nearest bound.
    """
    lo, hi = 1.0e-4, 0.99
    frac_lo = _nl20_radius_fraction(lo, m_ratio, x_fem)
    frac_hi = _nl20_radius_fraction(hi, m_ratio, x_fem)
    if radius_frac <= frac_lo:
        log.warning(
            'Requested core radius fraction %.3f is below the NL20-achievable '
            'minimum (%.3f); clamping the core mass fraction to %.0e.',
            radius_frac,
            frac_lo,
            lo,
        )
        return lo
    if radius_frac >= frac_hi:
        log.warning(
            'Requested core radius fraction %.3f exceeds the NL20-achievable '
            'maximum (%.3f); clamping the core mass fraction to %.2f. The '
            'realized core is smaller than requested and the mantle is a thin, '
            'low-density shell.',
            radius_frac,
            frac_hi,
            hi,
        )
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _nl20_radius_fraction(mid, m_ratio, x_fem) > radius_frac:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def iron_fractions(
    core_frac: float,
    core_frac_mode: str,
    fe_mantle: float = 0.1,
    mass_tot_M_earth: float | None = None,
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
    mass_tot_M_earth : float or None
        Planet mass in Earth masses. Required in ``'radius'`` mode, where the
        core mass fraction is found so the realized NL20 R_c/R_p matches
        ``core_frac`` (the ratio depends on mass); ignored in ``'mass'`` mode.

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

    # Molar masses from the shared table (kg/mol); x_fem is a ratio, so the
    # unit scale cancels and only the element values matter.
    m_fe = element_mmw['Fe']
    m_mg = element_mmw['Mg']
    m_si = element_mmw['Si']
    m_o = element_mmw['O']

    x_fem = (2 * fe_mantle * m_fe) / (
        2 * ((1 - fe_mantle) * m_mg + fe_mantle * m_fe) + m_si + 4 * m_o
    )

    if core_frac_mode == 'mass':
        x_cmf = core_frac
    else:
        # Choose the core mass fraction so the NL20 core radius fraction
        # R_c/R_p equals the requested core_frac, honouring the user's radial
        # CMF rather than approximating it with a fixed power law. The realized
        # ratio depends on mass, so it must be supplied here.
        if mass_tot_M_earth is None:
            raise ValueError(
                "core_frac_mode='radius' requires mass_tot_M_earth (planet mass "
                'in Earth masses) to invert the NL20 radius relations.'
            )
        x_cmf = _x_cmf_for_radius_fraction(core_frac, mass_tot_M_earth, x_fem)

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

    x_cmf, x_fe, _ = iron_fractions(core_frac, core_frac_mode, fe_mantle, mass_tot_M_earth)
    M_p = mass_tot_M_earth * M_earth
    m_ratio = mass_tot_M_earth

    R_p = nl20_planet_radius_km(x_fe, m_ratio) * 1e3
    R_c = nl20_core_radius_km(x_cmf, m_ratio) * 1e3
    if R_c >= R_p:
        R_c = 0.9 * R_p

    rho_m = (1.0 - x_cmf) * M_p / (4.0 / 3.0 * np.pi * (R_p**3 - R_c**3))
    g_surf = const_G * M_p / R_p**2
    g_cmb = const_G * x_cmf * M_p / R_c**2
    g_m_av = 0.5 * (g_surf + g_cmb)
    P_cmb = g_m_av * rho_m * (R_p - R_c)
    return float(P_cmb)
