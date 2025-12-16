#PHOENIX spectra utilities
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("fwl."+__name__)

def phoenix_param(x: float | int | str, kind: str) -> str:
    """
    Format a PHOENIX parameter.

    kind:
        "FeH"   -> FeH = 0.0  -> "-0.0" (handles inconsistency in filenames)
        "alpha" -> alpha = 0.0 -> "+0.0"
    """
    x = float(x)

    # zero case
    if abs(x) < 1e-9:
        if kind.lower() == "feh":
            return "-0.0"
        elif kind.lower() == "alpha":
            return "+0.0"

    # Normal case, e.g. +0.5, -1.0
    return f"{x:+0.1f}"

def phoenix_to_grid(*, FeH, alpha, Teff=None, logg=None):
    """
    Map (FeH, alpha, Teff, logg) to the nearest PHOENIX grid point.

    Parameters
    ----------
    FeH : float
        Metallicity [Fe/H].
    alpha : float
        [alpha/M].
    Teff : float, optional
        Effective temperature [K]. If provided, it is snapped to the PHOENIX
        Teff grid.
    logg : float, optional
        Surface gravity log10(g [cgs]). If provided, it is snapped to the
        PHOENIX log g grid.

    Rules
    -----
    - FeH grid: [-4.0, -3.0, -2.0, -1.5, ..., +1.0]
    - alpha grid: -0.2 .. +1.2 in steps of 0.2
    - Teff grid: 2300–7000 (100 K), 7200–12000 (200 K)
    - logg grid: 0.0–6.0 in steps of 0.5

    Alpha is allowed to be nonzero only if:
      - FeH <= 0, and
      - (if Teff is given) 3500 K <= Teff <= 8000 K (using the mapped Teff).

    If Teff is not provided, the Teff condition is ignored and only FeH <= 0
    is enforced.

    Returns
    -------
    dict
        Dictionary with keys: 'Teff', 'logg', 'FeH', 'alpha', where 'Teff' and
        'logg' may be None if not given.
    """
    FeH  = float(FeH)
    alpha = float(alpha)

    # Grids
    FeH_grid   = np.concatenate([[-4., -3.], np.arange(-2., 1. + 1e-6, 0.5)])
    alpha_grid = np.arange(-0.2, 1.2 + 1e-6, 0.2)

    Teff_grid = None
    logg_grid = None

    # Map FeH
    FeH_g = float(FeH_grid[np.abs(FeH_grid - FeH).argmin()])

    # Map Teff if provided
    Teff_g = None
    if Teff is not None:
        Teff = float(Teff)
        Teff_grid = np.concatenate([
            np.arange(2300., 7000. + 1e-6, 100.),
            np.arange(7200., 12000. + 1e-6, 200.),
        ])
        Teff_g = float(Teff_grid[np.abs(Teff_grid - Teff).argmin()])

    # Map logg if provided
    logg_g = None
    if logg is not None:
        logg = float(logg)
        logg_grid = np.arange(0., 6. + 1e-6, 0.5)
        logg_g = float(logg_grid[np.abs(logg_grid - logg).argmin()])

    # Decide whether alpha is allowed to be non-zero
    if Teff_g is not None:
        alpha_allowed = (FeH_g <= 0.0) and (FeH_g > -4.0) and (3500. <= Teff_g <= 8000.)
    else:
        # No Teff info -> only enforce FeH <= 0 and > -4.0
        alpha_allowed = (FeH_g <= 0.0) and (FeH_g > -4.0)

    if alpha_allowed:
        alpha_g = float(alpha_grid[np.abs(alpha_grid - alpha).argmin()])
    else:
        alpha_g = 0.0

        # warn user about alpha override
        if abs(alpha) > 1e-6 and Teff_g is not None:
            log.warning("Requested [alpha/M]=%+.1f is not available for [Fe/H]=%+.1f, Teff=%.0f; using [alpha/M]=0.0 instead.", alpha, FeH_g, Teff_g)
        elif abs(alpha) > 1e-6:
            log.warning("Requested [alpha/M]=%+.1f is not available for [Fe/H]=%+.1f; using [alpha/M]=0.0 instead.", alpha, FeH_g)

    return {
        "Teff":  Teff_g,
        "logg":  logg_g,
        "FeH":   FeH_g,
        "alpha": alpha_g,
    }

def phoenix_filename(Teff: float, logg: float, FeH: float, alpha: float) -> str:
    """
    Build PHOENIX filename like:
    LTE_T02300_logg1.00_FeH+0.5_alpha+0.0_phoenixMedRes_R05000.txt
    """
    Tstr    = f"{int(round(Teff)):05d}"      # e.g. 2300 -> "02300"
    logg_s  = f"{logg:.2f}"                  # "1.00"
    feh_s   = phoenix_param(FeH,   kind="FeH")
    alpha_s = phoenix_param(alpha, kind="alpha")

    return (f"LTE_T{Tstr}_logg{logg_s}_FeH{feh_s}_alpha{alpha_s}_phoenixMedRes_R05000.txt")
