#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate melting curves in pressure–temperature (P–T) and pressure–entropy (P–S)
space for several literature parametrizations of peridotite / silicate melting.

This version is designed to work with the legacy EOS lookup tables:

    - temperature_solid.dat
    - temperature_melt.dat

These tables provide temperature as a function of entropy and pressure,
T(S, P), on structured grids. The script therefore:

1. Builds solidus and liquidus curves in P–T space from literature fits.
2. Converts those curves into P–S space by inverting the EOS relation T(S, P).
3. Resamples the solidus and liquidus entropy curves onto a common pressure grid.
4. Saves both the P–T and P–S versions to disk.

Notes
-----
- Pressure is handled internally in GPa for the melting-curve parametrizations,
  unless a given published fit is explicitly defined in Pa and converted.
- The EOS tables are assumed to have the SPIDER-style format with scaling factors
  in the header.
- The exported entropy files use the same scaled SPIDER-like header so they can
  be re-used by downstream tools expecting that format.

References included here
------------------------
- Andrault et al. (2011), DOI: 10.1016/j.epsl.2011.02.006
- Monteux et al. (2016), DOI: 10.1016/j.epsl.2016.05.010
- Lin et al. (2024), DOI: 10.1038/s41561-024-01495-1
- Hirschmann (2000), DOI: 10.1029/2000GC000070
- Katz et al. (2003), DOI: 10.1029/2002GC000433
- Stixrude (2014), DOI: 10.1098/rsta.2013.0076
- Fei et al. (2021), DOI: 10.1038/s41467-021-21170-y
- Belonoshko et al. (2005), DOI: 10.1103/PhysRevLett.94.195701
- Fiquet et al. (2010), DOI: 10.1126/science.1192448
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d

# =============================================================================
# GENERAL HELPERS
# =============================================================================

def make_pressure_grid(Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 500) -> np.ndarray:
    r"""
    Create a uniformly sampled pressure grid in GPa.

    Parameters
    ----------
    Pmin : float, optional
        Minimum pressure in GPa.
    Pmax : float, optional
        Maximum pressure in GPa.
    n : int, optional
        Number of grid points.

    Returns
    -------
    np.ndarray
        One-dimensional pressure array in GPa.

    Notes
    -----
    This helper is used by all melting parametrizations so that they share
    a consistent pressure sampling unless otherwise specified.
    """
    return np.linspace(Pmin, Pmax, n)


def solidus_from_liquidus_stixrude(T_liq: np.ndarray) -> np.ndarray:
    r"""
    Estimate the solidus from a liquidus using the Stixrude ratio.

    Parameters
    ----------
    T_liq : np.ndarray
        Liquidus temperature in K.

    Returns
    -------
    np.ndarray
        Estimated solidus temperature in K.

    Formula
    -------
    .. math::

        T_{\mathrm{sol}} = \frac{T_{\mathrm{liq}}}{1 - \ln(0.79)}

    Notes
    -----
    This is used in cases where only a liquidus parametrization is available
    and an approximate solidus is needed.

    Reference
    ---------
    Stixrude (2014), DOI: 10.1098/rsta.2013.0076
    """
    return T_liq / (1.0 - np.log(0.79))


def liquidus_from_solidus_stixrude(T_sol: np.ndarray) -> np.ndarray:
    r"""
    Estimate the liquidus from a solidus using the inverse Stixrude ratio.

    Parameters
    ----------
    T_sol : np.ndarray
        Solidus temperature in K.

    Returns
    -------
    np.ndarray
        Estimated liquidus temperature in K.

    Formula
    -------
    .. math::

        T_{\mathrm{liq}} = T_{\mathrm{sol}} \left(1 - \ln(0.79)\right)

    Notes
    -----
    This is used in cases where only a solidus parametrization is available
    and an approximate liquidus is needed.

    Reference
    ---------
    Stixrude (2014), DOI: 10.1098/rsta.2013.0076
    """
    return T_sol * (1.0 - np.log(0.79))


# =============================================================================
# LITERATURE MELTING CURVES
# =============================================================================

def andrault_2011(kind: str = "solidus", Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 500):
    r"""
    Melting curve from Andrault et al. (2011).

    Parameters
    ----------
    kind : {"solidus", "liquidus"}
        Which branch to evaluate.
    Pmin, Pmax : float
        Pressure range in GPa.
    n : int
        Number of pressure samples.

    Returns
    -------
    P : np.ndarray
        Pressure in GPa.
    T : np.ndarray
        Temperature in K.

    Formula
    -------
    The parametrization is written as

    .. math::

        T(P) = T_0 \left( \frac{P}{a} + 1 \right)^{1/c}

    where the published coefficients are defined using pressure in Pa.
    In this implementation the user-facing pressure grid is generated in GPa
    and internally converted to Pa.

    Reference
    ---------
    Andrault et al. (2011), DOI: 10.1016/j.epsl.2011.02.006
    """
    P = make_pressure_grid(Pmin, Pmax, n)   # GPa
    P_pa = P * 1e9                          # Pa

    if kind == "solidus":
        T0, a, c = 2045, 92e9, 1.3
    elif kind == "liquidus":
        T0, a, c = 1940, 29e9, 1.9
    else:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    T = T0 * ((P_pa / a) + 1.0) ** (1.0 / c)
    return P, T


def fei_2021(kind: str = "liquidus", Pmin: float = 1.0, Pmax: float = 1000.0, n: int = 500):
    r"""
    Melting curve based on Fei et al. (2021).

    Parameters
    ----------
    kind : {"solidus", "liquidus"}
        Which branch to evaluate.
    Pmin, Pmax : float
        Pressure range in GPa.
    n : int
        Number of pressure samples.

    Returns
    -------
    P : np.ndarray
        Pressure in GPa.
    T : np.ndarray
        Temperature in K.

    Formula
    -------
    The liquidus is given by

    .. math::

        T_{\mathrm{liq}}(P) = 6000 \left(\frac{P}{140}\right)^{0.26}

    with pressure in GPa.

    If ``kind="solidus"``, the solidus is estimated from the liquidus using
    :func:`solidus_from_liquidus_stixrude`.

    Reference
    ---------
    Fei et al. (2021), DOI: 10.1038/s41467-021-21170-y
    """
    P = make_pressure_grid(Pmin, Pmax, n)
    T_liq = 6000.0 * (P / 140.0) ** 0.26

    if kind == "liquidus":
        T = T_liq
    elif kind == "solidus":
        T = solidus_from_liquidus_stixrude(T_liq)
    else:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    return P, T


def belonoshko_2005(kind: str = "liquidus", Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 500):
    r"""
    Melting curve based on Belonoshko et al. (2005).

    Parameters
    ----------
    kind : {"solidus", "liquidus"}
        Which branch to evaluate.
    Pmin, Pmax : float
        Pressure range in GPa.
    n : int
        Number of pressure samples.

    Returns
    -------
    P : np.ndarray
        Pressure in GPa.
    T : np.ndarray
        Temperature in K.

    Formula
    -------
    The liquidus is given by

    .. math::

        T_{\mathrm{liq}}(P) = 1831 \left(1 + \frac{P}{4.6}\right)^{0.33}

    with pressure in GPa.

    If ``kind="solidus"``, the solidus is estimated from the liquidus using
    :func:`solidus_from_liquidus_stixrude`.

    Reference
    ---------
    Belonoshko et al. (2005), DOI: 10.1103/PhysRevLett.94.195701
    """
    P = make_pressure_grid(Pmin, Pmax, n)
    T_liq = 1831.0 * (1.0 + P / 4.6) ** 0.33

    if kind == "liquidus":
        T = T_liq
    elif kind == "solidus":
        T = solidus_from_liquidus_stixrude(T_liq)
    else:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    return P, T


def fiquet_2010(kind: str = "liquidus", Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 500):
    r"""
    Melting curve based on Fiquet et al. (2010).

    Parameters
    ----------
    kind : {"solidus", "liquidus"}
        Which branch to evaluate.
    Pmin, Pmax : float
        Pressure range in GPa.
    n : int
        Number of pressure samples.

    Returns
    -------
    P : np.ndarray
        Pressure in GPa.
    T : np.ndarray
        Temperature in K.

    Formula
    -------
    The liquidus is implemented as a two-branch fit:

    for :math:`P \leq 20\ \mathrm{GPa}`:

    .. math::

        T_{\mathrm{liq}} = 1982.1 \left(\frac{P}{6.594} + 1\right)^{1/5.374}

    and for :math:`P > 20\ \mathrm{GPa}`:

    .. math::

        T_{\mathrm{liq}} = 78.74 \left(\frac{P}{0.004056} + 1\right)^{1/2.44}

    where pressure is in GPa.

    If ``kind="solidus"``, the solidus is estimated from the liquidus using
    :func:`solidus_from_liquidus_stixrude`.

    Reference
    ---------
    Fiquet et al. (2010), DOI: 10.1126/science.1192448
    """
    P = make_pressure_grid(Pmin, Pmax, n)
    T_liq = np.zeros_like(P, dtype=float)

    low = P <= 20.0
    high = P > 20.0

    T_liq[low] = 1982.1 * ((P[low] / 6.594) + 1.0) ** (1.0 / 5.374)
    T_liq[high] = 78.74 * ((P[high] / 0.004056) + 1.0) ** (1.0 / 2.44)

    if kind == "liquidus":
        T = T_liq
    elif kind == "solidus":
        T = solidus_from_liquidus_stixrude(T_liq)
    else:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    return P, T


def monteux_2016(kind: str = "solidus", Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 500):
    r"""
    Melting curve from Monteux et al. (2016).

    Parameters
    ----------
    kind : {"solidus", "liquidus"}
        Which branch to evaluate.
    Pmin, Pmax : float
        Pressure range in GPa.
    n : int
        Number of pressure samples.

    Returns
    -------
    P : np.ndarray
        Pressure in GPa.
    T : np.ndarray
        Temperature in K.

    Formula
    -------
    Both solidus and liquidus are implemented as piecewise power-law fits:

    .. math::

        T(P) = T_0 \left(\frac{P}{a} + 1\right)^{1/c}

    with one coefficient set below 20 GPa and another above 20 GPa.
    The published coefficients use pressure in Pa, so pressure is converted
    internally from GPa to Pa before evaluation.

    Reference
    ---------
    Monteux et al. (2016), DOI: 10.1016/j.epsl.2016.05.010
    """
    P = make_pressure_grid(Pmin, Pmax, n)   # GPa
    P_pa = P * 1e9                          # Pa

    params = {
        "solidus": {
            "low":  {"T0": 1661.2, "a": 1.336e9,  "c": 7.437},
            "high": {"T0": 2081.8, "a": 101.69e9, "c": 1.226},
        },
        "liquidus": {
            "low":  {"T0": 1982.1, "a": 6.594e9,  "c": 5.374},
            "high": {"T0": 2006.8, "a": 34.65e9,  "c": 1.844},
        }
    }

    if kind not in params:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    p = params[kind]
    T = np.zeros_like(P_pa, dtype=float)

    mask_low = P_pa <= 20e9
    mask_high = P_pa > 20e9

    T[mask_low] = p["low"]["T0"] * ((P_pa[mask_low] / p["low"]["a"]) + 1.0) ** (1.0 / p["low"]["c"])
    T[mask_high] = p["high"]["T0"] * ((P_pa[mask_high] / p["high"]["a"]) + 1.0) ** (1.0 / p["high"]["c"])

    return P, T


def hirschmann_2000(kind: str = "solidus", Pmin: float = 0.0, Pmax: float = 10.0, n: int = 500):
    r"""
    Melting curve from Hirschmann (2000).

    Parameters
    ----------
    kind : {"solidus", "liquidus"}
        Which branch to evaluate.
    Pmin, Pmax : float
        Pressure range in GPa.
    n : int
        Number of pressure samples.

    Returns
    -------
    P : np.ndarray
        Pressure in GPa.
    T : np.ndarray
        Temperature in K.

    Formula
    -------
    The fit is a quadratic polynomial in pressure:

    .. math::

        T_{\mathrm{^\circ C}}(P) = A_1 + A_2 P + A_3 P^2

    The result is then converted to Kelvin via

    .. math::

        T_{\mathrm{K}} = T_{\mathrm{^\circ C}} + 273.15

    This parametrization is only intended for relatively low pressures.

    Reference
    ---------
    Hirschmann (2000), DOI: 10.1029/2000GC000070
    """
    P = make_pressure_grid(Pmin, Pmax, n)

    coeffs = {
        "solidus":  (1085.7, 132.9, -5.1),
        "liquidus": (1475.0, 80.0, -3.2),
    }

    if kind not in coeffs:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    A1, A2, A3 = coeffs[kind]
    T_c = A1 + A2 * P + A3 * P**2
    T = T_c + 273.15

    return P, T


def stixrude_2014(kind: str = "liquidus", Pmin: float = 1.0, Pmax: float = 1000.0, n: int = 500):
    r"""
    Melting curve based on Stixrude (2014).

    Parameters
    ----------
    kind : {"solidus", "liquidus"}
        Which branch to evaluate.
    Pmin, Pmax : float
        Pressure range in GPa.
    n : int
        Number of pressure samples.

    Returns
    -------
    P : np.ndarray
        Pressure in GPa.
    T : np.ndarray
        Temperature in K.

    Formula
    -------
    The liquidus is given by

    .. math::

        T_{\mathrm{liq}}(P) = 5400 \left(\frac{P}{140}\right)^{0.480}

    with pressure in GPa.

    If ``kind="solidus"``, the solidus is estimated from the liquidus using
    :func:`solidus_from_liquidus_stixrude`.

    Reference
    ---------
    Stixrude (2014), DOI: 10.1098/rsta.2013.0076
    """
    P = make_pressure_grid(Pmin, Pmax, n)
    T_liq = 5400.0 * (P / 140.0) ** 0.480

    if kind == "liquidus":
        T = T_liq
    elif kind == "solidus":
        T = solidus_from_liquidus_stixrude(T_liq)
    else:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    return P, T


def wolf_bower_2018(kind: str = "solidus", Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 500):
    r"""
    Piecewise melting curve based on Wolf & Bower (2018) style fits.

    Parameters
    ----------
    kind : {"solidus", "liquidus"}
        Which branch to evaluate.
    Pmin, Pmax : float
        Pressure range in GPa.
    n : int
        Number of pressure samples.

    Returns
    -------
    P : np.ndarray
        Pressure in GPa.
    T : np.ndarray
        Temperature in K.

    Formula
    -------
    The curve is piecewise linear in pressure, with breakpoints
    :math:`P_{\mathrm{cp1}}` and :math:`P_{\mathrm{cp2}}`:

    .. math::

        T(P) =
        \begin{cases}
        s_1 P + c_1, & P < P_{\mathrm{cp1}} \\
        s_2 P + c_2, & P_{\mathrm{cp1}} \leq P < P_{\mathrm{cp2}} \\
        s_3 P + c_3, & P \geq P_{\mathrm{cp2}}
        \end{cases}

    where continuity is enforced through

    .. math::

        c_2 = c_1 + (s_1 - s_2) P_{\mathrm{cp1}}

    .. math::

        c_3 = c_2 + (s_2 - s_3) P_{\mathrm{cp2}}

    Notes
    -----
    This implementation uses coefficients already encoded in the script.

    Reference
    ---------
    Wolf & Bower (2018)-style parametrization used in the model workflow.
    """
    P = make_pressure_grid(Pmin, Pmax, n)

    params = {
        "solidus": (
            7.696777581585296,
            870.4767697319186,
            101.52655163737373,
            15.959022187236807,
            3.090844734784906,
            1417.4258954709148
        ),
        "liquidus": (
            8.864665249317456,
            408.58442302949794,
            46.288444869816615,
            17.549174419770257,
            3.679647802112376,
            2019.967799687511
        )
    }

    if kind not in params:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    cp1, cp2, s1, s2, s3, intercept = params[kind]

    c1 = intercept
    c2 = c1 + (s1 - s2) * cp1
    c3 = c2 + (s2 - s3) * cp2

    T = np.zeros_like(P, dtype=float)

    m1 = P < cp1
    m2 = (P >= cp1) & (P < cp2)
    m3 = P >= cp2

    T[m1] = s1 * P[m1] + c1
    T[m2] = s2 * P[m2] + c2
    T[m3] = s3 * P[m3] + c3

    return P, T


def katz_2003(kind: str = "solidus", X_h2o: float = 30.0, Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 500):
    r"""
    Hydrous melting-curve correction following Katz et al. (2003).

    Parameters
    ----------
    kind : {"solidus", "liquidus"}
        Which branch to evaluate.
    X_h2o : float
        Water content parameter used in the hydrous temperature depression.
    Pmin, Pmax : float
        Pressure range in GPa.
    n : int
        Number of pressure samples.

    Returns
    -------
    P : np.ndarray
        Pressure in GPa.
    T : np.ndarray
        Temperature in K.

    Formula
    -------
    Starting from the anhydrous Wolf & Bower curve, a constant depression is
    applied at all pressures:

    .. math::

        \Delta T = K X_{\mathrm{H_2O}}^{\gamma}

    .. math::

        T = T_{\mathrm{anhydrous}} - \Delta T

    where in this implementation

    .. math::

        \gamma = 0.75,\quad K = 43

    Reference
    ---------
    Katz et al. (2003), DOI: 10.1029/2002GC000433
    """
    gamma = 0.75
    K = 43.0

    if kind not in {"solidus", "liquidus"}:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    P, T_anhydrous = wolf_bower_2018(kind=kind, Pmin=Pmin, Pmax=Pmax, n=n)
    delta_T = K * X_h2o ** gamma
    T = T_anhydrous - delta_T

    return P, T


def lin_2024(kind: str = "solidus", fO2: float = -4.0, Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 500):
    r"""
    Oxygen-fugacity-dependent solidus following Lin et al. (2024).

    Parameters
    ----------
    kind : {"solidus", "liquidus"}
        Which branch to evaluate.
    fO2 : float
        Oxygen fugacity offset parameter used in the solidus shift.
    Pmin, Pmax : float
        Pressure range in GPa.
    n : int
        Number of pressure samples.

    Returns
    -------
    P : np.ndarray
        Pressure in GPa.
    T : np.ndarray
        Temperature in K.

    Formula
    -------
    The anhydrous solidus is first taken from :func:`wolf_bower_2018` and then
    shifted by

    .. math::

        \Delta T = \frac{340}{3.2} (2 - f\mathrm{O}_2)

    so that

    .. math::

        T_{\mathrm{sol}} = T_{\mathrm{anhydrous}} + \Delta T

    If ``kind="liquidus"``, the liquidus is estimated from the shifted solidus
    using :func:`liquidus_from_solidus_stixrude`.

    Reference
    ---------
    Lin et al. (2024), DOI: 10.1038/s41561-024-01495-1
    """
    P, T_anhydrous = wolf_bower_2018(kind="solidus", Pmin=Pmin, Pmax=Pmax, n=n)

    delta_T = (340.0 / 3.2) * (2.0 - fO2)
    T_sol = T_anhydrous + delta_T

    if kind == "solidus":
        T = T_sol
    elif kind == "liquidus":
        T = liquidus_from_solidus_stixrude(T_sol)
    else:
        raise ValueError("kind must be 'solidus' or 'liquidus'")

    return P, T


# =============================================================================
# PHYSICAL-INTERVAL FILTER
# =============================================================================

def truncate_to_physical_interval(func):
    r"""
    Wrap a melting-curve function so only the main interval with
    :math:`T_{\mathrm{sol}} < T_{\mathrm{liq}}` is retained.

    Parameters
    ----------
    func : callable
        A function returning a melting curve in the form ``P, T`` for
        ``kind="solidus"`` or ``kind="liquidus"``.

    Returns
    -------
    callable
        Wrapped function returning only the largest contiguous physically valid
        pressure interval.

    Notes
    -----
    Some parametrizations can cross or become unphysical at high pressure.
    This wrapper evaluates both solidus and liquidus, identifies all points
    where

    .. math::

        T_{\mathrm{sol}} < T_{\mathrm{liq}}

    and keeps only the largest contiguous block of such points.
    """
    def wrapped(kind="solidus", Pmin=0.0, Pmax=1000.0, n=2000, **kwargs):
        P_sol, T_sol = func(kind="solidus", Pmin=Pmin, Pmax=Pmax, n=n, **kwargs)
        P_liq, T_liq = func(kind="liquidus", Pmin=Pmin, Pmax=Pmax, n=n, **kwargs)

        good = T_sol < T_liq
        idx = np.where(good)[0]

        if len(idx) == 0:
            raise ValueError(f"{func.__name__}: no physical interval where solidus < liquidus")

        splits = np.where(np.diff(idx) > 1)[0] + 1
        blocks = np.split(idx, splits)
        main_block = max(blocks, key=len)

        if kind == "solidus":
            return P_sol[main_block], T_sol[main_block]
        elif kind == "liquidus":
            return P_liq[main_block], T_liq[main_block]
        else:
            raise ValueError("kind must be 'solidus' or 'liquidus'")

    return wrapped


# Wrapped versions for models where physical truncation is needed.
andrault_2011_cut = truncate_to_physical_interval(andrault_2011)
monteux_2016_cut = truncate_to_physical_interval(monteux_2016)
wolf_bower_2018_cut = truncate_to_physical_interval(wolf_bower_2018)
katz_2003_cut = truncate_to_physical_interval(katz_2003)


# =============================================================================
# MODEL DISPATCHER
# =============================================================================

def get_melting_curves(model_name: str, Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 2000, **kwargs):
    r"""
    Return solidus and liquidus curves for a given model.

    Parameters
    ----------
    model_name : str
        Identifier of the melting parametrization.
    Pmin, Pmax : float
        Pressure range in GPa.
    n : int
        Number of sampling points.
    **kwargs
        Additional keyword arguments passed to the selected model
        (for example ``X_h2o`` or ``fO2``).

    Returns
    -------
    P_sol, T_sol, P_liq, T_liq : tuple of np.ndarray
        Solidus and liquidus curves in P–T space.

    Raises
    ------
    ValueError
        If the requested model name is unknown.
    """
    models = {
        "andrault_2011": andrault_2011_cut,
        "monteux_2016": monteux_2016_cut,
        "wolf_bower_2018": wolf_bower_2018_cut,
        "katz_2003": katz_2003_cut,
        "fei_2021": fei_2021,
        "belonoshko_2005": belonoshko_2005,
        "fiquet_2010": fiquet_2010,
        "hirschmann_2000": hirschmann_2000,
        "stixrude_2014": stixrude_2014,
        "lin_2024": lin_2024,
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    func = models[model_name]
    P_sol, T_sol = func(kind="solidus", Pmin=Pmin, Pmax=Pmax, n=n, **kwargs)
    P_liq, T_liq = func(kind="liquidus", Pmin=Pmin, Pmax=Pmax, n=n, **kwargs)

    return P_sol, T_sol, P_liq, T_liq


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def save_PT_table(path: Path, P_gpa: np.ndarray, T_k: np.ndarray):
    r"""
    Save a pressure–temperature table to disk.

    Parameters
    ----------
    path : pathlib.Path
        Output filename.
    P_gpa : np.ndarray
        Pressure in GPa.
    T_k : np.ndarray
        Temperature in K.

    Notes
    -----
    The file is written as two columns:

    .. code-block:: text

        pressure temperature

    using a simple header compatible with later inspection.
    """
    data = np.column_stack([P_gpa, T_k])
    np.savetxt(path, data, fmt="%.18e %.18e", header="pressure temperature", comments="#")


# =============================================================================
# EOS LOOKUP TABLE SETTINGS
# =============================================================================
#
# The files temperature_solid.dat and temperature_melt.dat contain structured
# tables in (S, P) space. Each row stores:
#
#   pressure, entropy, temperature
#
# in scaled units. The constants below decode those tables back to SI units.

eos_solid_path = Path("temperature_solid.dat")
eos_liquid_path = Path("temperature_melt.dat")

# Number of pressure points in the EOS tables.
nP = 2020

# Number of entropy points for the solid and liquid tables, respectively.
nS_solid = 125
nS_liquid = 95

# Number of header lines to skip when reading the raw EOS files.
skip_header = 5

# Scaling factors used by the EOS tables.
SCALE_P_EOS = 1e9
SCALE_T_EOS = 1.0
SCALE_S_SOLID_EOS = 4.82426684604467e6
SCALE_S_LIQUID_EOS = 4.805046659407042e6

# Scaling factors used when exporting P–S tables back to SPIDER-like format.
SCALE_P_OUT = 1_000_000_000.0
SCALE_S_OUT = 4_824_266.84604467

# Header used for exported SPIDER-style entropy tables.
COMMON_HEADER = "\n".join([
    "# 5 400",
    "# Pressure, Entropy, Quantity",
    "# column * scaling factor should be SI units",
    "# scaling factors (constant) for each column given on line below",
    "# 1000000000.0 4824266.84604467",
])


def load_eos_T_of_SP(eos_path: Path, nS: int, scale_S_axis: float):
    r"""
    Load an EOS lookup table and build an interpolator for :math:`T(S, P)`.

    Parameters
    ----------
    eos_path : pathlib.Path
        Path to the EOS file.
    nS : int
        Number of entropy grid points in the table.
    scale_S_axis : float
        Scaling factor needed to convert the stored entropy column to SI units.

    Returns
    -------
    S_axis : np.ndarray
        Entropy axis in :math:`\mathrm{J\,kg^{-1}\,K^{-1}}`.
    P_axis_GPa : np.ndarray
        Pressure axis in GPa.
    T_interp : scipy.interpolate.RegularGridInterpolator
        Interpolator returning temperature in K for points in ``(S, P)``.

    Notes
    -----
    The EOS files are stored in a flattened 3-column format and reshaped into

    .. math::

        (n_S, n_P, 3)

    where the three columns correspond to pressure, entropy, and temperature.
    """
    raw = np.genfromtxt(eos_path, skip_header=skip_header)
    resh = raw.reshape((nS, nP, 3))

    P_axis_GPa = resh[0, :, 0] * SCALE_P_EOS / 1e9
    S_axis = resh[:, 0, 1] * scale_S_axis
    T_grid = resh[:, :, 2] * SCALE_T_EOS

    T_interp = RegularGridInterpolator(
        (S_axis, P_axis_GPa),
        T_grid,
        bounds_error=False,
        fill_value=np.nan,
    )
    return S_axis, P_axis_GPa, T_interp


def invert_to_entropy_along_profile(P_gpa: np.ndarray, T_k: np.ndarray, S_axis: np.ndarray, T_of_SP):
    r"""
    Convert a P–T curve into a P–S curve by inverting :math:`T(S, P)`.

    Parameters
    ----------
    P_gpa : np.ndarray
        Pressure profile in GPa.
    T_k : np.ndarray
        Temperature profile in K.
    S_axis : np.ndarray
        Entropy grid used by the EOS table, in
        :math:`\mathrm{J\,kg^{-1}\,K^{-1}}`.
    T_of_SP : callable
        Interpolator returning temperature for input points ``(S, P)``.

    Returns
    -------
    np.ndarray
        Entropy values corresponding to the input P–T profile. Values are set
        to NaN where inversion is not possible.

    Method
    ------
    For each point :math:`(P_i, T_i)` along the melting curve:

    1. Evaluate the EOS table along the full entropy axis at fixed pressure
       :math:`P_i`.
    2. Build a 1D relation :math:`T(S)` at that pressure.
    3. Invert that relation numerically using interpolation to obtain
       :math:`S(T_i)`.

    Notes
    -----
    Repeated temperature values are removed before interpolation. If linear
    inversion fails, a nearest-neighbor fallback is attempted.
    """
    S_out = np.full_like(T_k, np.nan, dtype=float)

    for i, (P_i_gpa, T_i) in enumerate(zip(P_gpa, T_k)):
        P_col = np.full_like(S_axis, P_i_gpa)
        T_vals = T_of_SP(np.column_stack([S_axis, P_col]))

        valid = np.isfinite(T_vals)
        if np.count_nonzero(valid) < 2:
            continue

        Tv = T_vals[valid]
        Sv = S_axis[valid]

        order = np.argsort(Tv)
        T_sorted = Tv[order]
        S_sorted = Sv[order]

        # Remove repeated temperature values before inversion.
        T_unique, idx_unique = np.unique(T_sorted, return_index=True)
        S_unique = S_sorted[idx_unique]

        if len(T_unique) < 2:
            continue

        # Skip points that lie outside the invertible temperature range.
        if T_i < T_unique[0] or T_i > T_unique[-1]:
            continue

        try:
            f = interp1d(T_unique, S_unique, kind="linear", assume_sorted=True)
            S_out[i] = float(f(T_i))
        except Exception:
            try:
                f = interp1d(T_unique, S_unique, kind="nearest", assume_sorted=True)
                S_out[i] = float(f(T_i))
            except Exception:
                continue

    return S_out


def build_common_entropy_grid(P_sol: np.ndarray, S_sol: np.ndarray, P_liq: np.ndarray, S_liq: np.ndarray, n_common: int | None = None):
    r"""
    Resample solidus and liquidus entropy curves onto a shared pressure grid.

    Parameters
    ----------
    P_sol, S_sol : np.ndarray
        Solidus pressure and entropy arrays.
    P_liq, S_liq : np.ndarray
        Liquidus pressure and entropy arrays.
    n_common : int or None, optional
        Number of points in the common pressure grid. If omitted, the smaller
        valid curve length is used.

    Returns
    -------
    P_common, S_sol_common, S_liq_common : tuple of np.ndarray
        Shared pressure grid in GPa, and the solidus/liquidus entropy values
        interpolated onto that grid.

    Notes
    -----
    The common pressure range is defined as the overlap between the valid
    solidus and liquidus pressure intervals:

    .. math::

        P_{\min,\mathrm{common}} = \max(P_{\min,\mathrm{sol}}, P_{\min,\mathrm{liq}})

    .. math::

        P_{\max,\mathrm{common}} = \min(P_{\max,\mathrm{sol}}, P_{\max,\mathrm{liq}})

    After sorting and removing duplicate pressures, both entropy curves are
    linearly interpolated onto the shared pressure array.
    """
    mask_sol = np.isfinite(S_sol)
    mask_liq = np.isfinite(S_liq)

    P_sol_v = np.asarray(P_sol[mask_sol], dtype=float)
    S_sol_v = np.asarray(S_sol[mask_sol], dtype=float)
    P_liq_v = np.asarray(P_liq[mask_liq], dtype=float)
    S_liq_v = np.asarray(S_liq[mask_liq], dtype=float)

    if len(P_sol_v) < 2 or len(P_liq_v) < 2:
        return np.array([]), np.array([]), np.array([])

    Pmin_common = max(np.min(P_sol_v), np.min(P_liq_v))
    Pmax_common = min(np.max(P_sol_v), np.max(P_liq_v))

    if not np.isfinite(Pmin_common) or not np.isfinite(Pmax_common) or Pmax_common <= Pmin_common:
        return np.array([]), np.array([]), np.array([])

    if n_common is None:
        n_common = min(len(P_sol_v), len(P_liq_v))

    order_sol = np.argsort(P_sol_v)
    order_liq = np.argsort(P_liq_v)

    P_sol_s = P_sol_v[order_sol]
    S_sol_s = S_sol_v[order_sol]
    P_liq_s = P_liq_v[order_liq]
    S_liq_s = S_liq_v[order_liq]

    P_sol_u, idx_sol = np.unique(P_sol_s, return_index=True)
    S_sol_u = S_sol_s[idx_sol]

    P_liq_u, idx_liq = np.unique(P_liq_s, return_index=True)
    S_liq_u = S_liq_s[idx_liq]

    if len(P_sol_u) < 2 or len(P_liq_u) < 2:
        return np.array([]), np.array([]), np.array([])

    P_common = np.linspace(Pmin_common, Pmax_common, n_common)

    f_sol = interp1d(P_sol_u, S_sol_u, kind="linear", bounds_error=False, fill_value=np.nan, assume_sorted=True)
    f_liq = interp1d(P_liq_u, S_liq_u, kind="linear", bounds_error=False, fill_value=np.nan, assume_sorted=True)

    S_sol_common = f_sol(P_common)
    S_liq_common = f_liq(P_common)

    mask = np.isfinite(S_sol_common) & np.isfinite(S_liq_common)
    return P_common[mask], S_sol_common[mask], S_liq_common[mask]


def save_entropy_table_with_header(path: Path, P_gpa: np.ndarray, S_jpk: np.ndarray):
    r"""
    Save a pressure–entropy table in SPIDER-style scaled format.

    Parameters
    ----------
    path : pathlib.Path
        Output filename.
    P_gpa : np.ndarray
        Pressure in GPa.
    S_jpk : np.ndarray
        Entropy in :math:`\mathrm{J\,kg^{-1}\,K^{-1}}`.

    Notes
    -----
    The data are rescaled before saving so that the output matches the expected
    SPIDER-style two-column format:

    - pressure divided by ``SCALE_P_OUT``
    - entropy divided by ``SCALE_S_OUT``

    The header also stores the scaling factors, allowing the file to be read
    back into SI units later.
    """
    P_pa = P_gpa * 1e9
    data = np.column_stack([P_pa / SCALE_P_OUT, S_jpk / SCALE_S_OUT])
    np.savetxt(path, data, fmt="%.18e %.18e", header=COMMON_HEADER, comments="")


# Load EOS interpolators once at import time so they can be reused efficiently.
S_axis_solid, P_axis_solid, T_of_SP_solid = load_eos_T_of_SP(
    eos_solid_path, nS_solid, SCALE_S_SOLID_EOS
)

S_axis_liquid, P_axis_liquid, T_of_SP_liquid = load_eos_T_of_SP(
    eos_liquid_path, nS_liquid, SCALE_S_LIQUID_EOS
)


# =============================================================================
# MAIN EXPORTER
# =============================================================================

def export_model_curves(model_name: str, out_root: str = "outputs_entropy_curves",
                        Pmin: float = 0.0, Pmax: float = 1000.0, n: int = 2000, **kwargs):
    r"""
    Export one melting model in both P–T and P–S space.

    Parameters
    ----------
    model_name : str
        Name of the melting parametrization.
    out_root : str, optional
        Root output directory.
    Pmin, Pmax : float, optional
        Pressure range in GPa.
    n : int, optional
        Number of points along the pressure grid.
    **kwargs
        Additional keyword arguments passed to the selected model.

    Returns
    -------
    dict
        Dictionary containing the raw and converted curves.

    Files written
    -------------
    The following files are created inside ``out_root/model_name``:

    - ``solidus_P-T.dat``
    - ``liquidus_P-T.dat``
    - ``solidus_P-S.dat``
    - ``liquidus_P-S.dat``

    Notes
    -----
    The exported P–S files are resampled onto a single common pressure grid so
    that both branches have identical length and aligned pressure sampling.
    """
    out_dir = Path(out_root) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    P_sol, T_sol, P_liq, T_liq = get_melting_curves(
        model_name, Pmin=Pmin, Pmax=Pmax, n=n, **kwargs
    )

    # Save the direct pressure–temperature curves.
    save_PT_table(out_dir / "solidus_P-T.dat", P_sol, T_sol)
    save_PT_table(out_dir / "liquidus_P-T.dat", P_liq, T_liq)

    # Convert both branches from P–T to P–S using the EOS tables.
    S_sol = invert_to_entropy_along_profile(
        P_sol, T_sol, S_axis_solid, T_of_SP_solid
    )
    S_liq = invert_to_entropy_along_profile(
        P_liq, T_liq, S_axis_liquid, T_of_SP_liquid
    )

    # Place both entropy curves on the same pressure grid.
    P_common, S_sol_common, S_liq_common = build_common_entropy_grid(
        P_sol, S_sol, P_liq, S_liq, n_common=n
    )

    # Save the entropy-space curves.
    save_entropy_table_with_header(
        out_dir / "solidus_P-S.dat",
        P_common,
        S_sol_common,
    )

    save_entropy_table_with_header(
        out_dir / "liquidus_P-S.dat",
        P_common,
        S_liq_common,
    )

    return {
        "P_sol": P_sol,
        "T_sol": T_sol,
        "P_liq": P_liq,
        "T_liq": T_liq,
        "S_sol": S_sol,
        "S_liq": S_liq,
        "P_entropy_common": P_common,
        "S_sol_common": S_sol_common,
        "S_liq_common": S_liq_common,
    }


# =============================================================================
# BATCH EXPORTER
# =============================================================================

def export_all_models(out_root: str = "outputs_entropy_curves"):
    r"""
    Export all supported melting parametrizations.

    Parameters
    ----------
    out_root : str, optional
        Root output directory where subdirectories for each model will be made.

    Notes
    -----
    Some models require custom keyword arguments or pressure limits:

    - ``katz_2003`` uses ``X_h2o=30.0``
    - ``lin_2024`` uses ``fO2=-4.0``
    - ``hirschmann_2000`` is truncated at ``Pmax=10.0``
    - ``fei_2021`` and ``stixrude_2014`` start from ``Pmin=1.0``

    The function prints progress messages to standard output.
    """
    models = [
        "andrault_2011",
        "monteux_2016",
        "wolf_bower_2018",
        "katz_2003",
        "fei_2021",
        "belonoshko_2005",
        "fiquet_2010",
        "hirschmann_2000",
        "stixrude_2014",
        "lin_2024",
    ]

    for model in models:
        print(f"[INFO] Exporting {model}")

        if model == "katz_2003":
            export_model_curves(model, out_root=out_root, X_h2o=30.0)
        elif model == "lin_2024":
            export_model_curves(model, out_root=out_root, fO2=-4.0)
        elif model == "hirschmann_2000":
            export_model_curves(model, out_root=out_root, Pmax=10.0)
        elif model == "fei_2021":
            export_model_curves(model, out_root=out_root, Pmin=1.0)
        elif model == "stixrude_2014":
            export_model_curves(model, out_root=out_root, Pmin=1.0)
        else:
            export_model_curves(model, out_root=out_root)

    print("[INFO] Done.")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
#
# To export all models:
#
#     export_all_models()
#
# To export a single model:
#
#     export_model_curves("wolf_bower_2018")
#
# This script currently runs the full export when executed directly.

export_all_models()
