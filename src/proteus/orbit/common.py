# Common tides model functions
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import netCDF4 as nc
import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, fftshift

log = logging.getLogger("fwl."+__name__)


@dataclass
class TidalInteraction:
    primary: Any
    perturber: Any

    nmk: Optional[NDArray[np.floating]] = None
    sigma: Optional[NDArray[np.floating]] = None
    LNk: Optional[NDArray[np.floating]] = None


@dataclass
class Tides_t:
    interactions: List[TidalInteraction] = field(default_factory=list)

    def add(self, primary, perturber):
        try:
            return self.get(primary, perturber)
        except KeyError:
            interaction = TidalInteraction(primary, perturber)
            self.interactions.append(interaction)
            return interaction

    def get(self, primary, perturber):
        for interaction in self.interactions:
            if interaction.primary == primary and interaction.perturber == perturber:
                return interaction
        raise KeyError(f"No tidal interaction: {primary} <- {perturber}")

    def add_from_file(self, primary, perturber, file_path: str):
        interaction = self.add(primary, perturber)

        with nc.Dataset(file_path, 'r') as ds:
            n = ds.variables["n"][:]
            m = ds.variables["m"][:]
            k = ds.variables["k"][:]

            interaction.nmk = np.column_stack([n, m, k]).astype(int)
            interaction.sigma = ds.variables['sigma'][:]
            interaction.LNk = (
                ds.variables["LNk_real"][:]
                + 1j * ds.variables["LNk_imag"][:]
            )

        return interaction


def nextpow2_int(x):
    """Return the integer p such that 2^p >= x.

    Attributes
    ----------
    x : int
        Input value.

    Returns
    -------
    p : int
        The smallest integer p such that 2^p >= x.
    """
    return int(np.ceil(np.log2(x))) if x > 0 else 0


def kepler_newton(M, e):
    """
    Solve Kepler's equation E - e*sin(E) = M using Newton iteration.

    Attributes
    ----------
    M : array_like
        Mean anomaly in radians.
    e : float
        Orbital eccentricity (0 <= e < 1).

    Returns
    -------
    E : ndarray
        Eccentric anomaly in radians, same shape as M.
    """
    M = np.array(M, dtype=float)
    E = np.copy(M)

    # Danby-style improved initial guess
    if e > 0:
        E = M + (e * np.sin(M)) / (1 - np.sin(M + e) + np.sin(M))

    # Newton iterations
    for _ in range(10):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        dE = -f / fp
        E += dE
        if np.max(np.abs(dE)) < 1e-13:
            break

    return np.mod(E, 2 * np.pi)


def hansen_fft(n, m, e, kmin, kmax, N=None):
    """Compute Hansen coefficients X_k^{n,m}(e) using FFT on mean anomaly.

    Attributes
    ----------
    n : int
        Degree of the Hansen coefficient.
    m : int
        Order of the Hansen coefficient.
    e : float
        Orbital eccentricity (0 <= e < 1).
    kmin : int
        Minimum k value for which to compute the coefficient.
    kmax : int
        Maximum k value for which to compute the coefficient.
    N : int, optional
        Number of points for FFT. If None, it will be chosen adaptively.

    Returns
    -------
    k : ndarray
        Array of k values from kmin to kmax.
    Xkm : ndarray
        Corresponding Hansen coefficients X_k^{n,m}(e).
    """
    # Choose FFT size adaptively
    if N is None:
        width = max(64, 4 * (kmax - kmin + 1))
        target = width * max(8, int(np.ceil(16 / (1 - e + np.finfo(float).eps))))
        p = max(12, int(np.ceil(np.log2(target))))
        N = 2**p
    else:
        p = nextpow2_int(N)
        N = 2**p

    # Mean anomaly grid
    M = np.arange(N) * (2 * np.pi / N)

    # Solve Kepler
    E = kepler_newton(M, e)

    ce = np.cos(E)
    se = np.sin(E)
    r_over_a = 1 - e * ce
    v = np.arctan2(np.sqrt(1 - e**2) * se, ce - e) # true anomaly

    # Hansen integrand
    f = (r_over_a**n) * np.exp(1j * m * v)

    # FFT, normalized like Python’s fft(f)/N
    F = fftshift(fft(f.astype(complex))) / N

    k_all = np.arange(-N // 2, N // 2)
    mask = (k_all >= kmin) & (k_all <= kmax)

    k = k_all[mask]
    Zk = F[mask]
    Xkm = np.real(Zk)

    return k, Xkm


def get_all_m_hansen(ecc, n, k_min, k_max):
    """
    Computes Hansen coefficients for all m = -n to n.
    Returns a dictionary where keys are m and values are (k, Xkm).

    Attributes
    ----------
    ecc : float
        Orbital eccentricity (0 <= ecc < 1).
    n : int
        Degree of the Hansen coefficient.
    k_min : int
        Minimum k value for which to compute the coefficient.
    k_max : int
        Maximum k value for which to compute the coefficient.

    Returns
    -------
    k_range : ndarray
        Array of k values from k_min to k_max.
    results : dict
        Dictionary where keys are m and values are corresponding Hansen coefficients X_k^{n,m}(ecc).
    """
    results = {}

    # We loop through all required values of m
    for m in range(-n, n + 1):
        # We use the same logic as your original wrapper,
        # but iterate m from -n to n
        k_range, X = hansen_fft(-(n + 1), m, ecc, k_min, k_max, N=2**18)
        results[m] = X

    return k_range, results
