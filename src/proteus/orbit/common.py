# Common tides model functions
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

import netCDF4 as nc
import numpy as np
from numpy.typing import NDArray

log = logging.getLogger("fwl."+__name__)

# Tides structure class
@dataclass
class TidalInteraction:
    primary: Any
    perturber: Any

    nmk: Optional[NDArray[np.floating]] = None
    sigma: Optional[NDArray[np.floating]] = None
    LNk: Optional[NDArray[np.floating]] = None
    Hansen: Optional[NDArray[np.floating]] = None


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
        # Get existing interaction or create a new one
        interaction = self.add(primary, perturber)

        # Open the NetCDF file and extract the arrays
        with nc.Dataset(file_path, 'r') as ds:
            # Using [:] converts the netCDF4 variable directly into a NumPy array
            interaction.nmk = ds.variables['nmk'][:]
            interaction.sigma = ds.variables['sigma'][:]
            interaction.LNk = ds.variables['LNk'][:]
            interaction.Hansen = ds.variables['Hansen'][:]

        return interaction

# Orbit structure class
@dataclass
class Orbit:
    primary: Any      # Central body (e.g. star, planet)
    secondary: Any    # Orbiting body (e.g. planet, moon)

    semimajor_axis: Optional[float] = None
    eccentricity: Optional[float] = None
    inclination: Optional[float] = None
    argument_periapsis: Optional[float] = None
    longitude_ascending_node: Optional[float] = None
    mean_anomaly: Optional[float] = None


@dataclass
class Orbit_t:
    orbits: List[Orbit] = field(default_factory=list)

    def add(self, primary, secondary):
        orbit = Orbit(primary, secondary)
        self.orbits.append(orbit)
        return orbit

    def get(self, primary, secondary):
        for orbit in self.orbits:
            if orbit.primary is primary and orbit.secondary is secondary:
                return orbit
        raise KeyError(f"No orbit found for {secondary} around {primary}")


def read_ncdf_profile(nc_fpath:str):
    """Read data from tides NetCDF output file.

    Automatically reads forcing frequency (σ) and imaginary part of k2 Love number (Imk2) arrays.

    Parameters
    ----------
        nc_fpath : str
            Path to NetCDF file.

    Returns
    ----------
        out : dict
            Dictionary containing numpy arrays of data from the file.
    """

    import netCDF4 as nc

    # open file
    if not os.path.isfile(nc_fpath):
        log.error(f"Could not find NetCDF file '{nc_fpath}'")
        return None
    ds = nc.Dataset(nc_fpath)

    σ    = np.array(ds.variables["σ"][:])
    Imk2 = np.array(ds.variables["Imk2"][:])

    # read data into dictionary values
    out = {}
    out["sigma"] = σ
    out["Imk2"]  = Imk2

    # close file
    ds.close()

    # convert to np arrays
    for key in out.keys():
        out[key] = np.array(out[key], dtype=float)

    return out


def read_orbit_data(output_dir:str, times:list):
    """
    Read all Imk2 spectra from NetCDF files in a PROTEUS output folder.
    """
    profiles = [
        read_ncdf_profile(os.path.join(output_dir, "data", "%.0f_orb.nc"%t))
        for t in times
    ]
    if None in profiles:
        log.warning("One or more NetCDF files could not be found")
        if os.path.exists(os.path.join(output_dir,"data","data.tar")):
            log.warning("You may need to extract archived data files")
        return

    return profiles


def nextpow2_int(x: int) -> int:
    return int(np.ceil(np.log2(x))) if x > 0 else 0


def kepler_newton(M, e: float):
    """
    Solve Kepler's equation E - e*sin(E) = M
    M is expected to be a 2D column vector for broadcasting.
    """
    E = np.copy(M)
    if e > 0:
        E = M + e * np.sin(M) / (1.0 - np.sin(M + e) + np.sin(M))

    for _ in range(10):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = -f / fp
        E += dE

        if np.max(np.abs(dE)) < 1e-13:
            break

    return np.mod(E, 2 * np.pi)


def get_all_hansen(power: int, m_max: int, e: float, kmin: int, kmax: int, N: int = None):
    """
    Computes Hansen coefficients X_k^{power, m}(e) for all m in [-m_max, ..., m_max]
    simultaneously using 2D array broadcasting and a single vectorized FFT.

    Returns:
        k_indices (1D array): The integer k-indices.
        X_dict (dict): A dictionary mapping an integer m to its 1D array of Hansen coefficients.
    """
    if N is None:
        width = max(64, 4 * (kmax - kmin + 1))
        target = width * max(8, int(np.ceil(16.0 / (1.0 - e + np.finfo(float).eps))))
        p = max(12, int(np.ceil(np.log2(target))))
        N = 2**p
    else:
        p = nextpow2_int(N)
        N = 2**p

    # 1. Setup grids: M as a column vector (N, 1) to broadcast against m (1, 2*m_max + 1)
    M = np.arange(N) * (2 * np.pi / N)
    M_col = M[:, np.newaxis]

    # 2. Solve geometry ONCE for all m
    E_col = kepler_newton(M_col, e)
    ce = np.cos(E_col)
    se = np.sin(E_col)

    r_over_a = 1.0 - e * ce
    v_col = np.arctan2(np.sqrt(1.0 - e**2) * se, ce - e)

    # 3. Setup m as a row vector to trigger 2D broadcasting
    m_row = np.arange(-m_max, m_max + 1)[np.newaxis, :]

    # 4. Hansen integrand: (N, 1) * (N, 2*m_max + 1) -> Broadcasts to shape (N, 2*m_max + 1)
    f = (r_over_a ** power) * np.exp(1j * m_row * v_col)

    # 5. 1D FFT over the anomaly axis (axis=0) for all m columns simultaneously
    F = np.fft.fftshift(np.fft.fft(f, axis=0), axes=0) / N

    # 6. Extract the requested k bounds
    k_all = np.arange(-N // 2, N // 2)
    mask = (k_all >= kmin) & (k_all <= kmax)
    k_indices = k_all[mask]

    # 7. Package neatly into a dictionary mapped by m
    X_sliced = np.real(F[mask, :])
    X_dict = {
        m: X_sliced[:, i]
        for i, m in enumerate(range(-m_max, m_max + 1))
    }

    return k_indices, X_dict
