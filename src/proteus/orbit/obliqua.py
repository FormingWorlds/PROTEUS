# Obliqua tidal heating module
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import juliacall
import netCDF4 as nc
import numpy as np
from juliacall import Main as jl

from proteus.interior_energetics.common import Interior_t
from proteus.orbit.common import Tides_t
from proteus.utils.helper import UpdateStatusfile

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def import_obliqua():
    log.debug("Import Obliqua...")
    jl.seval("using Obliqua")


def to_julia_dict(obj):
    """Recursively convert Python dict/list to native Julia Dict/Vector."""
    if isinstance(obj, dict):
        jd = jl.Dict()
        for k, v in obj.items():
            jd[k] = to_julia_dict(v)
        return jd
    elif isinstance(obj, list):
        return [to_julia_dict(v) for v in obj]
    else:
        return obj


def _jlarr(arr: np.ndarray):
    # Make copy of array, reverse order, and convert to Julia type
    cop = np.array(arr, copy=True, dtype=float).flatten()
    return juliacall.convert(jl.Array[jl.Obliqua.prec, 1], cop)


def _jlsca_float(sca: float):
    # Make a copy of a scalar, and convert to Julia type
    return juliacall.convert(jl.Obliqua.Float64, sca)


def _jlsca_prec(sca: float):
    # Make a copy of a scalar, and convert to Julia type
    return juliacall.convert(jl.Obliqua.prec, sca)


def run_obliqua(hf_row: dict, dirs: dict, interior_o: Interior_t, tides_o: Tides_t, config: Config) -> float:
    """Run the Obliqua tidal heating module.

    Sets the interior tidal heating and returns k-love number. All tidal love-numbers
    are stored in the tides_o object for the specific perturber.

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        dirs: dict
            Dictionary of directories.
        interior_o: Interior_t
            Struct containing interior arrays at current time.
        tides_o: Tides_t
            Struct containing tidal arrays at current time.
        config: Config
            PROTEUS config object
    Returns
    ----------
        Imk: float
            Averaged imaginary part of the k love numbers.
    """

    # Calculate axial frequency of rotation
    axial  = _jlsca_prec(2 * np.pi / hf_row["axial_period"])

    if config.orbit.perturber == 'star':
        log.debug("Running Obliqua for star-planet tides...")

        # Calculate orbital frequency of rotation
        omega  = _jlsca_prec(2 * np.pi / hf_row["orbital_period"])

        # Convert planet-star orbital eccentricity, semi-major axis, and mass
        ecc    = _jlsca_float(hf_row["eccentricity"])
        sma    = _jlsca_float(hf_row["semimajorax"])
        M_pert = _jlsca_float(hf_row["M_star"])

    elif config.orbit.perturber == 'satellite':
        log.debug("Running Obliqua for satellite-planet tides...")

        # Calculate orbital frequency of rotation
        omega  = _jlsca_prec(2 * np.pi / hf_row["orbital_period_sat"])

        # Convert planet-satellite orbital eccentricity, semi-major axis, and mass
        ecc    = _jlsca_float(hf_row["eccentricity_sat"])
        sma    = _jlsca_float(hf_row["semimajorax_sat"])
        M_pert = _jlsca_float(hf_row["M_sat"])


    # Copy arrays
    arr_keys = ("density", "visc", "shear", "bulk", "phi", "mass", "radius")
    lov = {k:np.array(getattr(interior_o, k), copy=True, dtype=float) for k in arr_keys}

    # Reverse arrays if using SPIDER
    #  Such that i=0 is at the CMB
    if config.interior_energetics.module == "spider":
        for k in arr_keys:
            lov[k] = lov[k][::-1]

    # Get viscous region and check if fully liquid
    i_top = 0  # index of topmost cell which has visc>visc_thresh
    if config.interior_energetics.module == "dummy":
        # Construct arrays for obliqua (we need two cells, three edges here)
        for k in arr_keys:
            if k == "radius":
                rmid = np.median(lov["radius"])
                arr = [lov["radius"][0], rmid, lov["radius"][1]]
                lov[k] = _jlarr(arr[:])
            else:
                arr = [lov[k][0],lov[k][0]]
                lov[k] = _jlarr(arr[:])

    else:
        # for spider/aragog
        # Construct arrays for obliqua
        i_top = interior_o.nlev_s
        for k in arr_keys:
            if k == "radius":
                i = i_top + 1
            else:
                i = i_top
            lov[k] = _jlarr(lov[k][:i])

    # Create configuration dictionary for Obliqua
    cfg = {
        "title": "PROTEUS_run_"+str(round(hf_row["Time"])),

        "params": {
            "out": {
                "path": dirs['output/data'],
                "time": round(hf_row["Time"]),
            },
        },

        "orbit": {
            "obliqua": {
                "store_3D": config.orbit.obliqua.store_3D,
                "enforce_ec": config.orbit.obliqua.enforce_ec,
                "optimize_scales": config.orbit.obliqua.optimize_scales,
                "solid_shell": config.orbit.obliqua.solid_shell,

                "min_frac": config.orbit.obliqua.min_frac,

                "visc_l": config.orbit.obliqua.visc_l,
                "visc_lus": config.orbit.obliqua.visc_lus,
                "visc_s": config.orbit.obliqua.visc_s,
                "visc_sus": config.orbit.obliqua.visc_sus,

                "n": config.orbit.obliqua.n,
                "m": config.orbit.obliqua.m,

                "spectrum": "adaptive", # Note that this is fixed, since spectrum = full is not useful for the current implementation of Obliqua in PROTEUS.

                "s_min": config.orbit.obliqua.k_min,
                "s_max": config.orbit.obliqua.k_max,

                "material_mu": config.orbit.obliqua.material_mu,
                "material_k": config.orbit.obliqua.material_k,
                "alpha": config.orbit.obliqua.alpha,

                "module_solid": config.orbit.obliqua.module_solid,
                "module_mushy": config.orbit.obliqua.module_mushy,
                "module_fluid": config.orbit.obliqua.module_fluid,

                "solid": {
                    "ncalc": config.orbit.obliqua.solid.ncalc,
                    "dr_min": config.orbit.obliqua.solid.dr_min,
                    "dr_max": config.orbit.obliqua.solid.dr_max,
                    "core": config.orbit.obliqua.solid.core,
                    "bulk_l": config.orbit.obliqua.solid.bulk_l,
                    "porosity_thresh": config.orbit.obliqua.solid.porosity_thresh,
                    "dbulk_power": config.orbit.obliqua.solid.dbulk_power,
                },

                "mushy": {
                    "b_width": config.orbit.obliqua.mushy.b_width,
                    "t_width": config.orbit.obliqua.mushy.t_width,
                },

                "fluid": {
                    "sigma_R": config.orbit.obliqua.fluid.sigma_R,
                    "sigma_R_inf": config.orbit.obliqua.fluid.sigma_R_inf,
                    "sigma_R_prf": config.orbit.obliqua.fluid.sigma_R_prf,
                    "H_R": config.orbit.obliqua.fluid.H_R,
                    "efficiency": config.orbit.obliqua.fluid.efficiency,
                },
            }
        },

        "planet": {
            "mass_tot": config.planet.mass_tot,
        },

        "interior_energetics": {
            "grain_size": config.interior_energetics.grain_size,
        },

        "struct": {
            "core_density": config.interior_energetics.boundary.core_density,
            "core_shear": config.interior_energetics.boundary.core_shear,
            "core_bulk": config.interior_energetics.boundary.core_bulk,
        }

    }

    cfg = to_julia_dict(cfg)

    # Calculate heating using obliqua
    try:
        # Extract arrays
        rho     = lov["density"]
        radius  = lov["radius"]
        visc    = lov["visc"]
        shear   = lov["shear"]
        bulk    = lov["bulk"]
        phi     = lov["phi"]

        # Add permeability and drained bulk modulus and limit porosity
        perm      = jl.Obliqua.interior.get_permeability(phi, cfg)
        perm, phi = jl.Obliqua.interior.limit_porosity(perm, phi, cfg)
        bulkd     = jl.Obliqua.interior.get_drained_bulk(bulk, phi, cfg)

        # Run Obliqua to get tidal heating profile and love number
        power_prf, power_blk, nmk, sigma, LNk = \
            jl.Obliqua.run_tides(
                omega,
                axial,
                ecc,
                sma,
                M_pert,
                rho,
                radius,
                visc,
                shear,
                bulk,
                bulkd,
                phi,
                perm,
                cfg
            )

    except juliacall.JuliaError as e:
        UpdateStatusfile(dirs, 26)
        log.error(e)
        raise RuntimeError("Encountered problem when running Obliqua module")

    if config.interior_energetics.module == "dummy":
        interior_o.tides[0] = power_prf[1]

    else:
        # Store result, flipping for SPIDER
        if config.interior_energetics.module == "spider":
            interior_o.tides[:] = power_prf[::-1]
        else:
            interior_o.tides[:] = power_prf[:]

        # Verify result against bulk calculation
        power_blk /= np.sum(lov["mass"])
        log.debug("    power from bulk calc: %.3e W kg-1"%power_blk)

    # Store results in tides_o structure
    storage = tides_o.add(primary="planet", perturber=config.orbit.perturber)
    storage.nmk    = np.vstack(nmk).astype(int)
    storage.sigma  = sigma
    storage.LNk    = LNk

    return np.mean(np.imag(LNk))


def LN_from_lookup(hf_row: dict, tides_o: Tides_t, config: Config):
    """Extract and populate Love numbers for a satellite from a pre-generated lookup table.

    Since Hansen coefficients dictate which modes are relevant for tidal forcing and depend
    only on orbital eccentricity, this function interpolates/matches the Love numbers (LNk)
    for relevant modes from an Obliqua lookup file spanning a broad frequency range.

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        tides_o: Tides_t
            Struct containing tidal arrays at current time.
        config: Config
            PROTEUS config object
    """
    # Fetch relevant modes from planet-side forcing
    nmk_p = np.asarray(
        tides_o.get(primary="planet", perturber="satellite").nmk
    )

    # Vectorized calculation of forcing frequencies (sigma_s = m * omega_rot - k * omega_orb)
    axial_freq_s = 2.0 * np.pi / hf_row["axial_period_sat"]
    orbit_freq_s = 2.0 * np.pi / hf_row["orbital_period_sat"]

    # nmk_p[:, 1] is 'm', nmk_p[:, 2] is 'k'
    sigma_s = nmk_p[:, 1] * axial_freq_s - nmk_p[:, 2] * orbit_freq_s

    # Retrieve or load satellite lookup data
    try:
        lookup = tides_o.get(primary="satellite_dict", perturber="planet")
    except KeyError:
        file_path = config.orbit.satellite.love_number_sat

        if not file_path:
            raise ValueError(
                "Satellite tidal data file path (`config.orbit.satellite.love_number_sat`) is not specified."
            )

        tides_o.add_from_file(
            primary="satellite_dict",
            perturber="planet",
            file_path=file_path
        )

        lookup = tides_o.get(primary="satellite_dict", perturber="planet")

    # Extract lookup arrays
    sigma_lookup = np.asarray(lookup.sigma)
    nmk_lookup = np.asarray(lookup.nmk)
    LNk_lookup = np.asarray(lookup.LNk)

    # Mirror lookup about sigma = 0 (symmetry valid for linear rheologies)
    # Generates a clean negative-to-positive frequency sequence: [-c, -b, -a, a, b, c]
    sigma_lookup = np.concatenate([
        -sigma_lookup[::-1],
        sigma_lookup
    ])

    nmk_lookup = np.concatenate([
        nmk_lookup[::-1],
        nmk_lookup
    ], axis=0)  # axis=0 explicitly ensures 2D array coordinates align properly

    LNk_lookup = np.concatenate([
        np.conjugate(LNk_lookup)[::-1],
        LNk_lookup
    ])

    # Pre-index lookup table by degree n
    unique_n = np.unique(nmk_p[:, 0])
    lookup_by_n = {}

    for n_val in unique_n:
        # Create a boolean mask for the current degree n
        mask = (nmk_lookup[:, 0] == n_val)

        if not np.any(mask):
            raise ValueError(
                f"Lookup table does not contain tidal data for degree n = {int(n_val)}."
            )

        lookup_by_n[n_val] = (
            sigma_lookup[mask],
            LNk_lookup[mask]
        )

    # Nearest-neighbor frequency matching
    LNk_s = np.empty(len(nmk_p), dtype=LNk_lookup.dtype)

    for i, (n, _, _) in enumerate(nmk_p):
        # Static tide
        if np.isclose(sigma_s[i], 0.0, atol=1e-15):
            LNk_s[i] = 0.0 + 0.0j
            continue

        sigma_filtered, LNk_filtered = lookup_by_n[n]

        # Find the index of the closest forcing frequency
        idx = np.argmin(np.abs(sigma_filtered - sigma_s[i]))
        LNk_s[i] = LNk_filtered[idx]

    # Store computed values back into the tides object
    storage = tides_o.add(primary="satellite", perturber="planet")
    storage.nmk = nmk_p
    storage.sigma = sigma_s
    storage.LNk = LNk_s

    pass


def read_ncdf(fpath: str):
    out = {}
    ds = nc.Dataset(fpath)

    for key in ds.variables.keys():
        out[key] = ds.variables[key][:]

    ds.close()
    return out


def read_ncdfs(output_dir: str, times: list):
    return [read_ncdf(os.path.join(output_dir, 'data', '%d_obliqua.nc' % t)) for t in times]
