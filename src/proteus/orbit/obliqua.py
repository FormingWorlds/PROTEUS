# Obliqua tidal heating module
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

import juliacall
import numpy as np
from juliacall import Main as jl

from proteus.interior_energetics.common import Interior_t
from proteus.utils.helper import UpdateStatusfile

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def import_obliqua():
    log.debug("Import Obliqua...")
    jl.seval("using Obliqua")


def _jlarr(arr: np.ndarray):
    # Make copy of array, reverse order, and convert to Julia type
    cop = np.array(arr, copy=True, dtype=float).flatten()
    return juliacall.convert(jl.Array[jl.Obliqua.prec, 1], cop)


def _jlsca(sca: float):
    # Make a copy of a scalar, and convert to Julia type
    return juliacall.convert(jl.Obliqua.prec, sca)


def run_obliqua(hf_row: dict, dirs: dict, interior_o: Interior_t, config: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the Obliqua tidal heating module.

    Sets the interior tidal heating and returns k-love number.

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        dirs: dict
            Dictionary of directories.
        interior_o: Interior_t
            Struct containing interior arrays at current time.
        config: Config
            PROTEUS config object
    Returns
    ----------
        nmk: np.ndarray
            Tidal degree, order, and harmonnic index for each frequency.
        σ_range: np.ndarray
            Forcing frequency for each tidal mode.
        Hansen: np.ndarray
            Hansen coefficient for each tidal mode.
        LNk: np.ndarray
            Complex k love number for each tidal mode.
    """

    # Calculate axial frequency of rotation
    axial  = _jlsca(2 * np.pi / hf_row["axial_period"])

    if config.orbit.perturber == 'star':
        log.debug("Running Obliqua for star-planet tides...")

        # Calculate orbital frequency of rotation
        omega  = _jlsca(2 * np.pi / hf_row["orbital_period"])

        # Convert planet-star orbital eccentricity, semi-major axis, and mass
        ecc    = _jlsca(hf_row["eccentricity"])
        sma    = _jlsca(hf_row["semimajorax"])
        M_pert = _jlsca(hf_row["M_star"])

    elif config.orbit.perturber == 'satellite':
        log.debug("Running Obliqua for satellite-planet tides...")

        # Calculate orbital frequency of rotation
        omega  = _jlsca(2 * np.pi / hf_row["orbital_period_sat"])

        # Convert planet-satellite orbital eccentricity, semi-major axis, and mass
        ecc    = _jlsca(hf_row["eccentricity_sat"])
        sma    = _jlsca(hf_row["semimajorax_sat"])
        M_pert = _jlsca(hf_row["M_sat"])


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
        "orbit": {
            "obliqua": {
                "min_frac": config.orbit.obliqua.min_frac,

                "visc_l": config.orbit.obliqua.visc_l,
                "visc_lus": config.orbit.obliqua.visc_lus,
                "visc_s": config.orbit.obliqua.visc_s,
                "visc_sus": config.orbit.obliqua.visc_sus,

                "n": config.orbit.obliqua.n,
                "m": config.orbit.obliqua.m,

                "N_sigma": config.orbit.obliqua.N_sigma,
                "p_min": config.orbit.obliqua.p_min,
                "p_max": config.orbit.obliqua.p_max,

                "k_min": config.orbit.obliqua.k_min,
                "k_max": config.orbit.obliqua.k_max,

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

            "boundary": {
                "core_density": config.interior_energetics.boundary.core_density,
                "core_shear": config.interior_energetics.boundary.core_shear,
                "core_bulk": config.interior_energetics.boundary.core_bulk,
            }
        }


    }

    # Calculate heating using obliqua
    try:
        # Extract arrays
        rho     = lov["density"],
        radius  = lov["radius"],
        visc    = lov["visc"],
        shear   = lov["shear"],
        bulk    = lov["bulk"],
        phi     = lov["phi"]

        # Add permeability and drained bulk modulus and limit porosity
        perm      = jl.get_permeability(phi, cfg)
        perm, phi = jl.limit_porosity(perm, phi, cfg)
        bulkd     = jl.get_drained_bulk(bulk, phi, cfg)

        # Run Obliqua to get tidal heating profile and love number
        power_prf, power_blk, nmk, σ_range, Hansen, LNk = \
            jl.run_tides(
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

    # Return the frequency dependent LNk spectrum with harmonnic indices, forcing frequencies, and Hansen coefficients
    return np.array(nmk), np.array(σ_range), np.array(Hansen), np.array(LNk)
