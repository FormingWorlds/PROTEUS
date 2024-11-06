# Functions used to handle atmosphere temperature structure (running AGNI, etc.)
from __future__ import annotations

import glob
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
from juliacall import Main as jl
from scipy.interpolate import PchipInterpolator

from proteus.atmos_clim.common import get_spfile_path
from proteus.utils.constants import gas_list
from proteus.utils.helper import UpdateStatusfile, create_tmp_folder, safe_rm
from proteus.utils.logs import GetCurrentLogfileIndex, GetLogfilePath

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

# Constant
AGNI_LOGFILE_NAME="agni_recent.log"

def sync_log_files(outdir:str):
    # Logfile paths
    agni_logpath = os.path.join(outdir, AGNI_LOGFILE_NAME)
    logpath = GetLogfilePath(outdir, GetCurrentLogfileIndex(outdir))

    # Copy logfile content
    with open(agni_logpath, "r") as infile:
        inlines = infile.readlines()

        with open(logpath, "a") as outfile:
            for i,line in enumerate(inlines):
                # First line of agni logfile has NULL chars at the start, for some reason
                if i == 0:
                    line = "[" + line.split("[", 1)[1]
                # copy the line
                outfile.write(line)

    # Remove logfile content
    with open(agni_logpath, "w") as hdl:
        hdl.write("")

def activate_julia(dirs:dict):

    log.debug("Activating Julia environment")
    jl.seval("using Pkg")
    jl.Pkg.activate(dirs["agni"])

    # Plotting configuration
    jl.seval('ENV["GKSwstype"] = "100"')
    jl.seval("using Plots")
    jl.seval('default(label=nothing)')

    # Import AGNI
    jl.seval("using AGNI")

    # Setup logging from AGNI
    #    This handle will be kept open throughout the PROTEUS simulation, so the file
    #    should not be deleted at runtime. However, it will be emptied when appropriate.
    verbosity = 1
    logpath = os.path.join(dirs["output"], AGNI_LOGFILE_NAME)
    jl.AGNI.setup_logging(logpath, verbosity)

    log.debug("AGNI will log to '%s'"%logpath)


def _construct_voldict(hf_row:dict, config:Config, dirs:dict):

    # get from hf_row
    vol_dict = {}
    vol_sum = 0.0
    for vol in gas_list:
        vol_dict[vol] = hf_row[vol+"_vmr"]
        vol_sum += vol_dict[vol]

    # Check that the total VMR is not zero
    if vol_sum < 1e-4:
        UpdateStatusfile(dirs, 20)
        raise Exception("All volatiles have a volume mixing ratio of zero")

    return vol_dict


def init_agni_atmos(dirs:dict, config:Config, hf_row:dict):
    """Initialise atmosphere struct for use by AGNI.

    Does not set the temperature profile.

    Parameters
    ----------
        dirs : dict
            Dictionary containing paths to directories
        config : Config
            Configuration options and other variables
        hf_row : dict
            Dictionary containing simulation variables for current iteration

    Returns
    ----------
        atmos : atmosphere.Atmos_t
            Atmosphere struct

    """

    log.debug("New AGNI atmosphere")

    atmos = jl.AGNI.atmosphere.Atmos_t()

    # Stellar spectrum path
    sflux_files = glob.glob(os.path.join(dirs["output"], "data", "*.sflux"))
    sflux_times = [ int(s.split("/")[-1].split(".")[0]) for s in sflux_files]
    sflux_path  = os.path.join(dirs["output"],
                                "data", "%d.sflux"%int(sorted(sflux_times)[-1]))

    # Spectral file path
    try_spfile = os.path.join(dirs["output"] , "runtime.sf")
    if os.path.exists(try_spfile):
        # exists => don't modify it
        input_sf =      try_spfile
        input_star =    ""
    else:
        # doesn't exist => AGNI will copy it + modify as required
        input_sf =      get_spfile_path(dirs["fwl"], config)
        input_star =    sflux_path

    # composition
    vol_dict = _construct_voldict(hf_row, config, dirs)

    # set condensation
    condensates = []
    if len(vol_dict) == 1:
        # single-gas case
        condensates = list(vol_dict.keys())
    else:
        # get sorted gases
        vol_sorted = sorted(vol_dict.items(), key=lambda item: item[1])

        # set all gases as condensates, except the least abundant gas
        condensates = [v[0] for v in vol_sorted[1:]]

    # Chemistry
    chem_type = config.atmos_clim.agni.chemistry
    include_all = False
    fc_dir = "_unset"
    if chem_type == 'eq':
        # equilibrium
        include_all = True
        condensates = []

        # working folder for fastchem coupling
        fc_dir = create_tmp_folder()
        log.debug("Fastchem work folder: '%s'"%fc_dir)

    # Surface single-scattering albedo
    surface_material = config.atmos_clim.agni.surf_material
    if "greybody" in str(surface_material).lower():
        # Grey value
        surface_material = "greybody"
        log.debug("Using grey single-scattering surface properties")

    else:
        # Empirical values
        log.debug(f"Using '{surface_material}' single-scattering surface properties")
        surface_material = os.path.join(dirs["fwl"], surface_material)
        if not os.path.isfile(surface_material):
            raise FileNotFoundError(surface_material)

    # Setup struct
    jl.AGNI.atmosphere.setup_b(atmos,
                        dirs["agni"], dirs["output"], input_sf,

                        hf_row["F_ins"],
                        config.orbit.s0_factor,
                        config.atmos_clim.albedo_pl,
                        config.orbit.zenith_angle,

                        hf_row["T_surf"],
                        hf_row["gravity"], hf_row["R_int"],

                        int(config.atmos_clim.agni.num_levels),
                        hf_row["P_surf"],
                        config.atmos_clim.agni.p_top,

                        vol_dict, "",

                        flag_rayleigh = config.atmos_clim.rayleigh,
                        flag_cloud    = config.atmos_clim.cloud_enabled,
                        overlap_method  = config.atmos_clim.agni.overlap_method,

                        albedo_s=config.atmos_clim.surf_greyalbedo,
                        surface_material=surface_material,
                        condensates=condensates,
                        use_all_gases=include_all,
                        fastchem_work = fc_dir,

                        skin_d=config.atmos_clim.surface_d,
                        skin_k=config.atmos_clim.surface_k,
                        tmp_magma=hf_row["T_surf"], tmp_floor=config.atmos_clim.tmp_minimum
                        )

    # Allocate arrays
    jl.AGNI.atmosphere.allocate_b(atmos,input_star)

    # Set temperature profile from old NetCDF if it exists
    nc_files = glob.glob(os.path.join(dirs["output"],"data","*.nc"))
    if len(nc_files) > 0:
        log.debug("Load NetCDF profile")

        nc_times = [ int(s.split("/")[-1].split("_")[0]) for s in nc_files]
        nc_path  = os.path.join(dirs["output"],
                                "data", "%d_atm.nc"%int(sorted(nc_times)[-1]))
        jl.AGNI.setpt.fromncdf_b(atmos, nc_path)

    # Otherwise, set to log-linear
    else:
        # jl.AGNI.setpt.isothermal_b(atmos, hf_row["T_surf"])
        jl.AGNI.setpt.loglinear_b(atmos, min(900.0, hf_row["T_surf"]))

    # Logging
    sync_log_files(dirs["output"])

    return atmos


def deallocate_atmos(atmos):
    """
    Deallocate atmosphere struct
    """
    jl.AGNI.atmosphere.deallocate_b(atmos)
    safe_rm(str(atmos.fastchem_work))


def update_agni_atmos(atmos, hf_row:dict, config:Config, dirs:dict):
    """Update atmosphere struct.

    Sets the new boundary conditions and composition.

    Parameters
    ----------
        atmos : atmosphere.Atmos_t
            Atmosphere struct
        hf_row : dict
            Dictionary containing simulation variables for current iteration
        config : Config
            Configuration options and other variables

    Returns
    ----------
        atmos : atmosphere.Atmos_t
            Atmosphere struct

    """

    # ---------------------
    # Update compositions
    vol_dict = _construct_voldict(hf_row, config, dirs)
    for g in vol_dict.keys():
        atmos.gas_vmr[g][:]  = vol_dict[g]
        atmos.gas_ovmr[g][:] = vol_dict[g]

    # ---------------------
    # Store old/current log-pressure vs temperature arrays
    p_old = list(atmos.p)
    t_old = list(atmos.tmp)
    nlev_c = len(p_old)

    #    extend to lower pressures
    p_old = [p_old[0]/10] + p_old
    t_old = [t_old[0]]    + t_old

    #    extend to higher pressures
    p_old = p_old + [p_old[-1]*10]
    t_old = t_old + [t_old[-1]]

    #    create interpolator
    itp = PchipInterpolator(np.log10(p_old), t_old)

    # ---------------------
    # Update surface pressure [Pa] and generate new grid
    atmos.p_boa = 1.0e5 * float(hf_row["P_surf"])
    jl.AGNI.atmosphere.generate_pgrid_b(atmos)

    # ---------------------
    # Update surface temperature(s)
    atmos.tmp_surf  = float(hf_row["T_surf"] )
    atmos.tmp_magma = float(hf_row["T_magma"])

    # ---------------------
    # Set temperatures at all levels
    for i in range(nlev_c):
        atmos.tmp[i]  = float( itp(np.log10(atmos.p[i]))  )
        atmos.tmpl[i] = float( itp(np.log10(atmos.pl[i])) )
    atmos.tmpl[-1]    = float( itp(np.log10(atmos.pl[-1])))

    # ---------------------
    # Update instellation flux
    atmos.instellation = float(hf_row["F_ins"])

    return atmos



def run_agni(atmos, loops_total:int, dirs:dict, config:Config, hf_row:dict):
    """Run AGNI atmosphere model.

    Calculates the temperature structure of the atmosphere and the fluxes, etc.
    Stores the new flux boundary condition to be provided to SPIDER.

    Parameters
    ----------
        atmos : atmosphere.Atmos_t
            Atmosphere struct
        loops_total : int
            Model total loops counter.
        dirs : dict
            Dictionary containing paths to directories
        config : Config
            Configuration options and other variables
        hf_row : dict
            Dictionary containing simulation variables for current iteration

    Returns
    ----------
        atmos : atmosphere.Atmos_t
            Atmosphere struct
        output : dict
            Output variables, as a dictionary

    """

    # Inform
    log.info("Running AGNI...")
    time_str = "%d"%hf_row["Time"]

    # atmosphere solver plotting frequency
    modplot = 0
    if config.params.out.logging == "DEBUG":
        modplot = 1

    # tracking
    agni_success = False  # success?
    attempts = 0          # number of attempts so far

    # make attempts
    while not agni_success:
        attempts += 1
        log.info("Attempt %d" % attempts)

        # default parameters
        linesearch = 2
        easy_start = False
        dx_max = config.interior.spider.tsurf_atol+10.0
        ls_increase = 1.02

        # try different solver parameters if struggling
        if attempts == 2:
            linesearch  = 1
            dx_max     *= 3.0
            ls_increase = 1.1

        # first iteration parameters
        if loops_total == 0:
            linesearch  = 2
            easy_start  = True
            dx_max      = 200.0
            ls_increase = 1.1

        log.debug("Solver parameters:")
        log.debug("    ls_method=%d, easy_start=%s, dx_max=%.1f, ls_increase=%.2f"%(
            linesearch, str(easy_start), dx_max, ls_increase
        ))

        # Try solving temperature profile
        agni_success = jl.AGNI.solver.solve_energy_b(atmos,
                            sol_type=config.atmos_clim.surf_state_int,
                            chem_type=config.atmos_clim.agni.chemistry_int,

                            conduct=False, convect=True, latent=True, sens_heat=True,

                            max_steps=130, max_runtime=900.0,
                            conv_atol=config.atmos_clim.agni.solution_atol,
                            conv_rtol=config.atmos_clim.agni.solution_rtol,

                            method=1, ls_increase=ls_increase,
                            dx_max=dx_max, ls_method=linesearch, easy_start=easy_start,

                            save_frames=False, modplot=modplot
                            )

        # Move AGNI logfile content into PROTEUS logfile
        sync_log_files(dirs["output"])

        # Model status check
        if agni_success:
            # success
            log.info("Attempt %d succeeded" % attempts)
            break
        else:
            # failure
            log.warning("Attempt %d failed" % attempts)

            # Max attempts
            if attempts >= 2:
                log.error("Maximum attempts when executing AGNI")
                break

    # Write output data
    ncdf_path = os.path.join(dirs["output"],"data",time_str+"_atm.nc")
    jl.AGNI.dump.write_ncdf(atmos, ncdf_path)

    # Make plots
    if (config.params.out.plot_mod > 0) \
            and (loops_total % config.params.out.plot_mod == 0):

        fmt = config.params.out.plot_fmt
        jl.AGNI.plotting.plot_fluxes(atmos, os.path.join(dirs["output"],
                                                  "plot_fluxes_atmosphere.%s"%fmt))
        jl.AGNI.plotting.plot_vmr(atmos, os.path.join(dirs["output"], "plot_vmr.%s"%fmt))

    # ---------------------------
    # Calculate observables
    # ---------------------------

    # observed height and derived bulk density
    jl.AGNI.atmosphere.calc_observed_rho_b(atmos)
    rho_obs = float(atmos.transspec_rho)
    z_obs   = float(atmos.transspec_r) - hf_row["R_int"] # transspec_r = R_int + z_obs

    # ---------------------------
    # Parse results
    # ---------------------------

    log.debug("Parse results")
    net_flux =      np.array(atmos.flux_n)
    LW_flux_up =    np.array(atmos.flux_u_lw)
    SW_flux_up =    np.array(atmos.flux_u_sw)
    SW_flux_down =  np.array(atmos.flux_d_sw)
    T_surf =        float(atmos.tmp_surf)

    # New flux from SOCRATES
    F_atm_new = net_flux[0]

    # Require that the net flux must be upward (positive)
    if config.atmos_clim.prevent_warming:
        F_atm_lim = max( 1e-8 , F_atm_new )
    if not np.isclose(F_atm_lim , F_atm_new ):
        log.warning("Change in F_atm [W m-2] limited in this step!")
        log.warning("    %g  ->  %g" % (F_atm_new , F_atm_lim))

    log.info("SOCRATES fluxes (net@BOA, net@TOA, OLR): %.2e, %.2e, %.2e  W m-2" %
                                        (net_flux[-1], net_flux[0] ,LW_flux_up[0]))

    output = {}
    output["F_atm"]  = F_atm_lim
    output["F_olr"]  = LW_flux_up[0]
    output["F_sct"]  = SW_flux_up[0]
    output["T_surf"] = T_surf
    output["z_obs"]  = z_obs
    output["rho_obs"]= rho_obs
    output["albedo"] = SW_flux_up[0]/SW_flux_down[0]

    return atmos, output
