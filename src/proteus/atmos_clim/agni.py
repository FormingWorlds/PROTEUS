# Functions used to handle atmosphere temperature structure (running AGNI, etc.)
from __future__ import annotations

import glob
import logging
import os

import numpy as np
from juliacall import Main as jl
from scipy.interpolate import PchipInterpolator

from proteus.utils.constants import dirs, volatile_species
from proteus.utils.helper import UpdateStatusfile, create_tmp_folder, safe_rm
from proteus.utils.logs import GetCurrentLogfileIndex, GetLogfilePath

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


def _construct_voldict(hf_row:dict, OPTIONS:dict):

    # get from hf_row
    vol_dict = {}
    for vol in volatile_species:
        if OPTIONS[vol+"_included"]:
            vmr = hf_row[vol+"_vmr"]
            if vmr > 1e-40:
                vol_dict[vol] = vmr

    # check values
    if len(vol_dict) == 0:
        UpdateStatusfile(dirs, 20)
        raise Exception("All volatiles have a volume mixing ratio of zero")

    return vol_dict


def init_agni_atmos(dirs:dict, OPTIONS:dict, hf_row:dict):
    """Initialise atmosphere struct for use by AGNI.

    Does not set the temperature profile.

    Parameters
    ----------
        dirs : dict
            Dictionary containing paths to directories
        OPTIONS : dict
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
    sflux_path  = os.path.join(dirs["output"], "data", "%d.sflux"%int(sorted(sflux_times)[-1]))

    # Spectral file path
    try_spfile = os.path.join(dirs["output"] , "runtime.sf")
    if os.path.exists(try_spfile):
        # exists => don't modify it
        input_sf =      try_spfile
        input_star =    ""
    else:
        # doesn't exist => AGNI will copy it + modify as required
        input_sf =      os.path.join(dirs["fwl"], OPTIONS["spectral_file"])
        input_star =    sflux_path

    # composition
    vol_dict = _construct_voldict(hf_row, OPTIONS)

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

        # set top two gases to be condensible
        # condensates = [v[0] for v in vol_sorted[-2:]]

    # Chemistry
    chem_type = OPTIONS["atmosphere_chemistry"]
    include_all = False
    fc_dir = "_unset"
    if chem_type == 1:
        # equilibrium
        include_all = True
        condensates = []

        # working folder for fastchem coupling
        fc_dir = create_tmp_folder()
        log.debug("Fastchem work folder: '%s'"%fc_dir)

    elif chem_type >= 2:
        # kinetics
        raise Exception("Chemistry type %d unsupported by AGNI"%chem_type)

    # Setup struct
    jl.AGNI.atmosphere.setup_b(atmos,
                        dirs["agni"], dirs["output"], input_sf,

                        hf_row["F_ins"],
                        OPTIONS["asf_scalefactor"],
                        OPTIONS["albedo_pl"],
                        OPTIONS["zenith_angle"],

                        hf_row["T_surf"],
                        hf_row["gravity"], hf_row["R_planet"],

                        int(OPTIONS["atmosphere_nlev"]),
                        hf_row["P_surf"],
                        OPTIONS["P_top"],

                        vol_dict, "",

                        flag_rayleigh=bool(OPTIONS["rayleigh"] == 1),
                        flag_cloud=bool(OPTIONS["water_cloud"] == 1),

                        albedo_s=OPTIONS["albedo_s"],
                        condensates=condensates,
                        use_all_gases=include_all,
                        fastchem_work = fc_dir,

                        skin_d=OPTIONS["skin_d"], skin_k=OPTIONS["skin_k"],
                        tmp_magma=hf_row["T_surf"], tmp_floor=OPTIONS["min_temperature"]
                        )

    # Allocate arrays
    jl.AGNI.atmosphere.allocate_b(atmos,input_star)

    # Set temperature profile from old NetCDF if it exists
    nc_files = glob.glob(os.path.join(dirs["output"],"data","*.nc"))
    if len(nc_files) > 0:
        log.debug("Load NetCDF profile")

        nc_times = [ int(s.split("/")[-1].split("_")[0]) for s in nc_files]
        nc_path  = os.path.join(dirs["output"], "data", "%d_atm.nc"%int(sorted(nc_times)[-1]))
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


def update_agni_atmos(atmos, hf_row:dict, OPTIONS:dict):
    """Update atmosphere struct.

    Sets the new boundary conditions and composition.

    Parameters
    ----------
        atmos : atmosphere.Atmos_t
            Atmosphere struct
        hf_row : dict
            Dictionary containing simulation variables for current iteration
        OPTIONS : dict
            Configuration options and other variables

    Returns
    ----------
        atmos : atmosphere.Atmos_t
            Atmosphere struct

    """

    # ---------------------
    # Update compositions
    vol_dict = _construct_voldict(hf_row, OPTIONS)
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



def run_agni(atmos, loops_total:int, dirs:dict, OPTIONS:dict, hf_row:dict):
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
        OPTIONS : dict
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
    if OPTIONS["log_level"] == "DEBUG":
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
        dx_max = OPTIONS["tsurf_poststep_change"]+10.0
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
                            sol_type=OPTIONS["atmosphere_surf_state"],
                            chem_type=OPTIONS["atmosphere_chemistry"],

                            conduct=False, convect=True, latent=True, sens_heat=True,

                            max_steps=130, max_runtime=900.0,
                            conv_atol=1e-3, conv_rtol=2e-2,

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
    if (OPTIONS["plot_iterfreq"] > 0) \
            and (loops_total % OPTIONS["plot_iterfreq"] == 0):

        fmt = OPTIONS["plot_format"]
        jl.AGNI.plotting.plot_fluxes(atmos, os.path.join(dirs["output"],
                                                  "plot_fluxes_atmosphere.%s"%fmt))
        jl.AGNI.plotting.plot_vmr(atmos, os.path.join(dirs["output"], "plot_vmr.%s"%fmt))

    # ---------------------------
    # Calculate observables
    # ---------------------------

    # observed height and derived bulk density
    jl.AGNI.atmosphere.calc_observed_rho_b(atmos)
    rho_obs = float(atmos.transspec_rho)
    z_obs   = float(atmos.transspec_r)

    # ---------------------------
    # Parse results
    # ---------------------------

    log.debug("Parse results")
    net_flux =      np.array(atmos.flux_n)
    LW_flux_up =    np.array(atmos.flux_u_lw)
    SW_flux_up =    np.array(atmos.flux_u_sw)
    T_surf =        float(atmos.tmp_surf)

    # New flux from SOCRATES
    if (OPTIONS["F_atm_bc"] == 0):
        F_atm_new = net_flux[0]
    else:
        F_atm_new = net_flux[-1]

    # Require that the net flux must be upward (positive)
    if (OPTIONS["prevent_warming"] == 1):
        F_atm_new = max( 1e-8 , F_atm_new )

    log.info("SOCRATES fluxes (net@BOA, net@TOA, OLR): %.2e, %.2e, %.2e  W m-2" %
                                        (net_flux[-1], net_flux[0] ,LW_flux_up[0]))

    output = {}
    output["F_atm"]  = F_atm_new
    output["F_olr"]  = LW_flux_up[0]
    output["F_sct"]  = SW_flux_up[0]
    output["T_surf"] = T_surf
    output["z_obs"]  = z_obs
    output["rho_obs"]= rho_obs

    return atmos, output
