# Functions used to handle atmosphere temperature structure (running AGNI, etc.)
from __future__ import annotations

import glob
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
from juliacall import Main as jl
from juliacall import convert
from scipy.interpolate import PchipInterpolator

from proteus.atmos_clim.common import get_radius_from_pressure, get_spfile_path
from proteus.utils.constants import gas_list
from proteus.utils.helper import UpdateStatusfile, create_tmp_folder, multiple, safe_rm
from proteus.utils.logs import GetCurrentLogfileIndex, GetLogfilePath

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

# Constant
AGNI_LOGFILE_NAME="agni_recent.log"
ALWAYS_DRY = ("CO", "N2", "H2")

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

def activate_julia(dirs:dict, verbosity:int):

    log.info("Activating Julia environment")
    jl.seval("using Pkg")
    jl.Pkg.activate(dirs["agni"])

    # Plotting configuration
    jl.seval('ENV["GKSwstype"] = "100"')
    jl.seval("using Plots")
    jl.seval('default(label=nothing, dpi=250)')

    # Import AGNI
    jl.seval("import AGNI")

    # Setup logging from AGNI
    #    This handle will be kept open throughout the PROTEUS simulation, so the file
    #    should not be deleted at runtime. However, it will be emptied when appropriate.
    logpath = os.path.join(dirs["output"], AGNI_LOGFILE_NAME)
    jl.AGNI.setup_logging(logpath, verbosity)

    log.debug("AGNI will log to '%s'"%logpath)


def _construct_voldict(hf_row:dict, dirs:dict):

    # get from hf_row
    vol_dict = {}
    vol_sum = 0.0
    for vol in gas_list:
        vol_dict[vol] = hf_row[vol+"_vmr"]
        vol_sum += vol_dict[vol]

    # Check that the total VMR is not zero
    if vol_sum < 1e-4:
        UpdateStatusfile(dirs, 20)
        raise ValueError("All volatiles have a volume mixing ratio of zero")

    return vol_dict

def _determine_condensates(vol_list:list):
    """Determine which gases will be condensable.

    Need to ensure that there's at least one 'dry' gas with non-zero opacity.

    Parameters
    -----------
        config : Config
            Configuration options and other variables
        vol_list: list
            List of included gases.

    Returns
    ----------
        condensates : list
            List of allowed-condensable gases
    """

    # single-gas case must be dry
    if len(vol_list) == 1:
        log.warning("Cannot include rainout condensation with only one gas!")
        return []

    # all dry gases...
    return [v for v in vol_list if v not in ALWAYS_DRY]


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

    # Fast I/O folder
    if config.atmos_clim.agni.verbosity >= 2:
        io_dir = dirs["output"]
    else:
        io_dir = create_tmp_folder()
    log.info(f"Temporary-file working dir: {io_dir}")

    # composition
    vol_dict = _construct_voldict(hf_row, dirs)

    # set condensation
    condensates = []
    if config.atmos_clim.agni.condensation:
        condensates = _determine_condensates(vol_dict.keys())

    # Chemistry
    include_all = bool(config.atmos_clim.agni.chemistry == 'eq')

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

    # Boundary pressures
    p_surf = hf_row["P_surf"]
    p_top  = config.atmos_clim.agni.p_top
    p_surf = max(p_surf, p_top * 1.1) # this will happen if the atmosphere is stripped

    # Setup struct
    jl.AGNI.atmosphere.setup_b(atmos,
                        dirs["agni"], dirs["output"], input_sf,

                        hf_row["F_ins"],
                        config.orbit.s0_factor,
                        float(hf_row["albedo_pl"]),
                        config.orbit.zenith_angle,

                        hf_row["T_surf"],
                        hf_row["gravity"], hf_row["R_int"],

                        int(config.atmos_clim.agni.num_levels),
                        p_surf,
                        p_top,

                        vol_dict, "",

                        IO_DIR = io_dir,
                        flag_rayleigh = config.atmos_clim.rayleigh,
                        flag_cloud    = config.atmos_clim.cloud_enabled,
                        overlap_method  = config.atmos_clim.agni.overlap_method,

                        albedo_s=config.atmos_clim.surf_greyalbedo,
                        surface_material=surface_material,

                        condensates=condensates,
                        phs_timescale=config.atmos_clim.agni.phs_timescale,
                        evap_efficiency=config.atmos_clim.agni.evap_efficiency,

                        use_all_gases=include_all,
                        fastchem_floor        = config.atmos_clim.agni.fastchem_floor,
                        fastchem_maxiter_chem = config.atmos_clim.agni.fastchem_maxiter_chem,
                        fastchem_maxiter_solv = config.atmos_clim.agni.fastchem_maxiter_solv,
                        fastchem_xtol_chem    = config.atmos_clim.agni.fastchem_xtol_chem,
                        fastchem_xtol_elem    = config.atmos_clim.agni.fastchem_xtol_elem,
                        real_gas = config.atmos_clim.agni.real_gas,
                        check_integrity = False, # don't check thermo files every time
                        mlt_criterion = convert(jl.Char,config.atmos_clim.agni.mlt_criterion),

                        skin_d=config.atmos_clim.surface_d,
                        skin_k=config.atmos_clim.surface_k,
                        tmp_magma=hf_row["T_surf"], tmp_floor=config.atmos_clim.tmp_minimum
                        )

    # Allocate arrays
    jl.AGNI.atmosphere.allocate_b(atmos,input_star)

    # Set temperature profile from old NetCDF if it exists
    nc_files = glob.glob(os.path.join(dirs["output"],"data","*_atm.nc"))
    if len(nc_files) > 0:
        log.debug("Load NetCDF profile")

        nc_times = [ int(s.split("/")[-1].split("_")[0]) for s in nc_files]
        nc_path  = os.path.join(dirs["output"],"data", f"{sorted(nc_times)[-1]:.0f}_atm.nc")
        jl.AGNI.setpt.fromncdf_b(atmos, nc_path)

    # Otherwise, set profile initial guess
    else:
        # do as requested by user in the config
        log.info(f"Initialising T(p) as {config.atmos_clim.agni.ini_profile}")
        match config.atmos_clim.agni.ini_profile:
            case "loglinear":
                jl.AGNI.setpt.loglinear_b(atmos, -0.5 * hf_row["T_surf"])
            case "isothermal":
                jl.AGNI.setpt.isothermal_b(atmos, hf_row["T_surf"])
            case "dry_adiabat":
                jl.AGNI.setpt.dry_adiabat_b(atmos)
            case "analytic":
                jl.AGNI.setpt.analytic_b(atmos)
            case _:
                log.error("Invalid initial profile selected")
                return

        # lower-limit on initial profile
        jl.AGNI.setpt.stratosphere_b(atmos, min(400.0, hf_row["T_surf"]))

    # Logging
    sync_log_files(dirs["output"])

    return atmos


def deallocate_atmos(atmos):
    """
    Deallocate atmosphere struct
    """
    jl.AGNI.atmosphere.deallocate_b(atmos)
    safe_rm(str(atmos.fastchem_work))


def update_agni_atmos(atmos, hf_row:dict, dirs:dict, config:Config):
    """Update atmosphere struct.

    Sets the new boundary conditions and composition.

    Parameters
    ----------
        atmos : AGNI.atmosphere.Atmos_t
            AGNI atmosphere struct
        hf_row : dict
            Dictionary containing simulation variables for current iteration
        dirs : dict
            Directories dictionary
        config: Config
            PROTEUS config object

    Returns
    ----------
        atmos : AGNI.atmosphere.Atmos_t
            Atmosphere struct
    """

    # ---------------------
    # Update instellation flux
    atmos.instellation = float(hf_row["F_ins"])

    # ---------------------
    # Update compositions
    vol_dict = _construct_voldict(hf_row, dirs)
    for g in vol_dict.keys():
        atmos.gas_vmr[g][:]  = vol_dict[g]
        atmos.gas_ovmr[g][:] = vol_dict[g]

    # ---------------------
    # Update surface temperature(s)
    atmos.tmp_surf  = float(hf_row["T_surf"] )
    atmos.tmp_magma = float(hf_row["T_magma"])

    # ---------------------
    # Transparent mode?
    if hf_row["P_surf"] < config.atmos_clim.agni.psurf_thresh:

        # set p_boa to threshold value [bar -> Pa]
        atmos.p_boa = float(config.atmos_clim.agni.psurf_thresh) * 1.0e5

        # update struct to handle this mode of operation
        jl.AGNI.atmosphere.make_transparent_b(atmos)
        jl.AGNI.setpt.isothermal_b(atmos, hf_row["T_surf"])

        # return here - don't do anything else to the `atmos` struct
        return atmos

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
    atmos.p_oboa = 1.0e5 * float(hf_row["P_surf"])
    atmos.p_boa = atmos.p_boa
    jl.AGNI.atmosphere.generate_pgrid_b(atmos)

    # ---------------------
    # Set temperatures at all levels
    for i in range(nlev_c):
        atmos.tmp[i]  = float( itp(np.log10(atmos.p[i]))  )
        atmos.tmpl[i] = float( itp(np.log10(atmos.pl[i])) )
    atmos.tmpl[-1]    = float( itp(np.log10(atmos.pl[-1])))

    return atmos


def _solve_energy(atmos, loops_total:int, dirs:dict, config:Config):
    """Use AGNI to solve for energy-conserving solution.

    Parameters
    ----------
        atmos : AGNI.atmosphere.Atmos_t
            Atmosphere struct
        loops_total : int
            Model total loops counter.
        dirs : dict
            Dictionary containing paths to directories
        config : Config
            Configuration options and other variables

    Returns
    ----------
        atmos : AGNI.atmosphere.Atmos_t
            Atmosphere struct
    """

    # atmosphere solver plotting frequency
    modplot = 0
    plot_jacobian = False
    if config.params.out.logging == "DEBUG":
        modplot = 1
        plot_jacobian = True

    # tracking
    agni_success = False  # success?
    attempts = 0          # number of attempts so far

    # make attempts
    while not agni_success:
        attempts += 1
        log.info("Attempt %d" % attempts)

        # default parameters
        linesearch   = int(config.atmos_clim.agni.ls_default)
        easy_start   = False
        grey_start   = False
        dx_max       = float(config.atmos_clim.agni.dx_max)
        ls_increase  = 1.01
        perturb_all  = bool(config.atmos_clim.agni.perturb_all)
        max_steps    = int(config.atmos_clim.agni.max_steps)
        chemistry    = bool(config.atmos_clim.agni.chemistry == 'eq')

        # parameters during initial few iterations
        if loops_total < 3:
            dx_max      = float(config.atmos_clim.agni.dx_max_ini)
            ls_increase = 1.1
            max_steps   = 200

        # parameters for the first iteration
        if loops_total == 0:
            easy_start  = True

        # try different solver parameters if struggling
        if attempts == 1:
            pass

        elif attempts == 2:
            linesearch  = 1
            dx_max     *= 2.0
            ls_increase = 1.1
            perturb_all = True

            if loops_total == 0:
                grey_start  = True

        elif attempts == 3:
            linesearch  = 1
            dx_max     *= 2.0
            ls_increase = 1.1
            perturb_all = True
            easy_start  = True

        else:
            # Max attempts
            log.error("Maximum attempts when executing AGNI")
            break

        log.debug("Solver parameters:")
        log.debug("    ls_method=%d, easy_start=%s, dx_max=%.1f, ls_increase=%.2f"%(
            linesearch, str(easy_start), dx_max, ls_increase
        ))

        # Update solver
        jl.AGNI.solver.ls_increase  = float(ls_increase)

        # Try solving temperature profile
        agni_success = jl.AGNI.solver.solve_energy_b(atmos,
                            sol_type  = int(config.atmos_clim.surf_state_int),
                            method    = int(1),
                            chem      = chemistry,

                            conduct=config.atmos_clim.agni.conduction,
                            convect=config.atmos_clim.agni.convection,
                            sens_heat=config.atmos_clim.agni.sens_heat,
                            latent=config.atmos_clim.agni.latent_heat,
                            rainout=config.atmos_clim.agni.condensation,

                            max_steps=int(max_steps), max_runtime=900.0,
                            conv_atol=float(config.atmos_clim.agni.solution_atol),
                            conv_rtol=float(config.atmos_clim.agni.solution_rtol),
                            fdo=int(config.atmos_clim.agni.fdo),

                            ls_method=int(linesearch),

                            dx_max=float(dx_max),
                            easy_start=easy_start, grey_start=grey_start,
                            perturb_all=perturb_all,

                            save_frames=False, modplot=int(modplot),
                            plot_jacobian=plot_jacobian
                            )

        # Move AGNI logfile content into PROTEUS logfile
        sync_log_files(dirs["output"])

        # Model status check
        if agni_success:
            # success
            log.info("Attempt %d succeeded" % attempts)
            break
        else:
            # failure, loop again...
            log.warning("Attempt %d failed" % attempts)

    return atmos

def _solve_once(atmos, config:Config):
    """Use AGNI to solve radiative transfer with prescribed T(p) profile

    Parameters
    ----------
        atmos : AGNI.atmosphere.Atmos_t
            Atmosphere struct
        config : Config
            PROTEUS config object

    Returns
    ----------
        atmos : AGNI.atmosphere.Atmos_t
            Atmosphere struct
    """

    # set temperature profile
    #    rainout volatiles at surface
    rained = jl.AGNI.chemistry.calc_composition_b(atmos,
                                                    config.atmos_clim.agni.condensation,
                                                    False, False)
    rained = bool(rained)
    if rained:
        log.info("    gases are condensing at the surface")
    #    dry convection
    jl.AGNI.setpt.dry_adiabat_b(atmos)
    #    condensation above
    if config.atmos_clim.agni.condensation:
        for gas in gas_list:
            jl.AGNI.setpt.saturation_b(atmos, str(gas))
    #    temperature floor in stratosphere
    jl.AGNI.setpt.stratosphere_b(atmos, 0.5)

    # do chemistry
    jl.AGNI.chemistry.calc_composition_b(atmos,
                                            config.atmos_clim.agni.condensation,
                                            config.atmos_clim.agni.chemistry == 'eq',
                                            config.atmos_clim.agni.condensation)

    # solve fluxes
    jl.AGNI.energy.calc_fluxes_b(atmos, radiative=True, convective=True)

    # fill kzz values
    jl.AGNI.energy.fill_Kzz_b(atmos)

    return atmos

def _solve_transparent(atmos, config:Config):
    """
    Use AGNI to solve for the surface temperature under a transparent atmosphere

    Parameters
    ----------
        atmos : AGNI.atmosphere.Atmos_t
            Atmosphere struct
        config : Config
            PROTEUS config object

    Returns
    ----------
        atmos : AGNI.atmosphere.Atmos_t
            Atmosphere struct
    """

    atol = float(config.atmos_clim.agni.solution_atol)
    rtol = float(config.atmos_clim.agni.solution_rtol)
    max_steps = 120

    jl.AGNI.solver.solve_transparent_b(atmos,
                                        sol_type=int(config.atmos_clim.surf_state_int),
                                        conv_atol=atol, conv_rtol=rtol,
                                        max_steps=int(max_steps))
    return atmos

def run_agni(atmos, loops_total:int, dirs:dict, config:Config,
                hf_row:dict):
    """Run AGNI atmosphere model.

    Calculates the temperature structure of the atmosphere and the fluxes, etc.
    Stores the new flux boundary condition to be provided to SPIDER.

    Parameters
    ----------
        atmos : AGNI.atmosphere.Atmos_t
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
        atmos : AGNI.atmosphere.Atmos_t
            Atmosphere struct
        output : dict
            Output variables, as a dictionary
    """

    # Inform
    log.debug("Running AGNI...")

    # ---------------------------
    # Solve atmosphere
    # ---------------------------

    # Transparent case
    if bool(atmos.transparent):
        # no opacity
        log.info("Using transparent solver")
        atmos.transspec_p = float(atmos.p_boa)
        atmos = _solve_transparent(atmos, config)

    # Opaque case
    else:

        # Set observed pressure
        atmos.transspec_p = float(config.atmos_clim.agni.p_obs * 1e5) # converted to Pa

        # full solver
        if config.atmos_clim.agni.solve_energy:
            log.info("Using nonlinear solver to conserve fluxes")
            atmos = _solve_energy(atmos, loops_total, dirs, config)

        # simplified T(p)
        else:
            log.info("Using prescribed temperature profile")
            atmos = _solve_once(atmos, config)

    # Calculate planet transit radius (to be stored in NetCDF)
    jl.AGNI.atmosphere.calc_observed_rho_b(atmos)

    # Write output data
    log.debug("AGNI write to NetCDF file")
    ncdf_path = os.path.join(dirs["output"],"data","%.0f_atm.nc"%hf_row["Time"])
    jl.AGNI.save.write_ncdf(atmos, ncdf_path)

    # Make plots
    if multiple(loops_total, config.params.out.plot_mod):
        cff = os.path.join(dirs["output/plots"], f"plot_cff.{config.params.out.plot_fmt}")
        jl.AGNI.plotting.plot_contfunc1(atmos, cff)


    # ---------------------------
    # Parse results
    # ---------------------------

    log.debug("Parse results")
    tot_flux =      np.array(atmos.flux_tot)
    LW_flux_up =    np.array(atmos.flux_u_lw)
    SW_flux_up =    np.array(atmos.flux_u_sw)
    SW_flux_down =  np.array(atmos.flux_d_sw)
    albedo = SW_flux_up[0]/SW_flux_down[0]
    if bool(atmos.transparent):
        R_obs = float(hf_row["R_int"])
    else:
        R_obs = float(atmos.transspec_r)

    # Print info to user
    if config.atmos_clim.agni.condensation:
        log.info("    oceans area frac   = %6.3f %%"%float(atmos.ocean_areacov*100))
        log.info("    oceans max depth   = %6.3f km"%float(atmos.ocean_maxdepth/1e3))
    log.info("    R_obs photosphere  = %6.1f km"%float(R_obs/1e3))
    log.info("    Planet Bond albedo = %6.3f %%"%float(albedo*100))

    # New flux from SOCRATES
    F_atm_new = tot_flux[0]

    # Enforce positive limit on F_atm, if enabled
    if config.atmos_clim.prevent_warming:
        F_atm_lim = max( 1e-8 , F_atm_new )
    else:
        F_atm_lim = F_atm_new
    if not np.isclose(F_atm_lim , F_atm_new ):
        log.warning("Change in F_atm [W m-2] limited in this step!")
        log.warning("    %g  ->  %g" % (F_atm_new , F_atm_lim))

    # XUV height in atm
    if config.escape.module == 'zephyrus':
        # escape level set by zephyrus config
        p_xuv = config.escape.zephyrus.Pxuv # [bar]
    else:
        # escape level set to surface
        p_xuv = hf_row["P_surf"] # [bar]
    p_xuv, r_xuv = get_radius_from_pressure(atmos.p, atmos.r, p_xuv*1e5) # [Pa], [m]

    # final things to store
    output = {}
    output["F_atm"]         = F_atm_lim
    output["F_olr"]         = LW_flux_up[0]
    output["F_sct"]         = SW_flux_up[0]
    output["T_surf"]        = float(atmos.tmp_surf)
    output["p_obs"]         = float(atmos.transspec_p)/1e5 # convert [Pa] to [bar]
    output["R_obs"]         = R_obs
    output["albedo"]        = albedo
    output["p_xuv"]         = p_xuv/1e5        # Closest pressure from Pxuv    [bars]
    output["R_xuv"]         = r_xuv            # Radius at Pxuv                [m]
    output["ocean_areacov"] = float(atmos.ocean_areacov)
    output["ocean_maxdepth"]= float(atmos.ocean_maxdepth)
    output["P_surf_clim"]   = float(atmos.p_boa) / 1e5 # Calculated Psurf [bar]

    for g in gas_list:
        try:
            output[g+"_ocean"] = float(atmos.ocean_tot[g])
        except:
            output[g+"_ocean"] = 0.0

    return atmos, output
