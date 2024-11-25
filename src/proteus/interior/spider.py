# Function and classes used to run SPIDER
from __future__ import annotations

import glob
import json
import logging
import os
import platform
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from proteus.interior.timestep import next_step
from proteus.utils.constants import radnuc_data
from proteus.utils.helper import UpdateStatusfile, natural_sort, recursive_get

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)


class MyJSON( object ):

    '''load and access json data'''

    def __init__( self, filename ):
        self.filename = filename
        self._load()

    def _load( self ):
        '''load and store json data from file'''
        try:
            json_data  = open( self.filename )
        except FileNotFoundError:
            log.error('cannot find file: %s' % self.filename )
            log.error('please specify times for which data exists')
            sys.exit(1)
        self.data_d = json.load( json_data )
        json_data.close()

    # was get_field_data
    def get_dict( self, keys ):
        '''get all data relating to a particular field'''
        try:
            dict_d = recursive_get( self.data_d, keys )
            return dict_d
        except NameError:
            log.error('dictionary for %s does not exist', keys )
            sys.exit(1)

    # was get_field_units
    def get_dict_units( self, keys ):
        '''get the units (SI) of a particular field'''
        dict_d = recursive_get( self.data_d, keys )
        units = dict_d['units']
        units = None if units == 'None' else units
        return units

    # was get_scaled_field_values
    def get_dict_values( self, keys, fmt_o='' ):
        '''get the scaled values for a particular quantity'''
        dict_d = recursive_get( self.data_d, keys )
        scaling = float(dict_d['scaling'])
        if len( dict_d['values'] ) == 1:
            values_a = float( dict_d['values'][0] )
        else:
            values_a = np.array( [float(value) for value in dict_d['values']] )
        scaled_values_a = scaling * values_a
        if fmt_o:
            scaled_values_a = fmt_o.ascale( scaled_values_a )
        return scaled_values_a

    # was get_scaled_field_value_internal
    def get_dict_values_internal( self, keys, fmt_o='' ):
        '''get the scaled values for the internal nodes (ignore top
           and bottom nodes)'''
        scaled_values_a = self.get_dict_values( keys, fmt_o )
        return scaled_values_a[1:-1]

    def get_mixed_phase_boolean_array( self, nodes='basic' ):
        '''this array enables us to plot different linestyles for
           mixed phase versus single phase quantities'''
        if nodes == 'basic':
            phi = self.get_dict_values( ['data','phi_b'] )
        elif nodes == 'basic_internal':
            phi = self.get_dict_values_internal( ['data','phi_b'] )
        elif nodes == 'staggered':
            phi = self.get_dict_values( ['data','phi_s'] )
        # define mixed phase by these threshold values
        MIX = (phi<0.95) & (phi>0.05)
        return MIX

    def get_melt_phase_boolean_array( self, nodes='basic' ):
        '''this array enables us to plot different linestyles for
           melt phase regions'''
        if nodes == 'basic':
            phi = self.get_dict_values( ['data','phi_b'] )
        elif nodes == 'basic_internal':
            phi = self.get_dict_values_internal( ['data','phi_b'] )
        elif nodes == 'staggered':
            phi = self.get_dict_values( ['data','phi_s'] )
        MELT = (phi>0.95)
        return MELT

    def get_solid_phase_boolean_array( self, nodes='basic' ):
        '''this array enables us to plot different linestyles for
           solid phase regions'''
        if nodes == 'basic':
            phi = self.get_dict_values( ['data','phi_b'] )
        elif nodes == 'basic_internal':
            phi = self.get_dict_values_internal( ['data','phi_b'] )
        elif nodes == 'staggered':
            phi = self.get_dict_values( ['data','phi_s'] )
        SOLID = (phi<0.05)
        return SOLID

def read_jsons(output_dir:str, times:list):
    return [MyJSON(os.path.join(output_dir, "data", "%d.json"%t)) for t in times]

def get_all_output_times(odir:str):
    '''
    Get all times (in yr) from the json files located in the output directory
    '''

    odir = odir+'/data/'

    # locate times to process based on files located in odir/
    file_l = [f for f in os.listdir(odir) if os.path.isfile(odir+f)]
    if not file_l:
        raise Exception('Output data directory contains no files')

    time_l = [fname for fname in file_l]
    time_l = list(filter(lambda a: a.endswith('json'), time_l))
    time_l = [int(time.split('.json')[0]) for time in time_l]

    # ascending order
    time_l = sorted( time_l, key=int)
    time_a = np.array( time_l )

    return time_a

def get_dict_surface_values_for_specific_time( keys_t, time, indir='output'):
    '''Similar to above, but only loop over all times once and get
       all requested (surface / zero index) data in one go'''

    data_l = []

    filename = indir + '/data/{}.json'.format(time)
    myjson_o = MyJSON( filename )
    for key in keys_t:
        value = myjson_o.get_dict_values( key )
        data_l.append( value )


    return np.array(data_l)

#====================================================================
def _try_spider( dirs:dict, config:Config,
                IC_INTERIOR:int,
                hf_all:pd.DataFrame, hf_row:dict,
                step_sf:float, atol_sf:float ):
    '''
    Try to run spider with the current configuration.
    '''

    # Check that SPIDER can be found
    spider_exec = os.path.join(dirs["spider"],"spider")
    if not os.path.isfile(spider_exec):
        raise FileNotFoundError("SPIDER executable could not be found at '%s'"%spider_exec)

    # Bounds on tolereances
    step_sf = min(1.0, max(1.0e-10, step_sf))
    atol_sf = min(1.0e10, max(1.0e-10, atol_sf))

    # Recalculate time stepping
    if IC_INTERIOR == 2:

        # Current step number
        json_file   = MyJSON( dirs["output"]+'data/{}.json'.format(int(hf_row["Time"])) )
        step        = json_file.get_dict(['step'])

        # Get new time-step
        dtswitch = next_step(config, dirs, hf_row, hf_all, step_sf)

        # Number of total steps until currently desired switch/end time
        nsteps = 1
        nstepsmacro = step + nsteps
        dtmacro = dtswitch

        log.debug("Time options in RunSPIDER: dt=%.2e yrs in %d steps (at i=%d)" %
                                                    (dtmacro, nsteps, nstepsmacro))

    # For init loop
    else:
        nstepsmacro = 1
        dtmacro     = 0
        dtswitch    = 0

    empty_file = os.path.join(dirs["output"],"data", ".spider_tmp")
    open(empty_file, 'w').close()

    ### SPIDER base call sequence
    call_sequence = [
                        spider_exec,
                        "-options_file",           empty_file,
                        "-outputDirectory",        dirs["output"]+'data/',
                        "-IC_INTERIOR",            "%d"  %(IC_INTERIOR),
                        "-OXYGEN_FUGACITY_offset", "%.6e"%(config.outgas.fO2_shift_IW),  # Relative to the specified buffer
                        "-surface_bc_value",       "%.6e"%(hf_row["F_atm"]),
                        "-teqm",                   "%.6e"%(hf_row["T_eqm"]),
                        "-n",                      "%d"  %(config.interior.spider.num_levels),
                        "-nstepsmacro",            "%d"  %(nstepsmacro),
                        "-dtmacro",                "%.6e"%(dtmacro),
                        "-radius",                 "%.6e"%(hf_row["R_int"]),
                        "-gravity",                "%.6e"%(-1.0 * hf_row["gravity"]),
                        "-coresize",               "%.6e"%(config.struct.corefrac),
                        "-grain",                  "%.6e"%(config.interior.grain_size),
                    ]

    # Min of fractional and absolute Ts poststep change
    if hf_row["Time"] > 0:
        dTs_frac = config.interior.spider.tsurf_rtol * float(hf_all["T_surf"].iloc[-1])
        dT_int_max = np.min([ float(config.interior.spider.tsurf_atol), float(dTs_frac) ])
        call_sequence.extend(["-tsurf_poststep_change", str(dT_int_max)])
    else:
        call_sequence.extend(["-tsurf_poststep_change", str(config.interior.spider.tsurf_atol)])

    # set surface and core entropy (-1 is a flag to ignore)
    call_sequence.extend(["-ic_surface_entropy", "-1"])
    call_sequence.extend(["-ic_core_entropy",    "-1"])

    # Initial condition
    if IC_INTERIOR == 2:
        # get last JSON File
        last_filename = natural_sort([os.path.basename(x) for x in glob.glob(dirs["output"]+"data/*.json")])[-1]
        last_filename = os.path.join(dirs["output"], "data", last_filename)
        call_sequence.extend([
                                "-ic_interior_filename", str(last_filename),
                                "-activate_poststep",
                                "-activate_rollback"
                             ])
    else:
        # set to adiabat
        call_sequence.extend([
                                "-ic_adiabat_entropy", str(config.interior.spider.ini_entropy),
                                "-ic_dsdr", str(config.interior.spider.ini_dsdr) # initial dS/dr everywhere
                            ])

    # Mixing length parameterization: 1: variable | 2: constant
    call_sequence.extend(["-mixing_length", str(config.interior.spider.mixing_length)])
    call_sequence.extend(["-ts_sundials_atol", str(config.interior.spider.tolerance * atol_sf)])
    call_sequence.extend(["-ts_sundials_rtol", str(config.interior.spider.tolerance * atol_sf)])

    # Rollback
    call_sequence.extend(["-activate_poststep", "-activate_rollback"])

    # Dimensional scalings
    call_sequence.extend(["-radius0",   "63710000.0"])
    call_sequence.extend(["-entropy0",  "2993.025100070677"])
    call_sequence.extend(["-time0",     "1.0E5"])
    call_sequence.extend(["-pressure0", "10.0E5"])

    # Energy transport physics
    call_sequence.extend(["-CONDUCTION", "1"]) # conduction
    call_sequence.extend(["-CONVECTION", "1"]) # convection
    call_sequence.extend(["-MIXING    ", "1"]) # mixing (latent heat transport)
    call_sequence.extend(["-SEPARATION", "1"]) # gravitational separation of solid/melt

    # Tidal heating
    tidal_value = 0.0
    if config.interior.tidal_heat:
        if config.orbit.dummy:
            tidal_value = config.orbit.dummy.H_tide

        call_sequence.extend(["-HTIDAL",       "1"])
        call_sequence.extend(["-htidal_value", "%.5e"%tidal_value])

    # Properties lookup data (folder relative to SPIDER src)
    folder = "lookup_data/1TPa-dK09-elec-free/"
    call_sequence.extend(["-phase_names",  "melt,solid"])

    call_sequence.extend(["-melt_TYPE", "1"])
    call_sequence.extend(["-melt_alpha_filename_rel_to_src",          folder+"thermal_exp_melt.dat"])
    call_sequence.extend(["-melt_cp_filename_rel_to_src",             folder+"heat_capacity_melt.dat"])
    call_sequence.extend(["-melt_dTdPs_filename_rel_to_src",          folder+"adiabat_temp_grad_melt.dat"])
    call_sequence.extend(["-melt_rho_filename_rel_to_src",            folder+"density_melt.dat"])
    call_sequence.extend(["-melt_temp_filename_rel_to_src",           folder+"temperature_melt.dat"])
    call_sequence.extend(["-melt_phase_boundary_filename_rel_to_src", folder+"liquidus_A11_H13.dat"])
    call_sequence.extend(["-melt_log10visc", "2.0"])
    call_sequence.extend(["-melt_cond", "4.0"]) # conductivity of melt

    call_sequence.extend(["-solid_TYPE", "1"])
    call_sequence.extend(["-solid_alpha_filename_rel_to_src",            folder+"thermal_exp_solid.dat"])
    call_sequence.extend(["-solid_cp_filename_rel_to_src",               folder+"heat_capacity_solid.dat"])
    call_sequence.extend(["-solid_dTdPs_filename_rel_to_src",            folder+"adiabat_temp_grad_solid.dat"])
    call_sequence.extend(["-solid_rho_filename_rel_to_src",              folder+"density_solid.dat"])
    call_sequence.extend(["-solid_temp_filename_rel_to_src",             folder+"temperature_solid.dat"])
    call_sequence.extend(["-solid_phase_boundary_filename_rel_to_src",   folder+"solidus_A11_H13.dat"])
    call_sequence.extend(["-solid_log10visc", "22.0"])
    call_sequence.extend(["-solid_cond", "4.0"]) # conductivity of solid

    # static pressure profile derived from Adams-Williamson equation of state
    # these parameters are from fitting PREM in the lower mantle (for Earth)
    call_sequence.extend(["-adams_williamson_rhos", "4078.95095544"]) # surface density
    call_sequence.extend(["-adams_williamson_beta", "1.1115348931000002e-07"]) # beta parameter

    # eddy diffusivity
    # if negative, this value is adopted (units m^2/s)
    # if positive, this value is used to scale the internally calculated eddy diffusivity
    call_sequence.extend(["-eddy_diffusivity_thermal",  "1.0"])
    call_sequence.extend(["-eddy_diffusivity_chemical", "1.0"])


    # smoothing of material properties across liquidus and solidus
    # units of melt fraction (non-dimensional)
    call_sequence.extend(["-matprop_smooth_width", "1.0E-2"])

    # viscosity
    call_sequence.extend(["-phi_critical", "0.4"])  # transition melt fraction (non-dimensional)
    call_sequence.extend(["-phi_width",    "0.15"]) # transition width (non-dimensional)

    # core-mantle boundary condition
    call_sequence.extend(["-CORE_BC",  "1"])
    call_sequence.extend(["-rho_core", "%.6e"%(config.struct.core_density)]) # core density
    call_sequence.extend(["-cp_core",  "880.0"]) # core heat capacity

    # surface boundary condition
    # [4] heat flux (prescribe value using surface_bc_value)
    call_sequence.extend(["-SURFACE_BC", "4"])

    # parameterise the upper thermal boundary layer
    call_sequence.extend(["-PARAM_UTBL", "0"]) # disabled
    call_sequence.extend(["-param_utbl_const", "1.0E-7"]) # value of parameterisation

    # fO2 buffer chosen to define fO2 (7: Iron-Wustite)
    call_sequence.extend(["-OXYGEN_FUGACITY", "7"])

    # radionuclides
    if config.interior.radiogenic_heat:
        # offset by age_ini, which converts model simulation time to the actual age
        radio_t0 = config.delivery.radio_tref - config.star.age_ini
        radio_t0 *= 1e9 # Convert Gyr to yr
        radnuc_names = []

        def _append_radnuc(_iso, _cnc):
            radnuc_names.append(_iso)
            call_sequence.extend([f"-{_iso}_t0",              "%.5e"%radio_t0])
            call_sequence.extend([f"-{_iso}_concentration",   "%.5f"%_cnc])
            call_sequence.extend([f"-{_iso}_abundance",       "%.5e"%radnuc_data[_iso]["abundance"]])
            call_sequence.extend([f"-{_iso}_heat_production", "%.5e"%radnuc_data[_iso]["heatprod"]])
            call_sequence.extend([f"-{_iso}_half_life",       "%.5e"%radnuc_data[_iso]["halflife"]])

        if config.delivery.radio_K > 0.0:
            _append_radnuc("k40", config.delivery.radio_K)

        if config.delivery.radio_Th > 0.0:
            _append_radnuc("th232", config.delivery.radio_Th)

        if config.delivery.radio_U > 0.0:
            _append_radnuc("u235", config.delivery.radio_U)
            _append_radnuc("u238", config.delivery.radio_U)

        call_sequence.extend(["-radionuclide_names", ",".join(radnuc_names)])

    # Runtime info
    flags = ""
    for flag in call_sequence:
        flags += " " + flag
    # log.debug("SPIDER call sequence: '%s'" % flags)

    call_string = " ".join(call_sequence)

    # Environment
    spider_env = os.environ.copy()
    if platform.system() == "Darwin":
        spider_env["PETSC_ARCH"] = "arch-darwin-c-opt"
    else:
        spider_env["PETSC_ARCH"] = "arch-linux-c-opt"
    spider_env["PETSC_DIR"] = os.path.join(dirs["proteus"], "petsc")

    # Run SPIDER
    log.debug("SPIDER output suppressed")
    spider_print = open(dirs["output"]+"spider_recent.log",'w')
    spider_print.write(call_string+"\n")
    spider_print.flush()
    proc = subprocess.run([call_string],shell=True,stdout=spider_print, env=spider_env)
    spider_print.close()

    # Check status
    return bool(proc.returncode == 0)


def RunSPIDER( dirs:dict, config:Config, IC_INTERIOR:int,
              hf_all:pd.DataFrame, hf_row:dict ):
    '''
    Wrapper function for running SPIDER.
    This wrapper handles cases where SPIDER fails to find a solution.
    '''

    # parameters
    max_attempts = 5        # maximum number of attempts
    step_sf = 1.0           # step scale factor at attempt 1
    atol_sf = 1.0           # tolerance scale factor at attempt 1

    # tracking
    spider_success = False  # success?
    attempts = 0            # number of attempts so far

    # make attempts
    while not spider_success:
        attempts += 1
        log.debug("Attempt %d" % attempts)

        # run SPIDER
        spider_success = _try_spider(dirs, config, IC_INTERIOR, hf_all, hf_row, step_sf, atol_sf)

        if spider_success:
            # success
            log.debug("Attempt %d succeeded" % attempts)
        else:
            # failure
            log.warning("Attempt %d failed" % attempts)
            if attempts >= max_attempts:
                # give up
                log.error("Giving up")
                break
            else:
                # try again (change tolerance and step size)
                log.warning("Trying again")
                step_sf *= 0.5
                atol_sf *= 4.0

    # check status
    if spider_success:
        # success after some attempts
        return True
    else:
        # failure of all attempts
        UpdateStatusfile(dirs, 21)
        raise Exception("An error occurred when executing SPIDER (made %d attempts)" % attempts)


def ReadSPIDER(dirs:dict, config:Config, R_int:float):
    '''
    Read variables from last SPIDER output JSON file into a dictionary
    '''

    # Store variables in this dict
    output = {}

    ### Read in last SPIDER base parameters
    sim_time = get_all_output_times(dirs["output"])[-1]  # yr, as an integer value

    # SPIDER keys from JSON file that are read in
    keys_t = ( ('atmosphere','mass_liquid'),
                ('atmosphere','mass_solid'),
                ('atmosphere','mass_mantle'),
                ('atmosphere','mass_core'),
                ('atmosphere','temperature_surface'),
                ('rheological_front_phi','phi_global'),
                ('atmosphere','Fatm'),
                ('rheological_front_dynamic','depth'),
                )

    data_a = get_dict_surface_values_for_specific_time( keys_t, sim_time, indir=dirs["output"] )

    json_file = MyJSON( dirs["output"]+'/data/{}.json'.format(sim_time) )
    area_b   = json_file.get_dict_values(['data','area_b'])
    Hradio_s = json_file.get_dict_values(['data','Hradio_s'])
    Htidal_s = json_file.get_dict_values(['data','Htidal_s'])
    mass_s   = json_file.get_dict_values(['data','mass_s'])

    # Fill the new dict
    output["M_mantle_liquid"] = float(data_a[0])
    output["M_mantle_solid"]  = float(data_a[1])
    output["M_mantle"]        = float(data_a[2])

    # Surface properties
    output["T_magma"]         = float(data_a[4])
    output["Phi_global"]      = float(data_a[5])  # global melt fraction
    output["F_int"]           = float(data_a[6])  # Heat flux from interior
    output["RF_depth"]        = float(data_a[7])/R_int  # depth of rheological front

    # Tidal heating is not supported by SPIDER, so this should always be zero
    output["F_tidal"] = np.dot(Htidal_s, mass_s)/area_b[0]

    # Radiogenic heating
    output["F_radio"] = np.dot(Hradio_s, mass_s)/area_b[0]

    # Manually calculate heat flux at near-surface from energy gradient
    # Etot        = json_file.get_dict_values(['data','Etot_b'])
    # rad         = json_file.get_dict_values(['data','radius_b'])
    # area        = json_file.get_dict_values(['data','area_b'])
    # E0          = Etot[1] - (Etot[2]-Etot[1]) * (rad[2]-rad[1]) / (rad[1]-rad[0])
    # F_int2      = E0/area[0]

    # Limit F_int to positive values
    if config.atmos_clim.prevent_warming:
        output["F_int"] = max(1.0e-8, output["F_int"])

    # Check NaNs
    if np.isnan(output["T_magma"]):
        raise Exception("Magma ocean temperature is NaN")

    return sim_time, output
