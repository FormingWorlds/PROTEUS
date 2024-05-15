# Function and classes used to run SPIDER

from utils.modules_ext import *
from utils.constants import *
from utils.helper import *

log = logging.getLogger("PROTEUS")
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
        # MIX = MIX * 1.0 # convert to float array
        # MIX[MIX==0] = np.nan  # set false region to nan to prevent plotting
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
        # MELT = MELT * 1.0 # convert to float array
        # MELT[MELT==0] = np.nan # set false region to nan to prevent plotting
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
        # SOLID = SOLID * 1.0 # convert to float array
        # SOLID[SOLID==0] = np.nan # set false region to nan to prevent plotting
        return SOLID

    def get_rho_interp1d( self ):
        '''return interp1d object for determining density as a
           function of pressure for static structure calculations'''
        pressure_a = self.get_dict_values( ['data','pressure_s'] )
        density_a = self.get_dict_values( ['data','rho_s'] )
        rho_interp1d = interp1d( pressure_a, density_a, kind='linear',
            fill_value='extrapolate' )
        return rho_interp1d

    def get_temp_interp1d( self ):
        '''return interp1d object for determining temperature as a
           function of pressure for static structure calculations'''
        pressure_a = self.get_dict_values( ['data','pressure_b'] )
        temp_a = self.get_dict_values( ['data','temp_b'] )
        temp_interp1d = interp1d( pressure_a, temp_a, kind='linear',
            fill_value='extrapolate' )
        return temp_interp1d

    def get_atm_struct_depth_interp1d( self ):
        '''return interp1d object for determining atmospheric height
           as a function of pressure for static structure calculations'''
        apressure_a = self.get_dict_values( ['atmosphere', 'atm_struct_pressure'] )
        adepth_a = self.get_dict_values( ['atmosphere', 'atm_struct_depth'] )
        atm_interp1d = interp1d( apressure_a, adepth_a, kind='linear' )
        return atm_interp1d

    def get_atm_struct_temp_interp1d( self ):
        '''return interp1d object for determining atmospheric temperature
           as a function of pressure'''
        apressure_a = self.get_dict_values( ['atmosphere', 'atm_struct_pressure'] )
        atemp_a = self.get_dict_values( ['atmosphere', 'atm_struct_temp'] )
        atm_interp1d = interp1d( apressure_a, atemp_a, kind='linear' )
        return atm_interp1d

#====================================================================

def get_column_data_from_SPIDER_lookup_file( infile ):
    '''Load column data from a text file and scale by the specified
    value (by position)'''

    # this approach prevents reading the whole file into memory
    # just to extract header information
    fp = open( infile, 'r' )
    for ii, line in enumerate( fp ):
        if ii == 0:
            splitline = list(map( float, line.lstrip('#').split() ))
            sline = int(splitline[0])
            size_a = splitline[1:]
        elif ii == sline-1:
            scalings = map( float, line.lstrip('#').split() )
        elif ii > sline:
            break
    fp.close()

    # read files, ignore headers (#), and make a 2-D array
    data_a = np.loadtxt( infile, ndmin=2 )

    # scale each column in the data array by the respective scaling
    for nn, scale in enumerate( scalings ):
        data_a[:,nn] *= scale

    return (data_a, size_a)

def get_SPIDER_1D_lookup( infile ):
    ''' return 1-D lookup object using linear interpolation'''

    data_a, size_a = get_column_data_from_SPIDER_lookup_file( infile )
    xx = data_a[:,0]
    yy = data_a[:,1]
    # will not allow extrpolation beyond the bounds without an extra
    # argument
    lookup_o = interp1d( xx, yy, kind='linear' )
    return lookup_o

def get_SPIDER_2D_lookup( infile ):
    '''return 2-D lookup object'''

    data_a, size_a = get_column_data_from_SPIDER_lookup_file( infile )
    xsize = int(size_a[0])
    ysize = int(size_a[1])

    xx = data_a[:,0][:xsize]
    yy = data_a[:,1][0::xsize]
    zz = data_a[:,2]
    zz = zz.reshape( (xsize, ysize), order='F' )
    lookup_o = RectBivariateSpline(xx, yy, zz, kx=1, ky=1, s=0 )

    return lookup_o

def get_all_output_times( odir='output' ):
    '''get all times (in Myrs) from the json files located in the
       output directory'''
    
    odir = odir+'/data/'

    # locate times to process based on files located in odir/
    file_l = [f for f in os.listdir(odir) if os.path.isfile(odir+f)]
    if not file_l:
        log.error('output directory contains no files')
        sys.exit(0)

    time_l = [fname for fname in file_l]
    time_l = list(filter(lambda a: a.endswith('json'), time_l))

    # Filter out original/non-hacked jsons
    time_l = [ file for file in time_l if not file.startswith("orig_")]

    time_l = [int(time.split('.json')[0]) for time in time_l]
    
    # ascending order
    time_l = sorted( time_l, key=int)
    time_a = np.array( time_l )

    return time_a

def get_all_output_atm_times( odir='output' ):
    '''get all times (in Myrs) from the nc files located in the
       output directory'''

    odir = odir+'/data/'

    stub_dirs = {"output": os.path.abspath(odir)}

    # locate times to process based on files located in odir/
    file_l = [f for f in os.listdir(odir) if os.path.isfile(odir+f)]
    if not file_l:
        UpdateStatusfile(stub_dirs, 20)
        raise Exception("Output directory contains no files")

    time_l = [fname for fname in file_l]
    time_l = list(filter(lambda a: a.endswith('nc'), time_l))
    if len(time_l) == 0:
        UpdateStatusfile(stub_dirs, 20)
        raise Exception("Could not find any nc files in the output directory")

    # Filter and split files
    time_l = [ file for file in time_l if not file.startswith("orig_")]
    time_l = [ int(time.split('_atm')[0]) for time in time_l ]
    
    # ascending order
    time_l = sorted( time_l, key=int)
    time_a = np.array( time_l )

    return time_a



def get_dict_values_for_times( keys, time_l, indir='output' ):
    data_l = []
    for time in time_l:
        filename = indir + '/data/{}.json'.format(time)
        myjson_o = MyJSON( filename )
        values_a = myjson_o.get_dict_values( keys )
        data_l.append( values_a )

    data_a = np.array( data_l )

    # rows time, cols data
    data_a.reshape( (len(time_l),-1 ) )
    # rows data, cols time
    data_a = data_a.transpose()

    return data_a

def get_dict_surface_values_for_times( keys_t, time_l, indir='output'):
    '''Similar to above, but only loop over all times once and get
       all requested (surface / zero index) data in one go'''

    data_l = []

    for time in time_l:
        filename = indir + '/data/{}.json'.format(time)
        myjson_o = MyJSON( filename )
        keydata_l = []
        for key in keys_t:
            values_a = myjson_o.get_dict_values( key )
            try:
                value = values_a[0]
            except TypeError:
                value = values_a
            keydata_l.append( value )
        data_l.append( keydata_l )

    data_a = np.array( data_l )

    # rows time, cols data
    data_a.reshape( (len(time_l),-1 ) )
    # rows data, cols time
    data_a = data_a.transpose()

    return data_a

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


def get_deriv_static_structure( z, r, *args ):
    '''get derivatives of pressure, mass, and gravity
       returns dp/dr, dm/dr, and dg/dr'''

    p = z[0] # pressure
    m = z[1] # mass
    g = z[2] # gravity

    rho_interp1d = args[5]
    rho = np.asscalar(rho_interp1d( p ))

    # derivatives
    dpdr = -rho*g
    dmdr = 4*np.pi*r**2*rho
    dgdr = 4*np.pi*const_G*rho - 2*const_G*m/r**3

    return [dpdr,dmdr,dgdr]

def get_radius_array_static_structure( radius, *myargs ):
    R_core = myargs[1]
    num = myargs[4]

    return np.linspace(radius,R_core,num)

def get_static_structure_for_radius( radius, *myargs ):
    '''get static structure (pressure, mass, and gravity) for an
       input radius'''

    M_earth = myargs[0]
    R_core = myargs[1]
    num = myargs[4]
    g_Earth = const_G*M_earth/radius**2

    z0 = [0,M_earth,g_Earth]
    r = get_radius_array_static_structure( radius, *myargs )
    z = odeint( get_deriv_static_structure, z0, r, args=myargs )

    return z

def get_difference_static_structure( radius, *myargs ):
    '''return root, difference between computed mass or gravity at
       the core-mantle boundary and the desired value'''

    # you can either compare mass or gravity
    z = get_static_structure_for_radius( radius, *myargs )
    g_core = z[:,2][-1]
    m_core = z[:,1][-1]

    # if m_core > M_core, then radius is too small
    # if m_core < M_core, then radius is too large
    #return m_core-M_core
    G_core = myargs[3]

    return g_core-G_core


def check_static_structure( radius, *myargs ):
    '''compute relative accuracy of gravity'''

    G_core = myargs[3]
    dg = get_difference_static_structure( radius, *myargs )
    reldg = np.abs( dg/G_core )
    if reldg > 1.0e-6:
        log.warning('g relative accuracy= {}'.format(reldg) )




#====================================================================
def _try_spider( time_dict, dirs, COUPLER_options, loop_counter, runtime_helpfile, step_sf, atol_sf ):
    '''
    Try to run spider with the current configuration.
    On success, return (True, COUPLER_options)
    On failure, return (False, {})
    '''

    step_sf = min(1.0, max(1.0e-10, step_sf))
    atol_sf = min(1.0e10, max(1.0e-10, atol_sf))

    SPIDER_options_file = dirs["output"]+"/init_spider.opts"
    SPIDER_options_file_orig = dirs["utils"]+"/init_spider.opts"

    # First run
    if (loop_counter["init"] == 0):
        if os.path.isfile(SPIDER_options_file):
            os.remove(SPIDER_options_file)
        shutil.copy(SPIDER_options_file_orig,SPIDER_options_file)

    # Define which volatiles to track in SPIDER
    species_call = ""
    for vol in volatile_species: 
        if COUPLER_options[vol+"_included"] == 1:
            species_call = species_call + "," + vol
    species_call = species_call[1:] # Remove "," in front

    # Recalculate time stepping
    if (COUPLER_options["IC_INTERIOR"] == 2):  

        # Current step number
        json_file   = MyJSON( dirs["output"]+'data/{}.json'.format(int(time_dict["planet"])) )
        step        = json_file.get_dict(['step'])

        # Previous steps
        run_int = runtime_helpfile.loc[runtime_helpfile['Input']=='Interior'].drop_duplicates(subset=['Time'], keep='last')
        run_atm = runtime_helpfile.loc[runtime_helpfile['Input']=='Atmosphere'].drop_duplicates(subset=['Time'], keep='last')

        # Time stepping adjustment
        if time_dict["planet"] < 2.0:
            # First year, use small step
            dtmacro = 1
            dtswitch = 1
            nsteps = 1
            log.info("Time-stepping intent: static")

        else:
            if (COUPLER_options["dt_method"] == 0):
                # Proportional time-step calculation
                log.info("Time-stepping intent: proportional")
                dtswitch = time_dict["planet"] / float(COUPLER_options["dt_propconst"])

            elif (COUPLER_options["dt_method"] == 1):
                # Dynamic time-step calculation

                # Try to maintain a minimum step size of dt_initial at first
                if time_dict["planet"] > COUPLER_options["dt_initial"]:
                    dtprev = float(run_int.iloc[-1]["Time"] - run_int.iloc[-2]["Time"])
                else:
                    dtprev = COUPLER_options["dt_initial"]

                # Change in F_int 
                F_int_2  = run_int.iloc[-2]["F_int"]
                F_int_1  = run_int.iloc[-1]["F_int"]
                F_int_12 = abs(F_int_1 - F_int_2) 

                # Change in F_atm
                F_atm_2  = run_atm.iloc[-2]["F_atm"]
                F_atm_1  = run_atm.iloc[-1]["F_atm"]
                F_atm_12 = abs(F_atm_1 - F_atm_2)  

                # Change in global melt fraction
                phi_2  = run_atm.iloc[-2]["Phi_global"]
                phi_1  = run_atm.iloc[-1]["Phi_global"]
                phi_12 = abs(phi_1 - phi_2)  

                # Determine new time-step given the tolerances
                dt_rtol = COUPLER_options["dt_rtol"]
                dt_atol = COUPLER_options["dt_atol"]
                speed_up = True 
                speed_up = speed_up and ( F_int_12 < dt_rtol*abs(F_int_2) + dt_atol )
                speed_up = speed_up and ( F_atm_12 < dt_rtol*abs(F_atm_2) + dt_atol )
                speed_up = speed_up and ( phi_12   < dt_rtol*abs(phi_2  ) + dt_atol )

                if speed_up:
                    dtswitch = dtprev * 1.05
                    log.info("Time-stepping intent: speed up")
                else:
                    dtswitch = dtprev * 0.9
                    log.info("Time-stepping intent: slow down")


            elif (COUPLER_options["dt_method"] == 2):
                # Always use the maximum time-step, which can be adjusted in the cfg file
                log.info("Time-stepping intent: maximum")
                dtswitch = COUPLER_options["dt_maximum"]

            else:
                UpdateStatusfile(dirs, 20)
                raise Exception("Invalid time-stepping method '%d'" % COUPLER_options["dt_method"])
            
            # Additional step-size ceiling when F_crit is used
            if abs(run_atm.iloc[-1]["F_atm"]) <= COUPLER_options["F_crit"]:
                dtswitch = min(dtswitch, COUPLER_options["dt_crit"])
                log.info("|F_atm| <= F_crit, so time-step is limited to %g years" % COUPLER_options["dt_crit"])

            # Step scale factor (is always <= 1.0)
            dtswitch *= step_sf

            # Step-size ceiling
            dtswitch = min(dtswitch, COUPLER_options["dt_maximum"] )                    # Absolute
            dtswitch = min(dtswitch, float(time_dict["target"] - time_dict["planet"]))  # Run-over

            # Step-size floor
            dtswitch = max(dtswitch, time_dict["planet"]*0.0001)        # Relative
            dtswitch = max(dtswitch, COUPLER_options["dt_minimum"] )    # Absolute

            # Calculate number of macro steps for SPIDER to perform within
            # this time-step of PROTEUS, which sets the number of json files.
            nsteps = 1
            dtmacro = math.ceil(dtswitch / nsteps)   # Ensures that dtswitch is divisible by nsteps
            dtswitch = nsteps * dtmacro

            log.info("New time-step is %1.2e years" % dtswitch)

        # Number of total steps until currently desired switch/end time
        nstepsmacro = step + nsteps

        log.debug("TIME OPTIONS IN RUNSPIDER: %g %g %d" % (dtmacro, dtswitch, nstepsmacro))

    # For init loop
    else:
        nstepsmacro = 1
        dtmacro     = 0
        dtswitch    = 0

    # Store time-step (for next iteration)
    COUPLER_options["dtswitch"] = dtswitch
    COUPLER_options["dtmacro"] = dtmacro

    # Set spider flux boundary condition
    net_loss = COUPLER_options["F_atm"]

    ### SPIDER base call sequence 
    call_sequence = [   
                        dirs["spider"]+"/spider", 
                        "-options_file",           SPIDER_options_file, 
                        "-outputDirectory",        dirs["output"]+'data/',
                        "-IC_INTERIOR",            "%d"  %(COUPLER_options["IC_INTERIOR"]),
                        "-OXYGEN_FUGACITY_offset", "%.6e"%(COUPLER_options["fO2_shift_IW"]),  # Relative to the specified buffer
                        "-surface_bc_value",       "%.6e"%(net_loss), 
                        "-teqm",                   "%.6e"%(COUPLER_options["T_eqm"]), 
                        "-n",                      "%d"  %(COUPLER_options["interior_nlev"]),
                        "-nstepsmacro",            "%d"  %(nstepsmacro), 
                        "-dtmacro",                "%.6e"%(dtmacro), 
                        "-radius",                 "%.6e"%(COUPLER_options["radius"]), 
                        "-gravity",                "%.6e"%(-1.0 * COUPLER_options["gravity"]), 
                        "-coresize",               "%.6e"%(COUPLER_options["planet_coresize"]),
                        "-grain",                  "%.6e"%(COUPLER_options["grain_size"]),
                        "-volatile_names",          str(species_call)
                    ]

    # Min of fractional and absolute Ts poststep change
    if time_dict["planet"] > 0:
        dTs_frac = float(COUPLER_options["tsurf_poststep_change_frac"]) * float(runtime_helpfile["T_surf"].iloc[-1])
        dT_int_max = np.min([ float(COUPLER_options["tsurf_poststep_change"]), float(dTs_frac) ])
        call_sequence.extend(["-tsurf_poststep_change", str(dT_int_max)])
    else:
        call_sequence.extend(["-tsurf_poststep_change", str(COUPLER_options["tsurf_poststep_change"])])

    # Define distribution coefficients and total mass/surface pressure for volatiles > 0
    log.info("Input surface volatile partial pressures:")
    for vol in volatile_species:
        if COUPLER_options[vol+"_included"] == 1:

            # Set atmospheric pressure based on helpfile output, if required
            if loop_counter["total"] > loop_counter["init_loops"]:
                key = vol+"_initial_atmos_pressure"
                val = float(runtime_helpfile[vol+"_mr"].iloc[-1]) * float(runtime_helpfile["P_surf"].iloc[-1]) * 1.0e5   # convert bar to Pa
                COUPLER_options[key] = val

            # Load volatiles
            pp = COUPLER_options[vol+"_initial_atmos_pressure"]
            call_sequence.extend(["-"+vol+"_initial_atmos_pressure", str(pp)])
            log.info("    p_%s = %.5f bar" % (vol,pp/1.0e5))

            # Exception for N2 case: reduced vs. oxidized
            if vol == "N2" and COUPLER_options["N2_partitioning"] == 1:
                volatile_distribution_coefficients["N2_henry"] = volatile_distribution_coefficients["N2_henry_reduced"]
                volatile_distribution_coefficients["N2_henry_pow"] = volatile_distribution_coefficients["N2_henry_pow_reduced"]

            call_sequence.extend(["-"+vol+"_henry", str(volatile_distribution_coefficients[vol+"_henry"])])
            call_sequence.extend(["-"+vol+"_henry_pow", str(volatile_distribution_coefficients[vol+"_henry_pow"])])
            call_sequence.extend(["-"+vol+"_kdist", str(volatile_distribution_coefficients[vol+"_kdist"])])
            call_sequence.extend(["-"+vol+"_kabs", str(volatile_distribution_coefficients[vol+"_kabs"])])
            call_sequence.extend(["-"+vol+"_molar_mass", str(molar_mass[vol])])
            call_sequence.extend(["-"+vol+"_SOLUBILITY 1"])  # Set to use Henry's law

    # With start of the main loop only:
    # Volatile specific options: post step settings, restart filename
    if COUPLER_options["IC_INTERIOR"] == 2:
        call_sequence.extend([ 
                                "-ic_interior_filename", 
                                str(dirs["output"]+"data/"+COUPLER_options["ic_interior_filename"]),
                                "-activate_poststep", 
                                "-activate_rollback"
                             ])
        for vol in volatile_species:
            if COUPLER_options[vol+"_included"] == 1:
                call_sequence.extend(["-"+vol+"_poststep_change", str(COUPLER_options[vol+"_poststep_change"])])
    else:
        call_sequence.extend([
                                "-ic_adiabat_entropy", str(COUPLER_options["ic_adiabat_entropy"]),
                                "-ic_dsdr", str(COUPLER_options["ic_dsdr"]) # initial dS/dr everywhere"
                            ])

    # Gravitational separation of solid and melt phase, 0: off | 1: on
    if COUPLER_options["SEPARATION"] == 1:
        call_sequence.extend(["-SEPARATION", str(1)])

    # Mixing length parameterization: 1: variable | 2: constant
    call_sequence.extend(["-mixing_length", str(COUPLER_options["mixing_length"])])

    # Ultra-thin thermal boundary layer at top, 0: off | 1: on
    if COUPLER_options["PARAM_UTBL"] == 1:
        call_sequence.extend(["-PARAM_UTBL", str(1)])
        call_sequence.extend(["-param_utbl_const", str(COUPLER_options["param_utbl_const"])])

    # Check for convergence, if not converging, adjust tolerances iteratively
    if (loop_counter["total"] > loop_counter["init_loops"]) and (len(runtime_helpfile) > 50):

        # Check convergence for interior cycles
        run_int = runtime_helpfile.loc[runtime_helpfile['Input']=='Interior'].drop_duplicates(subset=['Time'], keep='last')

        ref_idx = -3
        if len(run_int["Time"]) < abs(ref_idx)-1:
            ref_idx = 0

        # First, relax too restrictive dTs
        if run_int["Time"].iloc[-1] == run_int["Time"].iloc[ref_idx]:
            if COUPLER_options["tsurf_poststep_change"] <= 300:
                COUPLER_options["tsurf_poststep_change"] += 10
                log.warning(">>> Raise dT poststep_changes:", COUPLER_options["tsurf_poststep_change"], COUPLER_options["tsurf_poststep_change_frac"])
            else:
                log.warning(">> dTs_int too high! >>", COUPLER_options["tsurf_poststep_change"], "K")
                
        # Slowly limit again if time advances smoothly
        if (run_int["Time"].iloc[-1] != run_int["Time"].iloc[ref_idx]) and COUPLER_options["tsurf_poststep_change"] > 30:
            COUPLER_options["tsurf_poststep_change"] -= 10
            log.warning(">>> Lower tsurf_poststep_change poststep changes:", COUPLER_options["tsurf_poststep_change"], COUPLER_options["tsurf_poststep_change_frac"])

        if run_int["Time"].iloc[-1] == run_int["Time"].iloc[ref_idx]:
            if COUPLER_options["solver_tolerance"] < 1.0e-2:
                COUPLER_options["solver_tolerance"] = float(COUPLER_options["solver_tolerance"])*2.
                log.warning(">>> ADJUST tolerances:", COUPLER_options["solver_tolerance"])
            COUPLER_options["adjust_tolerance"] = 1
            log.warning(">>> CURRENT TOLERANCES:", COUPLER_options["solver_tolerance"])

        # If tolerance was adjusted, restart SPIDER w/ new tolerances
        if "adjust_tolerance" in COUPLER_options:
            log.warning(">>>>> >>>>> RESTART W/ ADJUSTED TOLERANCES")
            call_sequence.extend(["-atmosts_snes_atol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosts_snes_rtol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosts_ksp_atol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosts_ksp_rtol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosic_ksp_rtol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosic_ksp_atol", str(COUPLER_options["solver_tolerance"])])

    call_sequence.extend(["-ts_sundials_atol", str(COUPLER_options["solver_tolerance"] * atol_sf)])
    call_sequence.extend(["-ts_sundials_rtol", str(COUPLER_options["solver_tolerance"] * atol_sf)])

    # Runtime info
    flags = ""
    for flag in call_sequence:
        flags += " " + flag
    log.debug("SPIDER call sequence: '%s'" % flags)

    call_string = " ".join(call_sequence)

    # Run SPIDER
    log.info("Terminal output suppressed")
    spider_print = open(dirs["output"]+"spider_recent.log",'w')
    spider_print.write(call_string+"\n")
    spider_print.flush()
    proc = subprocess.run([call_string],shell=True,stdout=spider_print)
    spider_print.close()

    # Update restart filename for next SPIDER run
    COUPLER_options["ic_interior_filename"] = natural_sort([os.path.basename(x) for x in glob.glob(dirs["output"]+"data/*.json")])[-1]

    # Check status
    success = (proc.returncode == 0)

    # Handle failure by signalling for another _try_spider attempt
    if success:
        return True, COUPLER_options
    else:
        return False, {"failure":True}


def RunSPIDER( time_dict, dirs, COUPLER_options, loop_counter, runtime_helpfile ):
    '''
    Wrapper function for running SPIDER.
    This wrapper handles cases where SPIDER fails to find a solution.
    '''

    # info
    PrintHalfSeparator()
    log.info("Running SPIDER...")
    log.debug("IC_INTERIOR = " + str(COUPLER_options["IC_INTERIOR"]))

    # parameters
    max_attempts = 7        # maximum number of attempts
    step_sf = 1.0           # step scale factor at attempt 1
    atol_sf = 1.0           # tolerance scale factor at attempt 1

    # tracking
    spider_success = False  # success?
    temp_options = {}       # COUPLER_options dict to be used for attempts
    attempts = 0            # number of attempts so far

    # make attempts
    while not spider_success:
        attempts += 1
        log.info("Attempt %d" % attempts)

        # run SPIDER
        temp_options = copy.deepcopy(COUPLER_options)
        spider_success, temp_options = _try_spider(time_dict, dirs, temp_options, loop_counter, runtime_helpfile, step_sf, atol_sf)

        if spider_success:
            # success
            log.info("Attempt %d succeeded" % attempts)
        else:
            # failure
            log.warning("Attempt %d failed" % attempts)
            if attempts >= max_attempts:
                # give up
                break
            else:
                # try again (change tolerance and step size)
                step_sf *= 0.5 
                atol_sf *= 4.0
    
    # check status
    if spider_success:
        # success after some attempts
        return temp_options
    else:
        # failure of all attempts
        UpdateStatusfile(dirs, 21)
        raise Exception("An error occurred when executing SPIDER (made %d attempts)" % attempts)


