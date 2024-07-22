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

def get_all_output_times( odir='output' ):
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

#====================================================================
def _try_spider( time_dict:dict, dirs:dict, COUPLER_options:dict, 
                IC_INTERIOR:int, loop_counter:dict, 
                hf_all:pd.DataFrame, hf_row:dict, 
                step_sf:float, atol_sf:float ):
    '''
    Try to run spider with the current configuration.
    '''

    step_sf = min(1.0, max(1.0e-10, step_sf))
    atol_sf = min(1.0e10, max(1.0e-10, atol_sf))

    SPIDER_options_file      = os.path.join(dirs["output"], "init_spider.opts")
    SPIDER_options_file_orig = os.path.join(dirs["utils"], "templates", "init_spider.opts")

    # First run
    if (loop_counter["init"] == 0):
        if os.path.isfile(SPIDER_options_file):
            os.remove(SPIDER_options_file)
        shutil.copy(SPIDER_options_file_orig,SPIDER_options_file)

    # Recalculate time stepping
    if IC_INTERIOR == 2:  

        # Current step number
        json_file   = MyJSON( dirs["output"]+'data/{}.json'.format(int(time_dict["planet"])) )
        step        = json_file.get_dict(['step'])

        # Time stepping adjustment
        if time_dict["planet"] < 2.0:
            # First year, use small step
            dtmacro = 1.0
            dtswitch = 1.0
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
                    dtprev = float(hf_all.iloc[-1]["Time"] - hf_all.iloc[-2]["Time"])
                else:
                    dtprev = COUPLER_options["dt_initial"]
                log.debug("Previous step size: %.2e yr"%dtprev)

                # Change in F_int 
                F_int_2  = hf_all.iloc[-2]["F_int"]
                F_int_1  = hf_all.iloc[-1]["F_int"]
                F_int_12 = abs(F_int_1 - F_int_2) 

                # Change in F_atm
                F_atm_2  = hf_all.iloc[-2]["F_atm"]
                F_atm_1  = hf_all.iloc[-1]["F_atm"]
                F_atm_12 = abs(F_atm_1 - F_atm_2)  

                # Change in global melt fraction
                phi_2  = hf_all.iloc[-2]["Phi_global"]
                phi_1  = hf_all.iloc[-1]["Phi_global"]
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
            dtmacro = np.ceil(dtswitch / nsteps)   # Ensures that dtswitch is divisible by nsteps
            dtswitch = nsteps * dtmacro

            log.info("New time-step is %1.2e years" % dtswitch)

        # Number of total steps until currently desired switch/end time
        nstepsmacro = step + nsteps

        log.debug("Time options in RunSPIDER: %.2e yrs in %d steps" % (dtmacro, nstepsmacro))

    # For init loop
    else:
        nstepsmacro = 1
        dtmacro     = 0
        dtswitch    = 0

    ### SPIDER base call sequence 
    call_sequence = [   
                        dirs["spider"]+"/spider", 
                        "-options_file",           SPIDER_options_file, 
                        "-outputDirectory",        dirs["output"]+'data/',
                        "-IC_INTERIOR",            "%d"  %(IC_INTERIOR),
                        "-OXYGEN_FUGACITY_offset", "%.6e"%(COUPLER_options["fO2_shift_IW"]),  # Relative to the specified buffer
                        "-surface_bc_value",       "%.6e"%(hf_row["F_atm"]), 
                        "-teqm",                   "%.6e"%(hf_row["T_eqm"]), 
                        "-n",                      "%d"  %(COUPLER_options["interior_nlev"]),
                        "-nstepsmacro",            "%d"  %(nstepsmacro), 
                        "-dtmacro",                "%.6e"%(dtmacro), 
                        "-radius",                 "%.6e"%(COUPLER_options["radius"]), 
                        "-gravity",                "%.6e"%(-1.0 * hf_row["gravity"]), 
                        "-coresize",               "%.6e"%(COUPLER_options["planet_coresize"]),
                        "-grain",                  "%.6e"%(COUPLER_options["grain_size"]),
                    ]

    # Min of fractional and absolute Ts poststep change
    if time_dict["planet"] > 0:
        dTs_frac = float(COUPLER_options["tsurf_poststep_change_frac"]) * float(hf_all["T_surf"].iloc[-1])
        dT_int_max = np.min([ float(COUPLER_options["tsurf_poststep_change"]), float(dTs_frac) ])
        call_sequence.extend(["-tsurf_poststep_change", str(dT_int_max)])
    else:
        call_sequence.extend(["-tsurf_poststep_change", str(COUPLER_options["tsurf_poststep_change"])])

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
                                "-ic_adiabat_entropy", str(COUPLER_options["ic_adiabat_entropy"]),
                                "-ic_dsdr", str(COUPLER_options["ic_dsdr"]) # initial dS/dr everywhere
                            ])

    # Gravitational separation of solid and melt phase, 0: off | 1: on
    if COUPLER_options["SEPARATION"] == 1:
        call_sequence.extend(["-SEPARATION", str(1)])

    # Mixing length parameterization: 1: variable | 2: constant
    call_sequence.extend(["-mixing_length", str(COUPLER_options["mixing_length"])])

    # Check for convergence, if not converging, adjust tolerances iteratively
    # if (loop_counter["total"] > loop_counter["init_loops"]) and (len(runtime_helpfile) > 50):

        # # Check convergence for interior cycles
        # run_int = runtime_helpfile.loc[runtime_helpfile['Input']=='Interior'].drop_duplicates(subset=['Time'], keep='last')

        # ref_idx = -3
        # if len(run_int["Time"]) < abs(ref_idx)-1:
        #     ref_idx = 0

        # First, relax too restrictive dTs
        # if run_int["Time"].iloc[-1] == run_int["Time"].iloc[ref_idx]:
        #     if COUPLER_options["tsurf_poststep_change"] <= 300:
        #         COUPLER_options["tsurf_poststep_change"] += 10
        #         log.warning(">>> Raise dT poststep_changes:", COUPLER_options["tsurf_poststep_change"], COUPLER_options["tsurf_poststep_change_frac"])
        #     else:
        #         log.warning(">> dTs_int too high! >>", COUPLER_options["tsurf_poststep_change"], "K")
                
        # Slowly limit again if time advances smoothly
        # if (run_int["Time"].iloc[-1] != run_int["Time"].iloc[ref_idx]) and COUPLER_options["tsurf_poststep_change"] > 30:
        #     COUPLER_options["tsurf_poststep_change"] -= 10
        #     log.warning(">>> Lower tsurf_poststep_change poststep changes:", COUPLER_options["tsurf_poststep_change"], COUPLER_options["tsurf_poststep_change_frac"])

        # if run_int["Time"].iloc[-1] == run_int["Time"].iloc[ref_idx]:
        #     if COUPLER_options["solver_tolerance"] < 1.0e-2:
        #         COUPLER_options["solver_tolerance"] = float(COUPLER_options["solver_tolerance"])*2.
        #         log.warning(">>> ADJUST tolerances:", COUPLER_options["solver_tolerance"])
        #     COUPLER_options["adjust_tolerance"] = 1
        #     log.warning(">>> CURRENT TOLERANCES:", COUPLER_options["solver_tolerance"])

        # If tolerance was adjusted, restart SPIDER w/ new tolerances
        # if "adjust_tolerance" in COUPLER_options:
        #     log.warning(">>>>> >>>>> RESTART W/ ADJUSTED TOLERANCES")
        #     call_sequence.extend(["-atmosts_snes_atol", str(COUPLER_options["solver_tolerance"])])
        #     call_sequence.extend(["-atmosts_snes_rtol", str(COUPLER_options["solver_tolerance"])])
        #     call_sequence.extend(["-atmosts_ksp_atol", str(COUPLER_options["solver_tolerance"])])
        #     call_sequence.extend(["-atmosts_ksp_rtol", str(COUPLER_options["solver_tolerance"])])
        #     call_sequence.extend(["-atmosic_ksp_rtol", str(COUPLER_options["solver_tolerance"])])
        #     call_sequence.extend(["-atmosic_ksp_atol", str(COUPLER_options["solver_tolerance"])])

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

    # Check status
    return bool(proc.returncode == 0)


def RunSPIDER( time_dict:dict, dirs:dict, COUPLER_options:dict, 
              IC_INTERIOR:int, loop_counter:dict, 
              hf_all:pd.DataFrame, hf_row:dict ):
    '''
    Wrapper function for running SPIDER.
    This wrapper handles cases where SPIDER fails to find a solution.
    '''

    # info
    PrintHalfSeparator()
    log.info("Running SPIDER...")
    log.debug("    IC_INTERIOR = %d"%IC_INTERIOR)

    # parameters
    max_attempts = 7        # maximum number of attempts
    step_sf = 1.0           # step scale factor at attempt 1
    atol_sf = 1.0           # tolerance scale factor at attempt 1

    # tracking
    spider_success = False  # success?
    attempts = 0            # number of attempts so far

    # make attempts
    while not spider_success:
        attempts += 1
        log.info("Attempt %d" % attempts)

        # run SPIDER
        spider_success = _try_spider(time_dict, dirs, COUPLER_options, IC_INTERIOR, loop_counter, hf_all, hf_row, step_sf, atol_sf)

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
        return True
    else:
        # failure of all attempts
        UpdateStatusfile(dirs, 21)
        raise Exception("An error occurred when executing SPIDER (made %d attempts)" % attempts)


def ReadSPIDER(dirs:dict, time_dict:dict, COUPLER_options:dict, prev_T_magma:float):
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

    # Fill the new dict
    output["M_mantle_liquid"] = float(data_a[0])
    output["M_mantle_solid"]  = float(data_a[1])
    output["M_mantle"]        = float(data_a[2])
    output["M_core"]          = float(data_a[3])

    # Surface properties
    output["T_magma"]         = float(data_a[4])
    output["Phi_global"]      = float(data_a[5])  # global melt fraction
    output["F_int"]           = float(data_a[6])  # Heat flux from interior
    output["RF_depth"]        = float(data_a[7])/COUPLER_options["radius"]  # depth of rheological front

    # Do not allow warming after init stage has completed
    if (COUPLER_options["prevent_warming"]) and (time_dict["planet"] > 5.0):
        output["T_magma"] = min(output["T_magma"], prev_T_magma)

   
    # Manually calculate heat flux at near-surface from energy gradient
    json_file   = MyJSON( dirs["output"]+'/data/{}.json'.format(sim_time) )
    Etot        = json_file.get_dict_values(['data','Etot_b'])
    rad         = json_file.get_dict_values(['data','radius_b'])
    area        = json_file.get_dict_values(['data','area_b'])
    E0          = Etot[1] - (Etot[2]-Etot[1]) * (rad[2]-rad[1]) / (rad[1]-rad[0])
    F_int2      = E0/area[0]
    log.debug(">>>>>>> F_int2: %.2e, F_int: %.2e" % (F_int2, output["F_int"]) )

    # Limit F_int to positive values
    if COUPLER_options["prevent_warming"]:
        output["F_int"] = max(1.0e-8, output["F_int"])

    # Check NaNs
    if np.isnan(output["T_magma"]):
        raise Exception("Magma ocean temperature is NaN")

    return sim_time, output

