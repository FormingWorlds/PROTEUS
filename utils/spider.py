# Function and classes used to run SPIDER

from utils.modules_ext import *
from utils.constants import *
from utils.helper import *

#===================================================================
# CLASSES
#===================================================================

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
            print('cannot find file: %s' % self.filename )
            print('please specify times for which data exists')
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
            print('dictionary for %s does not exist', keys )
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
        MIX = (phi<0.999) & (phi>0.001)
        MIX = MIX * 1.0 # convert to float array
        # set single phase region to nan to prevent plotting
        MIX[MIX==0] = np.nan
        return MIX

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

#====================================================================
# FUNCTIONS
#====================================================================

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

#====================================================================
def get_SPIDER_1D_lookup( infile ):

    ''' return 1-D lookup object using linear interpolation'''

    data_a, size_a = get_column_data_from_SPIDER_lookup_file( infile )
    xx = data_a[:,0]
    yy = data_a[:,1]
    # will not allow extrpolation beyond the bounds without an extra
    # argument
    lookup_o = interp1d( xx, yy, kind='linear' )
    return lookup_o

#====================================================================
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

#====================================================================
def get_all_output_times( odir='output' ):

    '''get all times (in Myrs) from the json files located in the
       output directory'''

    # locate times to process based on files located in odir/
    file_l = [f for f in os.listdir(odir) if os.path.isfile(os.path.join(odir,f))]
    if not file_l:
        print('output directory contains no files')
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

#====================================================================
def get_all_output_pkl_times( odir='output' ):

    '''get all times (in Myrs) from the pkl files located in the
       output directory'''

    # locate times to process based on files located in odir/
    file_l = [f for f in os.listdir(odir) if os.path.isfile(os.path.join(odir,f))]
    if not file_l:
        print('output directory contains no PKL files')
        sys.exit(0)

    time_l = [fname for fname in file_l]
    time_l = list(filter(lambda a: a.endswith('pkl'), time_l))

    # Filter and split files
    time_l = [ file for file in time_l if not file.startswith("orig_")]
    time_l = [ time.split('.pkl')[0] for time in time_l ]
    time_l = [ int(time.split('_atm')[0]) for time in time_l ]
    
    # ascending order
    time_l = sorted( time_l, key=int)
    time_a = np.array( time_l )

    return time_a



#====================================================================
def get_dict_values_for_times( keys, time_l, indir='output' ):

    data_l = []

    for time in time_l:
        filename = os.path.join( indir, '{}.json'.format(time) )
        myjson_o = MyJSON( filename )
        values_a = myjson_o.get_dict_values( keys )
        data_l.append( values_a )

    data_a = np.array( data_l )

    # rows time, cols data
    data_a.reshape( (len(time_l),-1 ) )
    # rows data, cols time
    data_a = data_a.transpose()

    return data_a

#====================================================================
def get_dict_surface_values_for_times( keys_t, time_l, indir='output'):

    '''Similar to above, but only loop over all times once and get
       all requested (surface / zero index) data in one go'''

    data_l = []

    for time in time_l:
        filename = os.path.join( indir, '{}.json'.format(time) )
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

#====================================================================
def get_dict_surface_values_for_specific_time( keys_t, time, indir='output'):

    '''Similar to above, but only loop over all times once and get
       all requested (surface / zero index) data in one go'''

    data_l = []

    filename = os.path.join( indir, '{}.json'.format(time) )
    myjson_o = MyJSON( filename )
    for key in keys_t:
        value = myjson_o.get_dict_values( key )
        data_l.append( value )


    return np.array(data_l)


#====================================================================
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
    dgdr = 4*np.pi*phys.G*rho - 2*phys.G*m/r**3

    return [dpdr,dmdr,dgdr]

#====================================================================
def get_radius_array_static_structure( radius, *myargs ):

    R_core = myargs[1]
    num = myargs[4]

    return np.linspace(radius,R_core,num)

#====================================================================
def get_static_structure_for_radius( radius, *myargs ):

    '''get static structure (pressure, mass, and gravity) for an
       input radius'''

    M_earth = myargs[0]
    R_core = myargs[1]
    num = myargs[4]
    g_Earth = gravity( M_earth, radius )
    z0 = [0,M_earth,g_Earth]
    r = get_radius_array_static_structure( radius, *myargs )
    z = odeint( get_deriv_static_structure, z0, r, args=myargs )

    return z

#====================================================================
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

#====================================================================
def get_myargs_static_structure( rho_interp1d ):

    # some constants taken from here (not the best reference)
    # https://www.sciencedirect.com/topics/earth-and-planetary-sciences/earth-core

    # hard-coded parameters here
    M_earth = 5.972E24 # kg
    # we want to match the mass and gravity at the core radius
    # and the core is assumed static and unchanging
    R_core = 3485000.0 # m
    M_core = 1.94E24 # kg
    G_core = gravity( M_core, R_core )
    # number of layers
    # FIXME: for plotting this might explain mismatch between
    # atmosphere and mantle temperature at the surface?
    num = 1000

    # tuple of arguments required for functions
    myargs = (M_earth,R_core,M_core,G_core,num,rho_interp1d)

    return myargs

#====================================================================
def solve_for_planetary_radius( rho_interp1d ):

    '''simple integrator for static structure equations based on the
       approach outlined in Valencia et al. (2007)'''

    # initial guess
    R_earth = 6371000.0 # m

    myargs = get_myargs_static_structure( rho_interp1d )

    radius = newton( get_difference_static_structure, R_earth,
        args=myargs, maxiter=500 )

    check_static_structure( radius, *myargs )

    return radius

#====================================================================
def check_static_structure( radius, *myargs ):

    '''compute relative accuracy of gravity'''

    G_core = myargs[3]
    dg = get_difference_static_structure( radius, *myargs )
    reldg = np.abs( dg/G_core )
    if reldg > 1.0e-6:
        print( 'WARNING: g relative accuracy= {}'.format(reldg) )

#====================================================================
def plot_static_structure( radius, rho_interp1d ):

    myargs = get_myargs_static_structure( rho_interp1d )

    radius_a = get_radius_array_static_structure( radius, *myargs )
    radius_a *= 1.0E-3 # to km
    z = get_static_structure_for_radius( radius, *myargs )

    pressure_a = z[:,0]
    rho_a = rho_interp1d( pressure_a )
    pressure_a *= 1.0E-9 # to GPa
    rho_a *= 1.0E-3 # to g/cc
    mass_a = z[:,1]
    gravity_a = z[:,2]

    fig, axs = plt.subplots(2,2,sharex=True, sharey=False)
    fig.set_figheight(6)
    fig.set_figwidth(8)

    ax0 = axs[0,0]
    ax1 = axs[0,1]
    ax2 = axs[1,0]
    ax3 = axs[1,1]

    ax0.plot( radius_a, pressure_a, 'k-' )
    ax0.set_ylabel( 'Pressure (GPa)' )
    ax1.plot( radius_a, mass_a, 'k-' )
    ax1.set_ylabel( 'Mass (kg)' )
    ax2.plot( radius_a, gravity_a, 'k-' )
    ax2.set_xlabel( 'Radius (km)' )
    ax2.set_ylabel( 'Gravity (m/s^2)' )
    ax3.plot( radius_a, rho_a, 'k-' )
    ax3.set_xlabel( 'Radius (km)' )
    ax3.set_ylabel( 'Density (g/cc)' )

    radius_title = np.round(radius,0) * 1.0E-3 # to km
    fig.suptitle('Planetary radius= {} km'.format(radius_title))
    fig.savefig( "static_structure.pdf", bbox_inches="tight")

#====================================================================

def RunSPIDER( time_dict, dirs, COUPLER_options, loop_counter, runtime_helpfile ):

    SPIDER_options_file = dirs["output"]+"/init_spider.opts"
    SPIDER_options_file_orig = dirs["utils"]+"/init_spider.opts"

    print("IC_INTERIOR =",COUPLER_options["IC_INTERIOR"])

    # First run
    if (loop_counter["init"] == 0):
        if os.path.isfile(SPIDER_options_file):
            os.remove(SPIDER_options_file)
        shutil.copy(SPIDER_options_file_orig,SPIDER_options_file)

    # Define which volatiles to track in SPIDER
    species_call = ""
    for vol in volatile_species: 
        if COUPLER_options[vol+"_included"]:
            species_call = species_call + "," + vol
    species_call = species_call[1:] # Remove "," in front

    # Recalculate time stepping
    if COUPLER_options["IC_INTERIOR"] == 2:  

        # Current step
        json_file   = MyJSON( dirs["output"]+'/{}.json'.format(int(time_dict["planet"])) )
        step        = json_file.get_dict(['step'])

        dtmacro     = float(COUPLER_options["dtmacro"])
        dtswitch    = float(COUPLER_options["dtswitch"])

        # Time resolution adjustment in the beginning
        if time_dict["planet"] < 1000:
            dtmacro = 10
            dtswitch = 50
        if time_dict["planet"] < 100:
            dtmacro = 2
            dtswitch = 5
        if time_dict["planet"] < 10:
            dtmacro = 1
            dtswitch = 1

        # Runtime left
        dtime_max   = time_dict["target"] - time_dict["planet"]

        # Limit Atm-Int switch
        dtime       = np.min([ dtime_max, dtswitch ])

        # Number of total steps until currently desired switch/end time
        COUPLER_options["nstepsmacro"] =  step + math.ceil( dtime / dtmacro )

        print("TIME OPTIONS IN RUNSPIDER:")
        print(dtmacro, dtswitch, dtime_max, dtime, COUPLER_options["nstepsmacro"])


    # For init loop
    else:
        dtmacro     = 0

    # Prevent interior oscillations during last-stage freeze-out
    net_loss = COUPLER_options["F_atm"]
    if len(runtime_helpfile) > 100 and runtime_helpfile.iloc[-1]["Phi_global"] <= COUPLER_options["phi_crit"]:
        net_loss = np.amax([abs(COUPLER_options["F_atm"]), COUPLER_options["F_eps"]])
        if debug:
            print("Prevent interior oscillations during last-stage freeze-out: F_atm =", COUPLER_options["F_atm"], "->", net_loss)

    ### SPIDER base call sequence 
    call_sequence = [   
                        dirs["spider"]+"/spider", 
                        "-options_file",          SPIDER_options_file, 
                        "-outputDirectory",       dirs["output"],
                        "-IC_INTERIOR",           str(COUPLER_options["IC_INTERIOR"]),
                        "-IC_ATMOSPHERE",         str(COUPLER_options["IC_ATMOSPHERE"]),
                        "-SURFACE_BC",            str(COUPLER_options["SURFACE_BC"]), 
                        "-surface_bc_value",      str(net_loss), 
                        "-teqm",                  str(COUPLER_options["T_eqm"]), 
                        "-nstepsmacro",           str(COUPLER_options["nstepsmacro"]), 
                        "-dtmacro",               str(dtmacro), 
                        "-radius",                str(COUPLER_options["radius"]), 
                        "-gravity",               "-"+str(COUPLER_options["gravity"]), 
                        "-coresize",              str(COUPLER_options["planet_coresize"]),
                        "-volatile_names",        str(species_call)
                    ]

    # Min of fractional and absolute Ts poststep change
    if time_dict["planet"] > 0:
        dTs_frac = float(COUPLER_options["tsurf_poststep_change_frac"]) * float(runtime_helpfile["T_surf"].iloc[-1])
        dT_int_max = np.min([ float(COUPLER_options["tsurf_poststep_change"]), float(dTs_frac) ])
        call_sequence.extend(["-tsurf_poststep_change", str(dT_int_max)])
    else:
        call_sequence.extend(["-tsurf_poststep_change", str(COUPLER_options["tsurf_poststep_change"])])

    # Define distribution coefficients and total mass/surface pressure for volatiles > 0
    for vol in volatile_species:
        if COUPLER_options[vol+"_included"]:

            # Set atmospheric pressure based on helpfile output
            if loop_counter["total"] > loop_counter["init_loops"]:
                key = vol+"_initial_atmos_pressure"
                val = float(runtime_helpfile[vol+"_mr"].iloc[-1]) * float(runtime_helpfile["P_surf"].iloc[-1]) * 1.0e5   # convert bar to Pa
                COUPLER_options[key] = val

            # Load volatiles
            if COUPLER_options["IC_ATMOSPHERE"] == 1:
                call_sequence.extend(["-"+vol+"_initial_total_abundance", str(COUPLER_options[vol+"_initial_total_abundance"])])
            elif COUPLER_options["IC_ATMOSPHERE"] == 3:
                call_sequence.extend(["-"+vol+"_initial_atmos_pressure", str(COUPLER_options[vol+"_initial_atmos_pressure"])])

            # Exception for N2 case: reduced vs. oxidized
            if vol == "N2" and COUPLER_options["N2_partitioning"] == 1:
                volatile_distribution_coefficients["N2_henry"] = volatile_distribution_coefficients["N2_henry_reduced"]
                volatile_distribution_coefficients["N2_henry_pow"] = volatile_distribution_coefficients["N2_henry_pow_reduced"]

            call_sequence.extend(["-"+vol+"_henry", str(volatile_distribution_coefficients[vol+"_henry"])])
            call_sequence.extend(["-"+vol+"_henry_pow", str(volatile_distribution_coefficients[vol+"_henry_pow"])])
            call_sequence.extend(["-"+vol+"_kdist", str(volatile_distribution_coefficients[vol+"_kdist"])])
            call_sequence.extend(["-"+vol+"_kabs", str(volatile_distribution_coefficients[vol+"_kabs"])])
            call_sequence.extend(["-"+vol+"_molar_mass", str(molar_mass[vol])])
            call_sequence.extend(["-"+vol+"_SOLUBILITY 1"])

    # With start of the main loop only:
    # Volatile specific options: post step settings, restart filename
    if COUPLER_options["IC_INTERIOR"] == 2:
        call_sequence.extend([ 
                                "-ic_interior_filename", 
                                str(dirs["output"]+"/"+COUPLER_options["ic_interior_filename"]),
                                "-activate_poststep", 
                                "-activate_rollback"
                             ])
        for vol in volatile_species:
            if COUPLER_options[vol+"_included"]:
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
    if len(runtime_helpfile) > 30 and loop_counter["total"] > loop_counter["init_loops"] :

        # Check convergence for interior cycles
        run_int = runtime_helpfile.loc[runtime_helpfile['Input']=='Interior']

        # First, relax too restrictive dTs
        if run_int["Time"].iloc[-1] == run_int["Time"].iloc[-3]:
            if COUPLER_options["tsurf_poststep_change"] <= 300:
                COUPLER_options["tsurf_poststep_change"] += 10
                print(">>> Raise dT poststep_changes:", COUPLER_options["tsurf_poststep_change"], COUPLER_options["tsurf_poststep_change_frac"])
            else:
                print(">> dTs_int too high! >>", COUPLER_options["tsurf_poststep_change"], "K")
        # Slowly limit again if time advances smoothly
        if (run_int["Time"].iloc[-1] != run_int["Time"].iloc[-3]) and COUPLER_options["tsurf_poststep_change"] > 30:
            COUPLER_options["tsurf_poststep_change"] -= 10
            print(">>> Lower tsurf_poststep_change poststep changes:", COUPLER_options["tsurf_poststep_change"], COUPLER_options["tsurf_poststep_change_frac"])

        if run_int["Time"].iloc[-1] == run_int["Time"].iloc[-7]:
            if "solver_tolerance" not in COUPLER_options:
                COUPLER_options["solver_tolerance"] = 1.0e-10
            if COUPLER_options["solver_tolerance"] < 1.0e-2:
                COUPLER_options["solver_tolerance"] = float(COUPLER_options["solver_tolerance"])*2.
                print(">>> ADJUST tolerances:", COUPLER_options["solver_tolerance"])
            COUPLER_options["adjust_tolerance"] = 1
            print(">>> CURRENT TOLERANCES:", COUPLER_options["solver_tolerance"])

        # If tolerance was adjusted, restart SPIDER w/ new tolerances
        if "adjust_tolerance" in COUPLER_options:
            print(">>>>> >>>>> RESTART W/ ADJUSTED TOLERANCES")
            call_sequence.extend(["-ts_sundials_atol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-ts_sundials_rtol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosts_snes_atol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosts_snes_rtol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosts_ksp_atol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosts_ksp_rtol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosic_ksp_rtol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosic_ksp_atol", str(COUPLER_options["solver_tolerance"])])

    # Runtime info
    PrintSeparator()
    print("Running SPIDER... (loop counter = ", loop_counter, ")")
    if debug:
        print("   Flags:")
        for flag in call_sequence:
            print("   ",flag)
        print()

    call_string = " ".join(call_sequence)

    # Run SPIDER
    if debug:
        spider_print = sys.stdout
    else:
        spider_print = open(dirs["output"]+"spider_recent.log",'w')

    subprocess.run([call_string],shell=True,check=True,stdout=spider_print)

    if not debug:
        spider_print.close()

    # Update restart filename for next SPIDER run
    COUPLER_options["ic_interior_filename"] = natural_sort([os.path.basename(x) for x in glob.glob(dirs["output"]+"/*.json")])[-1]

    return COUPLER_options
