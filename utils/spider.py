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



# Solve partial pressures functions
# Written by Dan Bower
# See the related issue on the PROTEUS GitHub page:-
# https://github.com/FormingWorlds/PROTEUS/issues/42
# Paper to cite:-
# https://www.sciencedirect.com/science/article/pii/S0012821X22005301

# Solve for the equilibrium chemistry of a magma ocean atmosphere
# for a given set of solubility and redox relations

#====================================================================
class OxygenFugacity:
    """log10 oxygen fugacity as a function of temperature"""

    def __init__(self, model='oneill'):
        self.callmodel = getattr(self, model)

    def __call__(self, T, fO2_shift=0):
        '''Return log10 fO2'''
        return self.callmodel(T) + fO2_shift

    def fischer(self, T):
        '''Fischer et al. (2011) IW'''
        return 6.94059 -28.1808*1E3/T

    def oneill(self, T): 
        '''O'Neill and Eggin (2002) IW'''
        return 2*(-244118+115.559*T-8.474*T*np.log(T))/(np.log(10)*8.31441*T)

#====================================================================
class ModifiedKeq:
    """Modified equilibrium constant (includes fO2)"""

    def __init__(self, Keq_model, fO2_model='oneill'):
        self.fO2 = OxygenFugacity(fO2_model)
        self.callmodel = getattr(self, Keq_model)

    def __call__(self, T, fO2_shift=0):
        fO2 = self.fO2(T, fO2_shift)
        Keq, fO2_stoich = self.callmodel(T)
        Geq = 10**(Keq-fO2_stoich*fO2)
        return Geq

    def schaefer_CH4(self, T): 
        '''Schaefer log10Keq for CO2 + 2H2 = CH4 + fO2'''
        # second argument returns stoichiometry of O2
        return (-16276/T - 5.4738, 1)

    def schaefer_C(self, T): 
        '''Schaefer log10Keq for CO2 = CO + 0.5 fO2'''
        return (-14787/T + 4.5472, 0.5) 

    def schaefer_H(self, T): 
        '''Schaefer log10Keq for H2O = H2 + 0.5 fO2'''
        return (-12794/T + 2.7768, 0.5) 

    def janaf_C(self, T): 
        '''JANAF log10Keq, 1500 < K < 3000 for CO2 = CO + 0.5 fO2'''
        return (-14467.511400133637/T + 4.348135473316284, 0.5) 

    def janaf_H(self, T): 
        '''JANAF log10Keq, 1500 < K < 3000 for H2O = H2 + 0.5 fO2'''
        return (-13152.477779978302/T + 3.038586383273608, 0.5) 

#====================================================================
class Solubility:
    """Solubility base class.  All p in bar"""

    def __init__(self, composition):
        self.callmodel = getattr(self, composition)

    def power_law(self, p, const, exponent):
        return const*p**exponent

    def __call__(self, p, *args):
        '''Dissolved concentration in ppmw in the melt'''
        return self.callmodel(p, *args)

#====================================================================
class SolubilityH2O(Solubility):
    """H2O solubility models"""

    # below default gives the default model used
    def __init__(self, composition='peridotite'):
        super().__init__(composition)

    def anorthite_diopside(self, p):
        '''Newcombe et al. (2017)'''
        return self.power_law(p, 727, 0.5)

    def peridotite(self, p):
        '''Sossi et al. (2022)'''
        return self.power_law(p, 524, 0.5)

    def basalt_dixon(self, p):
        '''Dixon et al. (1995) refit by Paolo Sossi'''
        return self.power_law(p, 965, 0.5)

    def basalt_wilson(self, p):
        '''Hamilton (1964) and Wilson and Head (1981)'''
        return self.power_law(p, 215, 0.7)

    def lunar_glass(self, p):
        '''Newcombe et al. (2017)'''
        return self.power_law(p, 683, 0.5)

#====================================================================
class SolubilityCO2(Solubility):
    """CO2 solubility models"""

    def __init__(self, composition='basalt_dixon'):
        super().__init__(composition)

    def basalt_dixon(self, p, temp):
        '''Dixon et al. (1995)'''
        ppmw = (3.8E-7)*p*np.exp(-23*(p-1)/(83.15*temp))
        ppmw = 1.0E4*(4400*ppmw) / (36.6-44*ppmw)
        return ppmw

#====================================================================
class SolubilityN2(Solubility):
    """N2 solubility models"""

    def __init__(self, composition='libourel'):
        super().__init__(composition)

    def libourel(self, p):
        '''Libourel et al. (2003)'''
        ppmw = self.power_law(p, 0.0611, 1.0)
        return ppmw
    
#====================================================================
# FUNCTIONS
#====================================================================

#====================================================================
def solvepp_get_partial_pressures(pin, fO2_shift, global_d):
    """Partial pressure of all considered species"""

    # we only need to know pH2O, pCO2, and pN2, since reduced species
    # can be directly determined from equilibrium chemistry

    pH2O, pCO2, pN2 = pin

    # return results in dict, to be explicit about which pressure
    # corresponds to which volatile
    p_d = {}
    p_d['H2O'] = pH2O
    p_d['CO2'] = pCO2
    p_d['N2'] = pN2

    # pH2 from equilibrium chemistry
    gamma = ModifiedKeq('janaf_H')
    gamma = gamma(global_d['temperature'], fO2_shift)
    p_d['H2'] = gamma*pH2O

    # pCO from equilibrium chemistry
    gamma = ModifiedKeq('janaf_C')
    gamma = gamma(global_d['temperature'], fO2_shift)
    p_d['CO'] = gamma*pCO2

    if global_d['is_CH4'] is True:
        gamma = ModifiedKeq('schaefer_CH4')
        gamma = gamma(global_d['temperature'], fO2_shift)
        p_d['CH4'] = gamma*pCO2*p_d['H2']**2.0
    else:
        p_d['CH4'] = 0

    return p_d

#====================================================================
def solvepp_get_total_pressure(pin, fO2_shift, global_d):
    """Sum partial pressures to get total pressure"""

    p_d = solvepp_get_partial_pressures(pin, fO2_shift, global_d)
    ptot = sum(p_d.values())

    return ptot

#====================================================================
def solvepp_atmosphere_mass(pin, fO2_shift, global_d):
    """Atmospheric mass of volatiles and totals for H, C, and N"""

    p_d = solvepp_get_partial_pressures(pin, fO2_shift, global_d)
    mu_atm = solvepp_atmosphere_mean_molar_mass(pin, fO2_shift, global_d)

    mass_atm_d = {}
    for key, value in p_d.items():
        # 1.0E5 because pressures are in bar
        mass_atm_d[key] = value*1.0E5/global_d['little_g']
        mass_atm_d[key] *= 4.0*np.pi*global_d['planetary_radius']**2.0
        mass_atm_d[key] *= molar_mass[key]/mu_atm

    # total mass of H
    mass_atm_d['H'] = mass_atm_d['H2'] / molar_mass['H2']
    mass_atm_d['H'] += mass_atm_d['H2O'] / molar_mass['H2O']
    # note factor 2 below to account for stoichiometry
    mass_atm_d['H'] += mass_atm_d['CH4'] * 2 / molar_mass['CH4']
    # below converts moles of H2 to mass of H
    mass_atm_d['H'] *= molar_mass['H2']

    # total mass of C
    mass_atm_d['C'] = mass_atm_d['CO'] / molar_mass['CO']
    mass_atm_d['C'] += mass_atm_d['CO2'] / molar_mass['CO2']
    mass_atm_d['C'] += mass_atm_d['CH4'] / molar_mass['CH4']
    # below converts moles of C to mass of C
    mass_atm_d['C'] *= molar_mass['C']

    # total mass of N
    mass_atm_d['N'] = mass_atm_d['N2']

    return mass_atm_d

#====================================================================
def solvepp_atmosphere_mean_molar_mass(pin, fO2_shift, global_d):
    """Mean molar mass of the atmosphere"""

    p_d = solvepp_get_partial_pressures(pin, fO2_shift, global_d)
    ptot = solvepp_get_total_pressure(pin, fO2_shift, global_d)

    mu_atm = 0
    for key, value in p_d.items():
        mu_atm += molar_mass[key]*value
    mu_atm /= ptot

    return mu_atm

#====================================================================
def solvepp_dissolved_mass(pin, fO2_shift, global_d):
    """Volatile masses in the (molten) mantle"""

    mass_int_d = {}

    p_d = solvepp_get_partial_pressures(pin, fO2_shift, global_d)

    prefactor = 1E-6*global_d['mantle_mass']*global_d['mantle_melt_fraction']

    # H2O
    sol_H2O = SolubilityH2O() # gets the default solubility model
    ppmw_H2O = sol_H2O(p_d['H2O'])
    mass_int_d['H2O'] = prefactor*ppmw_H2O

    # CO2
    sol_CO2 = SolubilityCO2() # gets the default solubility model
    ppmw_CO2 = sol_CO2(p_d['CO2'], global_d['temperature'])
    mass_int_d['CO2'] = prefactor*ppmw_CO2

    # N2
    sol_N2 = SolubilityN2() # gets the default solubility model
    ppmw_N2 = sol_N2(p_d['N2'])
    mass_int_d['N2'] = prefactor*ppmw_N2

    # now get totals of H, C, N
    mass_int_d['H'] = mass_int_d['H2O']*(molar_mass['H2']/molar_mass['H2O'])
    mass_int_d['C'] = mass_int_d['CO2']*(molar_mass['C']/molar_mass['CO2'])
    mass_int_d['N'] = mass_int_d['N2']

    return mass_int_d

#====================================================================
def solvepp_func(pin, fO2_shift, global_d, mass_target_d):
    """Function to compute the residual of the mass balance"""

    # get atmospheric masses
    mass_atm_d = solvepp_atmosphere_mass(pin, fO2_shift, global_d)

    # get (molten) mantle masses
    mass_int_d = solvepp_dissolved_mass(pin, fO2_shift, global_d)

    # compute residuals
    res_l = []
    for vol in ['H','C','N']:
        # absolute residual
        res = mass_atm_d[vol] + mass_int_d[vol] - mass_target_d[vol]
        # if target is not zero, compute relative residual
        # otherwise, zero target is already solved with zero pressures
        if mass_target_d[vol]:
            res /= mass_target_d[vol]
        res_l.append(res)

    return res_l

#====================================================================
def solvepp_get_initial_pressures(target_d):
    """Get initial guesses of partial pressures"""

    # all bar
    pH2O = 1*np.random.random_sample() # H2O less soluble than CO2
    pCO2 = 10*np.random.random_sample() # just a guess
    pN2 = 10*np.random.random_sample()

    if target_d['H'] == 0:
        pH2O = 0
    if target_d['C'] == 0:
        pCO2 = 0
    if target_d['N'] == 0:
        pN2 = 0

    return pH2O, pCO2, pN2

#====================================================================
def solvepp_equilibrium_atmosphere(N_ocean_moles, CH_ratio, fO2_shift, global_d, Nitrogen):
    """Calculate equilibrium chemistry of the atmosphere"""

    H_kg = N_ocean_moles * global_d['ocean_moles'] * molar_mass['H2']
    C_kg = CH_ratio * H_kg
    N_kg = Nitrogen * 1.0E-6 * global_d['mantle_mass']
    target_d = {'H': H_kg, 'C': C_kg, 'N': N_kg}

    count = 0
    ier = 0
    # could in principle result in an infinite loop, if randomising
    # the ic never finds the physical solution (but in practice,
    # this doesn't seem to happen)
    while ier != 1:
        x0 = solvepp_get_initial_pressures(target_d)
        sol, info, ier, msg = fsolve(solvepp_func, x0, args=(fO2_shift, 
            global_d, target_d), full_output=True)
        count += 1
        # sometimes, a solution exists with negative pressures, which
        # is clearly non-physical.  Here, assert we must have positive
        # pressures.
        if any(sol<0):
            # if any negative pressures, report ier!=1
            ier = 0

    logging.info(f'Randomised initial conditions= {count}')

    p_d = solvepp_get_partial_pressures(sol, fO2_shift, global_d)
    # get residuals for output
    res_l = solvepp_func(sol, fO2_shift, global_d, target_d)

    # for convenience, add inputs to same dict
    p_d['N_ocean_moles'] = N_ocean_moles
    p_d['CH_ratio'] = CH_ratio
    p_d['fO2_shift'] = fO2_shift
    # for debugging/checking, add success initial condition
    # that resulted in a converged solution with positive pressures
    p_d['pH2O_0'] = x0[0]
    p_d['pCO2_0'] = x0[1]
    p_d['pN2_0'] = x0[2]
    # also for debugging/checking, report residuals
    p_d['res_H'] = res_l[0]
    p_d['res_C'] = res_l[1]
    p_d['res_N'] = res_l[2]

    return p_d


#====================================================================
def solvepp_doit(COUPLER_options):

    print("Solving for eqm partial pressures at surface")

    # Volatiles that are solved-for using this eqm calculation
    solvepp_vols = ['H2O', 'CO2', 'N2', 'H2', 'CO', 'CH4']

    # Dictionary for passing parameters around for the partial pressure calculations
    global_d = {}

    # Don't change these
    global_d['mantle_melt_fraction'] =  1.0 # fraction of mantle that is molten
    global_d['ocean_moles'] =           7.68894973907177e+22 # moles of H2 (or H2O) in one present-day Earth ocean
    global_d['is_CH4'] =                True # include CH4

    # These require initial guesses
    # global_d['mantle_mass'] =           4.208261222595111e+24 # kg
    # global_d['temperature'] =           2000.0 # K
    global_d['mantle_mass'] =       COUPLER_options['mantle_mass_guess'] # kg
    global_d['temperature'] =       COUPLER_options['T_surf_guess'] # K

    # These are defined by the proteus configuration file
    global_d['planetary_radius'] =  COUPLER_options['radius']
    global_d['little_g'] =          COUPLER_options['gravity']
    N_ocean_moles =                 COUPLER_options['hydrogen_earth_oceans']
    CH_ratio =                      COUPLER_options['CH_ratio']
    fO2_shift =                     COUPLER_options['fO2_shift_IW']
    Nitrogen =                      COUPLER_options['nitrogen_ppmw']

    # Solve for partial pressures
    p_d = solvepp_equilibrium_atmosphere(N_ocean_moles, CH_ratio, fO2_shift, global_d, Nitrogen)

    partial_pressures = {}
    for s in solvepp_vols:
        print("   p_%s = %f bar" % (s,p_d[s]))
        partial_pressures[s] = p_d[s] * 1.0e5 # Convert from bar to Pa

    return partial_pressures



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

        if debug:
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
                        "-OXYGEN_FUGACITY_offset",str(COUPLER_options["fO2_shift_IW"]),
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
        if COUPLER_options[vol+"_included"] == 1:

            # Set atmospheric pressure based on helpfile output, if required
            if loop_counter["total"] > loop_counter["init_loops"]:
                key = vol+"_initial_atmos_pressure"
                val = float(runtime_helpfile[vol+"_mr"].iloc[-1]) * float(runtime_helpfile["P_surf"].iloc[-1]) * 1.0e5   # convert bar to Pa
                COUPLER_options[key] = val

            # Load volatiles
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
    PrintHalfSeparator()
    print("Running SPIDER...")
    print("IC_INTERIOR =",COUPLER_options["IC_INTERIOR"])
    if debug:
        flags = ""
        for flag in call_sequence:
            flags += " " + flag
        print("SPIDER call sequence: '%s'" % flags)

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
