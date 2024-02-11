# Function and classes used to run SPIDER

from utils.modules_ext import *
from utils.constants import *
from utils.helper import *

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
#====================================================================


# Solve partial pressures functions
# Written by Dan Bower
# See the related issue on the PROTEUS GitHub page:-
# https://github.com/FormingWorlds/PROTEUS/issues/42
# Paper to cite:-
# https://www.sciencedirect.com/science/article/pii/S0012821X22005301

# Solve for the equilibrium chemistry of a magma ocean atmosphere
# for a given set of solubility and redox relations

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

class Solubility:
    """Solubility base class.  All p in bar"""

    def __init__(self, composition):
        self.callmodel = getattr(self, composition)

    def power_law(self, p, const, exponent):
        return const*p**exponent

    def __call__(self, p, *args):
        '''Dissolved concentration in ppmw in the melt'''
        return self.callmodel(p, *args)

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

class SolubilityCO2(Solubility):
    """CO2 solubility models"""

    def __init__(self, composition='basalt_dixon'):
        super().__init__(composition)

    def basalt_dixon(self, p, temp):
        '''Dixon et al. (1995)'''
        ppmw = (3.8E-7)*p*np.exp(-23*(p-1)/(83.15*temp))
        ppmw = 1.0E4*(4400*ppmw) / (36.6-44*ppmw)
        return ppmw

class SolubilityN2(Solubility):
    """N2 solubility models"""

    def __init__(self, composition='libourel'):
        super().__init__(composition)

    def libourel(self, p):
        '''Libourel et al. (2003)'''
        ppmw = self.power_law(p, 0.0611, 1.0)
        return ppmw
    

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

def solvepp_get_total_pressure(pin, fO2_shift, global_d):
    """Sum partial pressures to get total pressure"""

    p_d = solvepp_get_partial_pressures(pin, fO2_shift, global_d)
    ptot = sum(p_d.values())

    return ptot

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

def solvepp_atmosphere_mean_molar_mass(pin, fO2_shift, global_d):
    """Mean molar mass of the atmosphere"""

    p_d = solvepp_get_partial_pressures(pin, fO2_shift, global_d)
    ptot = solvepp_get_total_pressure(pin, fO2_shift, global_d)

    mu_atm = 0
    for key, value in p_d.items():
        mu_atm += molar_mass[key]*value
    mu_atm /= ptot

    return mu_atm

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

def solvepp_doit(COUPLER_options):
    """Solves for initial surface partial pressures assuming melt-vapour eqm

    Requires an initial guess to be made for some parameters, as provided in
    the dictionary COUPLER_options. 

    Parameters
    ----------
        COUPLER_options : dict
            Dictionary of coupler options variables

    Returns
    ----------
        partial_pressures : dict
            Dictionary of volatile partial pressures [Pa]
    """


    print("Solving for equilibrium partial pressures at surface")

    # Volatiles that are solved-for using this eqm calculation
    solvepp_vols = ['H2O', 'CO2', 'N2', 'H2', 'CO', 'CH4']

    # Dictionary for passing parameters around for the partial pressure calculations
    global_d = {}

    # These do not require guesses
    global_d['ocean_moles'] =           7.68894973907177e+22 # moles of H2 (or H2O) in one present-day Earth ocean
    global_d['is_CH4'] =                bool(COUPLER_options['CH4_included'] > 0)

    # These require initial guesses
    global_d['mantle_melt_fraction'] =  COUPLER_options['melt_fraction_guess'] 

    # Get core's average density using Earth values
    earth_fr = 0.55     # earth core radius fraction
    earth_fm = 0.325    # earth core mass fraction  (https://arxiv.org/pdf/1708.08718.pdf)
    earth_m  = 5.97e24  # kg
    earth_r  = 6.37e6   # m

    core_rho = (3.0 * earth_fm * earth_m) / (4.0 * np.pi * ( earth_fr * earth_r )**3.0 )  # core density [kg m-3]
    print("Estimating core density to be %g kg m-3" % core_rho)

    # Calculate mantle mass by subtracting core from total
    core_mass = core_rho * 4.0/3.0 * np.pi * (COUPLER_options["radius"] * COUPLER_options["planet_coresize"] )**3.0
    global_d['mantle_mass'] = COUPLER_options["mass"] - core_mass 
    print("Total mantle mass is %.2e kg" % global_d['mantle_mass'])
    if (global_d['mantle_mass'] <= 0.0):
        UpdateStatusfile(dirs, 20)
        raise Exception("Something has gone wrong (mantle mass is negative)")

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
        print("    solvepp: p_%s = %f bar" % (s,p_d[s]))
        partial_pressures[s] = p_d[s] * 1.0e5 # Convert from bar to Pa

    return partial_pressures

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
        print( 'WARNING: g relative accuracy= {}'.format(reldg) )




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

        # Current step
        json_file   = MyJSON( dirs["output"]+'data/{}.json'.format(int(time_dict["planet"])) )
        step        = json_file.get_dict(['step'])

        # Previous steps
        run_int = runtime_helpfile.loc[runtime_helpfile['Input']=='Interior'].drop_duplicates(subset=['Time'], keep='last')
        run_atm = runtime_helpfile.loc[runtime_helpfile['Input']=='Atmosphere'].drop_duplicates(subset=['Time'], keep='last')

        # Time stepping adjustment
        if time_dict["planet"] < 2.0:
            # First few years, use static time-step
            dtmacro = 1
            dtswitch = 1
            nsteps = 1
            print("Time-stepping intent: static")

        else:
            if (COUPLER_options["dt_method"] == 0):
                # Proportional time-step calculation
                print("Time-stepping intent: proportional")
                dtswitch = time_dict["planet"] / 35.0

            elif (COUPLER_options["dt_method"] == 1):
                # Dynamic time-step calculation
                F_clip = 1.e-4

                # Get time-step length from last iter
                dtprev = float(run_int.iloc[-1]["Time"] - run_int.iloc[-2]["Time"])
                # dtprev = float(COUPLER_options["dtswitch"])
                
                F_int_3  = max(run_int.iloc[-3]["F_int"],F_clip)
                F_int_2  = max(run_int.iloc[-2]["F_int"],F_clip)
                F_int_1  = max(run_int.iloc[-1]["F_int"],F_clip)
                F_int_23 = abs((F_int_2 - F_int_3)/F_int_3)  # Relative change from [-3] to [-2] steps
                F_int_12 = abs((F_int_1 - F_int_2)/F_int_2)  # Relative change from [-2] to [-1] steps

                F_atm_3  = max(run_int.iloc[-3]["F_atm"],F_clip)
                F_atm_2  = max(run_atm.iloc[-2]["F_atm"],F_clip)
                F_atm_1  = max(run_atm.iloc[-1]["F_atm"],F_clip)
                F_atm_23 = abs((F_atm_2 - F_atm_3)/F_atm_3)  # Relative change from [-3] to [-2] steps
                F_atm_12 = abs((F_atm_1 - F_atm_2)/F_atm_2)  # Relative change from [-2] to [-1] steps

                F_acc_max = max( F_int_12-F_int_23, F_atm_12-F_atm_23 ) * 100.0  # Maximum accel. (in relative terms)

                if F_acc_max > 20.0:
                    # Slow down!!
                    print("Time-stepping intent: slow down!!")
                    dtswitch = 0.10 * dtprev

                elif (F_acc_max > 10.0):
                    # Slow down
                    print("Time-stepping intent: slow down")
                    dtswitch = 0.90 * dtprev

                elif F_acc_max > 0.1:
                    # Steady (speed up a little bit to promote evolution)
                    print("Time-stepping intent: steady")
                    dtswitch = 1.01 * dtprev

                elif F_acc_max > -12.0:
                    # Speed up
                    print("Time-stepping intent: speed up")
                    dtswitch = 1.15 * dtprev

                else:
                    # Speed up!!
                    print("Time-stepping intent: speed up!!")
                    dtswitch = 1.50 * dtprev

            elif (COUPLER_options["dt_method"] == 2):
                # Always use the maximum time-step, which can be adjusted in the cfg file
                print("Time-stepping intent: maximum")
                dtswitch = COUPLER_options["dt_maximum"]

            else:
                UpdateStatusfile(dirs, 20)
                raise Exception("Invalid time-stepping method '%d'" % COUPLER_options["dt_method"])
            
            # Additional step-size ceiling when F_crit is used
            if abs(run_atm.iloc[-1]["F_atm"]) <= COUPLER_options["F_crit"]:
                dtswitch = min(dtswitch, COUPLER_options["dt_crit"])
                print("F_atm < F_crit, so time-step is being limited")

            # Step scale factor (is always <= 1.0)
            dtswitch *= step_sf

            # Step-size ceiling
            dtswitch = min(dtswitch, COUPLER_options["dt_maximum"] )                    # Absolute
            dtswitch = min(dtswitch, float(time_dict["target"] - time_dict["planet"]))  # Run-over

            # Step-size floor
            dtswitch = max(dtswitch, time_dict["planet"]*0.0003)        # Relative
            dtswitch = max(dtswitch, COUPLER_options["dt_minimum"] )    # Absolute

            # Calculate number of macro steps for SPIDER to perform within
            # this time-step of PROTEUS, which sets the number of json files.
            nsteps = 2
            dtmacro = math.ceil(dtswitch / nsteps)   # Ensures that dtswitch is divisible by nsteps
            dtswitch = nsteps * dtmacro

            print("New time-step is %1.2e years" % dtswitch)

        # Number of total steps until currently desired switch/end time
        nstepsmacro = step + nsteps

        if debug:
            print("TIME OPTIONS IN RUNSPIDER:", dtmacro, dtswitch, nstepsmacro)

    # For init loop
    else:
        nstepsmacro = 1
        dtmacro     = 0
        dtswitch    = 0

    # Store time-step (for next iteration)
    COUPLER_options["dtswitch"] = dtswitch
    COUPLER_options["dtmacro"] = dtmacro

    print("Surface volatile partial pressures:")
    for s in volatile_species:
        key_pp = str(s+"_initial_atmos_pressure")
        key_in = str(s+"_included")
        if (key_pp in COUPLER_options) and (COUPLER_options[key_in] == 1):
            print("    p_%s = %.5f bar" % (s,COUPLER_options[key_pp]/1.0e5))

    # Set spider flux boundary condition
    net_loss = COUPLER_options["F_atm"]

    ### SPIDER base call sequence 
    call_sequence = [   
                        dirs["spider"]+"/spider", 
                        "-options_file",          SPIDER_options_file, 
                        "-outputDirectory",       dirs["output"]+'data/',
                        "-IC_INTERIOR",           str(COUPLER_options["IC_INTERIOR"]),
                        "-OXYGEN_FUGACITY_offset",str(COUPLER_options["fO2_shift_IW"]),  # Relative to the specified buffer
                        "-surface_bc_value",      str(net_loss), 
                        "-teqm",                  str(COUPLER_options["T_eqm"]), 
                        "-nstepsmacro",           str(nstepsmacro), 
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
                print(">>> Raise dT poststep_changes:", COUPLER_options["tsurf_poststep_change"], COUPLER_options["tsurf_poststep_change_frac"])
            else:
                print(">> dTs_int too high! >>", COUPLER_options["tsurf_poststep_change"], "K")
                
        # Slowly limit again if time advances smoothly
        if (run_int["Time"].iloc[-1] != run_int["Time"].iloc[ref_idx]) and COUPLER_options["tsurf_poststep_change"] > 30:
            COUPLER_options["tsurf_poststep_change"] -= 10
            print(">>> Lower tsurf_poststep_change poststep changes:", COUPLER_options["tsurf_poststep_change"], COUPLER_options["tsurf_poststep_change_frac"])

        if run_int["Time"].iloc[-1] == run_int["Time"].iloc[ref_idx]:
            if COUPLER_options["solver_tolerance"] < 1.0e-2:
                COUPLER_options["solver_tolerance"] = float(COUPLER_options["solver_tolerance"])*2.
                print(">>> ADJUST tolerances:", COUPLER_options["solver_tolerance"])
            COUPLER_options["adjust_tolerance"] = 1
            print(">>> CURRENT TOLERANCES:", COUPLER_options["solver_tolerance"])

        # If tolerance was adjusted, restart SPIDER w/ new tolerances
        if "adjust_tolerance" in COUPLER_options:
            print(">>>>> >>>>> RESTART W/ ADJUSTED TOLERANCES")
            call_sequence.extend(["-atmosts_snes_atol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosts_snes_rtol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosts_ksp_atol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosts_ksp_rtol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosic_ksp_rtol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosic_ksp_atol", str(COUPLER_options["solver_tolerance"])])

    call_sequence.extend(["-ts_sundials_atol", str(COUPLER_options["solver_tolerance"] * atol_sf)])
    call_sequence.extend(["-ts_sundials_rtol", str(COUPLER_options["solver_tolerance"] * atol_sf)])

    # Runtime info
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
        spider_print.write(call_string+"\n")

    proc = subprocess.run([call_string],shell=True,stdout=spider_print)

    if not debug:
        spider_print.close()

    # Update restart filename for next SPIDER run
    COUPLER_options["ic_interior_filename"] = natural_sort([os.path.basename(x) for x in glob.glob(dirs["output"]+"data/*.json")])[-1]

    # Check status
    if proc.returncode == 0:
        # Success
        return True, COUPLER_options
    else:
        # Failure
        return False, {"failure":True}


def RunSPIDER( time_dict, dirs, COUPLER_options, loop_counter, runtime_helpfile ):
    '''
    Wrapper function for running SPIDER.
    This wrapper handles cases where SPIDER fails to find a solution.
    '''

    # info
    PrintHalfSeparator()
    print("Running SPIDER...")
    print("IC_INTERIOR =",COUPLER_options["IC_INTERIOR"])

    # parameters
    max_attempts = 4        # maximum number of attempts
    step_sf = 1.0           # step scale factor at attempt 1
    atol_sf = 1.0           # tolerance scale factor at attempt 1

    # tracking
    spider_success = False  # success?
    temp_options = {}       # COUPLER_options dict to be used for attempts
    attempts = 0            # number of attempts so far

    # make attempts
    while not spider_success:
        attempts += 1
        print("Attempt %d" % attempts)

        # run SPIDER
        temp_options = copy.deepcopy(COUPLER_options)
        spider_success, temp_options = _try_spider(time_dict, dirs, temp_options, loop_counter, runtime_helpfile, step_sf, atol_sf)

        if spider_success:
            # success
            print("Attempt %d succeeded" % attempts)
        else:
            # failure
            print("Attempt %d failed" % attempts)
            if attempts > max_attempts:
                # give up
                break
            else:
                # try again (change tolerance and step size)
                step_sf *= 0.8 
                atol_sf *= 4.0
    
    # check status
    if spider_success:
        # success after some attempts
        return temp_options
    else:
        # failure of all attempts
        UpdateStatusfile(dirs, 21)
        raise Exception("An error occurred when executing SPIDER (made %d attempts)" % attempts)


