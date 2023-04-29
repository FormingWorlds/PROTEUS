#!/usr/bin/env python3

# Import utils-specific modules
from utils.modules_utils import *

from utils.modules_plot import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.transforms as transforms
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.integrate import odeint
from scipy.optimize import newton
import numpy as np
import logging, os, sys, json

#====================================================================
# constants
bigG = 6.67408E-11 # m^3 / kg / s^2

# lookup data directories
# TODO: this is the current model, but could be different depending
# on what the user is doing
# FIXME: below will break for other users
lookupdir = '/Users/tim/bitbucket/SPIDER-DEV/lookup_data/1TPa-dK09-elec-free/'
# melting curves
liquidus_file = os.path.join( lookupdir, 'melting_curves/final/liquidus.dat')
solidus_file = os.path.join( lookupdir, 'melting_curves/final/solidus.dat')
# melt files
temperature_melt_file = os.path.join( lookupdir, 'temperature_melt.dat' )
density_melt_file = os.path.join( lookupdir, 'density_melt.dat' )
# solid files
temperature_solid_file = os.path.join( lookupdir, 'temperature_solid.dat' )
density_solid_file = os.path.join( lookupdir, 'density_solid.dat' )

#===================================================================
# CLASSES
#===================================================================


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
            logger.critical('cannot find file: %s', self.filename )
            logger.critical('please specify times for which data exists')
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
            logger.critical('dictionary for %s does not exist', keys )
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
        logger.critical('output directory contains no files')
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
        logger.critical('output directory contains no PKL files')
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
def find_xx_for_yy( xx, yy, yywant ):

    a = yy - yywant

    s = sign_change( a )

    # for ease, just add zero at the beginning to enable us to
    # have the same length array.  Could equally add to the end, or
    # interpolate

    s = np.insert(s,0,0)

    result = xx * s

    return result

#====================================================================
def get_first_non_zero_index( myList ):

    # https://stackoverflow.com/questions/19502378/python-find-first-instance-of-non-zero-number-in-list

    index = next((i for i, x in enumerate(myList) if x), None)

    return index

#====================================================================
def sign_change( a ):

    s = (np.diff(np.sign(a)) != 0)*1

    return s

#====================================================================
def recursive_get(d, keys):

    '''function to access nested dictionaries'''

    if len(keys) == 1:
        return d[keys[0]]
    return recursive_get(d[keys[0]], keys[1:])

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
def gravity( m, r ):

    g = bigG*m/r**2
    return g

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
    dgdr = 4*np.pi*bigG*rho - 2*bigG*m/r**3

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

    plt.show()

#====================================================================
