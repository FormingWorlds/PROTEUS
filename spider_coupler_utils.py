#!/usr/bin/env python

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
lookupdir = '/Users/dan/Programs/spider-dev/lookup_data/1TPa-dK09-elec-free/'
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
class MyFuncFormatter( object ):

    '''the default function formatter from
       matplotlib.ticker.FuncFormatter(func) only accepts two
       arguments, which is not enough to scale an arcsinh function.
       But by creating our own class here we can attach the scaling
       to the object which can then be accessed in __call__'''

    def __init__( self, arcsinh_scale ):
        self.const = arcsinh_scale

    def ascale( self, xx ):
        '''map input to log-like values (scaled arcsinh)'''
        yy = np.arcsinh( xx*self.const )
        return yy

    def _invascale( self, yy ):
        '''map input from log-like values (inverse transform)'''
        xx = np.sinh(yy) / self.const
        return xx

    def _sci_notation( self, num, decimal_digits=1, precision=None, exponent=None):
        """
        Returns a string representation of the scientific
        notation of the given number formatted for use with
        LaTeX or Mathtext, with specified number of significant
        decimal digits and precision (number of decimal digits
        to show). The exponent to be used can also be specified
        explicitly.
        """

        # plotting zero is useful to emphasize that we are plotting both
        # positive and negative values, e.g. for the heat fluxes
        if num==0:
            fmt = r"$0$"
            return fmt

        if not exponent:
            exponent = abs(num)
            exponent = np.log10( exponent )
            exponent = np.floor( exponent )
            exponent = int( exponent )

        coeff = round(num / float(10**exponent), decimal_digits)
        # sometimes, probably due to floating point precision? the coeff
        # is not less than ten.  Correct for that here
        if np.abs(coeff) >= 10.0:
            coeff /= 10.0
            exponent += 1
        if not precision:
            precision = decimal_digits

        if coeff < 0.0:
            fmt = r"$-10^{{{0}}}$".format(exponent)
            #fmt= r"${{{0}}}$".format(exponent)
        else:
            fmt = r"$10^{{{0}}}$".format(exponent)

        return fmt
        #return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)

    def __call__( self, x, pos ):
        y = self._invascale( x )
        fmt = self._sci_notation( y, 0 )
        return fmt

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


#===================================================================
class FigureData( object ):

    def __init__( self, nrows, ncols, width, height, outname='fig',
        times=None, units='kyr' ):
        dd = {}
        self.data_d = dd
        if times:
            dd['time_l'] = times
            self.process_time_list()
        if units:
            dd['time_units'] = units
            dd['time_decimal_places'] = 2 # hard-coded
        dd['outname'] = outname
        self.set_properties( nrows, ncols, width, height )

    def get_color( self, num ):
        dd = self.data_d
        return dd['colors_l'][num]

    def get_legend_label( self, time ):
        dd = self.data_d
        units = dd['time_units']
        dp = dd['time_decimal_places']
        age = float(time)
        if units == 'yr':
            age = round( age, 0 )
            label = '%d'
        elif units == 'kyr':
            age /= 1.0E3
            label = '%0.1f'
        elif units == 'Myr':
            age /= 1.0E6
            label = '%0.2f'
        elif units == 'Byr' or units == 'Gyr':
            age /= 1.0E9
            label = '%0.2f'
        #label = '%0.{}e'.format( dp )
        #label = '%0.{}f'.format( dp )
        label = label % age
        return label

    def process_time_list( self ):
        dd = self.data_d
        time_l = dd['time_l']
        try:
            time_l = [int(time_l)]
        except ValueError:
            time_l = [int(time) for time in time_l.split(',')]
        self.time = time_l

    def make_figure( self ):
        dd = self.data_d
        nrows = dd['nrows']
        ncols = dd['ncols']
        fig, ax = plt.subplots( nrows, ncols )
        fig.subplots_adjust(wspace=0.3,hspace=0.3)
        fig.set_size_inches( dd['width'], dd['height'] )
        self.fig = fig
        self.ax = ax

    def savefig( self, num ):
        dd = self.data_d
        if dd['outname']:
            outname = dd['outname'] + '.pdf'
        else:
            outname = 'fig{}.pdf'.format( num)
        self.fig.savefig(outname, transparent=True, bbox_inches='tight',
            pad_inches=0.05, dpi=dd['dpi'])

    def set_colors( self, num=8, cmap='bkr8' ):
        dd = self.data_d
        # color scheme from Tim.  Nice reds and blues
        colors_l = ['#2364A4',
                   '#1695F9',
                   '#95D5FD',
                   '#8B0000',
                   '#CD5C5C',
                   '#FA141B',
                   '#FFA07A']
        # color scheme 'bkr8' for light background from Crameri
        # see f_Colours.m at http://www.fabiocrameri.ch/visualisation.php
        # this is actually very similar (same?) as Tim's scheme above
        # used in Bower et al. (2018)
        if cmap=='bkr8' and num==3:
            colors_l = [(0.0,0.0,0.3),
                        #(0.1,0.1,0.5),
                        #(0.2,0.2,0.7),
                        (0.4,0.4,0.8),
                        #(0.8,0.4,0.4),
                        #(0.7,0.2,0.2),
                        (0.5,0.1,0.1)]#,
                        #(0.3,0.0,0.0)]
            colors_l.reverse()
        elif cmap=='bkr8' and num==5:
            colors_l = [(0.0,0.0,0.3),
                        #(0.1,0.1,0.5),
                        (0.2,0.2,0.7),
                        #(0.4,0.4,0.8),
                        (0.8,0.4,0.4),
                        #(0.7,0.2,0.2),
                        (0.5,0.1,0.1),
                        (0.3,0.0,0.0)]
            colors_l.reverse()
        elif cmap=='bkr8' and num==6:
            colors_l = [(0.0,0.0,0.3),
                        (0.1,0.1,0.5),
                        (0.2,0.2,0.7),
                        #(0.4,0.4,0.8),
                        #(0.8,0.4,0.4),
                        (0.7,0.2,0.2),
                        (0.5,0.1,0.1),
                        (0.3,0.0,0.0)]
            colors_l.reverse()
        elif cmap=='bkr8' and num==8:
            colors_l = [(0.0,0.0,0.3),
                        (0.1,0.1,0.5),
                        (0.2,0.2,0.7),
                        (0.4,0.4,0.8),
                        (0.8,0.4,0.4),
                        (0.7,0.2,0.2),
                        (0.5,0.1,0.1),
                        (0.3,0.0,0.0)]
            colors_l.reverse()
        else:
            try:
                cmap = plt.get_cmap( cmap )
            except ValueError:
                cmap = plt.get_cmap('viridis_r')
            colors_l = [cmap(i) for i in np.linspace(0, 1, num)]
        dd['colors_l'] = colors_l

    def set_properties( self, nrows, ncols, width, height ):
        dd = self.data_d
        dd['nrows'] = nrows
        dd['ncols'] = ncols
        dd['width'] = width # inches
        dd['height'] = height # inches
        # TODO: breaks for MacOSX, since I don't think Mac comes
        # with serif font.  But whatever it decides to switch to
        # also looks OK and LaTeX-like.
        font_d = {'family' : 'sans-serif',
                  #'style': 'normal',
                  #'weight' : 'bold'
                  'serif': ['Arial'],
                  'sans-serif': ['Arial'],
                  'size'   : '10'}
        mpl.rc('font', **font_d)
        # Do NOT use TeX font for labels etc.
        plt.rc('text', usetex=False)
        dd['dpi'] = 300
        dd['extension'] = 'png'
        dd['fontsize_legend'] = 8
        dd['fontsize_title'] = 10
        dd['fontsize_xlabel'] = 10
        dd['fontsize_ylabel'] = 10
        try:
            self.set_colors( len(self.time) )
        except AttributeError:
            self.set_colors( num=8 )
        self.make_figure()

    def set_myaxes( self, ax, title='', xlabel='', xticks='',
        ylabel='', yticks='', yrotation='', fmt='', xfmt='', xmin='', xmax='', ymin='', ymax='' ):
        if title:
            self.set_mytitle( ax, title )
        if xlabel:
            self.set_myxlabel( ax, xlabel )
        if xticks:
            self.set_myxticks( ax, xticks, xmin, xmax, xfmt )
        if ylabel:
            self.set_myylabel( ax, ylabel, yrotation )
        if yticks:
            self.set_myyticks( ax, yticks, ymin, ymax, fmt )

    def set_mylegend( self, ax, handles, loc=4, ncol=1, TITLE=None, **kwargs ):
        dd = self.data_d
        fontsize = self.data_d['fontsize_legend']
        # FIXME
        if not TITLE:
            legend = ax.legend(handles=handles, loc=loc, ncol=ncol, fontsize=fontsize, **kwargs )
            #units = dd['time_units']
            #title = r'Time ({0})'.format( units )
        else:
            title = TITLE
            legend = ax.legend(title=title, handles=handles, loc=loc,
                ncol=ncol, fontsize=fontsize, **kwargs)
        plt.setp(legend.get_title(),fontsize=fontsize)

    def set_mytitle( self, ax, title ):
        dd = self.data_d
        fontsize = dd['fontsize_title']
        title = r'{}'.format( title )
        ax.set_title( title, fontsize=fontsize )

    def set_myxlabel( self, ax, label ):
        dd = self.data_d
        fontsize = dd['fontsize_xlabel']
        label = r'{}'.format( label )
        ax.set_xlabel( label, fontsize=fontsize )

    def set_myylabel( self, ax, label, yrotation ):
        dd = self.data_d
        fontsize = dd['fontsize_ylabel']
        if not yrotation:
            yrotation = 'horizontal'
        label = r'{}'.format( label )
        ax.set_ylabel( label, fontsize=fontsize, rotation=yrotation )

    def set_myxticks( self, ax, xticks, xmin, xmax, fmt ):
        dd = self.data_d
        if fmt:
            xticks = fmt.ascale( np.array(xticks) )
            ax.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(fmt))
        ax.set_xticks( xticks)
        # set x limits to match extent of ticks
        if not xmax: xmax=xticks[-1]
        if not xmin: xmin=xticks[0]
        ax.set_xlim( xmin, xmax )

    def set_myyticks( self, ax, yticks, ymin, ymax, fmt ):
        dd = self.data_d
        if fmt:
            yticks = fmt.ascale( np.array(yticks) )
            ax.yaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(fmt))
        ax.set_yticks( yticks)
        # set y limits to match extent of ticks
        if not ymax: ymax=yticks[-1]
        if not ymin: ymin=yticks[0]
        ax.set_ylim( ymin, ymax )

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
    time_l = [int(time.split('.json')[0]) for time in time_l]
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
def get_my_logger( name ):

    '''setup logger configuration and handles'''

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('mylog.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

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

logger = get_my_logger(__name__)
