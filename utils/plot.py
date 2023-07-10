# Variables and functions to help with plotting functions
# These do not do the plotting themselves

from utils.modules_ext import *
from utils.constants import *
from utils.helper import *
# from utils.coupler import *

sci_colormaps = {}
for g in glob.glob(str(pathlib.Path(__file__).parent.absolute())+"/../AEOLUS/plotting_tools/colormaps/*.txt"):
    cm_data = np.loadtxt(g)
    name = g.split('/')[-1].split('.')[0]
    sci_colormaps[name]      = LinearSegmentedColormap.from_list(name, cm_data)
    sci_colormaps[name+"_r"] = LinearSegmentedColormap.from_list(name, cm_data[::-1])


# https://matplotlib.org/tutorials/colors/colormaps.html

no_colors   = 9

vol_zorder  = {
    "H2O"            : 11,
    "CO2"            : 10,
    "H2"             : 9,
    "CH4"            : 8,
    "N2"             : 7,
    "N2_reduced"     : 7,
    "O2"             : 5,
    "CO"             : 4,
    "S"              : 3,
    "He"             : 2,
    "NH3"            : 1,
}

dict_colors  = {
    "H2O"            : cm.get_cmap('PuBu', no_colors)(range(no_colors)),
    "CO2"            : cm.get_cmap("Reds", no_colors)(range(no_colors)),
    "H2"             : cm.get_cmap("Greens", no_colors)(range(no_colors)),
    "N2"             : cm.get_cmap("Purples", no_colors)(range(no_colors)),
    "N2_reduced"     : cm.get_cmap("Purples", no_colors)(range(no_colors)),
    "O2"             : cm.get_cmap("Wistia", no_colors+2)(range(no_colors+2)),
    "CH4"            : cm.get_cmap("RdPu", no_colors)(range(no_colors)),
    "CO"             : cm.get_cmap("pink_r", no_colors)(range(no_colors)),
    "S"              : cm.get_cmap("YlOrBr", no_colors)(range(no_colors)),
    "He"             : cm.get_cmap("Greys", no_colors)(range(no_colors)),
    "NH3"            : cm.get_cmap("cool", no_colors)(range(no_colors)),
    "greys"          : cm.get_cmap("Greys", no_colors)(range(no_colors)),
    "mixtures"       : cm.get_cmap("Set3", 9)(range(no_colors)),
    "H2O-CO2"        : cm.get_cmap("Set3", 9)(range(no_colors))[1],
    "CO2-H2O"        : cm.get_cmap("Set3", 9)(range(no_colors))[1],
    "H2O-H2"         : cm.get_cmap("Set3", 9)(range(no_colors))[2],
    "H2-H2O"         : cm.get_cmap("Set3", 9)(range(no_colors))[2],
    "H2-CO"          : cm.get_cmap("Set3", 9)(range(no_colors))[3],
    "CO-H2"          : cm.get_cmap("Set3", 9)(range(no_colors))[3],
    "H2-CO2"         : cm.get_cmap("Set3", 9)(range(no_colors))[4],
    "CO2-H2"         : cm.get_cmap("Set3", 9)(range(no_colors))[4],
    "H2-CH4"         : cm.get_cmap("Set3", 9)(range(no_colors))[5],
    "CH4-H2"         : cm.get_cmap("Set3", 9)(range(no_colors))[5],
    "H2-N2"          : cm.get_cmap("Set2", 9)(range(no_colors))[0],
    "N2-H2"          : cm.get_cmap("Set2", 9)(range(no_colors))[0],
    "CO2-N2"         : cm.get_cmap("Set2", 9)(range(no_colors))[1],
    "N2-CO2"         : cm.get_cmap("Set2", 9)(range(no_colors))[1],
    "black_1"        : "#000000",
    "black_2"        : "#323232",
    "black_3"        : "#7f7f7f",
    "H2O_1"          : "#8db4cb",
    "H2O_2"          : "#4283A9",
    "H2O_3"          : "#274e65",
    "CO2_3"          : "#811111",
    "CO2_2"          : "#B91919",
    "CO2_1"          : "#ce5e5e",
    "H2_1"           : "#a0d2cb",
    "H2_2"           : "#62B4A9",
    "H2_3"           : "#3a6c65",
    "H_1"            : "#909090",
    "H_2"            : "#606060",
    "H_3"            : "#404040",
    "OH_1"           : "#f02090",
    "OH_2"           : "#b01080",
    "OH_3"           : "#800040",
    "HCN_1"          : "#fa9000",
    "HCN_2"          : "#f86000",
    "HCN_3"          : "#d02000",
    "CH4_1"          : "#eb9194",
    "CH4_2"          : "#E6767A",
    "CH4_3"          : "#b85e61",
    "NH3_1"          : "#109010",
    "NH3_2"          : "#108000",
    "NH3_3"          : "#004000",
    "CO_1"           : "#ff11ff",
    "CO_2"           : "#ea00aa",
    "CO_3"           : "#ba0080",
    "NO_1"           : "#a0ffa0",
    "NO_2"           : "#90ef40",
    "NO_3"           : "#209f40",
    "N2_1"           : "#c29fb2",
    "N2_2"           : "#9A607F",
    "N2_3"           : "#4d303f",  
    "S_1"            : "#f1ca70",
    "S_2"            : "#EBB434",
    "S_3"            : "#a47d24",    
    "O2_1"           : "#57ccda",
    "O2_2"           : "#2EC0D1",
    "O2_3"           : "#2499a7",
    "He_1"           : "#acbbbf",
    "He_2"           : "#768E95",
    "He_3"           : "#465559",
    "qgray"          : "#768E95",
    "qgray2"         : "#888888",
    "qblue"          : "#4283A9", # http://www.color-hex.com/color/4283a9
    "qgreen"         : "#62B4A9", # http://www.color-hex.com/color/62b4a9
    "qred"           : "#E6767A",
    "qturq"          : "#2EC0D1",
    "qorange"        : "#ff7f0e",
    "qmagenta"       : "#9A607F",
    "qyellow"        : "#EBB434",
    "qgray_dark"     : "#465559",
    "qblue_dark"     : "#274e65",
    "qgreen_dark"    : "#3a6c65",
    "qred_dark"      : "#b85e61",
    "qturq_dark"     : "#2499a7",
    "qmagenta_dark"  : "#4d303f",
    "qyellow_dark"   : "#a47d24",
    "qgray_light"    : "#acbbbf",
    "qblue_light"    : "#8db4cb",
    "qgreen_light"   : "#a0d2cb",
    "qred_light"     : "#eb9194",
    "qturq_light"    : "#57ccda",
    "qmagenta_light" : "#c29fb2",
    "qyellow_light"  : "#f1ca70",
}

# Volatile Latex names
vol_latex = {
    "H2O"     : r"H$_2$O",
    "CO2"     : r"CO$_2$",
    "H2"      : r"H$_2$" ,
    "CH4"     : r"CH$_4$",
    "CO"      : r"CO",
    "N2"      : r"N$_2$",
    "N2_reduced" : r"N$_2^{-}$",
    "S"       : r"S",
    "O2"      : r"O$_2$",
    "He"      : r"He",
    "NH3"     : r"NH$_3$",
    "H2O-CO2" : r"H$_2$O–CO$_2$",
    "H2O-H2"  : r"H$_2$O–H$_2$",
    "H2O-CO"  : r"H$_2$O–CO",
    "H2O-CH4" : r"H$_2$O–CH$_4$",
    "H2O-N2"  : r"H$_2$O–N$_2$",
    "H2O-O2"  : r"H$_2$O–O$_2$",
    "H2-H2O"  : r"H$_2$–H$_2$O",
    "H2-CO"   : r"H$_2$–CO",
    "H2-CH4"  : r"H$_2$–CH$_4$",
    "H2-CO2"  : r"H$_2$–CO$_2$",
    "H2-N2"   : r"H$_2$–N$_2$",
    "H2-O2"   : r"H$_2$-O$_2$",
    "CO2-N2"  : r"CO$_2$–N$_2$",
    "CO2-H2O" : r"CO$_2$–H$_2$O",
    "CO2-CO"  : r"CO$_2$–CO",
    "CO2-CH4"  : r"CO$_2$–CH$_4$",
    "CO2-O2"  : r"CO$_2$–O$_2$",
    "CO2-H2"  : r"CO$_2$–H$_2$",
    "CO-H2O" : r"CO–H$_2$O",
    "CO-CO2" : r"CO–CO$_2$",
    "CO-H2"  : r"CO–H$_2$",
    "CO-CH4" : r"CO–CH$_4$",
    "CO-N2"  : r"CO–N$_2$",
    "CO-O2"  : r"CO–O$_2$",
    "CH4-H2O" : r"CH$_4$–H$_2$O",
    "CH4-CO2" : r"CH$_4$–CO$_2$",
    "CH4-H2"  : r"CH$_4$–H$_2$",
    "CH4-CO"  : r"CH$_4$–CO",
    "CH4-CH4" : r"CH$_4$–CH$_4$",
    "CH4-N2"  : r"CH$_4$–N$_2$",
    "CH4-O2"  : r"CH$_4$–O$_2$",
    "N2-H2O" : r"N$_2$–H$_2$O",
    "N2-CO2" : r"N$_2$–CO$_2$",
    "N2-H2"  : r"N$_2$–H$_2$",
    "N2-CO"  : r"N$_2$–CO",
    "N2-CH4" : r"N$_2$–CH$_4$",
    "N2-N2"  : r"N$_2$–N$_2$",
    "N2-O2"  : r"N$_2$–O$_2$",
    "O2-H2O" : r"O$_2$–H$_2$O",
    "O2-CO2" : r"O$_2$–CO$_2$",
    "O2-H2"  : r"O$_2$–H$_2$",
    "O2-CO"  : r"O$_2$–CO",
    "O2-CH4" : r"O$_2$–CH$_4$",
    "O2-N2"  : r"O$_2$–N$_2$",
    "O2-O2"  : r"O$_2$–O$_2$",
}

molar_mass      = {
          "H2O" : 0.01801528,           # kg mol−1
          "CO2" : 0.04401,              # kg mol−1
          "H2"  : 0.00201588,           # kg mol−1
          "CH4" : 0.01604,              # kg mol−1
          "CO"  : 0.02801,              # kg mol−1
          "N2"  : 0.028014,             # kg mol−1
          "O2"  : 0.031999,             # kg mol−1
          "SO2" : 0.064066,             # kg mol−1
          "H2S" : 0.0341,               # kg mol−1 
          "H"   : 0.001008,             # kg mol−1 
          "C"   : 0.012011,             # kg mol−1 
          "O"   : 0.015999,             # kg mol−1 
          "N"   : 0.014007,             # kg mol−1 
          "S"   : 0.03206,              # kg mol−1 
          "He"  : 0.0040026,            # kg mol−1 
          "NH3" : 0.017031,             # kg mol−1 
        }

volatile_species = [ "H2O", "CO2", "H2", "CH4", "CO", "N2", "O2", "S", "He" ]



# https://stackoverflow.com/questions/13490292/format-number-using-latex-notation-in-python
def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    else:
        return float_str

def AtmosphericHeight(atm, m_planet, r_planet):

    z_profile       = np.zeros(len(atm.p))
    P_s             = np.max(atm.p)
    grav_s          = gravity( m_planet, r_planet )

    # Reverse arrays to go from high to low pressure
    atm.p   = atm.p[::-1]
    atm.tmp = atm.tmp[::-1]
    for vol in atm.vol_list.keys():
        atm.x_gas[vol] = atm.x_gas[vol][::-1]

    # print(atm.p)

    for n in range(0, len(z_profile)-1):

        # Gravity with height
        grav_z = grav_s * ((r_planet)**2) / ((r_planet + z_profile[n])**2)

        # print(r_planet, grav_s, grav_z, z_profile[n])

        # Mean molar mass depending on mixing ratio
        mean_molar_mass = 0
        for vol in atm.vol_list.keys():
            mean_molar_mass += molar_mass[vol]*atm.x_gas[vol][n]

        # Temperature below present height
        T_mean_below    = np.mean(atm.tmp[n:])

        # # Direction calculation
        # z_profile[n] = - R_gas * T_mean_below * np.log(atm.p[n]/P_s) / ( mean_molar_mass * grav_s )

        # Integration
        dz = - R_gas * T_mean_below * np.log(atm.p[n+1]/atm.p[n]) / (mean_molar_mass*grav_z)
        
        # Next height
        z_profile[n+1] = z_profile[n] + dz

    # Reverse arrays again back to normal
    atm.p   = atm.p[::-1]
    atm.tmp = atm.tmp[::-1]
    for vol in atm.vol_list.keys():
        atm.x_gas[vol] = atm.x_gas[vol][::-1]
    z_profile = z_profile[::-1]

    return z_profile



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
class FigureData( object ):

    def __init__( self, nrows, ncols, width, height, outname='fig',
        times=[], units='kyr' ):
        dd = {}
        self.data_d = dd

        mpl.use('Agg')  # Prevent plots popping up (it's very annoying)

        if (type(times) == float) or (type(times) == int):
            times = np.array([times])

        if len(times) > 0:
            dd['time_l'] = times
            # self.process_time_list()

        if units:
            dd['time_units'] = units
            dd['time_decimal_places'] = 2 # hard-coded
        dd['outname'] = outname

        self.cmap = sci_colormaps['imola']
            
        self.set_properties( nrows, ncols, width, height )

    def get_color( self, frac ):
        return self.cmap(frac)

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

    def set_cmap( self, cmap_obj ):
        self.cmap = cmap_obj

    def set_properties( self, nrows, ncols, width, height ):
        dd = self.data_d
        dd['nrows'] = nrows
        dd['ncols'] = ncols
        dd['width'] = width # inches
        dd['height'] = height # inches

        # Set main font properties
        font_d = {
            'size': 10.0,
            'family': ['Arial','sans-serif']
            }

        # fonts  = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        # Has arial?
        # for f in fonts:
        #     if 'Arial' in f:
        #         font_d["family"]        = 'sans-serif'
        #         font_d['serif']         = ['Arial']
        #         font_d['sans-serif']    = ['Arial']
        #         break
        mpl.rc('font', **font_d)

        # Do NOT use TeX font for labels etc.
        plt.rc('text', usetex=False)

        # Other params
        dd['dpi'] = 300
        dd['extension'] = 'png'
        dd['fontsize_legend'] = 8
        dd['fontsize_title'] = 10
        dd['fontsize_xlabel'] = 10
        dd['fontsize_ylabel'] = 10
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



# Function to read PT profile and VULCAN output file and AEOLUS mixing ratios (if read_const=True)
def offchem_read_year(output_dir, year_int, mx_clip_min=1e-30, mx_clip_max=1.0, read_const=False):

    year_data = {}  # Dictionary of mixing ratios, pressure, temperature

    # Read VULCAN output file
    with open(output_dir+"offchem/%d/output.vul" % year_int, "rb") as handle:
        vul_data = pkl.load(handle)


    # Save year
    year_data["year"] = int(year_int)

    # Parse mixing ratios
    vul_var = vul_data['variable']
    for si,sp in enumerate(vul_var['species']):
        year_data["mx_"+sp] = np.clip(np.array(vul_var['ymix'])[:,si],mx_clip_min,mx_clip_max)

    # Parse PT profile
    vul_atm = vul_data['atm']
    year_data["pressure"] =     np.array(vul_atm["pco"]) / 1e6  # convert to bar
    year_data["temperature"] =  np.array(vul_atm["Tco"])

    # Read AEOLUS mixing ratios which were used to initialise VULCAN
    if read_const:
        vol_data_str = None
        with open(output_dir+"offchem/%d/vulcan_cfg.py" % year_int, "r") as f:
            lines = f.read().splitlines()
            for ln in lines:
                if "const_mix" in ln:
                    vol_data_str = ln.split("=")[1].replace("'",'"')

        if vol_data_str == None:
            raise Exception("Could not parse vulcan cfg file!")
        
        vol_data = json.loads(vol_data_str)
        for vol in vol_data:
            ae_mx = float(vol_data[vol])
            if ae_mx > 1e-12:
                year_data["ae_"+vol] = ae_mx

    return year_data


