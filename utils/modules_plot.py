#!/usr/bin/env python3

# Import utils-specific modules
from utils.modules_utils import *

from atm_rad_conv.SocRadConv import surf_Planck_nu

# Define Crameri colormaps (+ recursive)
from matplotlib.colors import LinearSegmentedColormap

for name in [ 'acton', 'bamako', 'batlow', 'berlin', 'bilbao', 'broc', 'buda',
           'cork', 'davos', 'devon', 'grayC', 'hawaii', 'imola', 'lajolla',
           'lapaz', 'lisbon', 'nuuk', 'oleron', 'oslo', 'roma', 'tofino',
           'tokyo', 'turku', 'vik' ]:
    file = os.path.join(str(pathlib.Path(__file__).parent.absolute())+"/ScientificColourMaps5/", name + '.txt')
    cm_data = np.loadtxt(file)
    vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)
    vars()[name+"_r"] = LinearSegmentedColormap.from_list(name, cm_data[::-1])

# Color definitions, https://chrisalbon.com/python/seaborn_color_palettes.html
qgray       = "#768E95"
qblue       = "#4283A9" # http://www.color-hex.com/color/4283a9
qgreen      = "#62B4A9" # http://www.color-hex.com/color/62b4a9
qred        = "#E6767A"
qturq       = "#2EC0D1"
qmagenta    = "#9A607F"
qyellow     = "#EBB434"
qgray_dark  = "#465559"
qblue_dark  = "#274e65"
qgreen_dark = "#3a6c65"
qred_dark   = "#b85e61"
qturq_dark  = "#2499a7"
qmagenta_dark = "#4d303f"
qyellow_dark  = "#a47d24"
qgray_light  = "#acbbbf"
qblue_light  = "#8db4cb"
qgreen_light = "#a0d2cb"
qred_light   = "#eb9194"
qturq_light  = "#57ccda"
qmagenta_light = "#c29fb2"
qyellow_light = "#f1ca70"

# color_cycle2 = [ "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c", "#7f7f7f", "#bcbd22", "#17becf" ]

from matplotlib import cm

try:
    import seaborn as sns
except:
    print("Seaborn package not installed.")

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

vol_colors  = {
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
    "CO2_1"          : "#811111",
    "CO2_2"          : "#B91919",
    "CO2_3"          : "#ce5e5e",
    "H2_1"           : "#a0d2cb",
    "H2_2"           : "#62B4A9",
    "H2_3"           : "#3a6c65",
    "CH4_1"          : "#eb9194",
    "CH4_2"          : "#E6767A",
    "CH4_3"          : "#b85e61",
    "CO_1"           : "#eab597",
    "CO_2"           : "#DD8452",
    "CO_3"           : "#844f31",
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
    grav_s          = su.gravity( m_planet, r_planet )

    # print(np.max(T_profile), np.min(T_profile))

    for n in range(0, len(z_profile)):

        mean_molar_mass = 0

        for vol in atm.vol_list.keys():
            mean_molar_mass += molar_mass[vol]*atm.mr_gas[vol][n]

        # print(n, atm.p[n], mean_molar_mass)

        T_mean_below    = np.mean(atm.tmp[n:])
        P_z             = atm.p[n]
        # print(T_mean_below, P_z)

        z_profile[n] = - R_gas * T_mean_below * np.log(P_z/P_s) / ( mean_molar_mass * grav_s )

    return z_profile


# Constants
R_gas           = 8.31446261815324      # J K−1 mol−1

def find_nearest(array, value):
    array   = np.asarray(array)
    idx     = (np.abs(array - value)).argmin()
    return array[idx], idx
