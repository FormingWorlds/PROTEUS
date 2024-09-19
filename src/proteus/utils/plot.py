# Variables and functions to help with plotting functions
# These do not do the plotting themselves
from __future__ import annotations

import glob
import os
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm

if TYPE_CHECKING:
    from proteus import Proteus

vol_zorder  = {
    "H2O"            : 11,
    "CO2"            : 10,
    "H2"             : 9,
    "CH4"            : 8,
    "N2"             : 7,
    "O2"             : 5,
    "CO"             : 4,
    "S"              : 3,
    "S2"             : 3,
    "SO2"            : 3,
    "He"             : 2,
    "NH3"            : 1,
}

# Standard plotting colours
dict_colors  = {
    # From Julia's default colours
    "H2O": "#027FB1",
    "CO2": "#D24901",
    "H2" : "#008C01",
    "CH4": "#C720DD",
    "CO" : "#D1AC02",
    "N2" : "#870036",
    "S2" : "#FF8FA1",
    "SO2": "#00008B",
    "He" : "#30FF71",
    "NH3": "#675200",
}
dict_colors["OLR"] = "crimson"
dict_colors["ASF"] = "royalblue"
dict_colors["sct"] = "seagreen"
dict_colors["atm"] = "#768E95"
dict_colors["int"] = "#ff7f0e"
dict_colors["core"] = "#4d303f"
dict_colors["atm_bkg"] = (0.95, 0.98, 1.0)
dict_colors["int_bkg"] = (1.0, 0.98, 0.95)

# Volatile Latex names
vol_latex = {
    "H2O"     : r"H$_2$O",
    "CO2"     : r"CO$_2$",
    "H2"      : r"H$_2$" ,
    "CH4"     : r"CH$_4$",
    "CO"      : r"CO",
    "N2"      : r"N$_2$",
    "S"       : r"S",
    "S2"      : r"S$_2$",
    "SO2"     : r"SO$_2$",
    "O2"      : r"O$_2$",
    "O3"      : r"O$_3$",
    "OH"      : r"OH",
    "HCN"     : r"HCN",
    "NH3"     : r"NH$_3$",
    "He"      : r"He",
    "H2O-CO2" : r"H$_2$O-CO$_2$",
    "H2O-H2"  : r"H$_2$O-H$_2$",
    "H2O-CO"  : r"H$_2$O-CO",
    "H2O-CH4" : r"H$_2$O-CH$_4$",
    "H2O-N2"  : r"H$_2$O-N$_2$",
    "H2O-O2"  : r"H$_2$O-O$_2$",
    "H2-H2O"  : r"H$_2$-H$_2$O",
    "H2-CO"   : r"H$_2$-CO",
    "H2-CH4"  : r"H$_2$-CH$_4$",
    "H2-CO2"  : r"H$_2$-CO$_2$",
    "H2-N2"   : r"H$_2$-N$_2$",
    "H2-O2"   : r"H$_2$-O$_2$",
    "CO2-N2"  : r"CO$_2$-N$_2$",
    "CO2-H2O" : r"CO$_2$-H$_2$O",
    "CO2-CO"  : r"CO$_2$-CO",
    "CO2-CH4"  : r"CO$_2$-CH$_4$",
    "CO2-O2"  : r"CO$_2$-O$_2$",
    "CO2-H2"  : r"CO$_2$-H$_2$",
    "CO-H2O" : r"CO-H$_2$O",
    "CO-CO2" : r"CO-CO$_2$",
    "CO-H2"  : r"CO-H$_2$",
    "CO-CH4" : r"CO-CH$_4$",
    "CO-N2"  : r"CO-N$_2$",
    "CO-O2"  : r"CO-O$_2$",
    "CH4-H2O" : r"CH$_4$-H$_2$O",
    "CH4-CO2" : r"CH$_4$-CO$_2$",
    "CH4-H2"  : r"CH$_4$-H$_2$",
    "CH4-CO"  : r"CH$_4$-CO",
    "CH4-CH4" : r"CH$_4$-CH$_4$",
    "CH4-N2"  : r"CH$_4$-N$_2$",
    "CH4-O2"  : r"CH$_4$-O$_2$",
    "N2-H2O" : r"N$_2$-H$_2$O",
    "N2-CO2" : r"N$_2$-CO$_2$",
    "N2-H2"  : r"N$_2$-H$_2$",
    "N2-CO"  : r"N$_2$-CO",
    "N2-CH4" : r"N$_2$-CH$_4$",
    "N2-N2"  : r"N$_2$-N$_2$",
    "N2-O2"  : r"N$_2$-O$_2$",
    "O2-H2O" : r"O$_2$-H$_2$O",
    "O2-CO2" : r"O$_2$-CO$_2$",
    "O2-H2"  : r"O$_2$-H$_2$",
    "O2-CO"  : r"O$_2$-CO",
    "O2-CH4" : r"O$_2$-CH$_4$",
    "O2-N2"  : r"O$_2$-N$_2$",
    "O2-O2"  : r"O$_2$-O$_2$",
}

# Bandpasses for instrumentation of interest [units of microns, um]
observer_bands = {

    # https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-filters-and-dispersers
    "MIRI" : {
        "F560W":   [5.054, 6.171],
        "F770W":   [6.581, 8.687],
        "F1000W":  [9.023, 10.891],
        "F1130W":  [10.953, 11.667],
        "F1280W":  [11.588, 14.115],
        "F1500W":  [13.527, 16.64],
        "F1800W":  [16.519, 19.502],
        "F2100W":  [18.477, 23.159],
        "F2550W":  [23.301, 26.733],
    },

    # https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-dispersers-and-filters#
    "NIRSpec" : {
        "F070LP": [0.70,1.27],
        "F100LP": [0.97,1.84],
        "F170LP": [1.66,3.07],
        "F290LP": [2.87,5.10],
        "PRISM" : [0.60,5.30],
    },

    # https://jwst-docs.stsci.edu/jwst-near-infrared-imager-and-slitless-spectrograph/niriss-instrumentation/niriss-filters
    "NIRISS" : {
        "F090W": [0.796 , 1.005],
        "F115W": [1.013 , 1.283],
        "F150W": [1.33  , 1.671],
        "F200W": [1.751 , 2.226],
        "F277W": [2.413 , 3.143],
        "F356W": [3.14  , 4.068],
        "F444W": [3.88  , 5.023],
        "GR700XD": [0.6, 2.8]
    },

    # https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters
    "NIRCam" : {
        "Short": [0.6, 2.3],
        "Long":  [2.4, 5.0],
    },

    # https://www.esa.int/Science_Exploration/Space_Science/Ariel/Ariel_s_instruments
    "ARIEL" : {
        "AIRS0": [1.95, 3.9],
        "AIRS1": [3.9, 7.8]
    },

    # https://en.wikipedia.org/wiki/Infrared_astronomy
    "IR" : {
        "R": [0.65 , 1.0 ],
        "J": [1.1  , 1.4 ],
        "H": [1.5  , 1.8 ],
        "K": [2.0  , 2.4 ],
        "L": [3.0  , 4.0 ],
        "M": [4.6  , 5.0 ],
        "N": [7.5  , 14.5],
        "Q": [17.0 , 25.0],
        "Z": [28.0 , 40.0],
    },

    # https://link.springer.com/article/10.1007/s10686-020-09660-1
    "PLATO" : {
        "blue": [0.500, 0.675],
        "red" : [0.675, 1.125]
    },

    # https://doi.org/10.1051/0004-6361/202140366
    "LIFE" : {
        "LIFE": [4.0, 18.5]
    },

    # https://ntrs.nasa.gov/api/citations/20240006497/downloads/HWO%20Engineering%20View%20Status%20Plans%20Opportunities.pdf
    "HWO" : {
        "Coronograph":      [0.4, 1.8],
        "Highres imager":   [0.2, 2.5],
        "Spectrograph":     [0.1, 1.0]
    }

}

# https://stackoverflow.com/questions/13490292/format-number-using-latex-notation-in-python
def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    else:
        return float_str

def sample_times(times:list, nsamp:int, tmin:float=1.0):
    from proteus.utils.helper import find_nearest

    # check count
    if len(times) <= nsamp:
        out_t, out_i = np.unique(times, return_index=True)
        return list(out_t), list(out_i)

    # lower limit
    tmin = max(tmin,np.amin(times))
    tmin = min(tmin, np.amax(times))
    tmin = max(tmin, 1.0)
    # upper limit
    tmax = max(tmin+1, np.amax(times))

    # do not allow times outside range
    allowed_times = [int(x) for x in times if tmin<=x<=tmax]

    # get samples on log-time scale
    sample_t = []
    sample_i = []
    for s in np.logspace(np.log10(tmin),np.log10(tmax),nsamp): # Sample on log-scale

        remaining = [int(v) for v in set(allowed_times) - set(sample_t)]
        if len(remaining) == 0:
            break

        # Get next nearest time
        val,_ = find_nearest(remaining,s)
        sample_t.append(int(val))

        # Get the index of this time in the original array
        idx,_ = find_nearest(times,val)
        sample_i.append(idx)

    # sort output
    mask = np.argsort(sample_t)
    out_t, out_i = [], []
    for i in mask:
        out_t.append(sample_t[i])
        out_i.append(sample_i[i])

    return out_t, out_i


def sample_output(handler: Proteus, ftype:str = "nc", tmin:float = 1.0, nsamp:int=8):
    from proteus.utils.helper import find_nearest

    # get all files
    files = glob.glob(os.path.join(handler.directories["output"], "data", "*."+ftype))
    if len(files) < 1:
        return []

    # get times
    if ftype == "nc":
        dlm = "_"
    else:
        dlm = "."
    times = [int(f.split("/")[-1].split(dlm)[0]) for f in files]

    out_t, out_i = sample_times(times, nsamp, tmin=tmin)
    out_f = [files[i] for i in out_i]

    # return times and file paths
    return out_t, out_f

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

