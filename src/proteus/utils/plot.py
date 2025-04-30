# Variables and functions to help with plotting functions
# These do not do the plotting themselves
from __future__ import annotations

import glob
import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from proteus.utils.archive import archive_exists
from proteus.utils.helper import mol_to_ele

log = logging.getLogger("fwl."+__name__)

if TYPE_CHECKING:
    from proteus import Proteus

# Standard plotting colours
_preset_colours  = {
    # Default colour
    "_fallback": "#ff00ff",

    # Volatile gases
    "H2O": "#027FB1",
    "CO2": "#D24901",
    "H2" : "#008C01",
    "CH4": "#C720DD",
    "CO" : "#D1AC02",
    "N2" : "#870036",
    "S2" : "#FF8FA1",
    "SO2": "#00008B",
    "H2S": "#2eff90",
    "NH3": "#675200",

    # Volatile elements
    "H": "#0000cc",
    "C": "#ff0000",
    "O": "#88bb22",
    "N": "#ffaa00",
    "S": "#ff22ff",
    "P": "#33ccff",

    # refractory elements
    "Fe": "#888888",
    "Si": "#aa2277",
    "Mg": "#996633",
    "Na": "#ccff00",

    # GLobal energy fluxes
    "OLR":   "#dc143c",
    "ASF":   "#4169e1",
    "sct":   "#2e8b57",
    "tidal": "#daa520",
    "radio": "#C720DD",
    "star":  "#FF8FA1",

    # Atmosphere energy fluxes (copied from AGNI)
    "flux_r": "#c0c0c0",
    "flux_n": "#000000",
    "flux_c": "#6495ed",
    "flux_t": "#ff4400",
    "flux_o": "#66CD00",
    "flux_p": "#ecb000",

    # Model components
    "atm"     : "#444444",
    "int"     : "#ff7f0e",
    "core"    : "#aa3077",
    "atm_bkg" : "#f2faff",
    "int_bkg" : "#fffaf2",
    "cor_bkg" : "#efefef",
}


def _generate_colour(gas:str):
    """
    Systematically generate a colour for a gas, from its composition.

    This method for mixing colours was taken from AGNI's phys.jl
    """

    # Break into atoms
    try:
        atoms = mol_to_ele(gas)
    except ValueError:
        log.warning(f"Using fallback colour for '{gas}'")
        return _preset_colours["_fallback"]

    # Red, green, blue components
    red = 0.0
    gre = 0.0
    blu = 0.0

    # For each atom, add the contriution of its rgb components
    for e in atoms.keys():
        if e not in _preset_colours.keys():
            log.warning(f"Using fallback colour for '{gas}'")
            return _preset_colours["_fallback"]
        red += int(_preset_colours[e][1:2],base=16)*atoms[e]
        gre += int(_preset_colours[e][3:4],base=16)*atoms[e]
        blu += int(_preset_colours[e][5:6],base=16)*atoms[e]

    # Normalisation constant
    norm = max((red,gre,blu))

    # Normalise colours to 0-255
    red = int(255*red/norm)
    gre = int(255*gre/norm)
    blu = int(255*blu/norm)

    # Prevent the colour getting too close to white, which is hard to see on a plot
    c_lim = 220
    if red+gre+blu > c_lim*3:
        scale = c_lim/255.0
        red = int(scale*red)
        gre = int(scale*gre)
        blu = int(scale*blu)

    # Convert to hex string
    colour = f"#{red:02x}{gre:02x}{blu:02x}"

    return colour


def get_colour(thing:str):
    """
    Get a colour for something which needs one (e.g. for plotting a particular gas)
    """

    # Try getting it from dictionary
    try:
        colour = _preset_colours[thing]

    # Otherwise, generate it systematically
    except KeyError:
        colour = _generate_colour(thing)

    return colour

def latexify(gas:str):
    """
    Convert gas name to latex-formatted string
    """

    out = ""
    for c in gas:
        c = str(c)
        if c.isnumeric():
            out += r"$_%s$"%c
        else:
            out += c
    return out

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
        "LRS":     [5, 14],
    },

    # https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-dispersers-and-filters#
    "NIRSpec" : {
        "F070LP": [0.70,1.27],
        "F100LP": [0.97,1.84],
        "F170LP": [1.66,3.07],
        "F290LP": [2.87,5.10],
        "PRISM" : [0.60,5.30],
    },

    # https://jwst-docs.stsci.edu/jwst-near-infrared-imager-and-slitless-spectrograph#gsc.tab=0
    "NIRISS" : {
        "SOSS": [0.6, 2.8],
        "WFSS": [0.8, 2.2],
        "AMI":  [2.8, 4.8]
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
    # https://www.eso.org/sci/facilities/paranal/instruments/crires/inst.html
    # https://www.eso.org/sci/facilities/paranal/instruments/gravity/overview.html
    "IR" : {
        "R": [0.65 , 1.0 ],
        "J": [1.115, 1.362 ],
        "H": [1.423, 1.769],
        "K": [1.972, 2.624],
        "L": [2.869, 4.188],
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
    },

    # https://www.gemini.edu/instrumentation/maroon-x
    # https://www.gemini.edu/instrumentation/igrins-2
    "GEMINI-N" : {
        "MAROON-X": [0.5, 0.92],
        "IGRINS-2": [1.49, 2.46]
    },

    # https://www.eso.org/sci/facilities/paranal/instruments/sphere.html
    # https://www.eso.org/sci/facilities/paranal/instruments/espresso/overview.html
    "VLT" : {
        "ESPRESSO": [0.38, 0.788],
        "SPHERE": [0.95, 2.32]
    },

    # https://carmenes.caha.es/ext/instrument/index.html
    "CARMENES": {
        "CARMENES": [0.520, 1.710],
    },

    # https://www.tng.iac.es/instruments/harps/
    "HARPS" : {
        "HARPS-N":   [0.383, 0.690],
    },

    # https://noirlab.edu/public/programs/kitt-peak-national-observatory/wiyn-35m-telescope/neid/
    "NEID" : {
        "NEID": [0.38, 0.93],
    },

    # https://elt.eso.org/instrument/
    "ELT" : {
        "HARMONI": [0.47, 2.45],
        "MICADO":  [0.8,  2.4],
        "ANDES":   [0.40, 1.80]
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
        _,idx = find_nearest(times,val)
        sample_i.append(int(idx))

    # sort output
    mask = np.argsort(sample_t)
    out_t, out_i = [], []
    for i in mask:
        out_t.append(sample_t[i])
        out_i.append(sample_i[i])

    return out_t, out_i


def sample_output(handler: Proteus, extension:str = "_atm.nc", tmin:float = 1.0, nsamp:int=8):

    # get all files in directory
    files = glob.glob(os.path.join(handler.directories["output/data"], "*"+extension))

    # No files found?
    if len(files) < 1:

        # Maybe archived...
        if archive_exists(handler.directories["output/data"]):
            log.error("No output files found, but tar archive exists. Extract it first.")
            return [], []

        # Return empty
        return [], []

    # get times
    times = [int(f.split("/")[-1].split(extension)[0]) for f in files]

    out_t, out_i = sample_times(times, nsamp, tmin=tmin)
    out_f = [files[i] for i in out_i]

    # return times and file paths
    return out_t, out_f
