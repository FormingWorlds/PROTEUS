# Code shared by all interior modules
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import erf

from proteus.utils.constants import B_ein

if TYPE_CHECKING:
    pass

log = logging.getLogger("fwl."+__name__)

@dataclass
class rheo_t():
    dotl:float
    delta:float
    xi:float
    gamma:float
    phist:float

# Lookup parameters for rheological properties
#     Taken from Kervazo+21 (https://doi.org/10.1051/0004-6361/202039433).
#     Note that the phi_star value of 0.4 differs from their Table 3,
#     however, this value is required to replicate their Figure 2 with
#     a rheological transition centred at 30% melt fraction.
par_visc  = rheo_t(1.0,  25.7, 1.17e-9, 5.0, 0.4)
par_shear = rheo_t(10.0, 2.10, 7.08e-7, 5.0, 0.4)
par_bulk  = rheo_t(1e9,  2.62, 0.102,   5.0, 0.4)

# Evalulate big Phi at a given layer
def _bigphi(phi:float, par:rheo_t):
    return (1.0-phi)/(1.0-par.phist)

# Evalulate big F at a given layer
def _bigf(phi:float, par:rheo_t):
    numer = np.pi**0.5 * _bigphi(phi, par) * (1.0 + _bigphi(phi, par)**par.gamma)
    denom = 2.0 * (1.0 - par.xi)
    return (1.0-par.xi) * erf(numer/denom)

# Evalulate rheological parameter at a given layer
def eval_rheoparam(phi:float, which:str):
    match which:
        case 'visc':
            par = par_visc
        case 'shear':
            par = par_shear
        case 'bulk':
            par = par_bulk
        case _:
            raise ValueError(f"Invalid rheological parameter 'f{which}'")
    # Evaluate parameter
    numer = 1.0 + _bigphi(phi, par)**par.delta
    denom = (1.0-_bigf(phi, par)) ** (B_ein*(1-par.phist))
    return par.dotl * numer / denom

# Path to location at which to save tidal heating array
def get_file_tides(outdir:str):
    return os.path.join(outdir, "data", "tides_recent.dat")

# Structure for holding interior variables at the current time-step
class Interior_t():
    def __init__(self, nlev_b:int):

        # Initial condition flag  (-1: init, 1: start, 2: running)
        self.ic = -1

        # Current time step length [yr]
        self.dt = 1.0

        self.aragog_solver = None

        # Number of levels
        self.nlev_b = int(nlev_b)
        self.nlev_s = self.nlev_b - 1

        # Arrays of interior properties at CURRENT  time-step.
        #    Radius has a length N+1. All others have length  N.
        self.radius     = np.zeros(self.nlev_b)      # Radius [m].
        self.tides      = np.zeros(self.nlev_s)      # Tidal power density [W kg-1].
        self.phi        = np.zeros(self.nlev_s)      # Melt fraction.
        self.visc       = np.zeros(self.nlev_s)      # Viscosity [Pa s].
        self.density    = np.zeros(self.nlev_s)      # Mass density [kg m-3]
        self.mass       = np.zeros(self.nlev_s)      # Mass of shell [kg]
        self.shear      = np.zeros(self.nlev_s)      # Shear modulus [Pa]
        self.bulk       = np.zeros(self.nlev_s)      # Bulk modulus [Pa]
        self.pres       = np.zeros(self.nlev_s)      # Pressure [Pa]
        self.temp       = np.zeros(self.nlev_s)      # Temperature [K]

    def print(self):
        log.info("Printing interior arrays....")
        for attr in ("radius","tides","phi","visc","density","mass","shear","bulk"):
            log.info("    %s = %s"%(attr,str(getattr(self,attr))))

    def resume_tides(self, outdir:str):
        # Read tidal heating array from file, when resuming from disk.

        # If tides_recent.dat exists, we must be resuming a simulation from a
        #     previous state on the disk. Load the `tides` and `phi` arrays
        #     into this object. If we do not do this, tidal heating will be zero
        #     during the first iteration after model is resumed.

        file_tides = get_file_tides(outdir)

        # If the file is missing, then something has gone wrong
        if not os.path.exists(file_tides):
            log.warning("Cannot find tides file to resume from")
            return

        # Read the file
        data = np.loadtxt(file_tides)
        if self.nlev_s == 1:
            # dummy interior
            self.phi   = np.array([data[0]])
            self.tides = np.array([data[1]])
        else:
            # resolved interior
            self.phi   = np.array(data[:,0])
            self.tides = np.array(data[:,1])

        # Check the length
        if len(self.phi) != self.nlev_s:
            log.error("Array length mismatch when reading old tidal data")

    def write_tides(self, outdir:str):
        # Write tidal heating array to file.
        with open(get_file_tides(outdir), 'w') as hdl:
            # header information
            hdl.write("# 3 %d \n"%self.nlev_s)
            hdl.write("# Melt fraction, Tidal heating [W/kg] \n")
            hdl.write("# 1.0 1.0 \n")
            # for each level...
            for i in range(self.nlev_s):
                hdl.write("%.7e %.7e \n"%(self.phi[i], self.tides[i]))

    def update_rheology(self, visc:bool=False):
        # Update shear and bulk moduli arrays based on the melt fraction at each layer.
        for i,p in enumerate(self.phi):
            self.shear[i] = eval_rheoparam(p, 'shear')
            self.bulk[i]  = eval_rheoparam(p, 'bulk')
            if visc:
                self.visc[i]  = eval_rheoparam(p, 'visc')
