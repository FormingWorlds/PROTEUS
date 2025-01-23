# Code shared by all interior modules
from __future__ import annotations

from typing import TYPE_CHECKING
import os
import logging
import numpy as np

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

# Constants
TIDES_FILENAME = "tides_recent.dat"

# Structure for holding interior variables at the current time-step
class Interior_t():
    def __init__(self, outdir:str, config:Config):

        # Initial condition flag  (-1: init, 1: start, 2: running)
        self.ic         = -1

        # Current time step length [yr]
        self.dt         = 1.0

        # Arrays of interior properties at CURRENT  time-step.
        #    Radius has a length N+1. All others have length  N.
        #    Set to None until initialised later.
        self.tides      = None      # Tidal power density [W kg-1].
        self.phi        = None      # Melt fraction.
        self.visc       = None      # Viscosity [Pa s].
        self.radius     = None      # Radius [m].
        self.density    = None      # Mass density [kg m-3]
        self.mass       = None      # Mass of shell [kg]

        # Number of levels
        match config.interior.module :
            case "spider":
                self.nlev_b = int(config.interior.spider.num_levels)
            case "aragog":
                self.nlev_b = int(config.interior.aragog.num_levels)
            case "dummy":
                self.nlev_b = 2
        self.nlev_s = self.nlev_b - 1

        # Tides file path
        self.file_tides = os.path.join(outdir, "data", TIDES_FILENAME)

    def resume_tides(self):
        # Read tidal heating array from file, when resuming from disk.

        # If tides_recent.dat exists, we must be resuming a simulation from a
        #     previous state on the disk. Load the `tides` and `phi` arrays
        #     into this object. If we do not do this, tidal heating will be zero
        #     during the first iteration after model is resumed.

        # If the file is missing, then something has gone wrong
        if not os.path.exists(self.file_tides):
            log.warning("Cannot find tides file to resume from")
            return

        # Read the file
        data = np.loadtxt(self.file_tides)
        self.phi   = np.array(data[:,0])
        self.tides = np.array(data[:,1])

        # Check the length
        if len(self.phi) != self.nlev_s:
            log.error("Array length mismatch when reading old tidal data")


    def write_tides(self):
        # Write tidal heating array to file.

        with open(self.file_tides, 'w') as hdl:
            # header information
            hdl.write("# 3 %d \n"%self.nlev_s)
            hdl.write("# Melt fraction, Tidal heating [W/kg] \n")
            hdl.write("# 1.0 1.0 \n")
            # for each level...
            for i in range(self.nlev_s):
                hdl.write("%.7e %.7e \n"%(self.phi[i], self.tides[i]))

