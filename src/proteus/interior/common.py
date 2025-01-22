# Code common to all interior modules

class Interior_t():
    def __init__(self):

        # Initial condition flag  (-1: init, 1: start, 2: running)
        self.ic         = -1

        # Current time step length [yr]
        self.dt         = 1.0

        # Arrays of interior properties at CURRENT  time-step.
        #    These have a length of 1 if using the dummy interior module.
        #    Radius has a length N+1. All others have length  N.
        self.tides      = None      # Tidal power density [W kg-1].
        self.phi        = None      # Melt fraction.
        self.visc       = None      # Viscosity [Pa s].
        self.radius     = None      # Radius [m].
        self.density    = None      # Mass density [kg m-3]
        self.mass       = None      # Mass of shell [kg]
