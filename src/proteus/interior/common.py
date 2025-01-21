# Code common to all interior modules

class Interior_t():
    def __init__(self):

        # Arrays of interior properties at CURRENT  time-step.
        #    These have a length of 1 if using the dummy interior module.
        self.tides      = None      # Tidal power density [W kg-1].
        self.phi        = None      # Melt fraction.
        self.visc       = None      # Viscosity [Pa s].
        self.radius     = None      # Radius [m].
        self.density    = None      # Mass density [kg m-3]
        self.shear_mod  = None      # Shear modulus [Pa].
        self.bulk_mod   = None      # Bulk modulus [Pa].
