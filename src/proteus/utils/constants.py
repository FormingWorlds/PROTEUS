# Physical, numerical, etc constants

# Avoid importing from any of the files within utils/

# Astronomical constants
from __future__ import annotations

L_sun           = 3.828e+26             # W, IAU definition
R_sun_cm        = 6.957e+10             # cm
AU              = 1.495978707e+11       # m
AU_cm           = AU * 100.0            # cm
R_gas           = 8.31446261815324      # J K−1 mol−1
M_earth         = 5.972E24              # kg
R_earth         = 6.335439e6 
R_core_earth    = 3485000.0             # m
M_core_earth    = 1.94E24               # kg
mol             = 6.02214076e+23        # mol definition
ocean_moles     = 7.68894973907177e+22 # moles of H2 (or H2O) in one present-day Earth ocean

# Time constants 
secs_per_year   = 365.25 * 24 * 60 * 60 # seconds per year 

# Values from phys.py
const_h = 6.626075540e-34    #Planck's constant
const_c = 2.99792458e8       #Speed of light
const_k =1.38065812e-23      #Boltzman thermodynamic constant
const_sigma = 5.67051196e-8  #Stefan-Boltzman constant
const_G = 6.67428e-11        #Gravitational constant (2006 measurements)
const_N_avogadro = 6.022136736e23  #Avogadro's number
const_R_gas = 8.31446261815324 # Universal gas constant, J.K-1.mol-1

molar_mass  = {
            "H"   : 0.001008,             # kg mol−1 
            "C"   : 0.012011,             # kg mol−1 
            "O"   : 0.015999,             # kg mol−1 
            "N"   : 0.014007,             # kg mol−1 
            "S"   : 0.03206,              # kg mol−1
            "He"  : 0.0040026,            # kg mol−1 

            "H2O" : 0.01801528,           # kg mol−1
            "CO2" : 0.04401,              # kg mol−1
            "H2"  : 0.00201588,           # kg mol−1
            "CH4" : 0.01604,              # kg mol−1
            "CO"  : 0.02801,              # kg mol−1
            "N2"  : 0.028014,             # kg mol−1
            "O2"  : 0.031999,             # kg mol−1
            "SO2" : 0.064066,             # kg mol−1
            "H2S" : 0.0341,               # kg mol−1 
            "S2"  : 0.0641,               # kg mol-1
            "NH3" : 0.017031,             # kg mol−1 
        }

# Supported volatiles and elements
volatile_species = [ "H2O", "CO2", "H2", "CH4", "CO", "N2", "S2", "SO2"]
element_list     = [ "H", "O", "C", "N", "S" ] 

dirs = {}  # Modified by coupler.py: SetDirectories().

## Constant from Zephyrus
ergcm2stoWm2  = 1e-3                  # convert [erg s-1 cm-2] to [W m-2]
s2yr          = 1/(3600*24*365)       # convert [seconds]      to [years]
# Sun parameters 
Rs      = 6.957e8                      # Solar radius                          [m]
Ms      = 1.98847e30                   # Solar mass                            [kg]
age_sun = 4.603e9                      # Age of the Sun                        [yr]
# Earth parameters 
Me_atm            = 5.15e18            # Mass of the Earth atmopshere          [kg]
Fxuv_earth_10Myr  = 14.67              # Fxuv received on Earth at t = 10 Myr -> see Fig 9. Wordsworth+18 [W m-2]
Fxuv_earth_today  = 4.64e-3            # Stellar flux received on Earth today  [W m-2]
age_earth         = 4.543e9            # Age of the Earth                      [yr]
e_earth           = 0.017              # Earth eccentricity                    [dimensionless]
a_earth           = 1                  # Earth semi-major axis                 [au]



# End of file
