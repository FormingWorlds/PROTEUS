# Physical, numerical, etc constants

# Avoid importing from any of the files within utils/

# Astronomical constants
from __future__ import annotations

L_sun           = 3.828e+26             # W, IAU definition
R_sun           = 6.957e8               # m
R_sun_cm        = 100 * R_sun           # cm
M_sun           = 1.988416e30           # kg
AU              = 1.495978707e+11       # m
AU_cm           = AU * 100.0            # cm
M_earth         = 5.972E24              # kg
R_earth         = 6.335439e6            # m
R_core_earth    = 3485000.0             # m
M_core_earth    = 1.94E24               # kg
mol             = 6.02214076e+23        # mol definition

# Earth heat flux, globally averaged [W m-2]
# https://se.copernicus.org/articles/1/5/2010/
F_earth         = 47.2e12 / (4 * 3.141 * R_earth * R_earth)

# Io radius [m] and tidal heat flux [W m-2]
R_io            = 1.82149e6     # https://ssd.jpl.nasa.gov/sats/phys_par/#refs
F_io            = 2.24          # https://www.nature.com/articles/nature08108

# https://nssdc.gsfc.nasa.gov/planetary/factsheet/
M_jupiter       = 1.898e27              # kg
R_jupiter       = 1.42984e8 / 2         # m

# Time constants
secs_per_minute = 60.0
secs_per_hour   = secs_per_minute * 60.0
secs_per_day    = secs_per_hour * 24.0
secs_per_year   = secs_per_day * 365.25

# Values from Pierrehumbert (2010) workbook
const_R = 8.31446261815324      # J K−1 mol−1
const_h = 6.626075540e-34    #Planck's constant
const_c = 2.99792458e8       #Speed of light
const_k =1.38065812e-23      #Boltzman thermodynamic constant
const_sigma = 5.67051196e-8  #Stefan-Boltzman constant
const_G = 6.67428e-11        #Gravitational constant (2006 measurements)

B_ein = 2.5

# Supported gases
vol_list = ["H2O", "CO2", "O2", "H2", "CH4", "CO", "N2", "NH3", "S2", "SO2", "H2S"]
vap_list = ["SiO", "SiO2", "MgO", "FeO2"]
gas_list = vol_list + vap_list

# Supported elements
element_list = ["H", "O", "C", "N", "S", "Si", "Mg", "Fe", "Na"]

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
e_earth           = 0.0167             # Earth eccentricity                    [dimensionless]
a_earth           = 1                  # Earth semi-major axis                 [au]

# Radionuclide values below are from Ruedas (2017), via SPIDER
# Natural concentrations (provided in config file) can
#      be obtained from Turcotte & Schubert, 2014, p. 170
radnuc_data = {
    "k40": {
        "abundance": 1.1668E-4, # 40K/K
        "heatprod":  2.8761E-5, # W/kg
        "halflife":  1.248e9    # yr
    },

    "th232": {
        "abundance": 1.0,       # 232Th/Th
        "heatprod":  2.6368E-5, # W/kg
        "halflife":  14e9       # yr
    },

    "u235": {
        "abundance": 0.0072045,  # 235U/U
        "heatprod":  5.68402E-4, # W/kg
        "halflife":  0.704e9     # yr
    },

    "u238": {
        "abundance": 0.9927955, # 40K/K
        "heatprod":  9.4946E-5, # W/kg
        "halflife":  4.468e9    # yr
    },
}
