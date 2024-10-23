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
R_gas           = 8.31446261815324      # J K−1 mol−1
M_earth         = 5.972E24              # kg
R_earth         = 6.335439e6
R_core_earth    = 3485000.0             # m
M_core_earth    = 1.94E24               # kg
mol             = 6.02214076e+23        # mol definition
ocean_moles     = 7.68894973907177e+22  # moles of H2 (or H2O) in one present-day Earth ocean

# https://nssdc.gsfc.nasa.gov/planetary/factsheet/
M_jupiter       = 1.898e27              # kg
R_jupiter       = 1.42984e8 / 2         # m

# Time constants
secs_per_year   = 365.25 * 24 * 60 * 60 # seconds per year

# Values from Pierrehumbert (2010) workbook
const_h = 6.626075540e-34    #Planck's constant
const_c = 2.99792458e8       #Speed of light
const_k =1.38065812e-23      #Boltzman thermodynamic constant
const_sigma = 5.67051196e-8  #Stefan-Boltzman constant
const_G = 6.67428e-11        #Gravitational constant (2006 measurements)
const_N_avogadro = 6.022136736e23  #Avogadro's number
const_R_gas = 8.31446261815324 # Universal gas constant, J.K-1.mol-1

# Elemental molar masses from https://iupac.qmul.ac.uk/AtWt/
molar_mass = {  "H":   1.00800e-3 ,  # all in [kg mol-1]
                "He":  4.00200e-3 ,
                "Li":  6.94000e-3 ,
                "Be":  9.01200e-3 ,
                "B":   1.0810e-2 ,
                "C":   1.2011e-2 ,
                "N":   1.4007e-2 ,
                "O":   1.5999e-2 ,
                "F":   1.8998e-2 ,
                "Ne":  2.01797e-2 ,
                "Na":  2.2989e-2 ,
                "Mg":  2.4305e-2 ,
                "Al":  2.6981e-2 ,
                "Si":  2.8085e-2 ,
                "P":   3.0973e-2 ,
                "S":   3.2060e-2 ,
                "Cl":  3.5450e-2 ,
                "Ar":  3.9950e-2 ,
                "K":   3.90983e-2 ,
                "Ca":  4.0078e-2 ,
                "Sc":  4.4955e-2 ,
                "Ti":  4.7867e-2 ,
                "V":   5.09415e-2 ,
                "Cr":  5.19961e-2 ,
                "Mn":  5.4938e-2 ,
                "Fe":  5.5845e-2 ,
                "Co":  5.8933e-2 ,
                "Ni":  5.86934e-2 ,
                "Cu":  6.3546e-2 ,
                "Zn":  6.5380e-2 ,
                "Ga":  6.9723e-2 ,
                "Ge":  7.2630e-2 ,
                "As":  7.4921e-2 ,
                "Se":  7.8971e-2 ,
                "Br":  7.9904e-2 ,
                "Kr":  8.3798e-2 ,
                "Rb":  8.54678e-2 ,
                "Sr":  8.7620e-2 ,
                "Y":   8.8905e-2 ,
                "Zr":  9.1224e-2 ,
                "Nb":  9.2906e-2 ,
                "Mo":  9.5950e-2 ,
                "Ru":  1.01070e-3 ,
                "Rh":  1.02905e-3 ,
                "Pd":  1.06420e-3 ,
                "Ag":  1.078682e-01 ,
                "Cd":  1.12414e-3 ,
                "In":  1.14818e-3 ,
                "Sn":  1.18710e-3 ,
                "Sb":  1.21760e-3 ,
                "Te":  1.27600e-3 ,
                "I":   1.26904e-3 ,
                "Xe":  1.31293e-3 ,
                "Cs":  1.32905e-3 ,
                "Ba":  1.37327e-3 ,
                "La":  1.38905e-3 ,
                "Ce":  1.40116e-3 ,
                "Pr":  1.40907e-3 ,
                "Nd":  1.44242e-3 ,
                "Sm":  1.50360e-3 ,
                "Eu":  1.51964e-3 ,
                "Gd":  1.57250e-3 ,
                "Tb":  1.58925e-3 ,
                "Dy":  1.62500e-3 ,
                "Ho":  1.64930e-3 ,
                "Er":  1.67259e-3 ,
                "Tm":  1.68934e-3 ,
                "Yb":  1.73045e-3 ,
                "Lu":  1.749668e-01 ,
                "Hf":  1.78486e-3 ,
                "Ta":  1.80947e-3 ,
                "W":   1.83840e-3 ,
                "Re":  1.86207e-3 ,
                "Os":  1.90230e-3 ,
                "Ir":  1.92217e-3 ,
                "Pt":  1.95084e-3 ,
                "Au":  1.96966e-3 ,
                "Hg":  2.00592e-3 ,
                "Tl":  2.04380e-3 ,
                "Pb":  2.07200e-3 ,
                "Bi":  2.08980e-3 ,
                "Th":  2.320377e-01 ,
                "Pa":  2.31035e-3 ,
                "U":   2.38028e-3,
        }

# Supported gases
vol_list = ["H2O", "CO2", "H2", "CH4", "CO", "N2", "S2", "SO2"]
vap_list = ["SiO", "SiO2", "MgO", "FeO2"]
gas_list = vol_list + vap_list

# Supported elements
element_list = ["H", "O", "C", "N", "S", "Si", "Mg", "Fe"]

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
e_earth           = 0.0167             # Earth eccentricity                    [dimensionless]
a_earth           = 1                  # Earth semi-major axis                 [au]
