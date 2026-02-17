# Physical, numerical, etc constants
from __future__ import annotations

# import config file from proteus to set the vapour species list

L_sun = 3.828e26  # W, IAU definition
R_sun = 6.957e8  # m
R_sun_cm = 100 * R_sun  # cm
M_sun = 1.988416e30  # kg
AU = 1.495978707e11  # m
AU_cm = AU * 100.0  # cm
M_earth = 5.972e24  # kg
R_earth = 6.335439e6  # m
R_core_earth = 3485000.0  # m
M_core_earth = 1.94e24  # kg
mol = 6.02214076e23  # mol definition

# Earth heat flux, globally averaged [W m-2]
# https://se.copernicus.org/articles/1/5/2010/
F_earth = 47.2e12 / (4 * 3.141 * R_earth * R_earth)

# Io radius [m] and tidal heat flux [W m-2]
R_io = 1.82149e6  # https://ssd.jpl.nasa.gov/sats/phys_par/#refs
F_io = 2.24  # https://www.nature.com/articles/nature08108

# https://nssdc.gsfc.nasa.gov/planetary/factsheet/
M_jupiter = 1.898e27  # kg
R_jupiter = 1.42984e8 / 2  # m

# Time constants
secs_per_minute = 60.0
secs_per_hour = secs_per_minute * 60.0
secs_per_day = secs_per_hour * 24.0
secs_per_year = secs_per_day * 365.25

# Values from Pierrehumbert (2010) workbook
const_R = 8.31446261815324  # Gas constant [J K−1 mol−1]
const_h = 6.626075540e-34  # Planck's constant
const_c = 2.99792458e8  # Speed of light
const_k = 1.38065812e-23  # Boltzman thermodynamic constant
const_sigma = 5.67051196e-8  # Stefan-Boltzman constant
const_G = 6.67428e-11  # Gravitational constant (2006 measurements)
const_Nav = 6.02214076e23  # Avogadro's constant [mol-1]
B_ein = 2.5

# Supported gases
vap_list = ['SiO', 'SiO2', 'MgO', 'FeO2']
vol_list = ['H2O', 'CO2', 'O2', 'H2', 'CH4', 'CO', 'N2', 'NH3', 'S2', 'SO2', 'H2S']
# vap_list = ["SiO", "SiO2", "MgO", "FeO2"]
# vap_list = ["SiO", "SiO2", "MgO", "FeO", "Na"]
# gas_list = vol_list + vap_list

# Supported elements
element_list = ['H', 'O', 'C', 'N', 'S', 'Si', 'Mg', 'Fe', 'Na', 'Al', 'Ti', 'Ca', 'K']
element_mmw = {
    'H': 1.008000000e-03,
    'He': 4.002000000e-03,
    'Li': 6.940000000e-03,
    'Be': 9.012000000e-03,
    'B': 1.081000000e-02,
    'C': 1.201100000e-02,
    'N': 1.400700000e-02,
    'O': 1.599900000e-02,
    'F': 1.899800000e-02,
    'Ne': 2.017970000e-02,
    'Na': 2.298900000e-02,
    'Mg': 2.430500000e-02,
    'Al': 2.698100000e-02,
    'Si': 2.808500000e-02,
    'P': 3.097300000e-02,
    'S': 3.206000000e-02,
    'Cl': 3.545000000e-02,
    'Ar': 3.995000000e-02,
    'K': 3.909830000e-02,
    'Ca': 4.007800000e-02,
    'Sc': 4.495500000e-02,
    'Ti': 4.786700000e-02,
    'V': 5.094150000e-02,
    'Cr': 5.199610000e-02,
    'Mn': 5.493800000e-02,
    'Fe': 5.584500000e-02,
    'Co': 5.893300000e-02,
    'Ni': 5.869340000e-02,
    'Cu': 6.354600000e-02,
    'Zn': 6.538000000e-02,
    'Ga': 6.972300000e-02,
    'Ge': 7.263000000e-02,
    'As': 7.492100000e-02,
    'Se': 7.897100000e-02,
    'Br': 7.990400000e-02,
    'Kr': 8.379800000e-02,
    'Rb': 8.546780000e-02,
    'Sr': 8.762000000e-02,
    'Y': 8.890500000e-02,
    'Zr': 9.122400000e-02,
    'Nb': 9.290600000e-02,
    'Mo': 9.595000000e-02,
    'Ru': 1.010700000e-01,
    'Rh': 1.029050000e-01,
    'Pd': 1.064200000e-01,
    'Ag': 1.078682000e-01,
    'Cd': 1.124140000e-01,
    'In': 1.148180000e-01,
    'Sn': 1.187100000e-01,
    'Sb': 1.217600000e-01,
    'Te': 1.276000000e-01,
    'I': 1.269040000e-01,
    'Xe': 1.312930000e-01,
    'Cs': 1.329050000e-01,
    'Ba': 1.373270000e-01,
    'La': 1.389050000e-01,
    'Ce': 1.401160000e-01,
    'Pr': 1.409070000e-01,
    'Nd': 1.442420000e-01,
    'Sm': 1.503600000e-01,
    'Eu': 1.519640000e-01,
    'Gd': 1.572500000e-01,
    'Tb': 1.589250000e-01,
    'Dy': 1.625000000e-01,
    'Ho': 1.649300000e-01,
    'Er': 1.672590000e-01,
    'Tm': 1.689340000e-01,
    'Yb': 1.730450000e-01,
    'Lu': 1.749668000e-01,
    'Hf': 1.784860000e-01,
    'Ta': 1.809470000e-01,
    'W': 1.838400000e-01,
    'Re': 1.862070000e-01,
    'Os': 1.902300000e-01,
    'Ir': 1.922170000e-01,
    'Pt': 1.950840000e-01,
    'Au': 1.969660000e-01,
    'Hg': 2.005920000e-01,
    'Tl': 2.043800000e-01,
    'Pb': 2.072000000e-01,
    'Bi': 2.089800000e-01,
    'Th': 2.320377000e-01,
    'Pa': 2.310350000e-01,
    'U': 2.380280000e-01,
}  # noqa
## Constant from Zephyrus
ergcm2stoWm2 = 1e-3  # convert [erg s-1 cm-2] to [W m-2]
s2yr = 1 / (3600 * 24 * 365)  # convert [seconds]      to [years]
# Sun parameters
Rs = 6.957e8  # Solar radius [m]
Ms = 1.98847e30  # Solar mass [kg]
age_sun = 4.568e9  # Age of the Sun [yr]
Teff_sun = 5780.0  # Effective temperature of the sun [K]
C_solar = 2.884e-4  # Solar C/H ratio
N_solar = 6.761e-5  # Solar N/H ratio
S_solar = 1.318e-5  # Solar S/H ratio
# Earth parameters
Me_atm = 5.15e18  # Mass of the Earth atmopshere [kg]
Fxuv_earth_10Myr = 14.67  # Fxuv [W m-2] on Earth at 10 Myr, Fig9 Wordsworth+18
Fxuv_earth_today = 4.64e-3  # Fxuv [W m-2] received on Earth today
age_earth = 4.543e9  # Age of the Earth [yr]
e_earth = 0.0167  # Earth orbital eccentricity [dimensionless]
a_earth = 1  # Earth orbital semi-major axis [au]

# Radionuclide values below are from Ruedas (2017), via SPIDER
# Natural concentrations (provided in config file) can
#      be obtained from Turcotte & Schubert, 2014, p. 170
radnuc_data = {
    'k40': {
        'abundance': 1.1668e-4,  # 40K/K
        'heatprod': 2.8761e-5,  # W/kg
        'halflife': 1.248e9,  # yr
    },
    'th232': {
        'abundance': 1.0,  # 232Th/Th
        'heatprod': 2.6368e-5,  # W/kg
        'halflife': 14e9,  # yr
    },
    'u235': {
        'abundance': 0.0072045,  # 235U/U
        'heatprod': 5.68402e-4,  # W/kg
        'halflife': 0.704e9,  # yr
    },
    'u238': {
        'abundance': 0.9927955,  # 40K/K
        'heatprod': 9.4946e-5,  # W/kg
        'halflife': 4.468e9,  # yr
    },
}
