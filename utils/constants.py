# Physical, numerical, etc constants

# Avoid importing from utils.*
import os
import AEOLUS.utils.phys as phys

### Constants ###
debug = True

# Astronomical constants
L_sun           = 3.828e+26             # W, IAU definition
AU              = 1.495978707e+11       # m
R_gas           = 8.31446261815324      # J K−1 mol−1
M_earth         = 5.972E24              # kg
R_core_earth    = 3485000.0             # m
M_core_earth    = 1.94E24               # kg
mol             = 6.02214076e+23        # mol definition

# Elements
H_mol_mass      = 0.001008              # kg mol−1
C_mol_mass      = 0.012011              # kg mol−1
O_mol_mass      = 0.015999              # kg mol−1
N_mol_mass      = 0.014007              # kg mol−1
S_mol_mass      = 0.03206               # kg mol−1
He_mol_mass     = 0.0040026             # kg mol−1
Ar_mol_mass     = 0.039948              # kg mol−1
Ne_mol_mass     = 0.020180              # kg mol−1
Kr_mol_mass     = 0.083798              # kg mol−1
Xe_mol_mass     = 0.131293              # kg mol−1

# Volatile molar masses
H2O_mol_mass    = 0.01801528            # kg mol−1
CO2_mol_mass    = 0.04401               # kg mol−1
H2_mol_mass     = 0.00201588            # kg mol−1
CH4_mol_mass    = 0.01604               # kg mol−1
CO_mol_mass     = 0.02801               # kg mol−1
N2_mol_mass     = 0.028014              # kg mol−1
O2_mol_mass     = 0.031999              # kg mol−1
SO2_mol_mass    = 0.064066              # kg mol−1
H2S_mol_mass    = 0.0341                # kg mol−1

molar_mass      = {
          "H2O" : 0.01801528,           # kg mol−1
          "CO2" : 0.04401,              # kg mol−1
          "H2"  : 0.00201588,           # kg mol−1
          "CH4" : 0.01604,              # kg mol−1
          "CO"  : 0.02801,              # kg mol−1
          "N2"  : 0.028014,             # kg mol−1
          "O2"  : 0.031999,             # kg mol−1
          "SO2" : 0.064066,             # kg mol−1
          "H2S" : 0.0341,               # kg mol−1 
          "H"   : 0.001008,             # kg mol−1 
          "C"   : 0.012011,             # kg mol−1 
          "O"   : 0.015999,             # kg mol−1 
          "N"   : 0.014007,             # kg mol−1 
          "S"   : 0.03206,              # kg mol−1 
          "He"  : 0.0040026,            # kg mol−1 
          "NH3" : 0.017031,             # kg mol−1 
        }

volatile_species = [ "H2O", "CO2", "H2", "CH4", "CO", "N2", "O2", "S", "He" ]
element_list     = [ "H", "O", "C", "N", "S", "He" ] 

# volatile_species = [ "H2O", "CO2", "H2", "CH4", "CO", "N2", "O2"]
# element_list     = [ "H", "O", "C", "N" ] 

# Henry's law coefficients
# Add to dataframe + save to disk
volatile_distribution_coefficients = {               # X_henry -> ppm/Pa
    # 'H2O_henry':           6.800e-02,              # Lebrun+13
    # 'H2O_henry_pow':       1.4285714285714286,     # Lebrun+13         
    # 'CO2_henry':           4.4E-6,                 # Lebrun+13
    # 'CO2_henry_pow':       1.0,                    # Lebrun+13
    'H2O_henry':             1.033e+00,                
    'H2O_henry_pow':         1.747e+00,       
    'CO2_henry':             1.937e-09,            
    'CO2_henry_pow':         7.140e-01,                
    'H2_henry':              2.572e-06, 
    'H2_henry_pow':          1.0, 
    'CH4_henry':             9.937e-08, 
    'CH4_henry_pow':         1.0, 
    'CO_henry':              1.600e-07, 
    'CO_henry_pow':          1.0, 
    'N2_henry':              7.000e-05, 
    'N2_henry_pow':          1.8,
    'N2_henry_reduced':      7.416e+01,
    'N2_henry_pow_reduced':  4.582e+00,
    'O2_henry':              0.001E-10, 
    'O2_henry_pow':          1.0, 
    'S_henry':               5.000e-03, 
    'S_henry_pow':           1.0, 
    'He_henry':              0.001E-9, 
    'He_henry_pow':          1.0,
    # 'H2O_kdist':             1.0E-4,                 # distribution coefficients
    # 'H2O_kabs':              0.01,                   # absorption (m^2/kg)
    # 'CO2_kdist':             5.0E-4, 
    # 'CO2_kabs':              0.05,
    'H2O_kdist':             0.0E-0,
    'H2O_kabs':              0.00,
    'CO2_kdist':             0.0E-0,  
    'CO2_kabs':              0.00,
    'H2_kdist':              0.0E-0,  
    'H2_kabs':               0.00,     
    'N2_kdist':              0.0E-0,  
    'N2_kabs':               0.00,     
    'CH4_kdist':             0.0E-0, 
    'CH4_kabs':              0.00,    
    'CO_kdist':              0.0E-0,  
    'CO_kabs':               0.00,     
    'O2_kdist':              0.0E-0,  
    'O2_kabs':               0.00,     
    'S_kdist':               0.0E-0,   
    'S_kabs':                0.00,      
    'He_kdist':              0.0E-0,  
    'He_kabs':               0.00
    }
    

# Coupler-specific paths
coupler_dir = os.getenv('COUPLER_DIR')
output_dir  = coupler_dir+"/output/"
vulcan_dir  = coupler_dir+"/VULCAN/"
radconv_dir = coupler_dir+"/AEOLUS/"
spider_dir  = coupler_dir+"/SPIDER/"
utils_dir   = coupler_dir+"/utils/"

dirs = {"output": output_dir, 
        "coupler": coupler_dir, 
        "rad_conv": radconv_dir, 
        "vulcan": vulcan_dir, 
        "spider": spider_dir, 
        "utils": utils_dir
        }