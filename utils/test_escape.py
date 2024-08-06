# Test to validate the RunZephyrus() function for Proteus integration

from utils.modules_ext import *
from utils.constants import *

from escape import RunZEPHYRUS

########################### Initialization #####################################

M_star                   = 1.0              # Star mass in solar mass                [M_sun, kg]
Omega_star               = 1.0              # Star rotation rate                     [Omega_sun, rad s-1]
semi_major_axis_cm       = 1.0*AU_cm        # Planetary semi-major axis              [cm]

tidal_contribution       = 'no'             # Tidal correction factor                [dimensionless]
semi_major_axis          = 1.0*AU           # Planetary semi-major axis              [m]
eccentricity             = e_earth          # Planetary eccentricity                 [dimensionless]
M_planet                 = M_earth          # Planetary mass                         [kg]
epsilon                  = 0.15             # Escape efficiency factor               [dimensionless]  
R_earth                  = R_earth          # Planetary radius                       [m]
Rxuv                     = R_earth          # XUV planetary radius                   [m]


########################### EL escape computation #####################################

escape,out =  RunZEPHYRUS(M_star,Omega_star,tidal_contribution, semi_major_axis, eccentricity, M_planet, epsilon, R_earth, Rxuv)   # Compute EL escape     [kg s-1]

print(out)

