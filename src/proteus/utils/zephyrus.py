# Functions to compute escape from the Zephyrus module

import numpy as np
from proteus.utils.constants import const_G

def EL_escape(tidal_contribution,a,e,Mp,Ms,epsilon,Rp,Rxuv,Fxuv):

    ''''
    Compute the mass-loss rate for Energy-Limited (EL) escape.
    Based on the formula from Lopez, Fortney & Miller, 2012 (Equation 2,3,4).
    
    Inputs :
        - tidal_contribution : 0 (None) or 1 (yes) -> K_tide = tidal correction factor (0 < K_tide < 1)                                    [dimensionless]
        - a                  : planetary semi-major axis                                                                                   [m]
        - e                  : planetary eccentricty                                                                                       [dimensionless]
        - Mp                 : planetary mass                                                                                              [kg]
        - Ms                 : Stellar mass                                                                                                [kg]
        - epsilon            : efficiency factor (varies typically 0.1 < epsilon < 0.6)                                                    [dimensionless]
        - Rp                 : planetary radius                                                                                            [m]
        - Rxuv               : planetary radius at which XUV radiation are opticaly thick (defined at 20 mbar in Baumeister et  al. 2023)  [m]
        - Fxuv               : XUV incident flux received on the planet from the host star                                                 [W m-2]
    Constants :
        - G                  : gravitational constant                                                                                      [m3 kg-1 s-2] 

    Output : Mass-loss rate for EL escape [kg s-1]
    '''
    # Tidal contribution
    if tidal_contribution == 1:                 # Take into account tidal contributions : Ktide
        Rhill = a * (1-e) * (Mp/(3*Ms))**(1/3)
        ksi = Rhill/Rxuv
        K_tide = 1 - (3/(2*ksi)) + (1/(2*(ksi**3)))
    else :                                           # No tidal contributions : Ktide = 1
        K_tide = 1

    # Mass-loss rate for EL escape
    escape_EL = (epsilon * np.pi * Rp * (Rxuv**2) * Fxuv) / (const_G * Mp * K_tide)

    return escape_EL
