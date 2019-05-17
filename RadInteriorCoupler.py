"""
RadInteriorCoupler.py

"""

import numpy as np
# import GGRadConv
import SocRadConv
import ReadInterior

# placeholder, interior code will produce surfaceT
surfaceT = 300.0*np.ones(1)  # interior code goes here
np.savetxt('surfaceT.txt', surfaceT)

# save surface T to file
ReadInterior.write_surface_temp()

# load in surface temperature
surfaceT = np.loadtxt('surfaceT.dat')

print(surfaceT)

# # calculate OLR flux given surface T
# OLRFlux = SocRadConv.RadConvEqm(surfaceT)
#
#
# # save OLR flux for interior code
# np.savetxt('OLRFlux.txt', OLRFlux*np.ones(1))
