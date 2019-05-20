"""
RadInteriorCoupler.py

"""

import numpy as np
# import GGRadConv
import SocRadConv
import ReadInterior
import subprocess

# run SPIDER
subprocess.call(["spider", "-options_file", "bu_input.opts"])

# save surface T to file
ReadInterior.write_surface_quantitites()

# load in surface temperature
surfaceT = np.loadtxt('surfaceT.dat')
time_current = surfaceT[-1][0]      # K
surfaceT_current = surfaceT[-1][1]  # K

# load in volatiles released from interior
volatiles_out = np.loadtxt('volatiles_out.dat')
h2o_current = volatiles_out[-1][1]  # kg
co2_current = volatiles_out[-1][2]  # kg

# print current values
print("time – T_surf – h2o – co2")
print(time_current, surfaceT_current, h2o_current, co2_current)

# calculate OLR flux given surface T
OLRFlux = SocRadConv.RadConvEqm(surfaceT_current)

# save OLR flux for interior code
np.savetxt('OLRFlux.txt', OLRFlux*np.ones(1))
