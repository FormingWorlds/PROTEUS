#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 17:07:22 2018

@author: markhammond
"""

import numpy as np
import GGRadConv

# placeholder, interior code will produce surfaceT
surfaceT = 300.0*np.ones(1) # interior code goes here
np.savetxt('surfaceT.txt',surfaceT)

surfaceT = np.loadtxt('surfaceT.txt')

# calculate OLR flux given surface T
OLRFlux = GGRadConv.RadConvEqm(surfaceT)


# save OLR flux for interior code
np.savetxt('OLRFlux.txt',OLRFlux)

