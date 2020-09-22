import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy import interpolate
import copy
import pathlib
import pickle as pkl
import json
import glob, re, os
import seaborn as sns
import phys
import GeneralAdiabat as ga
import SocRadModel
import SocRadConv
from atmosphere_column import atmos


### Initial conditions

# Planet age and orbit
time = { "planet": 0., "star": 100e+6 } # yr,

# Star age, yr
Tstar_range = [ 0.100e+9 ]          # yr , 4.567e+9

# Star mass, M_sun
Mstar       = 1.0 

# Planet-star distance, au
distance    = 1.0

# Surface pressure range (Pa)
P_surf    = 260e+5

# Surface temperature range (K)
T_surf    = 1000

# Volatiles considered
vol_dict    = { 
              "H2O" : .0, 
              "CO2" : .0,
              "H2"  : .0, 
              "N2"  : .0,  
              "CH4" : .0, 
              "O2"  : .0, 
              "CO"  : .0 
            }

# # Set up plot
# fig, ax1 = plt.subplots(1, 1, figsize=(7,6))
# fig2, ax2 = plt.subplots(1, 1, figsize=(7,6))
# sns.set_style("ticks")
# sns.despine()

ls_list = [ "-", "--", ":", "-." ]
lw      = 1.5
col_idx = 4

# Font sizes 
fs_l = 16
fs_m = 14
fs_s = 12

legend1_handles = []
legend2_handles = []

fig, ax1 = plt.subplots(1, 1, figsize=(9,6))
sns.set_style("ticks")
sns.despine()

T_sat_array    = np.linspace(10,T_surf*1.1,100) 

dirs =  {
           "output":   os.getcwd()+"/", 
           "rad_conv": "/Users/tim/bitbucket/pcd_couple-interior-atmosphere/atm_rad_conv"
        }

# Loop through volatiles, options: "H2O", "CO2", "H2", "N2", "CH4", "CO", "O2"
vol_list = [ "H2O", "CO2", "CH4", "O2", "CO", "N2", "H2" ]
# vol_list = [ "H2" ]
for vol_idx, vol in enumerate(reversed(vol_list)): 
# for vol_idx, vol in enumerate(reversed([ "H2" ])): 

    # Set current volatile to 1, others to zero
    for vol1 in vol_dict.keys():
        if vol1 == vol:
            vol_dict[vol1] = 1.0
        else:
            vol_dict[vol1] = 0.0

    # Saturation vapor pressure for given temperature
    Psat_array = [ ga.p_sat(vol, T)/1e+5 for T in T_sat_array  ]
    ax1.semilogy( T_sat_array, Psat_array, lw=lw, ls=":", color=ga.vol_colors[vol][col_idx])

    # Create atmosphere object
    atm                = atmos(T_surf, P_surf, vol_dict)
    atm.toa_heating    = SocRadConv.InterpolateStellarLuminosity(Mstar, time, distance, atm.albedo_pl)
    atm_dry, atm_moist = SocRadConv.RadConvEqm(dirs, time, atm, [], [], standalone=False, cp_dry=False, trpp=True) 
    
    # Temperature vs. pressure
    l1, = ax1.semilogy(atm_moist.tmp,(atm_moist.p)/1e+5, color=ga.vol_colors[vol][col_idx], ls="-", lw=lw, label=ga.vol_latex[vol])

    # Add to legend 1
    legend1_handles.append(l1)

# Legends 1
legend1 = ax1.legend(handles=legend1_handles, loc=1, ncol=3, fontsize=fs_m, framealpha=0.3, title="Volatiles")
ax1.add_artist(legend1)
title = legend1.get_title()
title.set_fontsize(fs_m)

# Legend 2
l2a, = ax1.plot([0], [0], lw=lw, ls="-", color=ga.vol_colors["qgray"], label="Adiabat")
l2b, = ax1.plot([0], [0], lw=lw, ls=":", color=ga.vol_colors["qgray_dark"], label=r"$p_\mathrm{sat}$")
legend2_handles.append(l2a)
legend2_handles.append(l2b)
legend2 = ax1.legend(handles=legend2_handles, loc=5, ncol=1, fontsize=fs_m, framealpha=0.0)
ax1.add_artist(legend2)

ax1.invert_yaxis()
ax1.set_xlabel(r'Temperature, $T$ (K)', fontsize=fs_m)
ax1.set_ylabel(r'Pressure, $P$ (bar)', fontsize=fs_m)
ax1.set_xlim(left=0, right=T_surf)
ax1.set_ylim(top=1e-5, bottom=atm_moist.ps*1.01/1e+5)
ax1.set_xticks([10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
ax1.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 260])
ax1.set_xticklabels([10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], fontsize=fs_s)
ax1.set_yticklabels([r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$", r"$10^{2}$", "260"], fontsize=fs_s)


plt.savefig(dirs["output"]+"fig_2.pdf", bbox_inches="tight")
plt.close(fig)
