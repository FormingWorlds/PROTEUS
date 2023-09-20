#!/usr/bin/env python3

# Debug the output of the SolveVolatileAbundances script
# by plotting its output variables against each other.

# Import libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

cmap = mpl.colormaps['viridis']

# Load csv file
df = pd.read_csv("equilibrium_atmosphere_MC.csv",sep=',',header=0)

# Parse data...
result = {}
#    Results
result["H2O_bar"]           = df['H2O'].values
result["CO2_bar"]           = df['CO2'].values
result["N2_bar"]            = df['N2'].values
result["H2_bar"]            = df['H2'].values
result["CO_bar"]            = df['CO'].values
result["CH4_bar"]           = df['CH4'].values
#    Parameters
result["H2_oc_eqv"]         = df['N_ocean_moles'].values
result["N_ppm"]             = df['Nitrogen_ppm'].values
result["CH_ratio"]          = df['CH_ratio'].values
result["Mantle_kg"]         = df['mantle_mass'].values
result["fO2-IW"]            = df['fO2_shift'].values
result["Tsurf"]             = df['temperature'].values

num_rows        = len(df['N_ocean_moles'].values)
print("Number of rows:",num_rows)

# Make plots

fig,ax = plt.subplots()

x = "fO2-IW"
y = "Tsurf"
z = "H2O_bar"

x_plt = result[x]
y_plt = result[y]
z_plt = result[z]

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3%', pad=0.05)
vmin = np.amin(z_plt)
vmax = np.amax(z_plt)
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

im = ax.scatter(x_plt,y_plt,c=z_plt,s=10,alpha=0.8,edgecolors='none', cmap=cmap, norm=norm)

cbar = fig.colorbar(sm, cax=cax, orientation='vertical') 
cbar.ax.set_ylabel(z)

ax.set_xlabel(x)
ax.set_ylabel(y)

fig.savefig("mc.pdf",bbox_inches='tight')

