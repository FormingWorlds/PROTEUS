#!/usr/bin/env python3

# Debug the output of the SolveVolatileAbundances script
# by plotting its output variables against each other.

# Import libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load csv file
df = pd.read_csv("equilibrium_atmosphere_MC.csv",sep=',',header=0)

# Parse data
p_H2O           = df['H2O'].values
p_CO2           = df['CO2'].values
p_N2            = df['N2'].values
p_H2            = df['H2'].values
p_CO            = df['CO'].values
p_CH4           = df['CH4'].values
N_ocean_moles   = df['N_ocean_moles'].values
CH_ratio        = df['CH_ratio'].values
fO2_shift       = df['fO2_shift'].values
num_rows        = len(N_ocean_moles)

print("Number of rows:",num_rows)

# Make plots

fig,ax = plt.subplots()

x = N_ocean_moles
y = fO2_shift
z = p_H2

im = ax.scatter(x,y,c=z,s=10,alpha=0.8,edgecolors=None)

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3%', pad=0.05)
vmin = np.amin(z)
vmax = np.amax(z)
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=mpl.cm.get_cmap('cividis'), norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax, orientation='vertical') 
cbar.ax.set_ylabel("CH4 partial pressure [bar]")

ax.set_xlabel("Hydrogen inventory (EO equiv.)")
ax.set_ylabel("fO2 - IW")

# ax.scatter(fO2_shift,p_CH4,label='CH4', s=5, alpha=0.7)
# ax.scatter(fO2_shift,p_H2O,label='H2O', s=5, alpha=0.7)
# ax.scatter(fO2_shift,p_CO2, label='CO2',  s=5, alpha=0.7)


# ax.set_xlabel('fO2 - IW')
# ax.set_ylabel('Partial pressure [bar]')

# ax.legend()

fig.savefig("mc.pdf",bbox_inches='tight')

