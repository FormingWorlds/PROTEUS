#!/usr/bin/env python3
# Postprocess output into CHILI-MIP format

# Import modules
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proteus.utils.constants import R_earth

# Simulation folder
if len(sys.argv) != 2:
    raise ValueError("Must provide path to simulation folder")
simdir = os.path.abspath(sys.argv[1])
print(f"Sim dir: {simdir}")

# Read simulation helpfile
hfpath = os.path.join(simdir, "runtime_helpfile.csv")
if not os.path.isfile(hfpath):
   raise FileNotFoundError(f"Cannot find {hfpath}")
hf_all = pd.read_csv(hfpath, delimiter=r'\s+')

# Write to expected format
out = {}
out["t(yr)"]            = np.array(hf_all["Time"].iloc[:])
out["Tsurf(K)"]         = np.array(hf_all["T_pot"].iloc[:])
out["Tpot(K)"]          = np.array(hf_all["T_surf"].iloc[:])
out["phi(vol_frac)"]    = np.array(hf_all["Phi_global_vol"].iloc[:])
out["massC_solid(kg)"]  = np.array(hf_all["C_kg_solid"].iloc[:])
out["massC_melt(kg)"]   = np.array(hf_all["C_kg_liquid"].iloc[:])
out["massH_solid(kg)"]  = np.array(hf_all["H_kg_solid"].iloc[:])
out["massH_melt(kg)"]   = np.array(hf_all["H_kg_liquid"].iloc[:])
out["massO_atm(kg)"]    = np.array(hf_all["O_kg_atm"].iloc[:])
out["massO_atm(kg)"]    = np.array(hf_all["O_kg_atm"].iloc[:])
out["pH2O(bar)"]        = np.array(hf_all["H2O_bar"].iloc[:])
out["pCO2(bar)"]        = np.array(hf_all["CO2_bar"].iloc[:])
out["pCO(bar)"]         = np.array(hf_all["CO_bar"].iloc[:])
out["pH2(bar)"]         = np.array(hf_all["H2_bar"].iloc[:])
out["pCH4(bar)"]        = np.array(hf_all["CH4_bar"].iloc[:])
out["pO2(bar)"]         = np.array(hf_all["O2_bar"].iloc[:])
out["Rtrans(Re)"]       = np.array(hf_all["R_obs"].iloc[:]) / R_earth

outpath = os.path.join(simdir, "chili.csv")
pd.DataFrame(out).to_csv(outpath, sep=',', index=False, float_format="%.10e")

print("Done")


# Plot results
fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=False)
# Panel 1 : Surface Temperature
axes[0, 0].plot(out["t(yr)"], out["Tsurf(K)"], linewidth=2)
axes[0, 0].set_xlabel('Time [yr]', fontsize=14)
axes[0, 0].set_ylabel(r'$\rm T_{surf}$ [K]', fontsize=14)
axes[0, 0].set_xscale('log')
axes[0, 0].tick_params(axis='both', labelsize=14)
axes[0, 0].legend(loc='best', fontsize=14)
# Panel 2 : Partial pressures
axes[0, 1].plot(out["t(yr)"], out["pH2O(bar)"], label=r'$\rm H_2O$', color='royalblue', linewidth=2)
axes[0, 1].plot(out["t(yr)"], out["pCO2(bar)"], label=r'$\rm CO_2$', color='red', linewidth=2)
axes[0, 1].plot(out["t(yr)"], out["pCO(bar)"], label=r'$\rm CO$', color='green', linewidth=2)
axes[0, 1].plot(out["t(yr)"], out["pH2(bar)"], label=r'$\rm H_2$', color='red', linewidth=2)
axes[0, 1].plot(out["t(yr)"], out["pCH4(bar)"], label=r'$\rm CH_4$', color='purple', linewidth=2)
axes[0, 1].plot(out["t(yr)"], out["pO2(bar)"], label=r'$\rm O_2$', color='brown', linewidth=2)
axes[0, 1].set_xlabel('Time [yr]', fontsize=14)
axes[0, 1].set_ylabel(r'$\rm p_{i}$ [bar]', fontsize=14)
axes[0, 1].set_xscale('log')
axes[0, 1].set_yscale('log')
axes[0, 1].tick_params(axis='both', labelsize=14)
axes[0, 1].legend(loc='best', fontsize=10)
# Panel 3 : Transit radius
axes[0, 2].plot(out["t(yr)"], out["Rtrans(Re)"], label='PROTEUS', linewidth=2)
axes[0, 2].set_xlabel('Time [yr]', fontsize=14)
axes[0, 2].set_ylabel(r'$\rm R_{trans}$ [R$_{\oplus}$]', fontsize=14)
axes[0, 2].set_xscale('log')
axes[0, 2].tick_params(axis='both', labelsize=14)
axes[0, 2].legend(loc='best', fontsize=14)
# Panel 4 : Potential temperature
axes[1, 0].plot(out["t(yr)"], out["Tpot(K)"], label='PROTEUS', linewidth=2)
axes[1, 0].set_xlabel('Time [yr]', fontsize=14)
axes[1, 0].set_ylabel(r'$\rm T_{pot}$ [K]', fontsize=14)
axes[1, 0].set_xscale('log')
axes[1, 0].tick_params(axis='both', labelsize=14)
axes[1, 0].legend(loc='best', fontsize=14)
# Panel 5 : Melt fraction
axes[1, 1].plot(out["t(yr)"], out["phi(vol_frac)"], label='PROTEUS', linewidth=2)
axes[1, 1].set_xlabel('Time [yr]', fontsize=14)
axes[1, 1].set_ylabel(r'$\rm \phi$ [mass frac]', fontsize=14)
axes[1, 1].set_xscale('log')
axes[1, 1].tick_params(axis='both', labelsize=14)
axes[1, 1].legend(loc='best', fontsize=14)
# Panel 6 : Mass of volatile in solid and melt
axes[1, 2].plot(out["t(yr)"], out["massC_solid(kg)"], label=r'$\rm C_{solid}$', linewidth=2)
axes[1, 2].plot(out["t(yr)"], out["massC_melt(kg)"], label=r'$\rm C_{melt}$', linewidth=2, linestyle='--')
axes[1, 2].plot(out["t(yr)"], out["massH_solid(kg)"], label=r'$\rm H_{solid}$', linewidth=2)
axes[1, 2].plot(out["t(yr)"], out["massH_melt(kg)"], label=r'$\rm H_{melt}$', linewidth=2, linestyle='--')
axes[1, 2].set_xlabel('Time [yr]', fontsize=14)
axes[1, 2].set_ylabel(r'$m_i$ [kg]', fontsize=14)
axes[1, 2].set_xscale('log')
axes[1, 2].set_yscale('log')
axes[1, 2].tick_params(axis='both', labelsize=14)
axes[1, 2].legend(loc='best', fontsize=10)
# Horizontal lines at Tsurf = 300 K and 1000 K
axes[0, 0].axhline(300, color='k', linestyle='--', linewidth=1)
axes[0, 0].text(4.5e2, 0.92*300, r'$\rm T_{surf}$ = 300 K', va='top', ha='right', fontsize=12)
axes[0, 0].axhline(1000, color='k', linestyle='--', linewidth=1)
axes[0, 0].text(4.5e2, 0.95*1000, r'$\rm T_{surf}$ = 1000 K', va='top', ha='right', fontsize=12)
# Horizontal lines at Phi = 0.5 and Phi = 0.01
axes[1, 1].axhline(0.5, color='k', linestyle='--', linewidth=1)
axes[1, 1].text(4.5e1, 0.98*0.5, r'$\rm \phi$ = 0.5', va='top', ha='right', fontsize=12)
axes[1, 1].axhline(0.01, color='k', linestyle='--', linewidth=1)
axes[1, 1].text(4.5e1, 0.5*0.01, r'$\rm \phi$ = 0.01', va='top', ha='right', fontsize=12)
# Vertical lines at 4.5 Gyr
axes[0, 0].axvline( 4.5e9, color='silver', linestyle='--', linewidth=1)
axes[0, 0].text( 4.5e9, 0.95*max(out["Tsurf(K)"]), '4.5 Gyr', rotation=90, color='silver', va='top', ha='right', fontsize=12)
axes[0, 1].axvline( 4.5e9, color='silver', linestyle='--', linewidth    =1)
axes[0, 1].text( 4.5e9, 0.95*max(out["pH2O(bar)"]), '4.5 Gyr', rotation=90, color   ='silver', va='top', ha='right', fontsize=12)
axes[0, 2].axvline( 4.5e9, color='silver', linestyle='--', linewidth    =1)
axes[0, 2].text( 4.5e9, 0.95*max(out["Rtrans(Re)"]), '4.5 Gyr', rotation=90, color  ='silver', va='top', ha='right', fontsize=12)
axes[1, 0].axvline( 4.5e9, color='silver', linestyle='--', linewidth    =1)
axes[1, 0].text( 4.5e9, 0.95*max(out["Tpot(K)"]), '4.5 Gyr', rotation=90, color    ='silver', va='top', ha='right', fontsize=12)
axes[1, 1].axvline( 4.5e9, color='silver', linestyle='--', linewidth    =1)
axes[1, 1].text( 4.5e9, 0.95*max(out["phi(vol_frac)"]), '4.5 Gyr', rotation=90, color     ='silver', va='top', ha='right', fontsize=12)
axes[1, 2]. axvline( 4.5e9, color='silver', linestyle='--', linewidth    =1)
axes[1, 2].text( 4.5e9, 0.95*max(out["massC_solid(kg)"]), '4.5 Gyr', rotation=90, color='silver', va='top', ha='right', fontsize=12)
# Save the figure
fig.suptitle('CHILI Protocol', fontsize=16)
plt.tight_layout()
plt.savefig('fig_chili_protocol.png', dpi=300)
