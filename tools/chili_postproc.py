#!/usr/bin/env python3
# Postprocess output into CHILI-MIP format

# Import modules
from __future__ import annotations

import os
import sys

import matplotlib as mpl

mpl.use("Agg") # noqa
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd

from proteus.config import read_config_object
from proteus.utils.constants import R_earth, vol_list
from proteus.utils.plot import get_colour, latexify

# Target times for sampling profile [years]
tau_target = [1e3, 1e4, 1e5, 1e7, 1e9]

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

# Read config
config = read_config_object(os.path.join(simdir, "init_coupler.toml"))

# Write to expected format
out = {}
out["t(yr)"]            = np.array(hf_all["Time"].iloc[:])
out["T_surf(K)"]        = np.array(hf_all["T_surf"].iloc[:])
out["T_pot(K)"]         = np.array(hf_all["T_pot"].iloc[:])
out["phi(vol_frac)"]    = np.array(hf_all["Phi_global_vol"].iloc[:])
out["massC_solid(kg)"]  = np.array(hf_all["C_kg_solid"].iloc[:])
out["massC_melt(kg)"]   = np.array(hf_all["C_kg_liquid"].iloc[:])
out["massC_atm(kg)"]    = np.array(hf_all["C_kg_atm"].iloc[:])
out["massH_solid(kg)"]  = np.array(hf_all["H_kg_solid"].iloc[:])
out["massH_melt(kg)"]   = np.array(hf_all["H_kg_liquid"].iloc[:])
out["massH_atm(kg)"]    = np.array(hf_all["H_kg_atm"].iloc[:])
out["massO_atm(kg)"]    = np.array(hf_all["O_kg_atm"].iloc[:])
out["mmw(kg/mol)"]      = np.array(hf_all["atm_kg_per_mol"].iloc[:])
out["p_surf(bar)"]      = np.array(hf_all["P_surf"].iloc[:])
for gas in vol_list:
    out[f"p_{gas}(bar)"]       = np.array(hf_all[f"{gas}_bar"].iloc[:])
out["fO2_liquid(bar)"]  = out["p_O2(bar)"]
out["fO2_solid(bar)"]   = np.zeros_like(out["t(yr)"])
out["R_trans(m)"]       = np.array(hf_all["R_obs"].iloc[:])
out["flux_surf(W/m2)"]  = np.array(hf_all["F_int"].iloc[:])
out["flux_OLR(W/m2)"]   = np.array(hf_all["F_olr"].iloc[:])
out["flux_ASR(W/m2)"]   = np.array(hf_all["F_ins"].iloc[:]) * config.orbit.s0_factor * (1-config.atmos_clim.albedo_pl)
out["thick_surf_bl(m)"] = np.ones_like(out["t(yr)"]) * config.atmos_clim.surface_d

# Write to scalar CSV
print("Write CSV file for scalars")
outpath = os.path.join(simdir, "chili.csv")
pd.DataFrame(out).to_csv(outpath, sep=',', index=False, float_format="%.10e")

# Write TPZ profiles at required times
print("Write CSV files for profiles")
for tau in tau_target:
    iclose = np.argmin(np.abs(tau - out["t(yr)"]))
    tclose = out["t(yr)"][iclose]
    print(f"    {tau:.2e} -> {tclose:.2e} yr")
    if (tclose > tau*10) or (tclose < tau/10):
        print("    too far from target time, skipping")
        continue

    ncfile = os.path.join(simdir, "data", f"{tclose:.0f}_atm.nc")
    with nc.Dataset(ncfile) as ds:
        tarr = np.array(ds.variables["tmp"][:])
        parr = np.array(ds.variables["p"][:]  ) * 1e-5
        zarr = np.array(ds.variables["r"][:]  ) - np.amin(ds.variables["r"])
    X = np.array([tarr, parr, zarr]).T
    H = "T(K),p(bar),z(m)"
    csvname = f"chili_logt={np.log10(tclose):.2f}".replace(".","p") + ".csv"
    f = os.path.join(simdir,csvname)
    print(f"      {f}")
    np.savetxt(f, X, fmt="%.10e", delimiter=',', header=H)


# Make plot...
print("Make plot")
fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=False)

# Panel 1 : Surface Temperature
axes[0, 0].plot(out["t(yr)"], out["T_surf(K)"], linewidth=2)
axes[0, 0].set_ylabel(r'$\rm T_{surf}$ [K]', fontsize=14)
axes[0, 0].tick_params(axis='both', labelsize=14)

# Panel 2 : Partial pressures
for g in ["H2O", "CO2", "CO", "H2", "CH4", "O2"]:
    axes[0, 1].plot(out["t(yr)"], out[f"p_{g}(bar)"], label=latexify(g), color=get_colour(g), linewidth=2)
axes[0, 1].set_ylabel(r'$\rm p_{i}$ [bar]', fontsize=14)
axes[0, 1].set_yscale('log')
axes[0, 1].tick_params(axis='both', labelsize=14)
axes[0, 1].legend(loc='best', fontsize=10)

# Panel 3 : Transit radius
axes[0, 2].plot(out["t(yr)"], out["R_trans(m)"]/R_earth, linewidth=2)
axes[0, 2].set_ylabel(r'$\rm R_{trans}$ [R$_{\oplus}$]', fontsize=14)
axes[0, 2].tick_params(axis='both', labelsize=14)

# Panel 4 : Potential temperature
axes[1, 0].plot(out["t(yr)"], out["T_pot(K)"],  linewidth=2)
axes[1, 0].set_ylabel(r'$\rm T_{pot}$ [K]', fontsize=14)
axes[1, 0].tick_params(axis='both', labelsize=14)

# Panel 5 : Melt fraction
axes[1, 1].plot(out["t(yr)"], out["phi(vol_frac)"],  linewidth=2)
axes[1, 1].set_ylabel(r'$\rm \phi$ [volume frac]', fontsize=14)
axes[1, 1].tick_params(axis='both', labelsize=14)

# Panel 6 : Mass of volatile in solid and melt
m_unit = 1e16
axes[1, 2].plot(out["t(yr)"], out["massC_solid(kg)"]/m_unit, label=r'$\rm C_{solid}$', linewidth=2, color=get_colour('C'))
axes[1, 2].plot(out["t(yr)"], out["massC_melt(kg)"]/m_unit,  label=r'$\rm C_{melt}$',  linewidth=2, linestyle='--', color=get_colour('C'))
axes[1, 2].plot(out["t(yr)"], out["massH_solid(kg)"]/m_unit, label=r'$\rm H_{solid}$', linewidth=2, color=get_colour('H'))
axes[1, 2].plot(out["t(yr)"], out["massH_melt(kg)"]/m_unit,  label=r'$\rm H_{melt}$',  linewidth=2, linestyle='--', color=get_colour('H'))
axes[1, 2].set_ylabel(r'$m_i$ [$10^{16}$ kg]', fontsize=14)
axes[1, 2].set_yscale('symlog')
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

# Extra things for plots
for ax in axes.flatten():

    # reference times
    for tau in tau_target:
        ax.axvline(tau, color='silver', linestyle='--', linewidth=1)

    # set scales
    ax.set_xscale('log')
    ax.set_xlabel('Time [yr]', fontsize=14)


# Save the figure
fig.suptitle('CHILI-MIP Protocol Simulation', fontsize=16)
fig.tight_layout()
fig.savefig(os.path.join(simdir, "chili.pdf"), dpi=300, bbox_inches='tight')

print("Done")
