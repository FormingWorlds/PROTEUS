#!/usr/bin/env python3
# Postprocess output into CHILI-MIP format

# Import modules
from __future__ import annotations

import os
import sys

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
out["Tpot(K)"]         = np.array(hf_all["T_surf"].iloc[:])
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
out["Rtrans(Re)"]       = np.array(hf_all["R_planet"].iloc[:]) / R_earth

outpath = os.path.join(simdir, "chili.csv")
pd.DataFrame(out).to_csv(outpath, sep=',', index=False, float_format="%.10e")

print("Done")
