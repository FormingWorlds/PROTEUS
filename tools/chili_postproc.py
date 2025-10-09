#!/usr/bin/env python3
# Postprocess output into CHILI-MIP format

# Import modules
import sys, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from proteus.interior.wrapper import get_all_output_times, read_interior_data
from proteus.utils.constants import secs_per_year

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
out["t(yr)"]         = np.array(hf_all["Time"].iloc[:]) / secs_per_year
out["Tsurf(K)"]      = np.array(hf_all["T_surf"].iloc[:])
out["pH2O(bar)"]     = np.array(hf_all["H2O_bar"].iloc[:])
out["phi(vol_frac)"] = np.array(hf_all["Phi_global"].iloc[:])

outpath = os.path.join(simdir, "chili.csv")
pd.DataFrame(out).to_csv(outpath, sep=',', index=False, float_format="%.10e")

print("Done")
