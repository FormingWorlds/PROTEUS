#!/usr/bin/env python3

# Script to generate PROTEUS config files for the CHILI intercomparison project.
# See `input/chili/readme.txt` for more information.

from __future__ import annotations

import os
from copy import deepcopy
from glob import glob

import toml

# Options
use_scratch = True      # Store output in scratch folder

# -----------------------------------------------

# Folder containing configuration files
intercomp = os.path.abspath(os.path.join(os.path.dirname(__file__),"..","input","chili","intercomp"))

# Load base config
pth_base = os.path.join(intercomp,"_base.toml")
cfg = {"base":toml.load(pth_base)}
print(f"Loaded base config from {pth_base}")

# Load grid config
pth_grid = os.path.join(intercomp,"_base.grid.toml")
grd = {"base":toml.load(pth_grid)}
print(f"Loaded base grid from {pth_grid}")

# ------------------------
# Remove old configs
print(" ")
for f in glob(os.path.join(intercomp,"*.toml")):
    if "_base" not in f:
        os.remove(f)
print("Removed old configs")

# ------------------------
# Earth and Venus (Table 2 of protocol paper)
for p in ("earth","venus"):
    cfg[p] = deepcopy(cfg["base"])
    cfg[p]["params"]["out"]["path"] = f"chili_{p}"
    cfg[p]["star"]["mass"] = 1.0  # Msun

# Planet mass
cfg["earth"]["struct"]["mass_tot"] = 1.0    # Mearth
cfg["venus"]["struct"]["mass_tot"] = 0.815  # Mearth

# Orbital distance
cfg["earth"]["orbit"]["semimajoraxis"] = 1.0    # AU
cfg["venus"]["orbit"]["semimajoraxis"] = 0.723  # AU

# Scale bolometric flux to match Baraffe tracks at t=t0
cfg["earth"]["star"]["bol_scale"] = 920.0 / 1005.3
cfg["venus"]["star"]["bol_scale"] = 1760.0 / 1923.1

# Grid configs (H and C inventories) for these two planets
for p in ("earth","venus"):
    grd[p] = deepcopy(grd["base"])
    grd[p]["output"]     = "scratch/" if use_scratch else ""
    grd[p]["output"]    += f"chili_{p}_grid/"
    grd[p]["ref_config"] = f"input/chili/intercomp/{p}.toml"

# ------------------------
# TRAPPIST-1 b/e/α (Table 4 of protocol paper)
for p in ("tr1a","tr1b","tr1e"):
    cfg[p] = deepcopy(cfg["base"])
    cfg[p]["params"]["out"]["path"]  = "scratch/" if use_scratch else ""
    cfg[p]["params"]["out"]["path"] += f"chili_{p}/"
    cfg[p]["params"]["stop"]["time"]["maximum"] = (7.6 - cfg[p]["star"]["age_ini"])*1e9
    cfg[p]["star"]["mass"] = 0.09      # Msun
cfg["tr1b"]["orbit"]["semimajoraxis"] = 1.154e-2 # AU
cfg["tr1e"]["orbit"]["semimajoraxis"] = 2.295e-2 # AU
cfg["tr1a"]["orbit"]["semimajoraxis"] = 6.750e-4 # AU     2.0e-3

# ------------------------
# Write configs for all planets
print(" ")
for p in cfg.keys():
    if p == "base":
        continue

    pth = os.path.join(intercomp, f"{p}.toml")
    with open(pth,'w') as hdl:
        toml.dump(cfg[p], hdl)

    print(f"Wrote new {p:5s} config to {pth}")

# Write configs for Earth+Venus grids
for p in ("earth", "venus"):

    pth = os.path.join(intercomp, f"{p}.grid.toml")
    with open(pth,'w') as hdl:
        toml.dump(grd[p], hdl)

    print(f"Wrote new {p:5s} grid to {pth}")


# Exit
print(" ")
print("Done!")
