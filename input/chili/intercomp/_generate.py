#!/usr/bin/env python3
# Generate PROTEUS config files for CHILI intercomparison project

import toml
import os
from copy import deepcopy
from numpy import pi

# Constants
cG    = 6.67430e-11    # m3 kg-1 s-2
cMsun = 1.989e30       # kg
cday  = 60*60*24.0     # s

# Folder containing configuration files
intercomp = os.path.dirname(os.path.abspath(__file__))

# Load base config
pth_base = os.path.join(intercomp,"_base.toml")
cfg = {"base":toml.load(pth_base)}
print(f"Loaded base config from {pth_base}")

# Load grid config
pth_grid = os.path.join(intercomp,"_base.grid.toml")
grd = {"base":toml.load(pth_grid)}
print(f"Loaded grid config from {pth_grid}")
print(" ")

# ------------------------
# Earth and Venus (Table 2 of protocol paper)
for p in ("earth","venus"):
    cfg[p] = deepcopy(cfg["base"])
    cfg[p]["output"] = f"chili_{p}/"
    cfg[p]["star"]["mass"] = 1.0  # Msun

# Planet mass
cfg["earth"]["struct"]["mass_tot"] = 1.0    # Mearth
cfg["venus"]["struct"]["mass_tot"] = 0.815  # Mearth

# Orbital distance
cfg["earth"]["orbit"]["semimajorax"] = 1.0    # AU
cfg["venus"]["orbit"]["semimajorax"] = 0.723  # AU

# Scale bolometric flux to match Baraffe tracks at t=t0
cfg["earth"]["star"]["bol_scale"] = 1005.3/920.0
cfg["venus"]["star"]["bol_scale"] = 1.0

# Write grid configs (H and C inventories) for these two planets
for p in ("earth","venus"):
    grd[p] = deepcopy(grd["base"])
    grd[p]["output"]     = f"chili_{p}_grid/"
    grd[p]["ref_config"] = f"input/chili/intercomp/{p}.toml"

    pth = os.path.join(intercomp, p+".grid.toml")
    with open(pth,'w') as hdl:
        toml.dump(grd[p], hdl)
    print(f"Wrote {p:5s} grid to {pth}")

# ------------------------
# TRAPPIST-1 b/e/alpha (Table 4 of protocol paper)
for p in ("tr1a","tr1b","tr1e"):
    cfg[p] = deepcopy(cfg["base"])
    cfg[p]["output"] = f"chili_{p}/"
    cfg[p]["star"]["mass"] = 0.1      # Msun
cfg["tr1b"]["orbit"]["semimajorax"] = 1.154e-2 # AU
cfg["tr1e"]["orbit"]["semimajorax"] = 2.295e-2 # AU
cfg["tr1a"]["orbit"]["semimajorax"] = 6.750e-4 # AU


# ------------------------
# Write all planet configs
for p in cfg.keys():
    pth = os.path.join(intercomp, p+".toml")
    with open(pth,'w') as hdl:
        toml.dump(cfg[p], hdl)
    print(f"Wrote {p:5s} config to {pth}")

# Exit
print(" ")
print("Done!")
