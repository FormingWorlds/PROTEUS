"""Generate and save initial dataset for Bayesian optimization.

This script sets up parameter bounds and true observables for the PROTEUS simulator,
builds the objective function via `prot_builder`, generates a small random sample
of points in the normalized input space, evaluates the objective to obtain outputs,
and saves the resulting dataset to disk for use as the initial data in the BO pipeline.
"""
from __future__ import annotations

import pickle

import torch
from objective import prot_builder

# Use double precision for all tensor computations
dtype = torch.double

output = "output/inference/"
ref_config = "input/demos/dummy.toml"

# Define parameter bounds as [low, high] pairs. These keys must match BO_config.toml.
params = {
    'struct.mass_tot': [0.5, 3.0],
    'struct.corefrac': [0.3, 0.9],
    'atmos_clim.dummy.gamma': [0.05, 0.95],
    'escape.dummy.rate': [1.0, 1e5],
    'interior.dummy.ini_tmagma': [2000.0, 4500.0],
    'outgas.fO2_shift_IW': [-4.0, 4.0],
}

# True observable values used to compute the objective
observables = {
    'R_int': 7629550.6175,
    'M_planet': 7.9643831975e24,
    'transit_depth': 0.00012026905833,
    'bond_albedo': 0.25,
}

# Build the PROTEUS-based objective function with fixed context
f = prot_builder(
    parameters=params,
    observables=observables,
    worker=0,
    iter=0,
    ref_config=ref_config,
    output=output
)

# Determine problem dimension (number of parameters)
d = len(params)
# Number of initial samples; can be adjusted as needed
n = 2

# Set random seed for reproducibility
torch.manual_seed(2)

# Generate n random points in [0,1]^d and evaluate the objective
X = torch.rand(n, d, dtype=dtype)
Y = torch.stack([f(x[None, :]) for x in X]).reshape(n, 1)

# Package into dataset dict
D = {"X": X, "Y": Y}
print(f"Generated initial dataset with {n} points in {d}-dim space")

# Save dataset for use in BO pipeline
fpath = f"{output}/prot.pth"
with open(fpath, "wb") as f_out:
    pickle.dump(D, f_out)
    print(f"Saved initial-guess data to file: {fpath}")
