"""Sanity check script for PROTEUS Bayesian optimization workflow.

This script performs a basic verification of the PROTEUS-based objective
by sampling a random point in the normalized parameter space, running the
simulator, computing the objective, and comparing against known true inputs.

It demonstrates:
  1. Parameter normalization and unnormalization.
  2. Invocation of run_proteus and prot_builder.
  3. Computation and display of objective values.

Usage:
    python test.py
"""
from __future__ import annotations

import torch
from botorch.utils.transforms import unnormalize
from objective import prot_builder, run_proteus

dtype = torch.double
# Fix random seed for reproducibility
torch.manual_seed(1)

# Reference configuration for the PROTEUS simulator
ref = "input/demos/dummy.toml"
output = "output/inference/"

print("\nperform sanity check")

# Define parameter bounds matching prot_builder setup
params = {
    'struct.mass_tot': [0.5, 3.0],
    'struct.corefrac': [0.3, 0.9],
    'atmos_clim.dummy.gamma': [0.05, 0.95],
    'escape.dummy.rate': [1.0, 1e5],
    'interior.dummy.ini_tmagma': [2000.0, 4500.0],
    'outgas.fO2_shift_IW': [-4.0, 4.0],
}
# Observabels to consider
obs = ["R_int", "M_planet", "transit_depth", "bond_albedo"]

# List of parameter keys for ordering
keys = list(params.keys())

# Determine dimensionality
d = len(keys)

# Generate a normalized random point in [0,1]^d
x = torch.rand(1, d, dtype=dtype)
# Unnormalize to raw parameter values for display
bounds = torch.tensor([[params[k][0] for k in keys],
                       [params[k][1] for k in keys]], dtype=dtype)
raw_x = unnormalize(x, bounds).flatten()
print("\nrandom inputs (raw):", raw_x)

# Build raw parameter dict for direct simulator invocation
rand_par = {keys[i]: raw_x[i].item() for i in range(d)}
# Run PROTEUS directly and display simulator outputs
rand_obs = run_proteus(
    parameters=rand_par,
    observables=obs,
    worker=0,
    iter=0,
    output=output,
    ref_config=ref
).to_dict()
print("\nsimulated observables:", rand_obs)

# Build prot_builder objective function
f = prot_builder(
    parameters=params,
    observables=rand_obs,
    worker=0,
    iter=0,
    ref_config=ref,
    output=output,
)

# Compute objective at random input
y = f(x)
print("\nobjective value at random input (should be optimal, i.e., 1.0):", y.item())
