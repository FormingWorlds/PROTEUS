# from objective import run_proteus, prot_builder
# import torch
# from botorch.utils.transforms import normalize, unnormalize
# dtype = torch.double

# torch.manual_seed(1)

# print("\nperform sanity check")

# ref = "input/demos/dummy.toml"

# params =  {'struct.mass_tot': [0.5, 3.0],
#            'struct.corefrac': [0.3, 0.9],
#            'atmos_clim.dummy.gamma': [0.05, 0.95],
#            'escape.dummy.rate': [1.0, 1e5],
#            'interior.dummy.ini_tmagma': [2000.0, 4500.0],
#            'outgas.fO2_shift_IW': [-4.0, 4.0],
#            }


# par =  {'struct.mass_tot': 1.5,
#         'struct.corefrac': 0.35,
#         'atmos_clim.dummy.gamma': 0.11,
#         'escape.dummy.rate': 38166.62,
#         'interior.dummy.ini_tmagma': 2923.57,
#         'outgas.fO2_shift_IW': 2.03
#         }

# d = len(par)
# x_bounds = torch.tensor([[list(params.values())[i][j] for i in range(d)] for j in range(2)],
#                         dtype = torch.double,
#                         )

# x_true = torch.tensor(list(par.values()), dtype = dtype).unsqueeze(0)
# print("\ntrue inputs\n", x_true.flatten())
# x_true = normalize(x_true, x_bounds)

# true_obs = run_proteus(parameters=par, worker=0, iter=0, ref_config=ref).to_dict()

# print("\ntrue observed:\n ", true_obs)

# f = prot_builder(   parameters=params, observables=true_obs,
#                     worker=0, iter=0,
#                     ref_config=ref
#                 )

# x = torch.rand(1, d)
# raw_x =  unnormalize(x, x_bounds).flatten()
# print("\nrandom inputs\n", raw_x)

# rand_par = {list(par.keys())[i]: raw_x[i].item() for i in range(d)}
# rand_obs = run_proteus(parameters=rand_par, worker=0, iter=0, ref_config=ref).to_dict()
# print("\nsim observed:\n ", true_obs)

# y = f(x)
# print("\nobjective value at random input:", y.item())

# y_opt = f(x_true)

# print("\nobjective value at true inputs:", y_opt.item())


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
    ref_config=ref
).to_dict()
print("\nsimulated observables:", rand_obs)

# Build prot_builder objective function
f = prot_builder(
    parameters=params,
    observables=rand_obs,
    worker=0,
    iter=0,
    ref_config=ref
)

# Compute objective at random input
y = f(x)
print("\nobjective value at random input (should be optimal, i.e., 1.0):", y.item())


