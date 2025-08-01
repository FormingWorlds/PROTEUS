"""Generate and save initial dataset for Bayesian optimization.

This script sets up parameter bounds and true observables for the PROTEUS simulator,
builds the objective function via `prot_builder`, generates a small random sample
of points in the normalized input space, evaluates the objective to obtain outputs,
and saves the resulting dataset to disk for use as the initial data in the BO pipeline.
"""
from __future__ import annotations

import pickle

import torch
import os
from objective import prot_builder, eval_obj
from proteus.utils.coupler import get_proteus_directories

# Use double precision for all tensor computations
dtype = torch.double

def create_init(config):
    '''
    Create initial guess data file in output folder, using the specified method.
    '''

    # validate options
    init_grid  = str(config["init_grid"])
    if init_grid.lower().strip() == "none":
        init_grid = None
        init_samps = int(config["init_samps"])
        if init_samps < 2:
            raise ValueError("Initial guess dataset must contain >1 sample")
    else:
        init_grid = os.path.join(get_proteus_directories()["proteus"], "output", init_grid)
        init_samps = None

    # create new initial guess data by sampling bounds
    if init_samps:
        print( "Source for initial guess: sampling parameter space")
        print(f"    nsamp = {init_samps}")
        D_init = sample_from_bounds(config["output"],
                                    config["ref_config"],
                                    config["parameters"], config["observables"],
                                    init_samps)

    # read from grid
    else:
        print( "Source for initial guess: pre-computed grid")
        print(f"    grid = {init_grid}")
        D_init = sample_from_grid(config["output"],
                                    config["ref_config"],
                                    config["parameters"], config["observables"],
                                    init_grid)


    # Inform user
    print(f"Saved initial guess data to D_init pickle file: {D_init}")

def sample_from_grid(output:str, ref_config:str,
                       params:dict, observables:dict,
                       grid_dir:str):
    '''
    Create initial guess data, using pre-computed grid of models as input
    '''

    # This will be used to evaluate the objective function to provide initial samples
    #     _x is a list of values, each one corresponding to a parameter
    #     They are normalised to [0,1] within the bounds of each parameter's axis
    #     The function returns the value of the objective function for these values
    def f(_x):
        return 0.74

    # Determine problem dimension (number of parameters)
    dims = len(params)

    # Set random seed for reproducibility
    torch.manual_seed(2)

    # Generate n random points in [0,1]^d and evaluate the objective
    #     Each of the parameters are evaluated in space 0-1, normalised to the bounds
    #     This variable is 2D, with shape [nsamp, dims]
    X = torch.rand(nsamp, dims, dtype=dtype)

    # Evaluate the objective function for each of the samples
    #     This variable is 1D, with shape [nsamp]
    Y = torch.stack([f(x[None, :]) for x in X]).reshape(nsamp, 1)

    # Package into dataset dict
    D = {"X": X, "Y": Y}
    print(f"Generated initial dataset with {nsamp} points in {dims}-dim space")

    # Save dataset for use in BO pipeline
    proteus_out = get_proteus_directories(output)["output"]
    D_init_path = f"{proteus_out}/init.pkl"
    with open(D_init_path, "wb") as f_out:
        pickle.dump(D, f_out)
    return D_init_path

def sample_from_bounds(output:str, ref_config:str,
                       params:dict, observables:dict,
                       nsamp:int):
    '''
    Create initial guess data, sampling the parameter space at random.
    '''

    # Build the PROTEUS-based objective function with fixed context
    #    This will be used to evaluate the objective function to provide initial samples
    f = prot_builder(
        parameters=params,
        observables=observables,
        worker=0,
        iter=0,
        ref_config=ref_config,
        output=output
    )

    # Determine problem dimension (number of parameters)
    dims = len(params)

    # Set random seed for reproducibility
    torch.manual_seed(2)

    # Generate n random points in [0,1]^d and evaluate the objective
    #     Each of the parameters are evaluated in space 0-1, normalised to the bounds
    #     This variable is 2D, with shape [nsamp, dims]
    X = torch.rand(nsamp, dims, dtype=dtype)

    # Evaluate the objective function for each of the samples
    #     This variable is 1D, with shape [nsamp]
    Y = torch.stack([f(x[None, :]) for x in X]).reshape(nsamp, 1)

    # Package into dataset dict
    D = {"X": X, "Y": Y}
    print(f"Generated initial dataset with {nsamp} points in {dims}-dim space")

    # Save dataset for use in BO pipeline
    proteus_out = get_proteus_directories(output)["output"]
    D_init_path = f"{proteus_out}/init.pkl"
    with open(D_init_path, "wb") as f_out:
        pickle.dump(D, f_out)
    return D_init_path
