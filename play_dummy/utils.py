import toml
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time

import torch
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP

from concurrent.futures import ProcessPoolExecutor
import itertools

def run_proteus(parameters, run_name, observables = None, ref_config = "input/demos/dummy.toml"):
    """
    Run the simulator using a .toml configuration file and return the results as a DataFrame.

    Input:
        parameters (dict): parameter, value pairs to change vis a vis reference config
        run_name (str): name of experiment, will be used to create input and output directories in input and output folder, can be "directory/name"
        observables (list): PROTEUS output to be return, defaults to some interesting ones
        ref_config (str): path to reference config, defaults to the dummy.toml

    Returns:
        pd.DataFrame: Output data loaded into a Pandas DataFrame.
    """

    # the default output
    if observables == None:
        observables = [    "z_obs",
                            "R_int",
                            "M_planet",
                            "transit_depth",
                            "bond_albedo",
                            "contrast_ratio"]

    # create config file for this run

    # path to proteus output
    out_path = "output/" + run_name + "/runtime_helpfile.csv"

    # set path to output
    parameters["params.out.path"] = run_name

    # set path to config file for this run
    config_path = "input/" + run_name + ".toml"

    # create
    update_toml(ref_config, parameters, config_path)

    # Run the simulator as a subprocess
    try:
        subprocess.run(
            ["proteus", "start", "-c", config_path],
            check=True,
            stdout=subprocess.DEVNULL # to suppress terminal output
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Simulator failed with error: {e}\n\n The inputs were: \n\n {parameters}")

    # Load the output CSV file into a Pandas DataFrame
    try:
        df = pd.read_csv(out_path, delimiter="\t")
    except FileNotFoundError:
        raise RuntimeError(f"Output file {out_path} not found. Check the simulator configuration.")

    out = df.iloc[-1][observables]
    return out.T

def sample_proteus(n, parameters = None, observables = None):

    """
    Inputs:
        n (int): The number of samples to draw
        parameters (dict | None): If supplied, a dictionary of parameters to sample and their ranges.
        observables (list | None): If supplied, a list of observables to return.

    Outputs:
        X (pd.DataFrame): Sampled inputs
        Y (pd.DataFrame): Calculated outputs
    """

    # config and output files will always be overwritten
    # name of new config file
    run_name = "gen_synth/sample_synth_data"

    if parameters == None:

        parameters = {"struct.mass_tot": [0.5, 3.0],
                        "struct.corefrac": [0.3, 0.9],

                        "atmos_clim.surf_greyalbedo": [0.0, 0.51],
                        "atmos_clim.dummy.gamma": [0.05, 0.95],

                        "escape.dummy.rate": [0.0, 1e5],

                        "interior.dummy.ini_tmagma": [2000, 4500],

                        "outgas.fO2_shift_IW": [-4.0, 4.0],

                        "delivery.radio_K": [50, 400],
                        "delivery.elements.H_oceans": [0.5, 18]}

    if observables == None:

        observables = [ "z_obs",
                        "R_int",
                        "M_planet",
                        "transit_depth",
                        "bond_albedo",
                        "contrast_ratio"]

    X = []
    Y = []

    for sample in tqdm(range(n)):

        par = {}
        xs = []
        for paramter, ran in parameters.items():

            x = np.random.uniform(ran[0], ran[1])

            xs.append(x)
            par[paramter] = x

        X.append(xs)


        ys = run_proteus(par, run_name, observables)
        Y.append(ys)

    X = pd.DataFrame(X, columns=parameters.keys())
    Y = pd.DataFrame(Y, columns=observables)

    return X, Y



def update_toml(config_file, updates, output_file=None):
    """
    Update values in a .toml configuration file.

    Parameters:
        config_file (str): Path to the existing .toml file.
        updates (dict): Dictionary of updates to apply, with support for nested keys (e.g., "section.key").
        output_file (str): Optional path to save the updated .toml file. If None, overwrites the input file.

    Returns:
        None
    """
    # Load the current configuration
    with open(config_file, 'r') as f:
        config = toml.load(f)

    # Apply updates
    for key, value in updates.items():
        keys = key.split('.')  # Support for nested keys with dot notation
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})  # Create nested dictionaries if they don't exist
        d[keys[-1]] = value

    # Ensure the directory for the output file exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the updated configuration
    output_file = output_file or config_file  # Default to overwriting the input file
    with open(output_file, 'w') as f:
        toml.dump(config, f)

    # print(f"Configuration file updated and saved to {output_file}.")

def run_proteus_wrapper(args, obs):
    return run_proteus(*args, observables = obs)

def run_proteus_parallel(inputs, n, obs = None):

    """
    Run n simulations in parallel.

    Inputs:
        inputs (list): A list of tuples, first element a dict of parameter and values, second element the run name (str)
        n (int): the number of workers requested
        obs (list): Which observables to return, defaults to some interesting ones

    Returns:
        outputs (list): A list of pd.Series, the PROTEUS outputs

    """

    t0 = time.time()

    # Use ProcessPoolExecutor to parallelize the task
    with ProcessPoolExecutor(max_workers=n) as executor:
        outputs = list(executor.map(run_proteus_wrapper, inputs, itertools.repeat(obs)))

    t1 = time.time()

    print(f"{len(inputs)} runs took: {t1 - t0:.2f} seconds")

    return outputs


def J(x, true_y, bounds, parameters, observables, out_scaler):

    """
    Get objecitve values at q-batch

    Input:
        x (tensor): normalized q x d dimensional input batch
        true_y (tensor): the observables of the plante under investigation
        bounds (tensor): 2 x d
        observables (list): the observables considered
        parameters (list): the parameters to be inferred
        out_scaler (StandardScaler): the scipy scaler used to standardize outputs

    Output:
        J (tensor): q x 1 dimensional objective values
        sim_list (list): list of q tensors, each a PROTEUS output
    """

    # proteus needs unnormalized inputs
    x = unnormalize(x, bounds)

    q = len(x)

    run_names = [f"BO/workers/worker{i}" for i in range(q)]

    par = [{parameters[j]: i[j].item() for j in range(len(parameters))} for i in x]

    inputs = list(zip(par, run_names))

    sim_list = run_proteus_parallel(inputs, q, obs = observables)

    sim_list = [torch.tensor(i.values).reshape(1, -1) for i in sim_list]
    sim = [out_scaler.transform(i) for i in sim_list]
    J = np.array([-((true_y-i)**2).sum().reshape(1,1) for i in sim])
    J = torch.tensor(J).reshape(q, 1)

    return J, torch.cat(sim_list)


def init_model(train_X, train_Y):

    gp = SingleTaskGP(  train_X=train_X,
                        train_Y=train_Y,
                        input_transform=Normalize(d=train_X.shape[-1]),
                        outcome_transform=Standardize(m=1)
    )


    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

    return mll, gp

def sample_inputs(parameters, batch_size):
    """
    sample the input space uniformly

    Inputs:
        parameters (dict): d parameters to sample and their ranges
        batch_size (int): how many samples to draw

    Returns:
        x (torch.tensor): sampled inputs of dimension batch_size x d
    """

    X = []
    for _ in range(batch_size):

        xs = []
        for paramter, ran in parameters.items():

            x = np.random.uniform(ran[0], ran[1])

            xs.append(x)

        X.append(xs)

    X = torch.tensor(X, dtype = torch.double)

    return X
