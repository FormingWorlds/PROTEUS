import toml
import subprocess
import pandas as pd
import os
from functools import partial

import torch
import numpy as np

from botorch.utils.transforms import unnormalize

dtype = torch.double




def update_toml(config_file, updates, output_file):
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
    with open(output_file, 'w') as f:
        toml.dump(config, f)

def run_proteus(parameters: dict,
                observables: list,
                worker: int, iter: int,
                ref_config = "input/demos/dummy.toml",
                max_attempts = 2):
    """
    Run the simulator using a .toml configuration file and return the results as a DataFrame.

    Input:
        parameters (dict): parameter, value pairs to change vis a vis reference config
        worker (int): worker id
        iter (int): worker iteration
        run_name (str): name of experiment, will be used to create input and output directories in input and output folder, can be "directory/name"
        observables (list): PROTEUS output to be return, defaults to some interesting ones
        ref_config (str): path to reference config, defaults to the dummy.toml
        max_attempts (int): number of tries till error

    Returns:
        pd.DataFrame: Output data loaded into a Pandas DataFrame.
    """


    run_id = f"w_{worker}/i_{iter}/"
    out_path = "output/inference/" + run_id + "/runtime_helpfile.csv"

    # set path to output
    parameters["params.out.path"] = run_id

    # set path to config file for this run
    config_path = "input/inference/" + run_id + ".toml"

    for attempt in range(1, max_attempts+1):
        try:
            # create
            update_toml(ref_config, parameters, config_path)

            # Run the simulator as a subprocess
            subprocess.run(
                ["proteus", "start", "-c", config_path],
                check=True,
                stdout=subprocess.DEVNULL # to suppress terminal output
            )

            # Load the output CSV file into a Pandas DataFrame
            df = pd.read_csv(out_path, delimiter="\t")

            return df.iloc[-1][observables].T

        except subprocess.CalledProcessError as e:
            if attempt < max_attempts:
                print(f"Attempt {attempt} failed: {e}. Retrying with perturbed parameters...")
                parameters["struct.corefrac"] += 1e-8
            else:
                raise RuntimeError(f"Simulator failed with error: {e}\n\n The inputs were: \n\n {parameters}")


def J(x,
      parameters,
      true_observables,
      worker, iter,
      ref_config,
      ):

    """
    Get objecitve value at x

    Input:
        x (tensor): normalized 1 x d dimensional input
        parameters (dict): the parameters to be inferred and their bounds
        true_observables (dict): the observables of the planet under investigation

    Output:
        J (tensor): 1 x 1 dimensional objective values
    """

    input = {parameters.keys()[i]: torch.atleast_2d(x)[0, i].item() for i in range(len(parameters))}
    observables = list(true_observables.keys())

    sim = run_proteus(input,
                      observables,
                      worker,
                      iter,
                      ref_config,
                      )

    sim = torch.tensor(sim.values).reshape(1, -1)
    true_y = torch.tensor(true_observables.values()).reshape(1, -1)

    # sim = out_scaler.transform(sim)
    # true_y = out_scaler.transform(true_y)

    J = np.array([((true_y-sim)**2).sum().reshape(1,1)])
    J = torch.tensor(J).reshape(1, 1)

    return 1 - J


def prot_builder(parameters, observables, worker, iter, ref_config):

    """
    Build the PROTEUS inferecen objective.

    Input
        params (dict[str:list]): keys the parameters to in infer, values the upper and lower bounds
        observables (dict[str:float]): keys the observables considered, values the observed values

    """

    d = len(parameters)

    x_bounds = torch.tensor([[list(parameters.values())[i][j] for i in range(d)] for j in range(2)],
                            dtype = dtype,
                            )

    def f(x):

        x = unnormalize(x, x_bounds)

        J_part = partial(J,
                    parameters = list(parameters.keys()),
                    true_observables = observables,
                    worker = worker,
                    iter = iter,
                    ref_config = ref_config,
                    )

        return J_part(x)

    return f
