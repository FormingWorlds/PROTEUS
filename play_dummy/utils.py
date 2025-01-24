import toml
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

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
        raise RuntimeError(f"Simulator failed with error: {e}")

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

                        "interior.dummy.magma": [2000, 4500],

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
