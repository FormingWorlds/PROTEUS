# import toml
# import subprocess
# import pandas as pd
# import os
# from functools import partial
# import torch
# from botorch.utils.transforms import unnormalize

# dtype = torch.double




# def update_toml(config_file, updates, output_file):
#     """
#     Update values in a .toml configuration file.

#     Parameters:
#         config_file (str): Path to the existing .toml file.
#         updates (dict): Dictionary of updates to apply, with support for nested keys (e.g., "section.key").
#         output_file (str): Path to save the updated .toml file. If None, overwrites the input file.

#     Returns:
#         None
#     """
#     # Load the current configuration
#     with open(config_file, 'r') as f:
#         config = toml.load(f)

#     # Apply updates
#     for key, value in updates.items():

#         keys = key.split('.')  # Support for nested keys with dot notation
#         d = config

#         for k in keys[:-1]:
#             d = d.setdefault(k, {})  # Create nested dictionaries if they don't exist

#         d[keys[-1]] = value


#     # Ensure the directory for the output file exists
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)

#     # Save the updated configuration
#     with open(output_file, 'w') as f:
#         toml.dump(config, f)

# def run_proteus(parameters: dict,
#                 worker: int,
#                 iter: int,
#                 observables: list|None = None,
#                 ref_config = "input/demos/dummy.toml",
#                 max_attempts = 2):
#     """
#     Run the simulator using a .toml configuration file and return the results as a DataFrame.

#     Input:
#         parameters (dict): parameter, value pairs to change vis a vis reference config
#         worker (int): worker id
#         iter (int): worker iteration
#         observables (list): PROTEUS output to be return, defaults to some interesting ones
#         ref_config (str): path to reference config, defaults to the dummy.toml
#         max_attempts (int): number of tries till error

#     Returns:
#         pd.DataFrame: Output data loaded into a Pandas DataFrame.
#     """

#     if observables is None:
#             observables = [
#                             "R_int",
#                             "M_planet",
#                             "transit_depth",
#                             "bond_albedo",
#                             ]

#     run_id = f"w_{worker}/i_{iter}/"
#     out_path = "output/inference/workers/" + run_id + "/runtime_helpfile.csv"

#     # set path to output
#     parameters["params.out.path"] = "inference/workers/" + run_id

#     # set path to config file for this run
#     config_path = "output/inference/workers/" + run_id + "input.toml"

#     for attempt in range(1, max_attempts+1):
#         try:

#             update_toml(ref_config, parameters, config_path)

#             subprocess.run(
#                 ["proteus", "start", "-c", config_path],
#                 check=True,
#                 # stdout=subprocess.DEVNULL # to suppress terminal output
#             )

#             # call second time to save the config, got overwritten by proteus call
#             update_toml(ref_config, parameters, config_path)

#             df = pd.read_csv(out_path, delimiter="\t")

#             # print(df.iloc[-1][observables].T)
#             return df.iloc[-1][observables].T

#         # temporary fix for nummerical issues
#         except subprocess.CalledProcessError as e:
#             if attempt < max_attempts:
#                 print(f"Attempt {attempt} failed: {e}. Retrying with perturbed parameters...")
#                 parameters["struct.corefrac"] += 1e-1
#             else:
#                 raise RuntimeError(f"Simulator failed with error: {e}\n\n The inputs were: \n\n {parameters}")


# def J(x,
#       parameters,
#       true_observables,
#       worker, iter,
#       ref_config,
#       ):

#     """
#     Get objecitve value at x

#     Input:
#         x (tensor): raw 1 x d dimensional input
#         parameters (list): the parameters to be inferred
#         true_observables (dict): the observables of the planet under investigation

#     Output:
#         J (tensor): 1 x 1 dimensional objective values
#     """

#     input = {parameters[i]: x[0, i].item() for i in range(len(parameters))}
#     observables = list(true_observables.keys())

#     sim = run_proteus(parameters=input,
#                       observables=observables,
#                       worker=worker,
#                       iter=iter,
#                       ref_config=ref_config,
#                       )


#     sim = torch.tensor(sim.values, dtype=dtype).reshape(1, -1)
#     true_y = torch.tensor(list(true_observables.values()), dtype=dtype).reshape(1, -1)

#     # print("sim:\n ", sim)
#     # print("true_y:\n ", true_y)

#     # sim = out_scaler.transform(sim)
#     # true_y = out_scaler.transform(true_y)

#     # nummerical stability
#     # diff = true_y.log()-sim.log()
#     # diff = 1./(true_y) - 1./(sim)
#     # diff = true_y-sim
#     # diff = torch.log(true_y/sim)
#     diff =  torch.ones(1,1, dtype=dtype) - sim/true_y
#     # print("diff:\n", diff)

#     sq_dist = (diff ** 2).sum(dim=1, keepdim=True)
#     J_ = torch.ones(1,1, dtype=dtype) - sq_dist

#     return J_


# def prot_builder(parameters, observables, worker, iter, ref_config):

#     """
#     Build the PROTEUS inference objective.

#     Input
#         params (dict[str, list]): keys the parameters to in infer, values the upper and lower bounds
#         observables (dict[str, float]): keys the observables considered, values the observed values

#     """

#     d = len(parameters)

#     x_bounds = torch.tensor([[list(parameters.values())[i][j] for i in range(d)] for j in range(2)],
#                             dtype = dtype
#                             )

#     def f(x):
#         """
#         Inference objective

#             Input
#                 x (torch.tensor): normalized inputs of shape (1,d)

#             Output
#                 y (torch.tensor): the objective value of shape (1,1)
#         """

#         x = unnormalize(x, x_bounds)

#         J_part = partial(J,
#                     parameters = list(parameters.keys()),
#                     true_observables = observables,
#                     worker = worker,
#                     iter = iter,
#                     ref_config = ref_config,
#                     )

#         y = J_part(x)

#         return y

#     return f
from __future__ import annotations

import os
import subprocess
from functools import partial

import pandas as pd
import toml
import torch
from botorch.utils.transforms import unnormalize

dtype = torch.double

def update_toml(config_file: str,
                updates: dict,
                output_file: str) -> None:
    """Update values in a TOML configuration file.

    Loads the configuration from `config_file`, applies updates provided in the
    `updates` dict (supporting nested keys via dot notation), and writes the
    modified configuration to `output_file`.

    Args:
        config_file (str): Path to the existing TOML file.
        updates (dict): Mapping of keys to new values; nested keys via "section.key".
        output_file (str): Destination path for the updated TOML file.
    """
    # Load existing config
    with open(config_file, 'r') as f:
        config = toml.load(f)

    # Apply nested updates
    for key, value in updates.items():
        parts = key.split('.')
        d = config
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value

    # Ensure destination directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Write updated config
    with open(output_file, 'w') as f:
        toml.dump(config, f)


def run_proteus(parameters: dict,
                worker: int,
                iter: int,
                observables: list[str],
                ref_config: str = "input/demos/dummy.toml",
                max_attempts: int = 2) -> pd.Series:

    """Run the PROTEUS simulator and return selected observables.

    Builds a per-run TOML file, invokes the `proteus` CLI, and reads the resulting CSV.
    Retries up to `max_attempts` times on failure, with a small perturbation.

    Args:
        parameters (dict): Parameter-value pairs to set in the simulation config.
        worker (int): Worker identifier for directory organization.
        iter (int): Iteration identifier for directory organization.
        observables (list[str]): Names of output columns to return.
        ref_config (str): Path to the reference TOML config template.
        max_attempts (int): Maximum retry count on simulator failure.

    Returns:
        pd.Series: Last row of the simulator output containing requested observables.
    """

    # Construct run-specific paths
    run_id = f"w_{worker}/i_{iter}/"
    out_dir = os.path.join("output/inference/workers/", run_id)
    out_path = os.path.join(out_dir, "runtime_helpfile.csv")
    config_path = os.path.join(out_dir, "input.toml")

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    # Inject output path into simulation parameters
    parameters["params.out.path"] = os.path.join("inference/workers/", run_id)

    for attempt in range(1, max_attempts + 1):
        try:
            # Generate config and run simulator
            update_toml(ref_config, parameters, config_path)
            subprocess.run(["proteus", "start", "-c", config_path], check=True)
            # Re-write config in case simulator mutates it
            update_toml(ref_config, parameters, config_path)
            # Read simulator output
            df = pd.read_csv(out_path, delimiter="\t")
            return df.iloc[-1][observables].T
        except subprocess.CalledProcessError as e:
            if attempt < max_attempts:
                # Slightly perturb to avoid numerical issues
                print(f"Attempt {attempt} failed: {e}. Retrying...")
                parameters["struct.corefrac"] = parameters.get("struct.corefrac", 0) + 1e-1
            else:
                raise RuntimeError(f"Simulator failed after {max_attempts} attempts: {e}\nInputs: {parameters}")


def J(x: torch.Tensor,
      parameters: list[str],
      true_observables: dict[str, float] ,
      worker: int,
      iter: int,
      ref_config: str) -> torch.Tensor:
    """Compute the objective value for a given normalized input.

    Transforms normalized `x` to raw parameters, runs the simulator,
    and computes the squared-error based objective:

        J = 1 - sum((1 - sim/true)^2)

    Args:
        x (torch.Tensor): Normalized input tensor of shape (1, d).
        parameters (list[str]): Ordered list of parameter keys corresponding to x.
        true_observables (dict): Mapping of observable names to target values.
        worker (int): Worker identifier.
        iter (int): Iteration number.
        ref_config (str): Reference TOML config path.

    Returns:
        torch.Tensor: Objective value tensor of shape (1, 1).
    """
    # Map normalized x to raw parameter dict
    raw = {parameters[i]: x[0, i].item() for i in range(len(parameters))}
    sim_vals = run_proteus(parameters=raw,
                           worker=worker,
                           iter=iter,
                           observables=list(true_observables.keys()),
                           ref_config=ref_config)
    # Convert to tensor and reshape
    sim = torch.tensor(sim_vals.values, dtype=dtype).reshape(1, -1)
    true_y = torch.tensor(list(true_observables.values()), dtype=dtype).reshape(1, -1)

    # Compute normalized difference and squared distance
    diff = torch.ones(1, 1, dtype=dtype) - sim / true_y
    sq_dist = (diff ** 2).sum(dim=1, keepdim=True)
    # Return final objective
    return torch.ones(1, 1, dtype=dtype) - sq_dist


def prot_builder(parameters: dict[str, list[float]],
                 observables: dict[str, float],
                 worker: int,
                 iter: int,
                 ref_config: str) -> callable:
    """Factory returning a BO-compatible objective function for PROTEUS inference.

    Precomputes bounds for unnormalization and embeds simulation context.

    Args:
        parameters (dict): Mapping of parameter keys to [low, high] bounds.
        observables (dict): Target observable values.
        worker (int): Worker identifier.
        iter (int): Iteration number (seed) for reproducibility.
        ref_config (str): Reference TOML config path.

    Returns:
        callable: Function f(x_norm) -> y_objective.
    """
    # Build bounds tensor for unnormalization
    d = len(parameters)
    bounds = torch.tensor([[list(parameters.values())[i][j] for i in range(d)] for j in range(2)],
                          dtype=dtype)

    def f(x_norm: torch.Tensor) -> torch.Tensor:
        """Inference objective function accepting normalized inputs."""
        # Convert normalized to raw inputs
        x_raw = unnormalize(x_norm, bounds)
        # Partially apply J with fixed context
        J_context = partial(J,
                            parameters=list(parameters.keys()),
                            true_observables=observables,
                            worker=worker,
                            iter=iter,
                            ref_config=ref_config)
        # Compute objective
        return J_context(x_raw)

    return f
