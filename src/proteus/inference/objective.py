from __future__ import annotations

import os
import subprocess
from functools import partial

import pandas as pd
import toml
import torch
from botorch.utils.transforms import unnormalize

from proteus.utils.coupler import get_proteus_directories

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
                ref_config: str,
                output: str,
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
        output (str): Path to output relative to PROTEUS output folder
        max_attempts (int): Maximum retry count on simulator failure.

    Returns:
        pd.Series: Last row of the simulator output containing requested observables.
    """

    # Construct run-specific paths
    run_id = f"w_{worker}/i_{iter}/"
    out_dir = os.path.join(output, "workers", run_id)

    out_abs = get_proteus_directories(out_dir)["output"]
    out_cfg = os.path.join(out_abs, "input.toml")
    out_csv = os.path.join(out_abs, "runtime_helpfile.csv")

    # Ensure output directory exists
    os.makedirs(out_abs, exist_ok=True)
    # Inject output path into simulation parameters
    parameters["params.out.path"] = out_dir

    for attempt in range(1, max_attempts + 1):
        try:
            # Generate config and run simulator
            update_toml(ref_config, parameters, out_cfg)
            subprocess.run(["proteus", "start", "-c", out_cfg, "--offline"], check=True)

            # Re-write config in case simulator mutates or removes it
            update_toml(ref_config, parameters, out_cfg)

            # Read simulator output
            df = pd.read_csv(out_csv, delimiter="\t")
            return df.iloc[-1][observables].T
        except subprocess.CalledProcessError as e:
            if attempt < max_attempts:
                # Slightly perturb to avoid numerical issues
                print(f"Attempt {attempt} failed: {e}. Retrying...")
                parameters["struct.corefrac"] = parameters.get("struct.corefrac", 0) + 1e-9
            else:
                raise RuntimeError(f"Simulator failed after {max_attempts} attempts: {e}\nInputs: {parameters}")


def J(x: torch.Tensor,
      parameters: list[str],
      true_observables: dict[str, float] ,
      worker: int,
      iter: int,
      output: str,
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
                           ref_config=ref_config,
                           output=output)
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
                 output: str,
                 ref_config: str) -> callable:
    """Factory returning a BO-compatible objective function for PROTEUS inference.

    Precomputes bounds for unnormalization and embeds simulation context.

    Args:
        parameters (dict): Mapping of parameter keys to [low, high] bounds.
        observables (dict): Target observable values.
        worker (int): Worker identifier.
        iter (int): Iteration number (seed) for reproducibility.
        output (str): Path to output folder relative to PROTEUS output folder
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
                            ref_config=ref_config,
                            output=output)
        # Compute objective
        return J_context(x_raw)

    return f
