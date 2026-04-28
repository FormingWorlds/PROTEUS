from __future__ import annotations

import logging
import subprocess
from functools import partial
from pathlib import Path

import pandas as pd
import toml
import torch
from botorch.utils.transforms import unnormalize
from numpy import log10

from proteus.utils.constants import element_list, gas_list
from proteus.utils.coupler import get_proteus_directories, variable_is_logarithmic

dtype = torch.double
LOG_CLIP = 1e-30
log = logging.getLogger('fwl.' + __name__)


def update_toml(config_file: str, updates: dict, output_file: str) -> None:
    """Update values in a TOML configuration file.

    Loads the configuration from `config_file`, applies updates provided in the
    `updates` dict (supporting nested keys via dot notation), and writes the
    modified configuration to `output_file`.

    Parameters
    ----------
    - config_file (str): Path to the existing TOML file.
    - updates (dict): Mapping of keys to new values; nested keys via "section.key".
    - output_file (str): Destination path for the updated TOML file.

    Returns
    ----------
    - None
    """
    config_path = Path(config_file)
    output_path = Path(output_file)

    # Load existing config
    with open(config_path, 'r') as f:
        config = toml.load(f)

    # Apply nested updates
    for key, value in updates.items():
        parts = key.split('.')
        d = config
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value

    # Ensure destination directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write updated config
    with open(output_path, 'w') as f:
        toml.dump(config, f)


def run_proteus(
    parameters: dict,
    worker: int,
    iter: int,
    observables: list[str],
    ref_config: str,
    output: str,
) -> pd.Series:
    """Run the PROTEUS simulator and return selected observables.

    Builds a per-run TOML file, invokes the `proteus` CLI, and reads the resulting CSV.

    Parameters
    ----------
    - parameters (dict): Parameter-value pairs to set in the simulation config.
    - worker (int): Worker identifier for directory organization.
    - iter (int): Iteration identifier for directory organization.
    - observables (list[str]): Names of output columns to return.
    - ref_config (str): Path to the reference TOML config template.
    - output (str): Path to output relative to PROTEUS output folder.

    Returns
    ----------
    - pd.Series: Last row of the simulator output containing requested observables.
    """

    # Construct run-specific paths
    run_id = Path('workers') / f'w_{worker}' / f'i_{iter}'
    out_dir = Path(output) / run_id

    out_abs = Path(get_proteus_directories(str(out_dir))['output'])
    out_cfg = out_abs / 'input.toml'
    out_csv = out_abs / 'runtime_helpfile.csv'

    # Ensure output directory exists
    out_abs.mkdir(parents=True, exist_ok=True)

    # Inject output path into simulation parameters
    parameters['params.out.path'] = str(out_dir)

    # Don't allow workers to make plots or logs
    parameters['params.out.plot_mod'] = 'none'
    parameters['params.out.logging'] = 'WARNING'

    # Generate config
    update_toml(ref_config, parameters, str(out_cfg))

    # Run PROTEUS
    command = ['proteus', 'start', '-c', str(out_cfg), '--offline']
    try:
        subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as err:
        log.error(f"Cannot execute '{command[0]}': command not found")
        raise RuntimeError("Failed to run PROTEUS: 'proteus' command not found") from err
    except subprocess.CalledProcessError as err:
        log.error(f'PROTEUS run failed for worker={worker} iter={iter} outdir={str(out_dir)}')
        raise RuntimeError(
            f'Failed to run PROTEUS for worker={worker} iter={iter}; exit code {err.returncode}'
        ) from err

    # Re-write config in case simulator mutates or removes it
    update_toml(ref_config, parameters, str(out_cfg))

    # Read simulator output
    df_row = dict(pd.read_csv(out_csv, delimiter=r'\s+').iloc[-1])

    # Handle case where atmosphere has escaped
    #   Set VMRs and MMW to zero
    if bool(df_row['P_surf'] < 1e-30):
        df_row['atm_kg_per_mol'] = 0.0
        for g in gas_list:
            df_row[g + '_vmr'] = 0.0
        for e in element_list:
            df_row[f'{e}_kg_atm'] = 0.0

    # Make into dict
    try:
        observables_dict = {obs: df_row[obs] for obs in observables}
    except KeyError as e:
        raise KeyError(f"Requested observable '{e.args[0]}' not found") from e

    return observables_dict


def log_warp(sq_dist):
    """Map squared distances to the optimization objective scale.

    Parameters
    ----------
    - sq_dist (torch.Tensor): Non-negative squared distance tensor.

    Returns
    ----------
    - torch.Tensor: Warped score, larger for closer matches.
    """
    warped_dist = -torch.log10(sq_dist + 1e-10)
    return warped_dist


def eval_obj(sim_dict, tru_dict):
    """Evaluate objective value from simulated and target observables.

    The metric compares each observable in either linear or log space,
    accumulates squared normalized differences, and applies `log_warp`.

    Parameters
    ----------
    - sim_dict (dict): Simulated observable values.
    - tru_dict (dict): Target (reference) observable values.

    Returns
    ----------
    - torch.Tensor: Objective tensor of shape (1, 1).
    """

    sim_vals = []
    tru_vals = []
    for k in sim_dict.keys():
        # some variables scale logarithmically
        if variable_is_logarithmic(k):
            sim_vals.append(log10(max(sim_dict[k], LOG_CLIP)))
            tru_vals.append(log10(max(tru_dict[k], LOG_CLIP)))
        # others are just linear
        else:
            sim_vals.append(sim_dict[k])
            tru_vals.append(tru_dict[k])

    # Convert to tensor and reshape
    sim = torch.tensor(sim_vals, dtype=dtype).reshape(1, -1)
    true_y = torch.tensor(tru_vals, dtype=dtype).reshape(1, -1)

    # Compute normalized difference and squared distance

    diff = torch.ones(1, 1, dtype=dtype) - sim / true_y
    sq_dist = (diff**2).sum(dim=1, keepdim=True)

    # obj = torch.ones(1, 1, dtype=dtype) - sq_dist

    # print("\n",sim, true_y, obj, "\n")

    obj = log_warp(sq_dist)
    return obj


def J(
    x: torch.Tensor,
    parameters: list[str],
    true_observables: dict[str, float],
    worker: int,
    iter: int,
    output: str,
    ref_config: str,
) -> torch.Tensor:
    """Run PROTEUS, and then compute the objective value for a given normalized input.

    Transforms normalized `x` to raw parameters, runs the simulator,
    and computes the squared-error based objective:

        J = log_10(sum((1 - sim/true)^2) + 1e-10)

    Parameters
    ----------
    - x (torch.Tensor): Normalized input tensor of shape (1, d).
    - parameters (list[str]): Ordered list of parameter keys corresponding to x.
    - true_observables (dict): Mapping of observable names to target values.
    - worker (int): Worker identifier.
    - iter (int): Iteration number.
    - output (str): Path to output folder relative to PROTEUS output folder.
    - ref_config (str): Reference TOML config path.

    Returns
    ----------
    - torch.Tensor: Objective value tensor of shape (1, 1).
    """

    # Map normalized x to raw parameter dict and run PROTEUS
    raw = {parameters[i]: x[0, i].item() for i in range(len(parameters))}
    sim_vals = run_proteus(
        parameters=raw,
        worker=worker,
        iter=iter,
        observables=list(true_observables.keys()),
        ref_config=ref_config,
        output=output,
    )

    # Return value of objective function given these results
    return eval_obj(sim_vals, true_observables)


def prot_builder(
    parameters: dict[str, list[float]],
    observables: dict[str, float],
    worker: int,
    iter: int,
    output: str,
    ref_config: str,
) -> callable:
    """Factory returning a BO-compatible objective function for PROTEUS inference.

    Precomputes bounds for unnormalization and embeds simulation context.

    Parameters
    ----------
    - parameters (dict): Mapping of parameter keys to [low, high] bounds.
    - observables (dict): Target observable values.
    - worker (int): Worker identifier.
    - iter (int): Iteration number (seed) for reproducibility.
    - output (str): Path to output folder relative to PROTEUS output folder.
    - ref_config (str): Reference TOML config path.

    Returns
    ----------
    - callable: Function f(x_norm) -> y_objective.
    """
    # Build bounds tensor for unnormalization
    d = len(parameters)
    bounds = torch.tensor(
        [[list(parameters.values())[i][j] for i in range(d)] for j in range(2)], dtype=dtype
    )

    def f(x_norm: torch.Tensor) -> torch.Tensor:
        """Inference objective function accepting normalized inputs.

        Parameters
        ----------
        - x_norm (torch.Tensor): Input tensor in normalized [0, 1]^d space.

        Returns
        ----------
        - torch.Tensor: Objective value tensor of shape (1, 1).
        """
        # Convert normalized to raw inputs
        x_raw = unnormalize(x_norm, bounds)
        # Partially apply J with fixed context
        J_context = partial(
            J,
            parameters=list(parameters.keys()),
            true_observables=observables,
            worker=worker,
            iter=iter,
            ref_config=ref_config,
            output=output,
        )
        # Compute objective
        return J_context(x_raw)

    return f
