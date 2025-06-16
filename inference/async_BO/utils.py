import pandas as pd
from botorch.utils.transforms import unnormalize
import torch
import toml

dtype = torch.double

def get_nested(config: dict, key: str, sep: str = "."):
    """
    Given a nested dict return flattened dict with dot-separated keys like "struct.mass_tot"
    """
    val = config
    for part in key.split(sep):
        val = val[part]
    return val

def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def print_results(D, logs, config):

    Y = D["Y"]
    i_opt = Y.argmax()
    log_opt = logs[i_opt]

    w = log_opt["worker"]
    id = log_opt["task_id"]

    out_path = f"output/inference/workers/w_{w}/i_{id}/runtime_helpfile.csv"
    df = pd.read_csv(out_path, delimiter="\t")
    true_y = pd.Series(config["observables"])
    observables = list(true_y.keys())
    sim_opt = df.iloc[-1][observables].T

    params = config["parameters"].keys()

    in_path = f"input/inference/workers/w_{w}/i_{id}/input.toml"

    with open(in_path, "r") as f:
        input = toml.load(f)

    input = flatten(input)

    par_opt = pd.Series({param : input[param] for param in params})

    print("\ntrue observables\n", true_y)
    print("\nsimulated observables\n", sim_opt)
    print("\ninferred inputs\n", par_opt)

