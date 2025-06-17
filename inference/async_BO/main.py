import os

# prevent workers from trying to use each others cpus
# this makes a massive speed up
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # for Mac

import time
import torch

import argparse
import toml
from datetime import datetime

from objective import prot_builder
from async_BO import parallel_process, checkpoint
from utils import print_results

dtype = torch.double

# python inference/async_BO/main.py --config inference/async_BO/BO_config.toml

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config file")
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = toml.load(file)

    os.makedirs(config["directory"], exist_ok=True)
    with open(config["directory"]+"config.toml", "w") as file:
        timestamp = datetime.now().astimezone().isoformat()
        file.write(f"# Created: {timestamp}\n\n")
        toml.dump(config, file)

    assert os.cpu_count()-1 >= config["n_workers"]

    print("\nstarting optimization\n")
    t_0 = time.perf_counter()
    D_final, logs, Ts = parallel_process(objective_builder=prot_builder,
                                         **config
                                         )
    t_1 = time.perf_counter()

    print(f"this took: {(t_1-t_0):.2f} seconds\n")

    print_results(D_final, logs, config)

    # save
    checkpoint(D_final, logs, Ts, config["directory"])


