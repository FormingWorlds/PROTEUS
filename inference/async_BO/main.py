import time
import torch


import argparse
import tomllib
from functools import partial


from async_BO import parallel_process, checkpoint
from objective import prot_builder

dtype = torch.double


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config file")
    args = parser.parse_args()

    with open(args.config, "rb") as file:
            config = tomllib.load(file)
    config = dict(config)

    t_0 = time.perf_counter()
    D_final, logs, Ts = parallel_process(objective_builder=prot_builder, **config)
    t_1 = time.perf_counter()

    print(f"this took: {(t_1-t_0):.2f} seconds")

    # save
    checkpoint(D_final, logs, Ts, config["directory"])




