from utils import run_proteus, update_toml

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import os

# run from PROTEUS directory with: python play_dummy/gen_synth.py
# creates plots and csv files with synthetic data by varying one parameter and keeping all else fixed at dummy config


dummy_path = "input/demos/dummy.toml"

run_name = "gen_synth_data"
config_path = "input/gen_synth/" + run_name + ".toml"
out_path = "output/" + run_name + "/runtime_helpfile.csv"

observables = [ "z_obs",
                "R_int",
                "M_planet",
                "transit_depth",
                "bond_albedo",
                "contrast_ratio"]

parameters = {"struct.mass_tot": [0.5, 3.0],
                        "struct.corefrac": [0.3, 0.9],

                        "atmos_clim.surf_greyalbedo": [0.0, 0.51],
                        "atmos_clim.dummy.gamma": [0.05, 0.95],

                        "escape.dummy.rate": [0.0, 1e5],

                        "interior.dummy.magma": [2000, 4500],

                        "outgas.fO2_shift_IW": [-4.0, 4.0],

                        "delivery.radio_K": [50, 400],
                        "delivery.elements.H_oceans": [0.5, 18]}

for parameter, ran in parameters.items():

    index = np.linspace(ran[0], ran[1], 20)

    graph_df = pd.DataFrame(columns=(["index"].append(observables)))

    print(f"\nGenerating data by varying {parameter}\n")

    for i in tqdm(index):
        dummy_params = {   "params.out.path": run_name,

                            "struct.mass_tot": 1.0,
                            "struct.corefrac": 0.55,

                            "atmos_clim.surf_greyalbedo": 0.1,
                            "atmos_clim.dummy.gamma": 0.7,

                            "escape.dummy.rate": 2.0e4,

                            "interior.dummy.magma": 3500,

                            "outgas.fO2_shift_IW": 2,

                            "delivery.radio_K": 310.0,
                            "delivery.elements.H_oceans": 6.0}

        update_params = dummy_params

        update_params[parameter] = i

        # uptdate config file
        update_toml(dummy_path, update_params, config_path)

        out = run_proteus(config_path, out_path)

        out["index"] = i

        graph_df = pd.concat([graph_df, out.to_frame().T], ignore_index=True)

    # save data
    name = f"{parameter}.csv"
    direct = "play_dummy/synth_data"
    os.makedirs(direct, exist_ok=True)

    path = os.path.join(direct,name)
    graph_df.to_csv(path, sep= "\t",index=False)


    for observable in graph_df.columns[:-1]:

        plt.figure()
        plt.plot(index, graph_df[observable], label=observable)
        plt.title(f'{observable} vs {parameter}')
        plt.xlabel(f"{parameter}")
        plt.ylabel(observable)
        plt.legend()
        plt.grid(True)

        name = f'{observable} vs {parameter}.png'
        direct = f"play_dummy/plots/{parameter}"
        os.makedirs(direct, exist_ok=True)

        path = os.path.join(direct, name)
        plt.savefig(path, dpi = 300)

        plt.close()

print("done")
