from utils import run_proteus, update_toml

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

# run from PROTEUS directory with: python play_dummy/gen_synth.py
# creates plots and csv files with synthetic data by varying one parameter and keeping all else fixed at dummy config


# name of new config file
run_name = "gen_synth/gen_synth_data"

observables = [ "z_obs",
                "R_int",
                "M_planet",
                "transit_depth",
                "bond_albedo",
                "contrast_ratio"]

parameters = {  "struct.mass_tot": [0.5, 3.0],
                "struct.corefrac": [0.3, 0.9],

                "atmos_clim.surf_greyalbedo": [0.0, 0.51],
                "atmos_clim.dummy.gamma": [0.05, 0.95],

                "escape.dummy.rate": [0.0, 1e5],

                "interior.dummy.ini_tmagma": [2000, 4500],

                "outgas.fO2_shift_IW": [-4.0, 4.0],

                "delivery.radio_K": [50, 400],
                "delivery.elements.H_oceans": [0.5, 18]}

for parameter, ran in parameters.items():

    index = np.linspace(ran[0], ran[1], 20)

    data = []

    print(f"\nGenerating data by varying {parameter}\n")

    for i in tqdm(index):

        # change parameter to i in dummy config
        par = {parameter : i.item()}

        t_0 = time.time()
        out = run_proteus(par, run_name, observables)
        t_1 = time.time()

        out["run_time"] = t_1 - t_0
        out["index"] = i

        out = out.to_list()

        data.append(out)

    col_names = observables + ["run_time", "index"]

    data = pd.DataFrame(data, columns=col_names)

    # save data
    name = f"{parameter}.csv"
    direct = "play_dummy/synth_data"
    os.makedirs(direct, exist_ok=True)

    path = os.path.join(direct,name)
    data.to_csv(path, index=False)


    for observable in data.columns[:-1]:

        plt.figure()
        plt.plot(index, data[observable], label=observable)
        plt.title(f'{observable} vs {parameter}')
        plt.xlabel(f"{parameter}")
        plt.ylabel(observable)
        plt.legend()
        plt.grid(True)

        name = f'{observable} vs {parameter}.png'
        direct = f"play_dummy/plots/data/{parameter}"
        os.makedirs(direct, exist_ok=True)

        path = os.path.join(direct, name)
        plt.savefig(path, dpi = 300)

        plt.close()

print("done")
