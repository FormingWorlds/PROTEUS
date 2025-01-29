from utils import sample_proteus

import os

# generate data by sampling inputs uniformly from their range

# observables to sample
observables = [ "z_obs",
                "R_int",
                "M_planet",
                "transit_depth",
                "bond_albedo",
                "contrast_ratio"]

# parameters to sample and their ranges
parameters = {  "struct.mass_tot": [0.5, 3.0],
                "struct.corefrac": [0.3, 0.9],

                "atmos_clim.surf_greyalbedo": [0.0, 0.51],
                "atmos_clim.dummy.gamma": [0.05, 0.95],

                "escape.dummy.rate": [0.0, 1e5],

                "interior.dummy.ini_tmagma": [2000, 4500],

                "outgas.fO2_shift_IW": [-4.0, 4.0],

                "delivery.radio_K": [50, 400],
                "delivery.elements.H_oceans": [0.5, 18]}
# number of samples
n = 10


# generate samples
X, Y = sample_proteus(n, parameters, observables)

# save data
name_x = "sampled_input.csv"
name_y = "sampled_output.csv"

direct = "play_dummy/synth_data"
os.makedirs(direct, exist_ok=True)

path_x = os.path.join(direct,name_x)
path_y = os.path.join(direct,name_y)

X.to_csv(path_x,index=False)
Y.to_csv(path_y,index=False)

print("done")
