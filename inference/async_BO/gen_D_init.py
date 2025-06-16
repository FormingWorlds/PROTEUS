import torch
import os
import pickle
from objective import prot_builder
dtype = torch.double

params =  {'struct.mass_tot': [0.5, 3.0],
           'struct.corefrac': [0.3, 0.9],
           'atmos_clim.dummy.gamma': [0.05, 0.95],
           'escape.dummy.rate': [1.0, 1e5],
           'interior.dummy.ini_tmagma': [2000.0, 4500.0],
           'outgas.fO2_shift_IW': [-4.0, 4.0],
           }

# this needs to match the ones in BO_config.toml!!
observables = {'R_int': 7629550.6175,
               'M_planet': 7.9643831975e+24,
               'transit_depth': 0.00012026905833,
               'bond_albedo': 0.25}

f = prot_builder(parameters=params,
                 observables=observables,
                 worker=0,
                 iter=0,
                 ref_config="input/demos/dummy.toml"
                 )


d = len(params)
n = 3*d


torch.manual_seed(1)

X = torch.rand(n,d, dtype=dtype)
Y = torch.tensor([f(x[None,:]) for x in X], dtype=dtype).reshape(n,1)
D = {"X": X,
     "Y": Y
     }

print(D)

os.makedirs("inference/data/", exist_ok=True)

with open("inference/data/prot.pth", "wb") as f:
    pickle.dump(D, f)
