from objective import run_proteus, prot_builder
from utils import get_nested
import torch
import toml

from botorch.utils.transforms import normalize, unnormalize
dtype = torch.double

torch.manual_seed(1)

print("\nperform sanity check")

ref = "input/demos/dummy.toml"

params =  {'struct.mass_tot': [0.5, 3.0],
           'struct.corefrac': [0.3, 0.9],
           'atmos_clim.dummy.gamma': [0.05, 0.95],
           'escape.dummy.rate': [1.0, 1e5],
           'interior.dummy.ini_tmagma': [2000.0, 4500.0],
           'outgas.fO2_shift_IW': [-4.0, 4.0],
           }


par =  {'struct.mass_tot': 1.5,
        'struct.corefrac': 0.35,
        'atmos_clim.dummy.gamma': 0.11,
        'escape.dummy.rate': 38166.62,
        'interior.dummy.ini_tmagma': 2923.57,
        'outgas.fO2_shift_IW': 2.03
        }

d = len(par)
x_bounds = torch.tensor([[list(params.values())[i][j] for i in range(d)] for j in range(2)],
                        dtype = torch.double,
                        )

x_true = torch.tensor(list(par.values()), dtype = dtype).unsqueeze(0)
print("\ntrue inputs\n", x_true.flatten())
x_true = normalize(x_true, x_bounds)

true_obs = run_proteus(parameters=par, worker=0, iter=0, ref_config=ref).to_dict()

print("\ntrue observed:\n ", true_obs)

f = prot_builder(   parameters=params, observables=true_obs,
                    worker=0, iter=0,
                    ref_config=ref
                )

x = torch.rand(1, d)
raw_x =  unnormalize(x, x_bounds).flatten()
print("\nrandom inputs\n", raw_x)

rand_par = {list(par.keys())[i]: raw_x[i].item() for i in range(d)}
rand_obs = run_proteus(parameters=rand_par, worker=0, iter=0, ref_config=ref).to_dict()
print("\nsim observed:\n ", true_obs)

y = f(x)
print("\nobjective value at random input:", y.item())

y_opt = f(x_true)

print("\nobjective value at true inputs:", y_opt.item())

# print("sanity check passed\n")
