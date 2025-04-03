
import torch
import numpy as np

from botorch.utils.transforms import unnormalize, normalize

from botorch.test_functions.synthetic import (Ackley,
                                              EggHolder,
                                              Cosine8,
                                              Hartmann,
                                              Rosenbrock,
                                              Griewank,
                                              Michalewicz,
                                              Branin,
                                              )

from utils import run_proteus, update_toml

import subprocess
import pandas as pd

from sklearn.preprocessing import StandardScaler

from functools import partial

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def ackley_builder(d):
    """
    Build normalized Ackley with unit square input
    """
    x_bounds = torch.tensor([[-32.768 for _ in range(d)],
                      [32.768 for _ in range(d)]],
                      dtype=dtype,
                      device=device
                      )

    def f(x):

        # unnormalize
        x = unnormalize(x, x_bounds)
        y = Ackley(dim = d)(x).unsqueeze(-1)
        return 1 - y/(20+torch.e)

    return f

def egg2_builder(d = 2):
    """
    Build normalized Egg holder with unit square input
    must be evaluted in 2D
    """
    if d != 2:
        raise ValueError("Egg holder only defined for 2D")

    x_bounds = torch.tensor([[-512 for _ in range(d)],
                      [512 for _ in range(d)]],
                      dtype=dtype,
                      device=device
                      )

    def f(x):

        # unnormalize
        x = unnormalize(x, x_bounds)
        y = EggHolder(negate=True)(x).unsqueeze(-1)
        return (y + 1049.1316) / (959.6407 + 1049.1316)

    return f

def cos8_builder(d=8):
    """
    Build normalized Cos8 holder with unit square input
    must be evaluted in 8D
    """
    if d != 8:
        raise ValueError("Cos8 only defined for 8D")

    x_bounds = torch.tensor([[-1 for _ in range(d)],
                      [1 for _ in range(d)]],
                      dtype=dtype,
                      device=device
                      )

    def f(x):

        # unnormalize
        x = unnormalize(x, x_bounds)
        y = Cosine8()(x).unsqueeze(-1)
        return (y + 8.8) / (0.8 + 8.8)

    return f

def hart6_builder(d=6):
    """
    Build normalized Hartmann-6 with unit square input
    """

    x_bounds = torch.tensor([[0 for _ in range(d)],
                      [1 for _ in range(d)]],
                      dtype=dtype,
                      device=device
                      )

    def f(x):

        # unnormalize
        x = unnormalize(x, x_bounds)
        y = Hartmann(dim = d, negate=True)(x).unsqueeze(-1)
        return (y - 0.) / (3.32237 - 0.)

    return f

def ros_builder(d):
    """
    Build normalized Rosenbrock with unit square input
    """

    if d < 2:
        raise ValueError("Rosenbrock only defined for at least 2D")

    x_bounds = torch.tensor([[-5 for _ in range(d)],
                      [10 for _ in range(d)]],
                      dtype=dtype,
                      device=device
                      )

    def f(x):

        # unnormalize
        x = unnormalize(x, x_bounds)
        y = Rosenbrock(dim = d, negate=True)(x).unsqueeze(-1)
        return (y + (d-1)*(100**3+81)) / (0. + (d-1)*(100**3+81))

    return f

def griewank_builder(d):
    """
    Build normalized Griewank with unit square input
    """


    x_bounds = torch.tensor([[-10 for _ in range(d)],
                      [10 for _ in range(d)]],
                      dtype=dtype,
                      device=device
                      )

    def f(x):

        # unnormalize
        x = unnormalize(x, x_bounds)
        y = Griewank(dim =d, negate=True)(x).unsqueeze(-1)
        return (y + 2) / (0. + 2)

    return f

def mic_builder(d):
    """
    Build normalized Michalewicz with unit square input
    """

    x_bounds = torch.tensor([[0 for _ in range(d)],
                             [torch.pi for _ in range(d)]],
                             dtype=dtype,
                             device=device,
                             )
    if d == 2:
        y_max = 1.8013
    elif d == 5:
        y_max = 4.687658
    elif d == 10:
        y_max = 9.66015

    else:
        raise ValueError("d needs to be 2, 5 or 10")

    def f(x):

        # unnormalize
        x = unnormalize(x, x_bounds)
        y = Michalewicz(dim = d, negate=True)(x).unsqueeze(-1)

        return y / y_max

    return f

def bran_builder(d=2):
    """
    Build normalized Branin with unit square input
    """

    x_bounds = torch.tensor([[-5, 0], [10, 15]],
                             dtype=dtype,
                             device=device,
                             )

    if d != 2:

        raise ValueError("input needs to be 2D")

    def f(x):

        # unnormalize
        x = unnormalize(x, x_bounds)
        y = Branin(negate=True)(x).unsqueeze(-1)

        return (y + 308.1291) / (-0.397887 + 308.1291)

    return f


observables = [ "contrast_ratio",
                "M_planet",
                "R_int",
                "transit_depth"]

params = {  "struct.mass_tot": [0.5, 3.0],
            "struct.corefrac": [0.3, 0.9],

            "atmos_clim.surf_greyalbedo": [0.0, 0.45],
            "atmos_clim.dummy.gamma": [0.05, 0.95],

            "escape.dummy.rate": [0.0, 1e5],

            "interior.dummy.ini_tmagma": [2000, 4500],

            "outgas.fO2_shift_IW": [-4.0, 4.0],

            "delivery.radio_K": [50, 400],
            "delivery.elements.H_oceans": [0.5, 18],
}

out_path = "play_dummy/synth_data/sampled_output.csv"
in_path = "play_dummy/synth_data/sampled_input.csv"

output = pd.read_csv(out_path)[observables]
out_scaler = StandardScaler()
out_scaler = out_scaler.fit(torch.tensor(output.values))

input = pd.read_csv(in_path)[list(params.keys())]

def J(x, true_y, parameters, out_scaler):

    """
    Get objecitve values at q-batch

    Input:
        x (tensor): normalized q x d dimensional input batch
        true_y (tensor): the observables of the planet under investigation
        observables (list): the observables considered
        parameters (list): the parameters to be inferred
        out_scaler (StandardScaler): the scipy scaler used to standardize outputs
        bounds (tensor): 2 x d, for MinMax normalization
        in_normalized (bool): inputs are normalized
        max_cores (int): the maximum number of cores to request from the system

    Output:
        J (tensor): q x 1 dimensional objective values
        sim_list (list): list of q tensors, each a PROTEUS output
    """

    run_name = "BO/worker"

    par = {parameters[i]: torch.atleast_2d(x)[0, i].item() for i in range(len(parameters))}

    sim = run_proteus(par, run_name, observables)
    sim = torch.tensor(sim.values).reshape(1, -1)
    true_y = torch.tensor(true_y.values).reshape(1, -1)
    sim = out_scaler.transform(sim)
    true_y = out_scaler.transform(true_y)

    J = np.array([((true_y-i)**2).sum().reshape(1,1) for i in sim])
    J = torch.tensor(J).reshape(1, 1)

    return 1 - J

def prot_builder(d = 9):

    if d != 9:
        raise ValueError("this is a 9D problem, set d=9")

    # print("True y: \n", output.iloc[0])
    # print("True x: \n", input.iloc[0].values)

    x_bounds = torch.tensor([[list(params.values())[i][j] for i in range(9)] for j in range(2)],
                            dtype = dtype,
                            device = device)

    # print("True norm_x: \n", normalize(torch.tensor(input.iloc[0].values), x_bounds))

    def f(x):

        x = unnormalize(x, x_bounds)

        J_part = partial(J,
                    true_y = output.iloc[0],
                    parameters = list(params.keys()),
                    out_scaler = out_scaler,
                    )

        return J_part(x)

    return f




OBJECTIVES = {"ackley": ackley_builder,
              "egg2": egg2_builder,
              "cos8": cos8_builder,
              "hart6": hart6_builder,
              "ros": ros_builder,
              "grie": griewank_builder,
              "mic": mic_builder,
              "bran": bran_builder,
              "prot": prot_builder,
              }
