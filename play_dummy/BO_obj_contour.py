from utils import J, sample_proteus, run_proteus

import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# this will take a while, it runs n**2 simulations (batched, but still)


par = {"struct.mass_tot": [0.5, 3.0],
              "struct.corefrac": [0.3, 0.9]
}

ks = list(par.keys())

observables = [ "contrast_ratio",
                "M_planet",
                "R_int",
                "transit_depth"]

if __name__ == "__main__":

    # this is just to build the out_scaler, need a better solution to standardize observables
    data_path = "play_dummy/synth_data/struct.mass_tot.csv"
    output = pd.read_csv(data_path)[observables]
    out_scaler = StandardScaler()

    # work with standardized observables to keep objective at reasonable size
    output = out_scaler.fit_transform(torch.tensor(output.values))


    np.random.seed(4)

    # print("\ngenerating planet")
    # true_X, obs = sample_proteus(1, par, observables)
    # obs = out_scaler.transform(torch.tensor(obs.values))
    # true_X = torch.tensor(true_X.values)

    true_X = torch.tensor([[1.75, 0.6]])
    par = {'struct.mass_tot': 1.75, 'struct.corefrac': 0.6}
    obs = run_proteus(par,"tmp", observables).values.reshape(1,-1)
    obs = out_scaler.transform(torch.tensor(obs))

    x1_true = true_X[0][0]
    x2_true = true_X[0][1]

    n = 40

    x1 = np.linspace(0.5, 3.0, n)
    x2 = np.linspace(0.3, 0.9, n)
    X1, X2 = np.meshgrid(x1, x2)

    grid_points = np.stack([X1.ravel(), X2.ravel()], axis=1)


    Z, Sims = J(grid_points, obs, ks, observables, out_scaler, in_normalized=False)
    Z = Z.reshape(X1.shape)
    np.save("play_dummy/synth_data/grided_objective.npy", Z)
    np.save("play_dummy/synth_data/grided_sims.npy", Sims)

    # Z = np.load("play_dummy/synth_data/grided_output.npy")


    plt.figure(figsize=(8, 6))
    contour = plt.contour(X1, X2, Z, levels=20, cmap="viridis")  # Contour lines
    plt.contourf(X1, X2, Z, levels=20, cmap="viridis", alpha=0.75)  # Filled contours
    plt.colorbar(label="objective for given planet")  # Add colorbar
    plt.xlabel("struct.mass_tot")
    plt.ylabel("struct.corefrac")
    plt.grid(True)

    plt.scatter(x1_true, x2_true, color="red", marker="o", s=100, edgecolor="black", label="true input")

    plt.legend()

    plt.savefig("play_dummy/plots/objective.png")
