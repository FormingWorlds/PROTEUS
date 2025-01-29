from utils import sample_proteus, init_model, J, sample_inputs
import numpy as np
import pandas as pd
import os
import time

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.sampling.normal import SobolQMCNormalSampler

from scipy.stats.qmc import Halton
from sklearn.preprocessing import StandardScaler

from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, qLogExpectedImprovement, qExpectedImprovement

from botorch.optim import optimize_acqf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
dtype = torch.double


############################################################################################
# settings
NAME = "first_try"

# BO settings
NUM_ITER = 6
BATCH_SIZE = 3
N_INIT = 3

# acqf optimization settings
NUM_RESTARTS = 10
RAW_SAMPLES = 50

# set the involved observables and parameters
observables = [ "contrast_ratio",
                "M_planet",
                "R_int",
                "transit_depth"]

params = {  "struct.mass_tot": [0.5, 3.0],
            "struct.corefrac": [0.3, 0.9],

            # "atmos_clim.surf_greyalbedo": [0.0, 0.45],
            # "atmos_clim.dummy.gamma": [0.05, 0.95],

            # "escape.dummy.rate": [0.0, 1e5],

            # "interior.dummy.ini_tmagma": [2000, 4500],

            # "outgas.fO2_shift_IW": [-4.0, 4.0],

            # "delivery.radio_K": [50, 400],
            # "delivery.elements.H_oceans": [0.5, 18],
}
# helper function
def opt_acqf_obs(acq_func, obs, q, d):
    """Optimizes the acquisition function, and returns a new candidate and observation."""

    # normalized inputs
    b = torch.tensor([[i for _ in range(d)] for i in range(2)], dtype=dtype)

    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=b,
        q=q,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 25, "maxiter": 200},
    )
    # observe new values
    new_x = candidates.detach()

    new_y, new_y_raw = J(new_x, obs, bounds, ks, observables, out_scaler)

    return new_x, new_y, new_y_raw


def random_acqf_obs(params, q):

    """
    aquire random batch and get objective

    Inputs:
        params (dict): parameters to sample and their ranges
        q (int): batch size

    Returns:

    """

    new_x = sample_inputs(params, q)
    new_y, new_y_raw = J(new_x, obs, bounds, ks, observables, out_scaler)

    return new_x, new_y, new_y_raw

# save for later
ks = list(params.keys())
vs = list(params.values())
bounds = torch.tensor([[j[i] for j in vs] for i in range(2)], device=device, dtype=dtype)

# guard for child processes
if __name__ == "__main__":

    # this is just to build the out_scaler, need a better solution to standardize observables
    data_path = "play_dummy/synth_data/struct.mass_tot.csv"
    output = pd.read_csv(data_path)[observables]
    out_scaler = StandardScaler()

    # work with standardized observables to keep objective at reasonable size
    output = out_scaler.fit_transform(torch.tensor(output.values))

    # generate planet
    np.random.seed(8)

    print("\ngenerating planet")
    true_X, obs = sample_proteus(1, params, observables)
    obs = out_scaler.transform(torch.tensor(obs.values))
    norm_true_X = normalize(true_X, bounds)

    ############################################################################################
    # Initilaize BO

    halton_sampler = Halton(d=len(params), scramble=True)
    train_X = halton_sampler.random(n=N_INIT)
    train_X = torch.tensor(train_X, dtype= torch.double) * (bounds[1] - bounds[0]) + bounds[0]

    # train_X = (torch.rand(n_init, dtype=dtype) * (bounds[1] - bounds[0]) + bounds[0])

    # normalize inputs
    train_X = normalize(train_X, bounds)

    ############################################################################################
    # Start BO

    # sanity check
    print("\nperforming sanity check\n")
    jj = J(torch.tensor(norm_true_X.values), obs, bounds, ks, observables, out_scaler)[0]
    print(f"{jj}\n")
    print(f"the objective is 0 at optimizer: {(jj == 0).item()}\n")

    print("getting inital batch")
    train_Y, output_raw = J(train_X, obs, bounds, ks, observables, out_scaler)
    best_y = train_Y.max()
    best_y_rand = train_Y.max()
    bests = [best_y] # track best objective value
    bests_rand = [best_y_rand]
    best_x = train_X[torch.argmax(train_Y)][None, :]

    mll, gp = init_model(train_X, train_Y)
    fit_gpytorch_mll(mll)

    # acqf = UpperConfidenceBound(model=gp, beta=0.7)
    # acqf = ExpectedImprovement(gp, best_y)

    qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
    acqf = qLogExpectedImprovement(gp, best_y, qmc_sampler)

    # set things up to calculate time spent by BO and random search, respectively
    t_0 = time.time()
    t_r = 0.
    times = [0.]
    times_rand = [0.]
    evals = [0]

    for i in range(NUM_ITER):
        print(f"\nstarting iteration {i+1}")

        new_x, new_y, new_output_raw = opt_acqf_obs(acqf, obs, BATCH_SIZE, len(params))
        t_bo = time.time()
        times.append(times[-1] + (t_bo - t_r))

        new_x_rand, new_y_rand, new_output_raw_rand = random_acqf_obs(params, BATCH_SIZE)
        t_r = time.time()
        times_rand.append(t_r - times[-1])

        M = new_y.max()
        if M > best_y:

            best_y = M
            best_x = new_x[torch.argmax(new_y)][None, :]

        M_rand = new_y_rand.max()
        if M_rand > best_y_rand:

            best_y_rand = M_rand
            best_x_rand = new_x_rand[torch.argmax(new_y_rand)][None, :]



        evals.append(evals[-1]+BATCH_SIZE)

        bests.append(best_y)
        bests_rand.append(best_y_rand)

        train_X = torch.cat([train_X, new_x])
        train_Y = torch.cat([train_Y, new_y])
        output_raw = torch.cat([output_raw, new_output_raw])

        mll, gp = init_model(train_X, train_Y)

        fit_gpytorch_mll(mll)

        # acqf = UpperConfidenceBound(model=gp, beta=0.7)
        # acqf = ExpectedImprovement(gp, best_y)
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        acqf = qLogExpectedImprovement(gp, best_y, qmc_sampler)

    times = [0.]+[t-t_0 for t in times[1:]]

    t_1 = time.time()

    print(f"\nthis took: {(t_1 - t_0)/60:.2f} minutes")

    print(f"\nbest objective value: {best_y}\n")

    raw_est = unnormalize(best_x, bounds)

    best_x = pd.DataFrame(best_x, index=["norm est"], columns=ks)
    norm_true_X = norm_true_X.set_index([["norm true"]])
    raw_est = pd.DataFrame(raw_est, index=["raw est"], columns=ks)
    true_X = true_X.set_index([["raw true"]])

    results = pd.concat([best_x, norm_true_X, raw_est, true_X])
    print(results)

    ############################################################################################
    # post process
    direct = "play_dummy/BO_output/" + NAME
    os.makedirs(direct, exist_ok=True)

    # save data to this BO experiement but also centrally
    x_name = "X.csv"
    obs_name = "obs.csv"
    x_path = os.path.join(direct, x_name)
    obs_path = os.path.join(direct, obs_name)

    train_X = unnormalize(train_X, bounds)

    pd.DataFrame(train_X, columns=ks).to_csv(x_path, index=False)
    pd.DataFrame(output_raw, columns=observables).to_csv(obs_path, index=False)

    # store centrally
    # central_dir = "play_dummy/synth_data/"
    # in_name = "cum_inputs.csv"
    # out_name = "cum_outputs.csv"

    # file_exists = os.path.exists()

    # save results
    res = "results.csv"
    obj = "best_obj.csv"

    res_path = os.path.join(direct, res)
    obj_path = os.path.join(direct, obj)

    results.to_csv(res_path)
    pd.DataFrame([best_y.item()]).to_csv(obj_path, index=False, header=False)

    # plot progress

    fig, ax = plt.subplots()
    ax.plot(range(NUM_ITER+1), bests, label = "BO")
    ax.plot(range(NUM_ITER+1), bests_rand, label = "random search")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Force integer ticks
    plt.xlabel("BO iteration")
    plt.ylabel("current best objective")
    plt.title("BO progress by iteration")
    plt.legend()

    fig_name = "/best_obj_vs_iter.png"

    plt.savefig((direct+fig_name), dpi = 300)

    plt.close()

    plt.plot(times, bests, label = "BO")
    plt.plot(times_rand, bests_rand, label = "random search")
    plt.xlabel("time (s)")
    plt.ylabel("current best objective")
    plt.title("BO progress by time")
    plt.legend()

    fig_name = "/best_obj_vs_time.png"

    plt.savefig((direct+fig_name), dpi = 300)

    plt.close()

    fig, ax = plt.subplots()
    ax.plot(evals, bests, label = "BO")
    ax.plot(evals, bests_rand, label = "random search")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Force integer ticks
    plt.xlabel("# simulator runs")
    plt.ylabel("current best objective")
    plt.title("BO progress by PROTEUS runs")
    plt.legend()

    fig_name = "/best_obj_vs_runs.png"

    plt.savefig((direct+fig_name), dpi = 300)

    plt.close()
