import torch
import numpy as np

from BO_utils import res_plot, unit_bounds, warn_with_traceback, gpu_warmup, str2bool
from BO import SeqBO, SyncBO, AsyncBO
from objectives import OBJECTIVES

import argparse
import shutil
import os
import pickle
import warnings
import time

######################################################################################################################
# example

# caffeinate python -u main.py --NUM_WORK=8 --T=30 --dim_list 5 --N_RUNS=1 --experiment_name "ackley" --objective ackley  > output/ackley.txt 2>&1
######################################################################################################################
# set up

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available(): gpu_warmup()
print(f"Using device: {device}")
print(f"There are {os.cpu_count()} cpus available.")

warnings.showwarning = warn_with_traceback

###################################################################################################################################
# exper params

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default="results")
parser.add_argument('--experiment_name', type=str, default="run00")
parser.add_argument('--NUM_WORK', type=int, nargs='+', default=1)
parser.add_argument('--T', type=int, default=2)
parser.add_argument('--N_RUNS', type=int, default=1)
parser.add_argument('--dim_list', type=int, nargs='+', default=1)
parser.add_argument('--objective', type=str, default="ackley")
parser.add_argument('--kernel', type=str, default="RBF")
parser.add_argument('--sim_or_real', type=str2bool, default=True)
args = parser.parse_args()

methods = [
        #   "qLogEI",
        #    "qUCB",
        #    "RS",
           "asyUCB",
        #    "asyLogEI",
        #    "asyTh",
        #    "LP-UCB",
        #    "LLP-UCB",
           ]

kernel = args.kernel # "RBF" | "Mat"

DIMS =  args.dim_list

NUM_WORK = args.NUM_WORK

T = args.T

N_RUNS = args.N_RUNS

experiment_name = args.experiment_name

experiment_dir = args.folder + "/" + experiment_name + "/"

objective_builder = OBJECTIVES[args.objective]

sim_run_time = args.sim_or_real
print("arg", sim_run_time)

SEED = 4
np.random.seed(SEED)

for d in DIMS:
    for q in NUM_WORK:

        t_0 = time.time()
        ###################################################################################################################################
        # set up

        NUM_RESTARTS = 10
        RAW_SAMPLES = d * 1000
        N_INIT = 3 # d * 3

        bounds = unit_bounds(d)

        f = objective_builder(d)
        y_oracle = 1.

        direct = experiment_dir + f"d={d}/" + f"q={q}/"
        plot_dir = direct + "plots/"
        os.makedirs(plot_dir, exist_ok=True)

        D = {}

        ###################################################################################################################################
        # start BO

        if "qLogEI" in methods:
            Y_best_qEI = []
            Times_qEI = []
            Evals_qEI = []
            pTimes_qEI = []
            meth = "qLogEI"

            print("\nstarting", meth, "\n")

            for i in range(N_RUNS):

                print(f"\nstarting run {i}\n")

                if i == 0:
                    path = plot_dir+meth+"/"
                else:
                    path = None

                bo = SyncBO(f, bounds, meth, kernel, q, path, sim_or_real=sim_run_time)
                print(bo.sim_or_real)
                b, t, e, pt = bo.run_opt(T, N_INIT, NUM_RESTARTS, RAW_SAMPLES)

                Y_best_qEI.append(b)
                Times_qEI.append(t)
                Evals_qEI.append(e)
                pTimes_qEI.append(pt)

            D[meth] = (Times_qEI, Evals_qEI, Y_best_qEI, pTimes_qEI)

        if "qUCB" in methods:
            Y_best_qUCB = []
            Times_qUCB = []
            Evals_qUCB = []
            pTimes_qUCB = []
            meth = "qUCB"

            print("\nstarting", meth, "\n")

            for i in range(N_RUNS):

                print(f"\nstarting run {i}\n")

                if i == 0:
                    path = plot_dir+meth+"/"
                else:
                    path = None

                bo = SyncBO(f, bounds, meth, kernel, q, path, sim_or_real=sim_run_time )
                b, t, e, pt = bo.run_opt(T, N_INIT, NUM_RESTARTS, RAW_SAMPLES)

                Y_best_qUCB.append(b)
                Times_qUCB.append(t)
                Evals_qUCB.append(e)
                pTimes_qUCB.append(pt)

            D[meth] = (Times_qUCB, Evals_qUCB, Y_best_qUCB, pTimes_qUCB)

        if "asyUCB" in methods:
            Y_best_asynUCB = []
            Times_asynUCB = []
            Evals_asynUCB = []
            Dists_asynUCB = []
            pTimes_asynUCB = []
            meth = "UCB"

            print("\nstarting", meth, "\n")

            for i in range(N_RUNS):

                print(f"\nstarting run {i}\n")

                if i == 0:
                    path = plot_dir+meth+"/"
                else:
                    path = None

                bo = AsyncBO(f, bounds, meth, kernel, q, path, sim_or_real=sim_run_time)
                b, t, e, ds, pt = bo.run_opt(T, N_INIT, NUM_RESTARTS, RAW_SAMPLES)

                Y_best_asynUCB.append(b)
                Times_asynUCB.append(t)
                Evals_asynUCB.append(e)
                Dists_asynUCB.append(ds)
                pTimes_asynUCB.append(pt)

            D[meth] = (Times_asynUCB, Evals_asynUCB, Y_best_asynUCB, pTimes_asynUCB, Dists_asynUCB)

        if "asyLogEI" in methods:
            Y_best_asynEI = []
            Times_asynEI = []
            Evals_asynEI = []
            Dists_asynEI = []
            pTimes_asynEI = []
            meth = "LogEI"

            print("\nstarting", meth, "\n")

            for i in range(N_RUNS):

                print(f"\nstarting run {i}\n")

                if i == 0:
                    path = plot_dir+meth+"/"
                else:
                    path = None

                bo = AsyncBO(f, bounds, meth, kernel, q, path, sim_or_real=sim_run_time)
                b, t, e, ds, pt = bo.run_opt(T, N_INIT, NUM_RESTARTS, RAW_SAMPLES)

                Y_best_asynEI.append(b)
                Times_asynEI.append(t)
                Evals_asynEI.append(e)
                Dists_asynEI.append(ds)
                pTimes_asynEI.append(pt)

            D[meth] = (Times_asynEI, Evals_asynEI, Y_best_asynEI, pTimes_asynEI, Dists_asynEI)

        if "asyTh" in methods:
            Y_best_asynTh = []
            Times_asynTh = []
            Evals_asynTh = []
            Dists_asynTh = []
            pTimes_asynTh = []
            meth = "Th"

            print("\nstarting", meth, "\n")

            for i in range(N_RUNS):

                print(f"\nstarting run {i}\n")

                if i == 0:
                    path = plot_dir+meth+"/"
                else:
                    path = None

                bo = AsyncBO(f, bounds, meth, kernel, q, path, sim_or_real=sim_run_time)
                b, t, e, ds, pt = bo.run_opt(T, N_INIT, NUM_RESTARTS, RAW_SAMPLES)

                Y_best_asynTh.append(b)
                Times_asynTh.append(t)
                Evals_asynTh.append(e)
                Dists_asynTh.append(ds)
                pTimes_asynTh.append(pt)

            D[meth] = (Times_asynTh, Evals_asynTh, Y_best_asynTh, pTimes_asynTh, Dists_asynTh)

        if "LP-UCB" in methods:
            Y_best_LP = []
            Times_LP = []
            Evals_LP = []
            Dists_LP = []
            pTimes_LP = []
            meth = "LP-UCB"

            print("\nstarting", meth, "\n")

            for i in range(N_RUNS):

                print(f"\nstarting run {i}\n")

                if i == 0:
                    path = plot_dir+meth+"/"
                else:
                    path = None

                bo = AsyncBO(f, bounds, meth, kernel, q, path, sim_or_real=sim_run_time)
                b, t, e, ds, pt = bo.run_opt(T, N_INIT, NUM_RESTARTS, RAW_SAMPLES)

                Y_best_LP.append(b)
                Times_LP.append(t)
                Evals_LP.append(e)
                Dists_LP.append(ds)
                pTimes_LP.append(pt)

            D[meth] = (Times_LP, Evals_LP, Y_best_LP, pTimes_LP, Dists_LP)

        if "LLP-UCB" in methods:
            Y_best_LLP = []
            Times_LLP = []
            Evals_LLP = []
            Dists_LLP = []
            pTimes_LLP = []
            meth = "LLP-UCB"

            print("\nstarting", meth, "\n")

            for i in range(N_RUNS):

                print(f"\nstarting run {i}\n")

                if i == 0:
                    path = plot_dir+meth+"/"
                else:
                    path = None

                bo = AsyncBO(f, bounds, meth, kernel, q, path, sim_or_real=sim_run_time)
                b, t, e, ds, pt = bo.run_opt(T, N_INIT, NUM_RESTARTS, RAW_SAMPLES)

                Y_best_LLP.append(b)
                Times_LLP.append(t)
                Evals_LLP.append(e)
                Dists_LLP.append(ds)
                pTimes_LLP.append(pt)

            D[meth] = (Times_LLP, Evals_LLP, Y_best_LLP, pTimes_LLP, Dists_LLP)

        if "RS" in methods:
            Y_best_RS = []
            Times_RS = []
            Evals_RS = []
            Dists_RS = []
            pTimes_RS = []
            meth = "RS"

            print("\nstarting", meth, "\n")

            for i in range(N_RUNS):

                print(f"\nstarting run {i}\n")

                if i == 0:
                    path = plot_dir+meth+"/"
                else:
                    path = None

                bo = AsyncBO(f, bounds, meth, kernel, q, path, sim_or_real=sim_run_time)
                b, t, e, ds, pt = bo.run_opt(T, N_INIT, NUM_RESTARTS, RAW_SAMPLES)

                Y_best_RS.append(b)
                Times_RS.append(t)
                Evals_RS.append(e)
                Dists_RS.append(ds)
                pTimes_RS.append(pt)

            D[meth] = (Times_RS, Evals_RS, Y_best_RS, pTimes_RS, Dists_RS)

        t_1 = time.time()
        print(f"This took: {t_1-t_0:.2f} s ")

        # Saving to a pickle file
        path = direct + "data.pkl"
        with open(path, "wb") as f:
            pickle.dump(D, f)

        # plot progress
        res_plot(D, y_oracle, plot_dir,
                 n_workers=q, objective=args.objective, dim=d,
                 iqr_or_1sig=True
                 )

print("done")



