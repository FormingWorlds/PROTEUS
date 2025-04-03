import os
import time
import traceback

import numpy as np
import torch
from torch import Tensor

import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.stats.qmc import Halton

from botorch.models.transforms import Normalize, Standardize
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.acquisition import (UpperConfidenceBound,
                                 ExpectedImprovement,
                                 LogExpectedImprovement,
                                 qUpperConfidenceBound,
                                 qExpectedImprovement,
                                 qLogExpectedImprovement)
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from botorch.sampling.pathwise.posterior_samplers import draw_matheron_paths

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, MaternKernel

from botorch.test_functions.synthetic import (Ackley,
                                              EggHolder,
                                              Cosine8,
                                              Hartmann,
                                              Rosenbrock,
                                              Griewank)

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def res_plot(D, y_oracle, plot_dir, n_workers, objective, dim,
             iqr_or_1sig = True, T_max = None, N_max = None):

    # D (dict): method (str): trajectories tuple(list(lists)), (times, evals, y_bests, costs)

    if T_max is None:
        T = []
        for m, r in D.items():
            T.append(max([n for tr in r[0] for n in tr]))

        T = max(T)

    else:
        T = T_max

    if N_max is None:
        N = []
        for m, r in D.items():
            N.append(max([n for tr in r[1] for n in tr]))
        N = max(N)

    else:
        N = N_max

    ts = np.linspace(0,T, N)
    ns = np.linspace(0, N, N)

    num_meth = len(D)
    rows = (num_meth + 1 )// 2

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots(rows,2)
    ax3 = ax3.flatten()
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()
    fig6, ax6 = plt.subplots(rows,2)
    ax6 = ax6.flatten()
    fig7, ax7 = plt.subplots(rows,2)
    ax7 = ax7.flatten()
    fig8, ax8 = plt.subplots(rows,2)
    ax8 = ax8.flatten()

    for i , (method, results) in enumerate(D.items()): # for each method

        """
        Plot best objective and log regret over time
        Make histogram of best fct value
        """
        Ys = []
        for t, y in zip(results[0], results[2]):
            f = interp1d(t, y, kind='linear', bounds_error=False, fill_value=np.nan)
            Ys.append(f(ts)) # values at locations ts outside of t will be np.nan

        Ys = np.array(Ys)
        n_runs = len(Ys)
        Ys = torch.tensor(Ys).reshape(n_runs, -1) # N_RUNS, len(ts)

        # along columns, as soon as one elemt in col is NaN, quantile for that col is NaN
        # i.e. everything is truncated to length of shortest out of N_RUNS trajectories

        if iqr_or_1sig:

            Y_m = Ys.quantile(0.5, dim = 0)
            Y_u = Ys.quantile(0.75, dim = 0)
            Y_l= Ys.quantile(0.25, dim = 0)

            Rs = torch.log(y_oracle - Ys + 1e-5) # add floor on log
            R_m = Rs.quantile(0.5, dim = 0)
            R_u = Rs.quantile(0.75, dim = 0)
            R_l = Rs.quantile(0.25, dim = 0)

        else:

            Y_m = Ys.mean(dim = 0)
            Y_u = Y_m + 1 * Ys.std(dim=0)
            Y_l= Y_m - 1 * Ys.std(dim=0)

            Rs = torch.log(y_oracle - Ys + 1e-5) # add floor on log
            R_m = Rs.mean(dim = 0)
            R_u = R_m + 1 * Rs.std(dim=0)
            R_l = R_m - 1 * Rs.std(dim=0)


        ax1.plot(ts, Y_m, label = method)
        ax1.fill_between(ts, Y_l, Y_u, alpha=0.3)

        ax2.plot(ts, R_m, label = method)
        ax2.fill_between(ts, R_l, R_u, alpha=0.3)

        ax3[i].hist([ results[2][j][-1] for j in range(len(results[2])) ])
        ax3[i].set_title(method)
        ax3[i].set_xlabel("best function value")


    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("current best objective")
    ax1.set_title(f"{objective}, d={dim}, q={n_workers}, runs={n_runs}")
    ax1.legend()
    fig1_name = "/best_obj_vs_t.png"
    fig1.savefig((plot_dir+fig1_name), dpi = 300)
    plt.close(fig1)

    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("log regret")
    ax2.set_title(f"{objective}, d={dim}, q={n_workers}, runs={n_runs}")
    ax2.legend()
    fig2_name = "/log_regret_vs_t.png"
    fig2.savefig((plot_dir+fig2_name), dpi = 300)
    plt.close(fig2)

    fig3_name = "/best_hist_BO.png"
    fig3.tight_layout()
    fig3.savefig((plot_dir+fig3_name), dpi = 300)
    plt.close(fig3)


    for i , (method, results) in enumerate(D.items()):

        """
        Plot best objective and log regret over number of fct evals
        Make histogram of number of fct evals
        """

        Ys = []
        for n, y in zip(results[1], results[2]):
            f = interp1d(n, y, kind='linear', bounds_error=False, fill_value=np.nan)
            Ys.append(f(ns))

        Ys = np.array(Ys)
        Ys = torch.tensor(Ys).reshape(len(Ys), -1)

        if iqr_or_1sig:

            Y_m = Ys.quantile(0.5, dim = 0)
            Y_u = Ys.quantile(0.75, dim = 0)
            Y_l= Ys.quantile(0.25, dim = 0)

            Rs = torch.log(y_oracle - Ys + 1e-5) # add floor on log
            R_m = Rs.quantile(0.5, dim = 0)
            R_u = Rs.quantile(0.75, dim = 0)
            R_l = Rs.quantile(0.25, dim = 0)

        else:

            Y_m = Ys.mean(dim = 0)
            Y_u = Y_m + 1 * Ys.std(dim=0)
            Y_l= Y_m - 1 * Ys.std(dim=0)

            Rs = torch.log(y_oracle - Ys + 1e-5) # add floor on log
            R_m = Rs.mean(dim = 0)
            R_u = R_m + 1 * Rs.std(dim=0)
            R_l = R_m - 1 * Rs.std(dim=0)

        ax4.plot(ns, Y_m, label = method)
        ax4.fill_between(ns, Y_l, Y_u, alpha=0.3)

        ax5.plot(ns, R_m, label = method)
        ax5.fill_between(ns, R_l, R_u, alpha=0.3)

        ax6[i].hist([ results[1][j][-1] for j in range(len(results[1])) ])
        ax6[i].set_title(method)
        ax6[i].set_xlabel("number of function evaluations")

        ax8[i].hist(np.hstack(results[3]))
        ax8[i].set_title(method)
        ax8[i].set_xlabel("fit and ac time (s)")

        # only async methods have dist
        if len(results) == 5:

            """
            Make histogram of distance of query to closest busy location
            """

            ax7[i].hist(np.hstack(results[4]))
            ax7[i].set_title(method)
            ax7[i].set_xlabel("dist query to busy locations")

    ax4.set_xlabel("function evaluations")
    ax4.set_ylabel("current best objective")
    ax4.set_title(f"{objective}, d={dim}, q={n_workers}, runs={n_runs}")
    ax4.legend()
    fig4_name = "/best_obj_vs_n.png"
    fig4.savefig((plot_dir+fig4_name), dpi = 300)
    plt.close(fig4)

    ax5.set_xlabel("function evaluations")
    ax5.set_ylabel("log regret")
    ax5.set_title(f"{objective}, d={dim}, q={n_workers}, runs={n_runs}")
    ax5.legend()
    fig5_name = "/log_regret_vs_n.png"
    fig5.savefig((plot_dir+fig5_name), dpi = 300)
    plt.close(fig5)

    fig6_name = "/num_eval_hist.png"
    fig6.tight_layout()
    fig6.savefig((plot_dir+fig6_name), dpi = 300)
    plt.close(fig6)

    fig7_name = "/dist_hist.png"
    fig7.tight_layout()
    fig7.savefig((plot_dir+fig7_name), dpi = 300)
    plt.close(fig7)

    fig8_name = "/fit_ac_hist.png"
    fig8.tight_layout()
    fig8.savefig((plot_dir+fig8_name), dpi = 300)
    plt.close(fig8)

def unit_bounds(d):

    bounds = torch.tensor([[i for _ in range(d)] for i in range(2)],
                          dtype=dtype,
                          device=device)

    return bounds


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    print("\n")
    print(f"{category.__name__}: \n {message} \n")
    traceback.print_stack()
    print("\n")

def gpu_warmup():
    device = torch.device("cuda")
    print(f"Warming up GPU: {device}")

    for _ in range(100):
        x = torch.randn(5000, 5000, device=device)
        y = torch.matmul(x, x)
        torch.cuda.synchronize()

    print("GPU warmed up and ready.")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
