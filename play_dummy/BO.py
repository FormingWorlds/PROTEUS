import os
import time


import torch


import matplotlib.pyplot as plt

from scipy.stats.qmc import Halton

from botorch.models.transforms import Normalize, Standardize
from botorch.models import SingleTaskGP

from botorch.acquisition import (UpperConfidenceBound,
                                 ExpectedImprovement,
                                 LogExpectedImprovement,
                                 qUpperConfidenceBound,
                                 qExpectedImprovement,
                                 qLogExpectedImprovement)

from acquisition import (PosteriorSample,
                         PenalizedUpperConfidenceBound)

from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, MaternKernel

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SeqBO():
    def __init__(self, objective, bounds, acqf, kernel,
                 plot_dir = None, sim_or_real = True):

        """
        Class for running, plotting and saving a basic BO experiment.
        """

        self.objective = objective
        self.bounds = bounds
        self.acqf = acqf
        self.kernel = kernel
        self.d = bounds.shape[1]
        self.plot_dir = plot_dir
        self.sim_or_real = sim_or_real

    def _kickstart(self, n_init: int, store_or_return = True):
        """
        Kickstart (and reset) the optimization

        Arguments:
            n_init (int): the number of initial points
            f_fast (Callable): optional fast version of objective

        Returns:
            None:  if store_or_return = True (default), data is saved to attributes
            X,Y (tuple): initial data
        """

        sampler = Halton(d=self.d, scramble=True)  # `scramble=True` adds randomness
        data = sampler.random(n=n_init)
        scale = (self.bounds[1] - self.bounds[0]).unsqueeze(0)
        offset = self.bounds[0].unsqueeze(0)
        X = torch.tensor(data, dtype=dtype, device=device) * scale + offset # n_init x d
        Y, C = self._get_obs_and_cost(X)

        if store_or_return:
            self.X = X
            self.Y = Y
            self.y_best = Y.max().item()

        else:

            return X, Y, C

    def _fit_gp(self):
        """
        Fit GP model to data
        """

        if self.acqf == "RS":
            return None

        if self.kernel == "RBF":
            cov = RBFKernel(ard_num_dims=self.d)

        elif self.kernel == "Mat":
            cov = MaternKernel(ard_num_dims = self.d)

        else:
            raise RuntimeError("unknown kernel")

        self.gp = SingleTaskGP(train_X=self.X,
                               train_Y=self.Y,
                               covar_module=cov,
                               input_transform=Normalize(d=self.d),
                               outcome_transform=Standardize(m=1),
                            ).to(device=device)

        self.lik = self.gp.likelihood.to(device=device)
        self.mll = ExactMarginalLogLikelihood(self.lik, self.gp).to(device=device)
        fit_gpytorch_mll(self.mll)

    def _opt_acf(self, acqf, restarts, samples, q = 1):

        if acqf == "RS":
            return unnormalize(torch.rand(q, self.d, dtype=dtype, device=device),
                               self.bounds)

        else:
            x, _ = optimize_acqf(acq_function=acqf,
                                        bounds=self.bounds,
                                        q=q,
                                        num_restarts=restarts,
                                        raw_samples=samples,
                                        options={"batch_limit": 50, "maxiter": 200},
                                        )

            return x

    def _plt_iter(self, acqf, next_x, busys = None):
        """
        Plot and save iteration (for 1D inputs)

        Arguments:
            acqf: the acquisition function to plot
            next_x: the next point to be queried

        """

        if self.acqf == "RS":
            return None # nothing to plot for RS

        fig, ax = plt.subplots(2,1)

        xs = (torch.linspace(0,1, 400)
              * (self.bounds[1] - self.bounds[0])
              + self.bounds[0]
              ).reshape(-1,1,1) # batch_shape x q=1 x d for acqf

        post = self.gp.posterior(xs)
        mu = post.mean.detach().flatten()
        sig = post.variance.sqrt().detach().flatten()

        ac = acqf(xs).detach().flatten()

        ax[0].plot(xs.detach().flatten(), mu, color = "black", label = "post mean")
        ax[0].fill_between(xs.detach().flatten(), mu-2*sig, mu+2*sig, alpha = 0.25 )
        ax[0].scatter(self.X.flatten(), self.Y.flatten(), label = "data", color = "blue")
        fs = self.objective(xs).flatten()
        ax[0].plot(xs.detach().flatten(), fs, color = "grey", label = "f", linewidth = 0.5)

        ax[1].plot(xs.detach().flatten(), ac, color = "cornflowerblue", label = self.acqf)
        ax[1].scatter(next_x, acqf(next_x).detach().flatten(),
                      label = "next query", color = "green", marker = "*")
        if busys is not None:
            y_min, y_max = ax[1].get_ylim()
            ax[1].scatter(busys, torch.full_like(busys, y_min),
                          label = "busy", color = "orange", marker = "*")

        ax[0].legend()
        ax[1].legend()

        path = self.plot_dir + f"BO_{len(self.X)}_.png"
        os.makedirs(self.plot_dir, exist_ok=True)
        fig.savefig(path, dpi = 300)
        plt.close(fig)

    def _get_obs_and_cost(self, X):

        """
        Get function values and run time(s)

        """

        Y = []
        C = []
        for x in X:

            t_0 = time.time()
            y = self.objective(x)
            t_1 = time.time()

            Y.append(y)
            C.append(t_1-t_0)

        Y = torch.tensor(Y, dtype=dtype, device=device).unsqueeze(-1)

        if self.sim_or_real:
            n = len(X)
            C = (torch.pi/2)**(0.5) * torch.abs(torch.randn(n, dtype=dtype, device=device))
        else:
            C = torch.tensor(C, dtype=dtype, device=device)

        print("call(s) took", C)
        return Y, C

    def run_opt(self, T, n_init, num_restarts: int = 10, raw_samples: int = 100):

        """
        Run one trial

        Arguments:
            T (float): time budget
            n_init (int): number of initial data points
            num_restarts (int): number of promising acqf candidates to optimize
            raw_samples (int): number of candidates to generate from acqf

        Returns:
            bests (list): best fct value after each evaluation
            times (list): times (s) corresponding to fct evaluations
            evals (list): indexing the fct evaluations
        """

        self._kickstart(n_init)

        bests = [self.y_best]
        times = [0]
        eval_count = 0

        t = 0
        while t < T:

            self._fit_gp()

            if self.acqf == "EI":
                ac = ExpectedImprovement(self.gp, self.y_best)

            elif self.acqf == "UCB":
                ac = UpperConfidenceBound(self.gp, beta = 2) # maybe dont hard code the beta

            elif self.acqf == "Th":
                ac = PosteriorSample(self.gp)

            elif self.acqf == "RS":
                ac = "RS"

            else:
                raise RuntimeError("unknown acqf")

            x = self._opt_acf(ac, num_restarts, raw_samples)
            y, c = self._get_obs_and_cost(x)

            if self.plot_dir is not None and self.d == 1:
                self._plt_iter(ac, x)

            self.X = torch.cat([self.X, x])
            self.Y = torch.cat([self.Y, y])

            if y > self.y_best:
                self.y_best = y.item()
                x_best = x

            t += c
            eval_count += 1
            bests.append(self.y_best)
            times.append(t.item())

        evals = [i for i in range(eval_count+1)]

        return bests, times, evals

class SyncBO(SeqBO):
    def __init__(self, objective, bounds, acqf, kernel, n_workers,
                 plot_dir = None, sim_or_real = True):

        """
        Class for running, plotting and saving a synchronous batch BO experiment.
        """

        self.objective = objective
        self.bounds = bounds
        self.acqf = acqf
        self.kernel = kernel
        self.d = bounds.shape[1]
        self.n_workers = n_workers
        self.plot_dir = plot_dir
        self.sim_or_real = sim_or_real

    def _plt_iter(self, next_x):
        """
        Plot and save iteration (for 1D inputs)

        Arguments:
            acqf: the acquisition function to plot
            next_x: the next point to be queried

        """

        fig, ax = plt.subplots(1,1)

        xs = (torch.linspace(0,1, 400, dtype=dtype, device=device)
              * (self.bounds[1] - self.bounds[0])
              + self.bounds[0]
              ).reshape(-1,1,1) # batch_shape x q=1 x d for acqf

        post = self.gp.posterior(xs)
        mu = post.mean.detach().flatten().cpu()
        sig = post.variance.sqrt().detach().flatten().cpu()


        ax.plot(xs.detach().flatten().cpu(), mu, color = "black", label = "post mean")
        ax.fill_between(xs.detach().flatten().cpu(), mu-2*sig, mu+2*sig, alpha = 0.25 )
        ax.scatter(self.X.flatten().cpu(), self.Y.flatten().cpu(), label = "data", color = "blue")
        fs = self.objective(xs).flatten().cpu()
        ax.plot(xs.detach().flatten().cpu(), fs, color = "grey", label = "f", linewidth = 0.5)

        y_min, y_max = ax.get_ylim()
        ax.scatter(next_x.cpu(), torch.full_like(next_x.cpu(), y_min,), label = "next queries", color = "green", marker = "*")

        ax.legend()

        path = self.plot_dir + f"BO_{len(self.X)}_.png"
        os.makedirs(self.plot_dir, exist_ok=True)
        fig.savefig(path, dpi = 300)
        plt.close(fig)

    def _get_acqf(self):

        if self.acqf == "qEI":
                ac = qExpectedImprovement(self.gp, self.y_best)

        elif self.acqf == "qLogEI":
            ac = qLogExpectedImprovement(self.gp, self.y_best)

        elif self.acqf == "qUCB":
            ac = qUpperConfidenceBound(self.gp, beta = 2) # maybe dont hard code the beta

        elif self.acqf == "RS":
            ac = "RS"

        else:
            raise RuntimeError("unknown acqf")

        return ac


    def run_opt(self, T, n_init, num_restarts: int = 10, raw_samples: int = 100):

        """
        Run one trial

        Arguments:
            T (float): time budget
            n_init (int): number of initial data points
            num_restarts (int): number of promising acqf candidates to optimize
            raw_samples (int): number of candidates to generate from acqf

        Returns:
            bests (list): best fct value after each evaluation
            times (list): times (s) corresponding to fct evaluations
            evals (list): indexing the fct evaluations
        """

        self._kickstart(n_init)

        bests = [self.y_best]
        times = [0]
        eval_count = 0
        proc_times = []

        t = 0
        while t < T:

            t_0 = time.time()

            self._fit_gp()
            ac = self._get_acqf()
            x = self._opt_acf(ac, num_restarts, raw_samples, q = self.n_workers)

            t_1 = time.time()

            y, cs = self._get_obs_and_cost(x)
            c = cs.max() # wait for the slowest evaluation

            if self.plot_dir is not None and self.d == 1:
                self._plt_iter(x)

            self.X = torch.cat([self.X, x])
            self.Y = torch.cat([self.Y, y])

            if any(y > self.y_best):
                self.y_best = y.max().item()
                x_best = x

            dt = t_1 - t_0
            t += c + dt
            eval_count += 1
            bests.append(self.y_best)
            times.append(t.item())
            proc_times.append(dt)

        evals = [self.n_workers * i for i in range(eval_count+1)]

        return bests, times, evals, proc_times

class AsyncBO(SeqBO):

    def __init__(self, objective, bounds, acqf, kernel, n_workers,
                 plot_dir = None, sim_or_real = True):

        """
        Class for running, plotting and saving an asynchronous batch BO experiment.
        """

        self.objective = objective
        self.bounds = bounds
        self.acqf = acqf
        self.kernel = kernel
        self.d = bounds.shape[1]
        self.n_workers = n_workers
        self.plot_dir = plot_dir
        self.sim_or_real = sim_or_real

    def _init_workers(self):
        """
        Ininitialize workers

        Returns:
            worker_times (Tensor): times of workers (n_workers)
            worker_data (list): evaluations workers are "currently" busy with, list of tuples
        """

        X, Y, worker_times = self._kickstart(self.n_workers, store_or_return=False)

        worker_data = []
        for i in range(self.n_workers):
            datum = (X[i][None, :], Y[i][None, :])
            worker_data.append(datum)

        return worker_times, worker_data

    def _get_busys(self, worker_data: tuple, w):

        """
        Get the busy input loactions

        Arguments
            worker_data (list): evaluations workers are "currently" busy with, list of tuples
            w (int): index of worker next to accquire

        Returns
            B (Tensor): busy locations (num_workers-1, d)
        """

        B = worker_data[:w] + worker_data[w+1:]
        B = torch.cat([dat[0] for dat in B])

        return B

    def _get_acqf(self):

        if self.acqf == "EI":
            ac = ExpectedImprovement(self.gp, self.y_best)

        elif self.acqf == "LogEI":
            ac = LogExpectedImprovement(self.gp, self.y_best)

        elif self.acqf == "UCB":
            ac = UpperConfidenceBound(self.gp, beta = 2) # maybe dont hard code the beta

        elif self. acqf == "Th":
            ac = PosteriorSample(self.gp)

        elif self.acqf == "LP-UCB":
            ac = PenalizedUpperConfidenceBound(self.gp,
                                                beta = 2,
                                                bounds = self.bounds,
                                                busy=self.busys,
                                                y_max=self.y_best,
                                                local=False
                                                )

        elif self.acqf == "LLP-UCB":
            ac = PenalizedUpperConfidenceBound(self.gp,
                                                beta = 2,
                                                bounds = self.bounds,
                                                busy=self.busys,
                                                y_max=self.y_best,
                                                local=True
                                                )

        elif self.acqf == "RS":
            ac = "RS"

        else:
            raise RuntimeError("unknown acqf")

        return ac

    # def _update_waste(self, x):

    #     """
    #     Count the number of redundant (wasted) queries in the run.
    #     Note: the threshold 1e-5 is quite conservative under the null hyp "busys==x"
    #     If we don't see waste, even in this conservative setting,
    #     we can reject the null confidently
    #     """
    #     return torch.max(torch.abs(self.busys-x)) < 1e-5

    def run_opt(self, T, n_init, num_restarts: int = 10, raw_samples: int = 100):

        """
        Run one trial

        Arguments:
            T (float): time budget
            n_init (int): number of initial data points
            num_restarts (int): number of promising acqf candidates to optimize
            raw_samples (int): number of candidates to generate from acqf

        Returns:
            bests (list): best fct value after each evaluation
            times (list): times (s) corresponding to fct evaluations
            evals (list): indexing the fct evaluations
        """

        self._kickstart(n_init)

        worker_times, worker_data = self._init_workers()

        bests = [self.y_best]
        times = [0]
        eval_count = 0
        dists = []
        proc_times = []

        t = 0
        while t < T:

            # find lagging worker
            w = torch.argmin(worker_times)
            t_w = torch.min(worker_times)
            x, y = worker_data[w] # the datum to be added

            self.busys = self._get_busys(worker_data, w)

            # print("done x: ", x)

            self.X = torch.cat([self.X, x])
            self.Y = torch.cat([self.Y, y])

            if y > self.y_best:
                self.y_best = y.max().item()
                x_best = x

            t_0 = time.time()

            try:
                self._fit_gp()
                ac = self._get_acqf()
                x = self._opt_acf(ac, num_restarts, raw_samples)

            except Exception as e:

                print(f"caught:\n {e} ")

                x = unnormalize(torch.rand(1, self.d, dtype=dtype, device=device),
                               self.bounds)

            t_1 = time.time()

            dist = torch.min(torch.cdist(self.busys, x))
            dists.append(dist.item())

            y, c = self._get_obs_and_cost(x)
            dt = t_1 - t_0
            worker_times[w] += c.item() + dt
            proc_times.append(dt)

            if self.plot_dir is not None and self.d == 1:

                self._plt_iter(ac, x, self.busys)

            worker_data[w] = (x, y)

            t = torch.min(worker_times)
            eval_count += 1
            bests.append(self.y_best)
            times.append(t_w.item())

        evals = [i for i in range(eval_count+1)]

        return bests, times, evals, dists, proc_times
