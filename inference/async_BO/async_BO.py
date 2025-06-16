from multiprocessing import Process, Manager
import time
import torch
from scipy.stats.qmc import Halton

import pickle
import os

from functools import partial

from BO import BO_step
from plots import plot_times, plot_res

dtype = torch.double

def checkpoint(D, l, T, dir):

    os.makedirs(dir, exist_ok=True)
    path = dir + "data.pkl"
    with open(path, "wb") as f:
        pickle.dump(dict(D), f)

    path = dir + "logs.pkl"
    with open(path, "wb") as f:
        pickle.dump(list(l), f)

    path = dir + "Ts.pkl"
    with open(path, "wb") as f:
        pickle.dump(list(T), f)

def init_locs(n_workers, D):
    d = D["X"].shape[-1]
    sampler = Halton(d=d, scramble=True)
    data = sampler.random(n=n_workers)
    X = torch.tensor(data, dtype=dtype).reshape(n_workers,d)

    return X


def worker(process_fun, f_w_i, D_shared, B, T, T_0, x_init, lock, max_len, worker_id, log_list, direct):
    task_id = 0
    torch.manual_seed(os.getpid())

    while True:

        with lock:
            X = D_shared["X"]
        if len(X) >= max_len:
            break

        x_in = x_init if task_id == 0 else None

        f = f_w_i(worker = worker_id,
                  iter = task_id)

        start_time = time.perf_counter()
        x, y, t_bo, t_eval, t_lock, t_fit, t_ac, dist = process_fun(f = f,
                                                                    D = D_shared,
                                                                    B = B,
                                                                    x_in = x_in,
                                                                    lock = lock,
                                                                    worker_id = worker_id,
                                                                    )
        end_time = time.perf_counter()

        # Note: if workers finish at very similar times lock contention may occur,
        # causing new query under same data.
        # This means that one worker "waits" (fractions of a second)
        # for the other worker to add their evaluation to the shared data set.
        # (Could flip BO step to only allow new query after evaluation is provided?)

        with lock:
            D_shared["X"] = torch.concatenate((D_shared["X"], x))
            D_shared["Y"] = torch.concatenate((D_shared["Y"], y))

            T.append(time.perf_counter())
            log_list.append({
                                "worker": worker_id,
                                "task_id": task_id,
                                "start_time": start_time,
                                "end_time": end_time,
                                "duration": end_time-start_time,
                                "BO_time": t_bo,
                                "t_eval": t_eval,
                                "t_lock": t_lock,
                                "t_fit": t_fit,
                                "t_ac": t_ac,
                                "dist": dist,
                                "x_value": x.tolist()[0],
                                "y_value": float(y),
                            })
            Ts = [i - T_0 for i in list(T)]
            checkpoint(D_shared, log_list, Ts, direct)


        task_id += 1

def parallel_process(objective_builder, *,
                     kernel, n_restarts, n_samples,
                     n_workers, max_len,
                     D_init_path, directory,
                     ref_config,
                     observables,
                     parameters
                     ):

    f_w_i = partial(objective_builder,
                    observables = observables,
                    parameters = parameters,
                    ref_config = ref_config
                    )

    process_fun = partial(BO_step,
                          k = kernel,
                          n_restarts = n_restarts,
                          n_samples = n_samples,
                          )


    t_0 = time.perf_counter()
    mgr = Manager()

    with open(D_init_path, "rb") as f:
        D_init = pickle.load(f)
    D_shared = mgr.dict()
    D_shared["X"] = D_init["X"]
    D_shared["Y"] = D_init["Y"]

    n_init = len(D_shared["X"])

    lock = mgr.Lock()
    log_list = mgr.list()

    X_init = init_locs(n_workers, D_shared)
    B = mgr.dict()

    T = mgr.list()
    T_0 = time.perf_counter()

    procs = []
    for i in range(n_workers):

        x_init = X_init[i][None,:]

        with lock:
            B[i] = x_init

        p = Process(target=worker, args=(process_fun,
                                         f_w_i,
                                         D_shared,
                                         B,
                                         T,
                                         T_0,
                                         x_init,
                                         lock,
                                         max_len,
                                         i,
                                         log_list,
                                         directory))
        p.start()
        procs.append(p)
    t_1 = time.perf_counter()

    print(f"manager time: {(t_1-t_0):.2f}")

    for p in procs:
        t_0 = time.perf_counter()
        p.join()
        t_1 = time.perf_counter()

        # print(f"time for {p}: {(t_1-t_0):.2f}")

    D_final = dict(D_shared)
    logs = list(log_list)
    T = [i - T_0 for i in list(T)]

    plot_times(logs, directory)
    plot_res(D_final, T, n_init, directory)


    return D_final, logs, T

