"""Optimisation with one GP per observable instead of one GP of the objective.

Each GP models one observable in the space the objective compares it in (log10 or linear),
and the acquisition scores a candidate by the objective of joint samples of these GPs, so the
narrow valley of the objective near the best fit comes from its formula, not from a GP fit.
"""

from __future__ import annotations

import logging

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.monte_carlo import qProbabilityOfImprovement, qUpperConfidenceBound
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import ModelListGP

from proteus.inference.correlation import CorrelationWhitener
from proteus.inference.objective import (
    BAD_OBJ_VALUE,
    objective_from_observables,
    observable_space,
    upper_mask,
)
from proteus.inference.posterior import fit_gp

dtype = torch.double
log = logging.getLogger('fwl.' + __name__)

# Acquisition functions with a version that takes the objective of the observables
OBSERVABLE_ACQFS = ('LogEI', 'LogPI', 'UCB')

# Hyperparameter attempts per observable GP, each with its noise learned
OBSERVABLE_GP_ATTEMPTS = 3


class ObservableObjective:
    """The objective of observables given in the space eval_obj compares them in.

    Parameters
    ----------
    - observables (dict): Target value of each observable.
    - sigma (dict | None): Sigma of each observable; None for the relative-difference objective.
    - correlation (dict | None): Correlations between the observable uncertainties.
    - loss (tuple[str, float] | None): Robust loss; None for chi-squared.
    """

    def __init__(self, observables: dict, sigma=None, correlation=None, loss=None):
        self.names = list(observables)
        _, tru, sig = observable_space(observables, observables, sigma)
        self.tru = torch.tensor(tru, dtype=dtype)
        self.sig = torch.tensor(sig, dtype=dtype) if sigma is not None else None
        self.whitener = (
            None if correlation is None else CorrelationWhitener(self.names, correlation)
        )
        self.loss = loss
        self.upper = upper_mask(self.names, sigma)

    def __call__(self, S: torch.Tensor) -> torch.Tensor:
        """Objective of each row of S, shape (..., m), as shape (...)."""
        J = objective_from_observables(
            S, self.tru, self.sig, self.names, self.whitener, self.loss, self.upper
        )
        return J[..., 0]


class ObservableSurrogate:
    """One GP per observable, and the acquisition through the objective of their samples.

    Parameters
    ----------
    - objective (ObservableObjective): Objective of the observables.
    - kernel (str): Kernel name, as `get_kernel` takes it.
    - attempts (int): Hyperparameter attempts per GP.
    """

    def __init__(self, objective: ObservableObjective, kernel: str, attempts=None):
        self.objective = objective
        self.kernel = kernel
        self.attempts = OBSERVABLE_GP_ATTEMPTS if attempts is None else attempts

    def fit(self, X: torch.Tensor, Y: torch.Tensor, S: torch.Tensor) -> ModelListGP:
        """Fit one GP per column of S to the scored runs where that observable is finite.

        Parameters
        ----------
        - X (torch.Tensor): Evaluated inputs in the unit cube, shape (n, d).
        - Y (torch.Tensor): Objective values, shape (n, 1); BAD_OBJ_VALUE marks unscored runs.
        - S (torch.Tensor): Observables, shape (n, m), NaN where a run has no usable value.

        Returns
        ----------
        - ModelListGP: The fitted GPs, in the column order of S.
        """
        scored = Y[:, 0] > BAD_OBJ_VALUE
        gps = []
        for j, name in enumerate(self.objective.names):
            ok = scored & torch.isfinite(S[:, j])
            if int(ok.sum()) < int(scored.sum()):
                log.debug(f'GP of {name}: {int(ok.sum())} of {int(scored.sum())} scored runs')
            try:
                gps.append(fit_gp(X[ok], S[ok][:, [j]], self.kernel, None, None, self.attempts))
            except ValueError as err:
                raise ValueError(f'GP of observable {name!r}: {err}') from err
        return ModelListGP(*gps)

    def acqf(self, name: str, model: ModelListGP, best: float, X_pending=None):
        """Monte Carlo acquisition of the objective of joint samples of the observable GPs.

        Parameters
        ----------
        - name (str): One of OBSERVABLE_ACQFS.
        - model (ModelListGP): GPs from `fit`.
        - best (float): Best objective value of the scored runs.
        - X_pending (torch.Tensor | None): Points other workers are evaluating, shape (n, d).

        Returns
        ----------
        - AcquisitionFunction: The acquisition, to maximise over the unit cube.
        """
        objective = GenericMCObjective(lambda samples, X=None: self.objective(samples))
        if name == 'LogEI':
            return qLogExpectedImprovement(
                model, best_f=best, objective=objective, X_pending=X_pending
            )
        if name == 'LogPI':
            return qProbabilityOfImprovement(
                model, best_f=best, objective=objective, X_pending=X_pending
            )
        if name == 'UCB':
            return qUpperConfidenceBound(
                model, beta=2.0, objective=objective, X_pending=X_pending
            )
        raise ValueError(
            f'Acquisition {name!r} has no per-observable version; use one of {OBSERVABLE_ACQFS}'
        )


def validate_bo_per_observable(flag, acqf: str, n_init: int | None = None, d: int = 0) -> bool:
    """Check `bo_per_observable`: a bool, default false, with an acquisition that supports it
    and, when `n_init` random initial samples are drawn, more of them than the `d` parameters,
    which each observable GP needs (`fit_gp`). A grid's size is checked when it is loaded."""
    if flag is None:
        return False
    if not isinstance(flag, bool):
        raise ValueError(f'bo_per_observable must be true or false, got {flag!r}')
    if flag and acqf not in OBSERVABLE_ACQFS:
        raise ValueError(f'bo_per_observable needs acqf in {OBSERVABLE_ACQFS}, got {acqf!r}')
    if flag and n_init is not None and n_init < d + 1:
        raise ValueError(
            f'bo_per_observable needs init_samps of at least {d + 1} for {d} parameters, '
            f'got {n_init}'
        )
    return flag
