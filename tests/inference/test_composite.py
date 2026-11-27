"""
Unit tests for ``src/proteus/inference/composite.py``: the optimisation with one GP per
observable.

Checks that the objective of the observables reproduces eval_obj in every objective form,
that each observable GP leaves out unscored runs and its own unusable values, that the
acquisition through the objective ranks the target above a poor fit, and the config checks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('botorch')
pytest.importorskip('gpytorch')

import proteus.inference.composite as comp  # noqa: E402
import proteus.inference.posterior as post  # noqa: E402
from proteus.inference.correlation import CorrelationWhitener  # noqa: E402
from proteus.inference.objective import (  # noqa: E402
    BAD_OBJ_VALUE,
    Sigma,
    eval_obj,
    observable_row,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# o3 is compared in log10 because its sigma was given in dex
TARGET = {'o1': 1.4, 'o2': -0.3, 'o3': 250.0}
SIGMA = Sigma({'o1': 0.02, 'o2': 0.05, 'o3': 0.1 * 250.0 * math.log(10)}, log=['o3'])


def _observables(u):
    """Smooth observables of points u in [0, 1]^2, equal to TARGET at (0.4, 0.6)."""
    u = np.atleast_2d(u)
    return {
        'o1': 1.0 + u[:, 0],
        'o2': -0.3 + 2.0 * (u[:, 1] - 0.6) * (1 + u[:, 0]),
        'o3': 250.0 * 10 ** (1.5 * (u[:, 0] - 0.4) - (u[:, 1] - 0.6)),
    }


def _rows(u):
    """Observables of points u as `observable_row` gives them, shape (n, 3)."""
    obs = _observables(u)
    return torch.cat(
        [
            observable_row({k: v[i] for k, v in obs.items()}, TARGET, SIGMA)
            for i in range(len(u))
        ]
    )


@pytest.mark.parametrize(
    'sigma, correlation, loss',
    [
        (SIGMA, None, None),
        (SIGMA, {'o1': {'o2': 0.6}}, None),
        (SIGMA, {'o1': {'o3': -0.4}}, ('student-t', 4.0)),
        (SIGMA, None, ('huber', 1.5)),
        (None, None, None),
    ],
    ids=['chi2', 'correlated', 'correlated_student_t', 'huber', 'relative'],
)
def test_observable_objective_reproduces_eval_obj(sigma, correlation, loss):
    """J of the observables in the compared space equals eval_obj of the raw observables, for
    every objective form, at points from a near-perfect to a poor fit."""
    objective = comp.ObservableObjective(TARGET, sigma, correlation, loss)
    whitener = None if correlation is None else CorrelationWhitener(list(TARGET), correlation)
    u = np.array([[0.4, 0.6], [0.42, 0.55], [0.1, 0.9], [0.8, 0.2]])
    obs = _observables(u)
    raw = [{k: v[i] for k, v in obs.items()} for i in range(len(u))]
    rows = torch.cat([observable_row(r, TARGET, sigma) for r in raw])
    ref = torch.tensor(
        [float(eval_obj(r, TARGET, sigma, whitener, loss)) for r in raw], dtype=torch.double
    )
    got = objective(rows)
    assert got.shape == (4,)
    # The exact match scores above both poor fits in every objective form
    assert got[0] > got[2:].max() + 1
    torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-12)
    # Batched samples (sample, batch, q, m) give the same values as rows
    torch.testing.assert_close(objective(rows.reshape(2, 2, 1, 3))[..., 0].reshape(-1), got)


def test_observable_objective_compares_dex_observables_in_log():
    """o3, with a sigma in dex, enters through its log10: comparing it linearly instead moves
    J by far more than round-off at a point 0.1 dex off the target."""
    objective = comp.ObservableObjective(TARGET, SIGMA)
    off = dict(TARGET, o3=TARGET['o3'] * 10**0.1)
    J = objective(observable_row(off, TARGET, SIGMA))
    # 0.1 dex off with a 0.1 dex sigma is chi2 = 1, so J = -log10(1 + 1e-10)
    assert J.item() == pytest.approx(0.0, abs=1e-9)
    linear = (off['o3'] - TARGET['o3']) ** 2 / SIGMA['o3'] ** 2
    assert abs(-math.log10(linear) - J.item()) > 0.1


def test_surrogate_fits_each_observable_to_its_own_usable_runs(monkeypatch):
    """Unscored runs are left out of every GP; a NaN observable only out of its own GP."""
    torch.manual_seed(0)
    np.random.seed(0)
    monkeypatch.setattr(post, 'fit_gpytorch_mll', lambda *a, **kw: None)
    U = torch.rand(20, 2, dtype=torch.double)
    S = _rows(U.numpy())
    Y = comp.ObservableObjective(TARGET, SIGMA)(S).reshape(-1, 1)
    Y[0] = BAD_OBJ_VALUE  # a failed run with garbage observables
    S[0] = 1e9
    S[1, 2] = float('nan')  # an escaped atmosphere with no usable o3
    model = comp.ObservableSurrogate(comp.ObservableObjective(TARGET, SIGMA), 'MAT3/2').fit(
        U, Y, S
    )
    sizes = [m.train_inputs[0].shape[0] for m in model.models]
    assert sizes == [19, 19, 18]
    # The garbage value of the failed run reaches no GP
    assert all(m.train_targets.abs().max() < 1e3 for m in model.models)


def test_surrogate_refuses_too_few_usable_runs(monkeypatch):
    """With fewer usable runs than d + 1 for one observable the fit raises, as fit_gp does."""
    monkeypatch.setattr(post, 'fit_gpytorch_mll', lambda *a, **kw: None)
    U = torch.rand(6, 2, dtype=torch.double)
    S = _rows(U.numpy())
    S[2:, 1] = float('nan')
    Y = torch.zeros(6, 1, dtype=torch.double)
    surrogate = comp.ObservableSurrogate(comp.ObservableObjective(TARGET, SIGMA), 'RBF')
    with pytest.raises(ValueError, match="observable 'o2': 2 scored evaluations"):
        surrogate.fit(U, Y, S)
    assert surrogate.attempts == comp.OBSERVABLE_GP_ATTEMPTS


@pytest.mark.parametrize('acqf', ['LogEI', 'LogPI', 'UCB'])
def test_acquisition_through_the_objective_prefers_the_target(acqf):
    """Fitted to a grid that misses the target, each acquisition scores the target above a
    poor fit, with and without a pending point far from both."""
    torch.manual_seed(1)
    np.random.seed(1)
    g = torch.linspace(0.05, 0.95, 6, dtype=torch.double)
    U = torch.cartesian_prod(g, g)
    S = _rows(U.numpy())
    objective = comp.ObservableObjective(TARGET, SIGMA)
    Y = objective(S).reshape(-1, 1)
    surrogate = comp.ObservableSurrogate(objective, 'MAT5/2', attempts=1)
    model = surrogate.fit(U, Y, S)
    target = torch.tensor([[[0.4, 0.6]]], dtype=torch.double)
    poor = torch.tensor([[[0.9, 0.1]]], dtype=torch.double)
    pending = torch.tensor([[0.05, 0.95]], dtype=torch.double)
    for X_pending in (None, pending):
        a = surrogate.acqf(acqf, model, Y.max().item(), X_pending)
        assert a(target).item() > a(poor).item()
    # The best grid point is a poor fit, so the target improves on it by a wide margin
    assert Y.max().item() < objective(_rows(np.array([[0.4, 0.6]]))).item() - 1


def test_acquisition_without_a_per_observable_version_raises():
    """Thompson sampling has no version through the objective, so it is refused by name."""
    surrogate = comp.ObservableSurrogate(comp.ObservableObjective(TARGET, SIGMA), 'RBF')
    with pytest.raises(ValueError, match="'TS'"):
        surrogate.acqf('TS', None, 0.0)
    assert 'TS' not in comp.OBSERVABLE_ACQFS


def test_validate_bo_per_observable():
    """Absent is false; a bool passes with an acquisition that supports it; anything else raises."""
    assert comp.validate_bo_per_observable(None, 'TS') is False
    assert comp.validate_bo_per_observable(True, 'LogEI') is True
    assert comp.validate_bo_per_observable(False, 'TS') is False
    with pytest.raises(ValueError, match='true or false'):
        comp.validate_bo_per_observable('yes', 'LogEI')
    with pytest.raises(ValueError, match='acqf'):
        comp.validate_bo_per_observable(True, 'TS')


def test_validate_bo_per_observable_needs_more_initial_samples_than_parameters():
    """Each observable GP needs d + 1 scored runs: 6 initial samples pass for 5 parameters, 5 do
    not; a grid (n_init None) and the flag off are not checked here."""
    assert comp.validate_bo_per_observable(True, 'LogEI', 6, 5) is True
    with pytest.raises(ValueError, match='at least 6 for 5 parameters, got 5'):
        comp.validate_bo_per_observable(True, 'LogEI', 5, 5)
    assert comp.validate_bo_per_observable(True, 'LogEI', None, 5) is True
    assert comp.validate_bo_per_observable(False, 'LogEI', 3, 5) is False


def test_observable_objective_reproduces_eval_obj_with_an_upper_limit():
    """With o3 an upper limit, J of the observable rows equals eval_obj of the raw observables,
    and points below the limit differ from the two-sided objective."""
    sigma = Sigma(dict(SIGMA), log=SIGMA.log, upper=['o3'])
    objective = comp.ObservableObjective(TARGET, sigma)
    u = np.array([[0.4, 0.6], [0.42, 0.55], [0.1, 0.9], [0.8, 0.2]])
    obs = _observables(u)
    raw = [{k: v[i] for k, v in obs.items()} for i in range(len(u))]
    rows = torch.cat([observable_row(r, TARGET, sigma) for r in raw])
    ref = torch.tensor([float(eval_obj(r, TARGET, sigma)) for r in raw], dtype=torch.double)
    torch.testing.assert_close(objective(rows), ref, rtol=1e-12, atol=1e-12)
    assert objective(rows)[0] > objective(rows)[2] + 1
    two_sided = comp.ObservableObjective(TARGET, SIGMA)(rows)
    assert (objective(rows) - two_sided).abs().max() > 0.1
