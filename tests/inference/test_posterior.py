"""
Unit tests for ``src/proteus/inference/posterior.py``.

An inference utility (``tests/AGENTS.md``, "Physics modules"); the statistical tests
still reject the most plausible wrong likelihood, ln L = -S.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('torch')
pytest.importorskip('botorch')
pytest.importorskip('gpytorch')

import torch  # noqa: E402

import proteus.inference.posterior as post  # noqa: E402
from proteus.inference.objective import BAD_OBJ_VALUE, eval_obj, log_warp  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.fixture
def no_hyperfit(monkeypatch):
    """Skip the slow GP hyperparameter fit; returns the kwargs of each skipped call."""
    calls = []
    monkeypatch.setattr(post, 'fit_gpytorch_mll', lambda *a, **kw: calls.append(kw))
    return calls


@pytest.fixture
def small_rounds(monkeypatch):
    """Refined rounds of 50 000 draws on 100 centres, an eighth of the cost of the default."""
    monkeypatch.setattr(post, 'ADAPT_PROPOSAL', 50_000)
    monkeypatch.setattr(post, 'ADAPT_CENTRES', 100)


def _gaussian_objective(mu, sd):
    """Objective Y = log_warp(S) of an axis-aligned Gaussian chi-squared in the unit cube."""
    mu_t = torch.tensor(mu, dtype=torch.double)
    sd_t = torch.tensor(sd, dtype=torch.double)

    def objective(U):
        S = (((U - mu_t) / sd_t) ** 2).sum(dim=1, keepdim=True)
        return log_warp(S).reshape(-1)

    return objective


# ---------------------------------------------------------------------------
# importance_resample
# ---------------------------------------------------------------------------


def test_importance_resample_recovers_a_gaussian_posterior():
    """A Gaussian likelihood is recovered; ln L = -S would give sd / sqrt(2)."""
    mu, sd = [0.35, 0.6], [0.05, 0.08]
    torch.manual_seed(0)
    U, ess = post.importance_resample(_gaussian_objective(mu, sd), 2, 8000, 400_000, seed=3)
    # ESS of a 2-D Gaussian under a uniform proposal: N * 4 pi sd1 sd2, about 2e4
    assert ess == pytest.approx(400_000 * 4 * math.pi * 0.05 * 0.08, rel=0.1)
    m, s = U.mean(dim=0).numpy(), U.std(dim=0).numpy()
    # 8000 samples from about 2e4 effective proposals: sd of the sd estimate about 1.5 %
    np.testing.assert_allclose(m, mu, atol=0.005)
    np.testing.assert_allclose(s, sd, rtol=0.06)
    assert np.all(np.abs(s - np.array(sd) / math.sqrt(2.0)) > 0.15 * np.array(sd))
    assert torch.all((U >= 0.0) & (U <= 1.0))


def test_importance_resample_refines_the_proposal_for_a_narrow_posterior(small_rounds):
    """A Gaussian with sd 0.004 fills about 1e-4 of the square, so 50 000 uniform draws
    give an effective sample size near 10; the refined rounds bring it above the 2000
    samples and recover the mean and sd, which 10 effective draws cannot.
    """
    mu, sd = [0.4, 0.7], [0.004, 0.004]
    U, ess = post.importance_resample(_gaussian_objective(mu, sd), 2, 2000, 50_000, seed=5)
    assert 50_000 * 4 * math.pi * 0.004**2 < 20  # what the uniform draws alone would give
    assert ess > 2000
    assert len(torch.unique(U, dim=0)) > 1000  # distinct samples, not one point repeated
    np.testing.assert_allclose(U.mean(dim=0).numpy(), mu, atol=0.0005)
    np.testing.assert_allclose(U.std(dim=0).numpy(), sd, rtol=0.1)


def test_importance_resample_keeps_both_of_two_narrow_modes(small_rounds):
    """Two equal narrow peaks at (0.2, 0.2) and (0.8, 0.8): each keeps about half the
    samples. A single refined Gaussian on the weighted mean would sit between them.
    """
    a = _gaussian_objective([0.2, 0.2], [0.005, 0.005])
    b = _gaussian_objective([0.8, 0.8], [0.005, 0.005])

    def two_peaks(U):
        # Y of the summed likelihoods: S = -2 ln(exp(-Sa/2) + exp(-Sb/2))
        sa, sb = 10.0 ** (-a(U)), 10.0 ** (-b(U))
        s = -2 * torch.logaddexp(-sa / 2, -sb / 2) + 2 * math.log(2.0)
        return log_warp(s.clamp(min=0).reshape(-1, 1)).reshape(-1)

    U, ess = post.importance_resample(two_peaks, 2, 500, 50_000, seed=8)
    near_a = (U - 0.2).norm(dim=1) < 0.05
    near_b = (U - 0.8).norm(dim=1) < 0.05
    assert ess > 500
    assert (near_a | near_b).all()  # nothing between the peaks
    assert near_a.float().mean().item() == pytest.approx(0.5, abs=0.1)


def test_importance_resample_is_reproducible_for_a_seed():
    """The same seed gives the same samples; another seed gives different ones."""
    obj = _gaussian_objective([0.5], [0.1])
    a, _ = post.importance_resample(obj, 1, 50, 10_000, seed=11)
    b, _ = post.importance_resample(obj, 1, 50, 10_000, seed=11)
    c, _ = post.importance_resample(obj, 1, 50, 10_000, seed=12)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


# ---------------------------------------------------------------------------
# fit_surrogate
# ---------------------------------------------------------------------------


def _study_data(n=40, d=2, seed=0):
    """Evaluations of a Gaussian objective centred at (0.3, 0.7), sd 0.15."""
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, d, generator=g, dtype=torch.double)
    Y = _gaussian_objective([0.3, 0.7][:d], [0.15] * d)(X).reshape(-1, 1)
    return X, Y


def test_fit_surrogate_leaves_failed_evaluations_out(no_hyperfit):
    """Evaluations scored BAD_OBJ_VALUE do not enter the GP training set."""
    X, Y = _study_data()
    Y[:5] = BAD_OBJ_VALUE
    gp = post.fit_surrogate(X, Y, 'MAT3/2', seed=0)
    assert gp.train_inputs[0].shape[0] == 35
    # fitted as the BO steps fit theirs: best of 10 restarts
    assert no_hyperfit[0]['kwargs'] == {'pick_best_of_all_attempts': True, 'max_attempts': 10}
    # the GP mean stays near the scored values, not near -20
    assert post.gp_mean(gp, X[5:10]).min().item() > -5.0


def test_fit_surrogate_refuses_too_few_scored_evaluations(no_hyperfit):
    """Two scored points in two parameters raise before any fit."""
    X, Y = _study_data(n=6)
    Y[2:] = BAD_OBJ_VALUE
    with pytest.raises(ValueError, match='2 scored evaluations'):
        post.fit_surrogate(X, Y, 'RBF', seed=0)
    assert no_hyperfit == []


# ---------------------------------------------------------------------------
# sample_posterior
# ---------------------------------------------------------------------------


def test_sample_posterior_returns_config_units_with_log_parameters_log_uniform():
    """A flat objective returns the prior: H_budget log-uniform (median sqrt(2e7), not 10500)."""
    parameters = {
        'planet.elements.H_budget': [1e3, 2e4],
        'interior_struct.core_frac': [0.3, 0.7],
    }
    flat = lambda U: torch.ones(len(U), dtype=torch.double)  # noqa: E731
    samples, ess = post.sample_posterior(flat, parameters, 4000, seed=1, n_proposal=20_000)
    assert list(samples.columns) == list(parameters)
    assert ess == pytest.approx(20_000, rel=0.01)  # equal weights everywhere
    h = samples['planet.elements.H_budget']
    assert h.between(1e3, 2e4).all()
    assert h.median() == pytest.approx(math.sqrt(1e3 * 2e4), rel=0.05)
    assert abs(h.median() - 10_500) > 4000
    assert samples['interior_struct.core_frac'].median() == pytest.approx(0.5, abs=0.02)


# ---------------------------------------------------------------------------
# validate_posterior_samples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'n, sigma, expected',
    [
        (None, {'R_obs': 1.0e5}, post.DEFAULT_POSTERIOR_SAMPLES),
        (None, None, 0),
        (0, {'R_obs': 1.0e5}, 0),
        (500, {'R_obs': 1.0e5}, 500),
    ],
    ids=['default-with-sigma', 'default-without-sigma', 'off', 'explicit'],
)
def test_validate_posterior_samples_accepts(n, sigma, expected):
    """Default on with [sigma], off without it; explicit counts pass through."""
    out = post.validate_posterior_samples(n, sigma)
    assert out == expected
    assert type(out) is int


@pytest.mark.parametrize(
    'n, sigma, match',
    [
        (100, None, 'needs \\[sigma\\]'),
        (-1, {'R_obs': 1.0e5}, 'non-negative integer'),
        (2.5, {'R_obs': 1.0e5}, 'non-negative integer'),
        (True, {'R_obs': 1.0e5}, 'non-negative integer'),
        ('2000', {'R_obs': 1.0e5}, 'non-negative integer'),
    ],
    ids=['no-sigma', 'negative', 'float', 'bool', 'string'],
)
def test_validate_posterior_samples_rejects(n, sigma, match):
    """Non-integer or negative counts, and a posterior without sigma."""
    with pytest.raises(ValueError, match=match) as err:
        post.validate_posterior_samples(n, sigma)
    assert str(err.value).startswith('posterior_samples ')  # names the config key


# ---------------------------------------------------------------------------
# write_posterior
# ---------------------------------------------------------------------------


def test_write_posterior_writes_the_samples(tmp_path, monkeypatch, no_hyperfit):
    """posterior.csv holds the requested samples in bounds, centred on the best fit."""
    monkeypatch.setattr(post, 'N_PROPOSAL', 50_000)
    X, Y = _study_data(n=60)
    parameters = {'interior_struct.core_frac': [0.3, 0.7], 'outgas.fO2_shift_IW': [-4.0, 4.0]}
    config = {
        'posterior_samples': 300,
        'parameters': parameters,
        'kernel': 'MAT3/2',
        'seed': 5,
        'truth': {'interior_struct.core_frac': 0.42, 'outgas.fO2_shift_IW': 1.6},
    }
    out = post.write_posterior(config, {'X': X, 'Y': Y}, str(tmp_path))

    disk = pd.read_csv(tmp_path / 'posterior.csv')
    assert disk.shape == (300, 2)
    pd.testing.assert_frame_equal(disk, out, check_exact=False, rtol=1e-12)
    assert disk['interior_struct.core_frac'].between(0.3, 0.7).all()
    # objective centred at unit (0.3, 0.7): core_frac 0.42, fO2 1.6
    assert disk['interior_struct.core_frac'].median() == pytest.approx(0.42, abs=0.03)
    assert disk['outgas.fO2_shift_IW'].median() == pytest.approx(1.6, abs=0.5)


# ---------------------------------------------------------------------------
# per-observable posterior
# ---------------------------------------------------------------------------

_OBS_PARS = {'interior_struct.core_frac': [0.3, 0.7], 'outgas.fO2_shift_IW': [-4.0, 4.0]}


def _observables(u):
    """Smooth observables of the unit-cube parameters u (n, 2); C/O spans two decades."""
    return {
        'R_obs': 1e7 * (1 + 0.5 * u[:, 0]),
        'T_obs': 300 + 200 * u[:, 1],
        'C/O_atm': 10 ** (2 * (u[:, 0] - u[:, 1])),
    }


def _observable_config():
    target = {k: float(v[0]) for k, v in _observables(np.array([[0.3, 0.75]])).items()}
    return {
        'observables': target,
        'sigma': {k: 0.05 * v for k, v in target.items()},
        'parameters': _OBS_PARS,
        'kernel': 'MAT3/2',
        'seed': 0,
        'correlation': None,
        'loss': None,
        'posterior_samples': 300,
        'posterior_per_observable': True,
    }


def _observable_cases(monkeypatch):
    """A 7 x 7 grid of cases, none at the target, plus a failed case with garbage values;
    case_parameters is replaced by the grid in config units."""
    g = np.linspace(0.02, 0.98, 7)
    u = np.array([[a, b] for a in g for b in g] + [[0.3, 0.75]])
    cases = pd.DataFrame(_observables(u))
    cases.loc[len(u) - 1, ['R_obs', 'T_obs']] = [-1.0, 1e9]
    cases['excluded'] = False
    cases.loc[len(u) - 1, 'excluded'] = True
    cases['J'] = 0.0
    raw = np.column_stack([0.3 + 0.4 * u[:, 0], -4 + 8 * u[:, 1]])
    monkeypatch.setattr(post, 'case_parameters', lambda d, c, k: raw[c.index])
    return cases


def test_observables_objective_matches_eval_obj_of_the_observables(monkeypatch, no_hyperfit):
    """J from one GP per observable equals eval_obj of the true observables between the cases,
    and the failed case, whose values would wreck the GPs, is left out."""
    config = _observable_config()
    objective = post.observables_objective(config, 'unused', _observable_cases(monkeypatch))

    U = torch.tensor([[0.3, 0.75], [0.5, 0.5], [0.65, 0.35], [0.8, 0.6]], dtype=torch.double)
    true = _observables(U.numpy())
    ref = [
        float(
            eval_obj({k: v[i] for k, v in true.items()}, config['observables'], config['sigma'])
        )
        for i in range(len(U))
    ]
    Y = objective(U)
    assert Y.shape == (4,)
    # chi2 = 10**-Y within 3 % where the fit is poor, and below 2 at the target
    chi2, chi2_ref = 10 ** -Y.numpy(), 10 ** -np.array(ref)
    assert chi2[1:] == pytest.approx(chi2_ref[1:], rel=0.03)
    assert chi2[0] < 2
    assert chi2_ref[1:].min() > 50  # the other points are poor fits
    # one GP per observable, each with one hyperparameter attempt
    assert [c['kwargs']['max_attempts'] for c in no_hyperfit] == [1, 1, 1]


def test_write_posterior_per_observable_peaks_at_the_target(tmp_path, monkeypatch, no_hyperfit):
    """With posterior_per_observable, posterior.csv centres on the target, which no case hit."""
    # One round of 30 000 draws gives an ESS of about 300, plenty for the medians
    monkeypatch.setattr(post, 'N_PROPOSAL', 30_000)
    monkeypatch.setattr(post, 'ADAPT_ROUNDS', 0)
    config = _observable_config()
    cases = _observable_cases(monkeypatch)
    out = post.write_posterior(config, {}, str(tmp_path), cases)

    assert out.shape == (300, 2)
    assert (tmp_path / 'posterior.csv').is_file()
    assert out['interior_struct.core_frac'].median() == pytest.approx(0.3 + 0.4 * 0.3, abs=0.02)
    assert out['outgas.fO2_shift_IW'].median() == pytest.approx(-4 + 8 * 0.75, abs=0.4)


@pytest.mark.parametrize(
    'flag, n, expected',
    [(None, 2000, True), (None, 0, False), (True, 2000, True), (False, 2000, False)],
)
def test_validate_posterior_per_observable_accepts(flag, n, expected):
    """None is on with posterior samples and off without; true needs posterior samples."""
    assert post.validate_posterior_per_observable(flag, n) is expected
    assert isinstance(post.validate_posterior_per_observable(flag, n), bool)


@pytest.mark.parametrize(
    'flag, n, match',
    [(1, 2000, 'true or false'), ('yes', 2000, 'true or false'), (True, 0, 'off')],
)
def test_validate_posterior_per_observable_rejects(flag, n, match):
    """A non-boolean value, or true with the posterior off, is an error naming the key."""
    with pytest.raises(ValueError, match=match) as err:
        post.validate_posterior_per_observable(flag, n)
    assert str(err.value).startswith('posterior_per_observable ')
