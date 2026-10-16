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
from proteus.inference.objective import BAD_OBJ_VALUE, log_warp  # noqa: E402

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


def test_sample_posterior_returns_config_units_with_log_parameters_log_uniform(no_hyperfit):
    """A flat objective returns the prior: H_budget log-uniform (median sqrt(2e7), not 10500)."""
    torch.manual_seed(0)
    X = torch.rand(30, 2, dtype=torch.double)
    D = {'X': X, 'Y': torch.full((30, 1), 1.0, dtype=torch.double)}
    parameters = {
        'planet.elements.H_budget': [1e3, 2e4],
        'interior_struct.core_frac': [0.3, 0.7],
    }
    samples, ess = post.sample_posterior(D, parameters, 'RBF', 4000, seed=1, n_proposal=20_000)
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


def test_write_posterior_writes_samples_and_corner_plot(tmp_path, monkeypatch, no_hyperfit):
    """posterior.csv holds the requested samples in bounds, centred on the best fit."""
    monkeypatch.setattr(post, 'N_PROPOSAL', 50_000)
    (tmp_path / 'plots').mkdir()
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
    assert (tmp_path / 'plots' / f'result_posterior.{post.fmt}').stat().st_size > 1000
