"""
Unit tests for ``src/proteus/inference/noise.py``.

Checks that the noise has the configured standard deviation in the space the objective compares
each observable in (linear, or log10 for element ratios), the configured correlations, and a
dex error that the rescaled sigma preserves.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

pytest.importorskip('torch')

import proteus.inference.noise as noise_mod  # noqa: E402
from proteus.inference.noise import apply_noise, noisy_observables  # noqa: E402
from proteus.inference.objective import validate_sigma  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

OBS = {'R_obs': 6.0e6, 'C/O_atm': 0.33, 'S/O_atm': 0.05, 'O/H_atm': 0.002}
DEX = 0.3  # wide enough that linear noise would give some negative ratios
SIGMA = {'R_obs': 3.0e5} | {k: DEX * OBS[k] * math.log(10) for k in OBS if '/' in k}
CORR = {'C/O_atm': {'S/O_atm': 0.5, 'O/H_atm': -0.5}, 'S/O_atm': {'O/H_atm': -0.5}}


def _draws(n):
    """n noisy draws, one seed each, as an array with the columns of OBS."""
    return np.array(
        [list(noisy_observables(OBS, SIGMA, CORR, s)[0].values()) for s in range(n)]
    )


def test_noise_has_sigma_and_correlation_in_the_compared_space(monkeypatch):
    """R gets linear noise of sigma; ratios get 0.3 dex of log-normal noise, so they stay
    positive and their linear mean sits exp((0.3 ln 10)^2 / 2) = 1.27 times the value, not at it
    as linear noise would give, and their log10 values carry the configured correlations."""
    # One whitener for all draws: its Cholesky factor is the same, and slow to rebuild
    w = noise_mod.CorrelationWhitener(list(OBS), CORR)
    monkeypatch.setattr(noise_mod, 'CorrelationWhitener', lambda keys, corr: w)
    D = _draws(4000)
    L = np.log10(D[:, 1:])
    assert D[:, 0].mean() == pytest.approx(OBS['R_obs'], rel=4 * 0.05 / math.sqrt(len(D)))
    assert D[:, 0].std() == pytest.approx(SIGMA['R_obs'], rel=0.07)
    assert (D[:, 1:] > 0).all()
    assert L.std(axis=0) == pytest.approx([DEX] * 3, rel=0.07)
    lognormal = math.exp(0.5 * (DEX * math.log(10)) ** 2)
    assert D[:, 1].mean() / OBS['C/O_atm'] == pytest.approx(lognormal, abs=0.1)
    assert abs(D[:, 1].mean() / OBS['C/O_atm'] - 1.0) > 0.15  # linear noise
    c = np.corrcoef(np.column_stack([D[:, 0], L]).T)
    assert [c[1, 2], c[1, 3], c[2, 3]] == pytest.approx([0.5, -0.5, -0.5], abs=0.08)
    assert abs(c[0, 1]) < 0.1


def test_rescaled_sigma_keeps_the_dex_error():
    """After the draw, sigma / (x ln 10) of each ratio is still 0.3 dex at its noisy value x,
    and the linear sigma of R is unchanged."""
    noisy, sigma = noisy_observables(OBS, SIGMA, CORR, 7)
    for k in ('C/O_atm', 'S/O_atm', 'O/H_atm'):
        assert sigma[k] / (noisy[k] * math.log(10)) == pytest.approx(DEX, rel=1e-12)
        assert noisy[k] != pytest.approx(OBS[k], rel=1e-3)
    assert sigma['R_obs'] == SIGMA['R_obs']


def test_dex_sigma_gives_log_normal_noise_on_a_linear_observable():
    """g_obs with '0.25 dex' gets noise in log10: every draw stays positive, where linear
    noise of the same size (58 % of g) would turn about 4 % of them negative, and the
    rescaled sigma keeps both the 0.25 dex and the log10 comparison."""
    obs = {'g_obs': 9.8}
    sigma = validate_sigma(obs, {'g_obs': '0.25 dex'})
    draws = [noisy_observables(obs, sigma, None, s) for s in range(2000)]
    g = np.array([d[0]['g_obs'] for d in draws])
    assert (g > 0).all()
    assert np.log10(g / 9.8).std() == pytest.approx(0.25, rel=0.07)
    assert np.median(g) == pytest.approx(9.8, rel=0.05)  # log-normal: median at the truth
    noisy, rescaled = draws[0]
    assert rescaled.log == {'g_obs'}
    assert rescaled['g_obs'] / (noisy['g_obs'] * math.log(10)) == pytest.approx(0.25, rel=1e-12)


def _config(**kw):
    """Inference config with the fields apply_noise reads."""
    return {'observables': dict(OBS), 'sigma': dict(SIGMA), 'correlation': CORR, 'seed': 3} | kw


def test_apply_noise_keeps_the_truth_and_repeats_with_the_seed():
    """noise = true keeps the noise-free observables, defaults noise_seed to seed, and the same
    seed gives the same draw while another seed does not; noise = false changes nothing."""
    a, b, c = _config(noise=True), _config(noise=True), _config(noise=True, noise_seed=4)
    for cfg in (a, b, c):
        apply_noise(cfg)
    off = _config(noise=False)
    apply_noise(off)

    assert a['observables_true'] == OBS and a['noise_seed'] == 3
    assert a['observables'] == b['observables'] != c['observables']
    assert off == _config(noise=False)


@pytest.mark.parametrize(
    'kw, match',
    [
        ({'noise': 'yes'}, 'true or false'),
        ({'noise': True, 'sigma': None}, 'needs'),
        ({'noise': True, 'noise_seed': 1.5}, 'integer'),
    ],
    ids=['not-bool', 'no-sigma', 'fractional-seed'],
)
def test_apply_noise_refuses_bad_settings_without_changing_the_config(kw, match):
    """Each bad setting raises before the observables change."""
    cfg = _config(**kw)
    before = copy.deepcopy(cfg)
    with pytest.raises(ValueError, match=match):
        apply_noise(cfg)
    assert cfg['observables'] == before['observables']
    assert cfg['sigma'] == before['sigma']


def test_noise_leaves_upper_limits_unchanged():
    """An upper limit keeps its value and sigma under noise, and stays an upper limit, while the
    measured observables get their draw."""
    obs = {'R_obs': 7.0e6, 'C/O_atm': 0.1}
    sigma = validate_sigma(
        obs, {'R_obs': 1.0e5, 'C/O_atm': '0.3 dex'}, upper_limits=['C/O_atm']
    )
    noisy, out = noisy_observables(obs, sigma, None, 7)
    assert noisy['C/O_atm'] == obs['C/O_atm'] and out['C/O_atm'] == sigma['C/O_atm']
    assert out.upper == {'C/O_atm'}
    assert noisy['R_obs'] != obs['R_obs']
