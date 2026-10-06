"""
Unit tests for the correlated observable uncertainties of the inference objective.

References:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip('torch')

import proteus.inference.likelihood as likelihood_mod  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_validate_correlation_rejects_invalid_matrices():
    """A valid table comes back as floats; every kind of invalid table is refused."""
    obs = {'R_obs': 6.0e6, 'T_obs': 400.0, 'g_obs': 9.8}
    sigma = {'R_obs': 1.0e5, 'T_obs': 10.0, 'g_obs': 0.5}

    out = likelihood_mod.validate_correlation(obs, sigma, {'R_obs': {'g_obs': -0.3}})
    assert out == {'R_obs': {'g_obs': pytest.approx(-0.3)}}
    assert likelihood_mod.validate_correlation(obs, sigma, None) is None

    with pytest.raises(ValueError, match='sigma'):
        likelihood_mod.validate_correlation(obs, None, {'R_obs': {'g_obs': 0.3}})
    for bad in (1.0, -1.0, 1.5, float('nan')):
        with pytest.raises(ValueError, match=r'\(-1, 1\)'):
            likelihood_mod.validate_correlation(obs, sigma, {'R_obs': {'g_obs': bad}})
    with pytest.raises(KeyError, match='P_surf'):
        likelihood_mod.validate_correlation(obs, sigma, {'R_obs': {'P_surf': 0.3}})
    with pytest.raises(ValueError, match='itself'):
        likelihood_mod.validate_correlation(obs, sigma, {'R_obs': {'R_obs': 0.3}})
    with pytest.raises(ValueError, match='twice'):
        likelihood_mod.validate_correlation(
            obs, sigma, {'R_obs': {'g_obs': 0.3}, 'g_obs': {'R_obs': 0.4}}
        )
    # Each pair is valid alone, but the matrix has eigenvalue 1 - 1.8 < 0.
    with pytest.raises(ValueError, match='positive definite'):
        likelihood_mod.validate_correlation(
            obs, sigma, {'R_obs': {'T_obs': 0.9, 'g_obs': 0.9}, 'T_obs': {'g_obs': -0.9}}
        )
