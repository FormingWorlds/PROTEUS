"""
Unit tests for the correlated observable uncertainties of the inference objective.

References:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip('torch')

import proteus.inference.correlation as correlation_mod  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def test_validate_correlation_rejects_invalid_matrices():
    """A valid table comes back as floats; every kind of invalid table is refused."""
    obs = {'R_obs': 6.0e6, 'T_obs': 400.0, 'g_obs': 9.8}
    sigma = {'R_obs': 1.0e5, 'T_obs': 10.0, 'g_obs': 0.5}

    out = correlation_mod.validate_correlation(obs, sigma, {'R_obs': {'g_obs': -0.3}})
    assert out == {'R_obs': {'g_obs': pytest.approx(-0.3)}}
    assert correlation_mod.validate_correlation(obs, sigma, None) is None

    with pytest.raises(ValueError, match='sigma'):
        correlation_mod.validate_correlation(obs, None, {'R_obs': {'g_obs': 0.3}})
    for bad in (1.0, -1.0, 1.5, float('nan')):
        with pytest.raises(ValueError, match=r'\(-1, 1\)'):
            correlation_mod.validate_correlation(obs, sigma, {'R_obs': {'g_obs': bad}})
    with pytest.raises(KeyError, match='P_surf'):
        correlation_mod.validate_correlation(obs, sigma, {'R_obs': {'P_surf': 0.3}})
    with pytest.raises(ValueError, match='itself'):
        correlation_mod.validate_correlation(obs, sigma, {'R_obs': {'R_obs': 0.3}})
    with pytest.raises(ValueError, match='twice'):
        correlation_mod.validate_correlation(
            obs, sigma, {'R_obs': {'g_obs': 0.3}, 'g_obs': {'R_obs': 0.4}}
        )
    # Each pair is valid alone, but the matrix has eigenvalue 1 - 1.8 < 0.
    with pytest.raises(ValueError, match='positive definite'):
        correlation_mod.validate_correlation(
            obs, sigma, {'R_obs': {'T_obs': 0.9, 'g_obs': 0.9}, 'T_obs': {'g_obs': -0.9}}
        )


def test_composition_from_names_parses_ratios_and_rejects_unknown_elements():
    """Each '/' name becomes +1 numerator, -1 denominator; other names are skipped."""
    obs = {'R_obs': 9.18e6, 'C/O_atm': 0.62, 'S/O_atm': 0.25, 'O/H_atm': 5.4, 'Si/Mg_atm': 1.1}
    assert correlation_mod.composition_from_names(obs) == {
        'C/O_atm': {'C': 1, 'O': -1},
        'S/O_atm': {'S': 1, 'O': -1},
        'O/H_atm': {'O': 1, 'H': -1},
        'Si/Mg_atm': {'Si': 1, 'Mg': -1},
    }
    for bad in ('C/Xx_atm', 'CO/H_atm', 'C/O/H_atm', 'C/C_atm', '/O_atm', 'C/O'):
        with pytest.raises(ValueError, match='two different elements'):
            correlation_mod.composition_from_names({'R_obs': 1.0, bad: 1.0})


@pytest.mark.physics_invariant
def test_ratio_correlation_shares_elements_with_sign():
    """Equal element errors give rho = +-1/2 for one shared element, signed by
    whether it sits on the same side of both ratios.
    """
    obs = {'R_obs': 9.18e6, 'C/O_atm': 0.624, 'S/O_atm': 0.249, 'O/H_atm': 5.37}
    comp = correlation_mod.composition_from_names(obs)
    corr = correlation_mod.ratio_correlation(comp)

    assert corr['C/O_atm']['S/O_atm'] == pytest.approx(0.5, abs=1e-12)
    assert corr['C/O_atm']['O/H_atm'] == pytest.approx(-0.5, abs=1e-12)
    assert corr['S/O_atm']['O/H_atm'] == pytest.approx(-0.5, abs=1e-12)
    assert sum(len(row) for row in corr.values()) == 3
    assert 'R_obs' not in corr and all('R_obs' not in row for row in corr.values())

    # Positive definite: eigenvalues of [[1, .5, -.5], [.5, 1, -.5], [-.5, -.5, 1]] are 2, .5, .5.
    names = list(comp)
    mat = torch.eye(3, dtype=torch.double)
    for a, row in corr.items():
        for b, rho in row.items():
            mat[names.index(a), names.index(b)] = mat[names.index(b), names.index(a)] = rho
    eig = torch.linalg.eigvalsh(mat)
    assert eig.tolist() == pytest.approx([0.5, 0.5, 2.0], abs=1e-12)
    sigma = {k: 0.1 * v for k, v in obs.items()}
    assert correlation_mod.validate_correlation(obs, sigma, corr) == corr


@pytest.mark.parametrize(
    'names, dependent',
    [
        (['C/O_atm', 'O/H_atm', 'C/H_atm'], 'C/H_atm'),
        (['C/O_atm', 'N/O_atm', 'C/N_atm'], 'C/N_atm'),
        (['C/O_atm', 'S/O_atm', 'C/S_atm', 'O/H_atm'], 'C/S_atm'),
    ],
    ids=['ch_from_co_and_oh', 'cn_from_co_and_no', 'cs_before_an_independent_ratio'],
)
def test_ratio_correlation_rejects_a_ratio_that_follows_from_the_others(names, dependent):
    """A dependent ratio set makes R singular, which Cholesky accepts on round-off.

    Without the check the last pivot is about 2e-8, so chi2 would amplify one
    residual direction by about 1e15 instead of failing.
    """
    comp = correlation_mod.composition_from_names(names)
    with pytest.raises(ValueError, match=f"'{dependent}' follows from"):
        correlation_mod.ratio_correlation(comp)
    # Dropping the dependent ratio leaves a valid, positive definite set.
    kept = [n for n in names if n != dependent]
    corr = correlation_mod.ratio_correlation(correlation_mod.composition_from_names(kept))
    sigma = dict.fromkeys(kept, 0.1)
    assert correlation_mod.validate_correlation(dict.fromkeys(kept, 1.0), sigma, corr) == corr


@pytest.mark.physics_invariant
def test_chi_squared_matches_residuals_to_names_in_any_order():
    """Residuals listed in another order than the matrix are reordered first.
    With three observables and rho on one pair, a skipped reorder changes chi2;
    with two and a unit diagonal it would not, since u^T R^-1 u is then symmetric.
    """
    whitener = correlation_mod.CorrelationWhitener(['A', 'B', 'C'], {'A': {'B': 0.5}})
    # u_A, u_B, u_C = 1, 2, 3, passed as (B, C, A).
    u = torch.tensor([[2.0, 3.0, 1.0]], dtype=torch.double)

    chi2 = whitener.chi_squared(['B', 'C', 'A'], u).item()

    # (1 - 2*0.5*1*2 + 4) / 0.75 + 9 = 13.
    assert chi2 == pytest.approx(13.0, rel=1e-12)
    # Without the reorder rho would pair u_B with u_C: (4 - 6 + 9) / 0.75 + 1 = 10.33.
    assert abs(chi2 - 31.0 / 3.0) > 2.0
    # Independent residuals give 14, so rho is applied.
    assert abs(chi2 - 14.0) > 0.5
    with pytest.raises(KeyError, match='do not match'):
        whitener.chi_squared(['A', 'B', 'D'], u)


@pytest.mark.physics_invariant
def test_whiten_is_the_symmetric_inverse_root_and_preserves_chi_squared():
    """For rho on (A, B), R^(-1/2) has diagonal (a + b) / 2 and off-diagonal (a - b) / 2 with
    a = (1 + rho)^(-1/2), b = (1 - rho)^(-1/2). The squares of z sum to the chi-squared, C
    (correlated with none) passes unchanged, and z_A differs from the Cholesky value u_A = 1.
    """
    whitener = correlation_mod.CorrelationWhitener(['A', 'B', 'C'], {'A': {'B': 0.5}})
    u = torch.tensor(
        [[2.0, 3.0, 1.0]], dtype=torch.double
    )  # (B, C, A): u_A, u_B, u_C = 1, 2, 3

    z = whitener.whiten(['B', 'C', 'A'], u)[0].tolist()

    a, b = 1.5**-0.5, 0.5**-0.5
    diag, off = (a + b) / 2, (a - b) / 2
    assert z == pytest.approx([diag * 1 + off * 2, off * 1 + diag * 2, 3.0], rel=1e-12)
    assert sum(v * v for v in z) == pytest.approx(
        whitener.chi_squared(['B', 'C', 'A'], u).item(), rel=1e-12
    )
    assert abs(z[0] - 1.0) > 0.3
    with pytest.raises(KeyError, match='do not match'):
        whitener.whiten(['A', 'B', 'D'], u)


def test_chi_squared_of_several_rows_matches_each_row_alone():
    """A batch of residual rows gives one chi2 per row, each as the row alone gives it."""
    whitener = correlation_mod.CorrelationWhitener(['A', 'B', 'C'], {'A': {'B': 0.5}})
    u = torch.tensor([[2.0, 3.0, 1.0], [0.0, 1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.double)

    chi2 = whitener.chi_squared(['B', 'C', 'A'], u)

    assert chi2.shape == (3, 1)
    rows = [whitener.chi_squared(['B', 'C', 'A'], r[None]).item() for r in u]
    assert chi2[:, 0].tolist() == pytest.approx(rows, rel=1e-12)
    # A sum over all rows, as the single-row form did, would give their total for each
    assert abs(chi2[0, 0].item() - sum(rows)) > 1.0


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('C/O_atm', True),
        ('Si/Mg_atm', True),
        ('C/Xx_atm', False),
        ('O/H_kg', False),
        ('C/O/H_atm', False),
        ('R_obs', False),
    ],
)
def test_is_element_ratio_needs_two_elements_and_the_atm_suffix(name, expected):
    """Only '<element>/<element>_atm' names count as element ratios."""
    assert correlation_mod.is_element_ratio(name) is expected


def test_validate_correlation_allows_at_most_one_correlated_upper_limit():
    """One upper limit may have coefficients and a zero coefficient does not count; two
    correlated limits are refused, as they would need a multivariate normal CDF."""
    from proteus.inference.objective import validate_sigma

    obs = {'R_obs': 7.0e6, 'T_obs': 600.0, 'C/O_atm': 0.1, 'S/O_atm': 0.05}
    given = {'R_obs': 1.0e5, 'T_obs': 20.0, 'C/O_atm': '0.1 dex', 'S/O_atm': '0.1 dex'}
    sigma = validate_sigma(obs, given, upper_limits=['C/O_atm', 'S/O_atm'])
    one = {'T_obs': {'C/O_atm': 0.4, 'S/O_atm': 0.0}, 'R_obs': {'T_obs': 0.2}}
    assert correlation_mod.validate_correlation(obs, sigma, one) == one
    with pytest.raises(ValueError, match=r"at most one.*\['C/O_atm', 'S/O_atm'\]"):
        correlation_mod.validate_correlation(
            obs, sigma, {'T_obs': {'C/O_atm': 0.4, 'S/O_atm': 0.3}}
        )


@pytest.mark.physics_invariant
def test_censored_chi_squared_matches_the_integral_of_the_bivariate_normal():
    """For one detection and one correlated upper limit, the chi-squared equals -2 ln of the
    bivariate normal density integrated over the limit's residual above u_c (scipy quad), up
    to ln(2 pi). Ignoring the correlation, or flipping the sign of the conditional mean,
    misses by more than 1."""
    from scipy.integrate import quad
    from scipy.stats import multivariate_normal

    rho, u_d, u_c = 0.6, 1.5, 0.3
    w = correlation_mod.CorrelationWhitener(['T_obs', 'C/O_atm'], {'T_obs': {'C/O_atm': rho}})
    u = torch.tensor([[u_d, u_c]], dtype=torch.double)
    got = w.chi_squared(['T_obs', 'C/O_atm'], u, {'C/O_atm'}).item()

    pdf = multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]]).pdf
    L, _ = quad(lambda r: pdf([u_d, r]), u_c, math.inf, epsabs=1e-13)
    assert got == pytest.approx(-2.0 * math.log(L) - math.log(2 * math.pi), rel=1e-8)

    from scipy.stats import norm

    s = math.sqrt(1 - rho**2)
    uncorrelated = u_d**2 - 2 * norm.logcdf(-u_c)
    flipped = u_d**2 - 2 * norm.logcdf((-rho * u_d - u_c) / s)
    assert abs(got - uncorrelated) > 1.0 and abs(got - flipped) > 1.0


@pytest.mark.physics_invariant
def test_censored_chi_squared_reduces_without_correlation_of_the_limit():
    """A limit correlated with none adds -2 ln Phi(-u_c) to the correlated chi-squared of the
    detections, in every row and whatever the column order; a second, uncorrelated limit adds
    its own term."""
    from scipy.stats import norm

    names = ['C/O_atm', 'R_obs', 'S/O_atm', 'T_obs']
    w = correlation_mod.CorrelationWhitener(names, {'R_obs': {'T_obs': -0.5}})
    u = torch.tensor([[0.3, 1.0, -2.0, 0.5], [-1.0, -0.4, 1.2, 2.0]], dtype=torch.double)
    got = w.chi_squared(names, u, {'C/O_atm', 'S/O_atm'})[:, 0]
    R_inv = torch.linalg.inv(torch.tensor([[1.0, -0.5], [-0.5, 1.0]], dtype=torch.double))
    det = u[:, [1, 3]]
    ref = (det @ R_inv * det).sum(1) - 2 * torch.tensor(
        norm.logcdf(-u[:, [0, 2]].numpy()).sum(1), dtype=torch.double
    )
    torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-12)
    # Dropping the limits, or scoring them two-sided as u^2, misses by more than 0.1
    assert (got - (det @ R_inv * det).sum(1)).min() > 0.1
    assert (got - (det @ R_inv * det).sum(1) - (u[:, [0, 2]] ** 2).sum(1)).abs().min() > 0.5
    perm = [3, 2, 1, 0]
    swapped = w.chi_squared([names[i] for i in perm], u[:, perm], {'C/O_atm', 'S/O_atm'})
    torch.testing.assert_close(swapped[:, 0], got, rtol=1e-12, atol=1e-12)
