"""Correlated observable uncertainties for the chi-squared in `objective.eval_obj`."""

from __future__ import annotations

import math

import torch


class CorrelationWhitener:
    """Factored correlation matrix R of the observables, for chi2 = u^T R^-1 u.

    `correlation` holds dimensionless coefficients as `{name_a: {name_b: rho}}`,
    each pair once; unlisted pairs are zero. Using R rather than the covariance
    keeps the matrix well conditioned whatever the observables' units.
    """

    def __init__(self, names: list[str], correlation: dict):
        self.names = list(names)
        index = {k: i for i, k in enumerate(self.names)}
        mat = torch.eye(len(self.names), dtype=torch.double)
        seen = {}
        for a, row in correlation.items():
            if not isinstance(row, dict):
                raise ValueError(f"correlation['{a}'] must be a table of {{name: rho}}")
            for b, rho in row.items():
                rho = float(rho)
                unknown = [k for k in (a, b) if k not in index]
                if unknown:
                    raise KeyError(f'correlation names unknown observables: {unknown}')
                if a == b:
                    raise ValueError(f"correlation of '{a}' with itself is fixed at 1")
                if not (math.isfinite(rho) and -1.0 < rho < 1.0):
                    raise ValueError(
                        f"correlation of '{a}' and '{b}' must lie in (-1, 1), got {rho}"
                    )
                if seen.setdefault(frozenset((a, b)), rho) != rho:
                    raise ValueError(f"correlation of '{a}' and '{b}' given twice, differently")
                mat[index[a], index[b]] = mat[index[b], index[a]] = rho

        # Pairwise bounds do not guarantee a valid matrix
        self._chol, info = torch.linalg.cholesky_ex(mat)
        if info != 0:
            eig_min = torch.linalg.eigvalsh(mat).min().item()
            raise ValueError(
                f'correlation matrix is not positive definite (eigenvalue {eig_min:.3g})'
            )

    def chi_squared(self, names: list[str], u: torch.Tensor) -> torch.Tensor:
        """Chi-squared, shape (1, 1), of residuals u = (sim - true) / sigma ordered as `names`."""
        if sorted(names) != sorted(self.names):
            raise KeyError(f'residuals for {sorted(names)} do not match {sorted(self.names)}')
        pos = {k: i for i, k in enumerate(names)}
        u = u.reshape(-1)[[pos[k] for k in self.names]].reshape(-1, 1)
        z = torch.linalg.solve_triangular(self._chol, u, upper=False)
        return (z**2).sum().reshape(1, 1)


def validate_correlation(
    observables: dict, sigma: dict | None, correlation: dict | None
) -> dict | None:
    """Check the config's `[correlation]` table; return it as floats, or None if absent."""
    if correlation is None:
        return None
    if sigma is None:
        raise ValueError('correlation is given but sigma is not: correlations scale sigma')
    CorrelationWhitener(list(observables), correlation)
    return {a: {b: float(rho) for b, rho in row.items()} for a, row in correlation.items()}
