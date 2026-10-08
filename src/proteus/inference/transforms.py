"""Parameter transforms for inference normalization and unnormalization.

This module applies log10 scaling for parameters that are naturally
logarithmic, using the shared `variable_is_logarithmic()` convention.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.utils.transforms import normalize, unnormalize

from proteus.utils.coupler import variable_is_logarithmic


def _log_mask(keys: Sequence[str]) -> torch.Tensor:
    return torch.tensor([variable_is_logarithmic(k) for k in keys], dtype=torch.bool)


# Get log10 of the bounds for the parameters that are configured as logarithmic
def _log10_bounds(
    bounds: torch.Tensor, keys: Sequence[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    log_mask = _log_mask(keys)
    if not log_mask.any():
        return bounds, log_mask

    log_bounds = bounds.clone()
    for i, is_log in enumerate(log_mask.tolist()):
        if not is_log:
            continue
        if torch.any(log_bounds[:, i] <= 0):
            raise ValueError(
                f'Log-scaled parameter bounds must be > 0 for {keys[i]}: '
                f'{log_bounds[:, i].tolist()}'
            )
        log_bounds[:, i] = torch.log10(log_bounds[:, i])

    return log_bounds, log_mask


# Get log10 of the parameter values for the parameters that are configured as logarithmic
def _log10_values(
    values: torch.Tensor, keys: Sequence[str], log_mask: torch.Tensor
) -> torch.Tensor:
    if not log_mask.any():
        return values

    logged = values.clone()
    for i, is_log in enumerate(log_mask.tolist()):
        if not is_log:
            continue
        if torch.any(logged[..., i] <= 0):
            raise ValueError(
                f'Log-scaled parameter values must be > 0 for {keys[i]}: '
                f'{logged[..., i].flatten().tolist()}'
            )
        logged[..., i] = torch.log10(logged[..., i])

    return logged


# Get 10^values for the parameters that are configured as logarithmic
def _exp10_values(values: torch.Tensor, log_mask: torch.Tensor) -> torch.Tensor:
    if not log_mask.any():
        return values

    expanded = values.clone()
    ten = torch.tensor(10.0, dtype=expanded.dtype, device=expanded.device)
    for i, is_log in enumerate(log_mask.tolist()):
        if not is_log:
            continue
        expanded[..., i] = torch.pow(ten, expanded[..., i])

    return expanded


# Normalise parameters to [0, 1], applying log10 scaling when configured
def normalize_parameters(
    raw_x: torch.Tensor, bounds: torch.Tensor, keys: Sequence[str]
) -> torch.Tensor:
    """Normalize parameters to [0, 1], applying log10 scaling when configured."""
    log_bounds, log_mask = _log10_bounds(bounds, keys)
    raw_logged = _log10_values(raw_x, keys, log_mask)
    return normalize(raw_logged, log_bounds)


# Unnormalise parameters from [0, 1], applying exp10 scaling when configured
def unnormalize_parameters(
    x_norm: torch.Tensor, bounds: torch.Tensor, keys: Sequence[str]
) -> torch.Tensor:
    """Unnormalize parameters from [0, 1], returning dimensional values."""
    log_bounds, log_mask = _log10_bounds(bounds, keys)
    x_logged = unnormalize(x_norm, log_bounds)
    return _exp10_values(x_logged, log_mask)
