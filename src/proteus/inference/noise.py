"""Gaussian noise on the observables, so that a study of a simulated planet sees data as real
observations would give them."""

from __future__ import annotations

import numpy as np

from proteus.inference.correlation import CorrelationWhitener
from proteus.inference.objective import Sigma, compared_in_log, observable_space


def noisy_observables(
    observables: dict, sigma: dict, correlation: dict | None, seed: int
) -> tuple[dict, dict]:
    """One draw of the observables with Gaussian noise of standard deviation sigma, in the space
    the objective compares each in (log10 or linear), correlated as in `correlation`. Returns
    the noisy observables and sigma rescaled so that the error in dex stays the same. An upper
    limit is a reported value, not a measurement, so it keeps its value and sigma."""
    keys = list(observables)
    _, x, s = observable_space(observables, observables, sigma)
    z = np.random.default_rng(seed).standard_normal(len(keys))
    if correlation:
        z = CorrelationWhitener(keys, correlation).correlate(z)
    upper = getattr(sigma, 'upper', frozenset())
    z = np.where([k in upper for k in keys], 0.0, z)
    y = np.asarray(x) + np.asarray(s) * z
    noisy = {k: float(10**v if compared_in_log(k, sigma) else v) for k, v in zip(keys, y)}
    scale = {k: noisy[k] / observables[k] if compared_in_log(k, sigma) else 1.0 for k in keys}
    noisy.update({k: observables[k] for k in upper})
    scale.update({k: 1.0 for k in upper})
    return noisy, Sigma(
        {k: sigma[k] * scale[k] for k in keys}, getattr(sigma, 'log', ()), upper
    )


def apply_noise(config: dict) -> None:
    """With `noise = true`, replace config's observables by a noisy draw (seed `noise_seed`,
    default `seed`) and keep the noise-free values as `observables_true`."""
    noise = config.get('noise', False)
    if not isinstance(noise, bool):
        raise ValueError(f'noise must be true or false, got {noise!r}')
    if not noise:
        return
    if config['sigma'] is None:
        raise ValueError('noise needs [sigma] for its standard deviations')
    if 'observables_true' in config:
        raise ValueError('the observables already carry noise (observables_true is set)')
    seed = config.setdefault('noise_seed', config['seed'])
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f'noise_seed must be an integer, got {seed!r}')
    config['observables_true'] = config['observables']
    config['observables'], config['sigma'] = noisy_observables(
        config['observables'], config['sigma'], config['correlation'], seed
    )
