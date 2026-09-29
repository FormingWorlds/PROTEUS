# Orbital migration module (no tides)
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.utils.constants import AU

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def instant_migration(
    t: float, sma_init: float, sma_final: float, time_migration: float
) -> float:
    """
    Step-function for instant orbital migration.

    Parameters
    ----------
    t : float
        Current simulation time.
    sma_init : float
        Initial semi-major axis before migration [m].
    sma_final : float
        Final semi-major axis after migration [m].
    time_migration : float
        Time of orbital migration.

    Returns
    -------
    float
        Semi-major axis [m].
    """

    if t < time_migration:
        return sma_init
    else:
        return sma_final


def sigmoid_migration(
    t: float, sma_init: float, sma_final: float, time_migration: float, tau_mig: float
) -> float:
    """
    Smooth orbital migration over a migration window of finite length.

    The orbit is held at sma_init until time_migration, crosses to sma_final
    over the following tau_mig, and is held there afterwards. Inside the
    window it follows the cubic S curve 3u^2 - 2u^3 in the window fraction
    u, whose derivative vanishes at both ends, so the semi-major axis and
    its rate of change are both continuous across the whole track.

    Parameters
    ----------
    t : float
        Current simulation time.
    sma_init : float
        Initial semi-major axis [m].
    sma_final : float
        Final semi-major axis [m].
    time_migration : float
        Time at which migration starts [yr].
    tau_mig : float
        Length of the migration window (must be positive) [yr].

    Returns
    -------
    float
        Semi-major axis [m].
    """

    if tau_mig <= 0:
        raise ValueError(f'Migration speed tau_mig must be > 0, got {tau_mig}')

    if t <= time_migration:
        return sma_init

    if t >= time_migration + tau_mig:
        return sma_final

    u = (t - time_migration) / tau_mig

    return sma_init + (sma_final - sma_init) * u * u * (3.0 - 2.0 * u)


def high_eccentricity_migration(
    t: float,
    ecc: float,
    sma_init: float,
    sma_final: float,
    time_migration: float,
    tau_mig: float,
) -> float:
    """
    Orbital migration triggered by a high-eccentricity event, with a time transition.

    Parameters
    ----------
    t : float
        Current simulation time.
    ecc : float
        Initial eccentricity.
    sma_init : float
        Initial semi-major axis [m].
    sma_final : float
        Final semi-major axis [m].
    time_migration : float
        Midpoint time of migration.
    tau_mig : float
        Migration speed parameter (must be positive) [yr-1].

    Returns
    -------
    float
        Semi-major axis [m].
    float
        Eccentricity [].
    """

    if tau_mig <= 0:
        raise ValueError(f'Migration speed tau_mig must be > 0, got {tau_mig}')

    if sma_final > sma_init:
        raise ValueError(
            'High-eccentricity migration is inward only and requires '
            f'sma_final <= sma_init, got sma_init={sma_init} and sma_final={sma_final}'
        )

    if t < time_migration:
        return sma_init, ecc
    else:
        e_mig = np.sqrt(1.0 - sma_final / sma_init)
        sma = sma_final / (1.0 - e_mig**2 * np.exp(-2 * (t - time_migration) / tau_mig))
        ecc = np.sqrt(max(0, 1.0 - sma_final / sma))
        return sma, ecc


def run_parameterized_orbital_migration(hf_row: dict, config: Config):
    """
    Run the parameterized orbital migration module.

    Evaluate semi-major axis for a selected orbital migration option.

    Parameters
    ----------
    hf_row : dict
        Dictionary of current runtime variables
    config : Config
        Configuration options

    Returns
    -------
    float
        Semi-major axis [m].
    float
        Eccentricity [].
    """

    # Initial parameters from config
    eccentricity = config.orbit.eccentricity
    migration = config.orbit.parameterized.migration
    t_mig = config.orbit.parameterized.time_migration
    tau_mig = config.orbit.parameterized.tau_migration

    # Both endpoints are schema-optional, so name the missing one here rather
    # than failing inside the unit conversion below.
    if config.orbit.parameterized.sma_init is None:
        raise ValueError('Parameterized migration requires orbit.parameterized.sma_init')
    sma_i = config.orbit.parameterized.sma_init * AU

    if migration in ('instant', 'sigmoid', 'high_ecc'):
        if config.orbit.parameterized.sma_final is None:
            raise ValueError(
                f'Migration option {migration!r} requires orbit.parameterized.sma_final'
            )
        sma_f = config.orbit.parameterized.sma_final * AU
    else:
        sma_f = sma_i

    # Time step
    current_time = float(hf_row['Time'])

    # Evaluate migration regime. Every regime below writes both orbital
    # elements, so the row is either fully updated or left untouched.

    if migration == 'none':  # no migration
        hf_row['semimajorax'] = sma_i
        hf_row['eccentricity'] = eccentricity
    elif migration == 'instant':  # instant migration
        hf_row['semimajorax'] = instant_migration(
            t=current_time, sma_init=sma_i, sma_final=sma_f, time_migration=t_mig
        )
        hf_row['eccentricity'] = eccentricity
    elif migration == 'sigmoid':  # sigmoid migration
        if tau_mig is None:
            raise ValueError('Sigmoid migration requires timescale tau_mig')
        else:
            hf_row['semimajorax'] = sigmoid_migration(
                t=current_time,
                sma_init=sma_i,
                sma_final=sma_f,
                time_migration=t_mig,
                tau_mig=tau_mig,
            )
            hf_row['eccentricity'] = eccentricity

    elif migration == 'high_ecc':  # high-eccentricity migration
        if tau_mig is None:
            raise ValueError('High-eccentricity migration requires timescale tau_mig')
        else:
            hf_row['semimajorax'], hf_row['eccentricity'] = high_eccentricity_migration(
                t=current_time,
                ecc=eccentricity,
                sma_init=sma_i,
                sma_final=sma_f,
                time_migration=t_mig,
                tau_mig=tau_mig,
            )
    else:
        raise ValueError(
            f'Unknown migration option: {migration!r}. '
            'Expected "none", "instant", "sigmoid" or "high_ecc".'
        )

    return hf_row['semimajorax'], hf_row['eccentricity']
