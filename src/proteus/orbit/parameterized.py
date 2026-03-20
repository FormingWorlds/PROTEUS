# Orbital migration module (no tides)
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.utils.constants import AU

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def instant_migration(t: float, sma_init: float, sma_final: float, time_migration: float) -> float:
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


def sigmoid_migration(t: float, sma_init: float, sma_final: float, time_migration: float, v_mig: float) -> float:
    """
    Sigmoid function for orbital migration with a time transition.

    Parameters
    ----------
    t : float
        Current simulation time.
    sma_init : float
        Initial semi-major axis [m].
    sma_final : float
        Final semi-major axis [m].
    time_migration : float
        Midpoint time of migration.
    v_mig : float
        Migration speed parameter (must be positive) [yr-1].

    Returns
    -------
    float
        Semi-major axis [m].
    """

    if v_mig <= 0:
        raise ValueError(f'Migration speed v_mig must be > 0, got {v_mig}')

    if t < time_migration:
        return sma_init
    else:
        sma = ((sma_init - sma_final) / (1.0 + np.exp((t - time_migration) * v_mig))) + sma_final
        return sma

def run_parameterized_orbital_migration(hf_row: dict, config: Config, dt: float):
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
    """

    # Initial parameters from config
    migration = config.orbit.parameterized.migration
    sma_i = config.orbit.parameterized.sma_init * AU
    sma_f = config.orbit.parameterized.sma_final * AU
    t_mig = config.orbit.parameterized.time_migration
    v_mig = config.orbit.parameterized.speed_migration

    # Time step
    current_time = float(hf_row['Time'])

    # Use config parameters as initial guess
    if current_time <= 1:
        hf_row['semimajorax'] = sma_i
    elif migration == "instant": # instant migration
        hf_row['semimajorax'] = instant_migration(t=current_time, sma_init=sma_i, sma_final=sma_f, time_migration=t_mig)
    elif migration == "sigmoid": # continuous migration
        if v_mig is None:
            raise ValueError('Sigmoid migration requires speed v_mig')
        else:
            hf_row['semimajorax'] = sigmoid_migration(t=current_time,
                                                sma_init=sma_i,
                                                sma_final=sma_f,
                                                time_migration=t_mig,
                                                v_mig=v_mig,
                                            )
    elif migration is None:
        raise ValueError(f'Unknown migration option: {migration}. Expected None, "instant", or "sigmoid"')
