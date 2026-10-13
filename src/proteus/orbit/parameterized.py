# Orbital migration module (no tides)
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.utils.constants import AU, const_G, secs_per_year

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
        Current simulation time [yr].
    sma_init : float
        Initial semi-major axis before migration [m].
    sma_final : float
        Final semi-major axis after migration [m].
    time_migration : float
        Time of orbital migration [yr].

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
        Current simulation time [yr].
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
        raise ValueError(f'Migration timescale tau_mig must be > 0, got {tau_mig}')

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
) -> tuple[float, float]:
    """
    Orbital migration driven by a high-eccentricity event.

    Before ``time_migration`` the orbit is untouched. At that time the
    eccentricity steps discontinuously from the configured value to
    ``sqrt(1 - sma_final / sma_init)``, which is the eccentricity that
    carries the orbital angular momentum of the final circular orbit at
    the initial semi-major axis. It then decays towards zero while the
    semi-major axis falls to ``sma_final``, conserving ``a (1 - e^2)``.

    ``tau_mig`` sets the decay after the event, not the onset, which is a
    step. Downstream that step is visible: the orbit-averaged stellar
    flux rises while the time-averaged separation ``a (1 + e^2 / 2)``
    rises too, so instellation and separation move in opposite
    directions across the epoch.

    Parameters
    ----------
    t : float
        Current simulation time [yr].
    ecc : float
        Eccentricity before the event [].
    sma_init : float
        Initial semi-major axis [m].
    sma_final : float
        Final semi-major axis [m].
    time_migration : float
        Time at which the high-eccentricity event occurs [yr].
    tau_mig : float
        Eccentricity decay timescale, must be positive [yr].

    Returns
    -------
    tuple[float, float]
        Semi-major axis [m] and eccentricity [].
    """

    if tau_mig <= 0:
        raise ValueError(f'Migration timescale tau_mig must be > 0, got {tau_mig}')

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


def orbital_energy_rate(
    t: float,
    migration: str,
    sma_init: float,
    sma_final: float,
    time_migration: float,
    tau_mig: float,
    mass_star: float,
    mass_planet: float,
) -> float:
    """
    Rate of change of the orbital energy along a prescribed migration track.

    The orbital energy is ``E = -G M_star M_planet / (2 a)``, so its rate is
    ``dE/dt = G M_star M_planet (da/dt) / (2 a^2)``, evaluated with the
    closed-form ``da/dt`` of the selected law at fixed masses. It is negative
    while the orbit shrinks. The parameterized model deposits this energy
    nowhere, so the value is a diagnostic of the power a tidal model would
    have to dissipate to follow the same track.

    For ``high_ecc`` the rate is ``-G M_star M_planet e_mig^2 x / (sma_final
    tau_mig)`` with ``x = exp(-2 (t - time_migration) / tau_mig)``, largest at
    the epoch, where it equals twice the total energy change divided by
    ``tau_mig``. ``none`` and ``instant`` return zero: the instant step
    releases its energy at a single time, which a rate cannot carry.

    Parameters
    ----------
    t : float
        Current simulation time [yr].
    migration : str
        Migration law: "none", "instant", "sigmoid" or "high_ecc".
    sma_init : float
        Initial semi-major axis [m].
    sma_final : float
        Final semi-major axis [m].
    time_migration : float
        Time at which migration starts [yr].
    tau_mig : float
        Migration window for "sigmoid", decay timescale for "high_ecc" [yr].
    mass_star : float
        Stellar mass [kg].
    mass_planet : float
        Planet mass [kg].

    Returns
    -------
    float
        Rate of change of the orbital energy [W].
    """

    if migration in ('none', 'instant'):
        return 0.0

    if tau_mig <= 0:
        raise ValueError(f'Migration timescale tau_mig must be > 0, got {tau_mig}')

    if migration == 'sigmoid':
        if not time_migration < t < time_migration + tau_mig:
            return 0.0
        u = (t - time_migration) / tau_mig
        sma = sigmoid_migration(t, sma_init, sma_final, time_migration, tau_mig)
        dadt = (sma_final - sma_init) * 6.0 * u * (1.0 - u) / tau_mig
    elif migration == 'high_ecc':
        # Evaluated first so that an outward track is refused at every time
        sma, _ = high_eccentricity_migration(
            t, 0.0, sma_init, sma_final, time_migration, tau_mig
        )
        if t < time_migration:
            return 0.0
        e_mig_sq = 1.0 - sma_final / sma_init
        decay = np.exp(-2.0 * (t - time_migration) / tau_mig)
        dadt = -2.0 * sma * sma * e_mig_sq * decay / (sma_final * tau_mig)
    else:
        raise ValueError(
            f'Unknown migration option: {migration!r}. '
            'Expected "none", "instant", "sigmoid" or "high_ecc".'
        )

    # da/dt is in m per year here, the energy rate in W
    return float(const_G * mass_star * mass_planet * dadt / (2.0 * sma * sma * secs_per_year))


def track_endpoints(config: Config) -> tuple[float, float]:
    """
    Semi-major axes at the two ends of the prescribed track.

    The track starts from ``orbit.semimajoraxis``, the same value that seeds
    the orbit at the initial condition, so the run has one source for a(0).
    A static track (``migration = 'none'``) needs no final value and ends
    where it starts.

    Parameters
    ----------
    config : Config
        Configuration options

    Returns
    -------
    tuple[float, float]
        Initial and final semi-major axis [m].
    """

    params = config.orbit.parameterized
    sma_i = float(config.orbit.semimajoraxis) * AU

    if params.sma_final is None:
        if params.migration != 'none':
            raise ValueError(
                f'Migration option {params.migration!r} requires orbit.parameterized.sma_final'
            )
        return sma_i, sma_i

    return sma_i, float(params.sma_final) * AU


def update_orbital_energy_rate(hf_row: dict, config: Config) -> float:
    """
    Write the orbital energy rate of the prescribed track to ``hf_row['dEdt_orb']``.

    The track runs from ``orbit.semimajoraxis`` to ``orbit.parameterized.sma_final``.

    Parameters
    ----------
    hf_row : dict
        Dictionary of current runtime variables, carrying ``Time`` [yr],
        ``M_star`` [kg] and ``M_planet`` [kg].
    config : Config
        Configuration options

    Returns
    -------
    float
        Rate of change of the orbital energy [W].
    """

    params = config.orbit.parameterized
    sma_i, sma_f = track_endpoints(config)

    hf_row['dEdt_orb'] = orbital_energy_rate(
        t=float(hf_row['Time']),
        migration=params.migration,
        sma_init=sma_i,
        sma_final=sma_f,
        time_migration=float(params.time_migration),
        tau_mig=float(params.tau_migration),
        mass_star=float(hf_row['M_star']),
        mass_planet=float(hf_row['M_planet']),
    )
    return hf_row['dEdt_orb']


def run_parameterized_orbital_migration(hf_row: dict, config: Config) -> tuple[float, float]:
    """
    Run the parameterized orbital migration module.

    Advance the orbit along the selected migration law. A law writes only the
    elements it sets: ``instant`` and ``sigmoid`` the semi-major axis,
    ``high_ecc`` the semi-major axis and the eccentricity. The static law and
    every law before ``time_migration`` leave the row as it is, carrying the
    orbit seeded from the config at the initial condition or set by the
    previous step.

    Parameters
    ----------
    hf_row : dict
        Dictionary of current runtime variables, carrying ``Time`` [yr],
        ``semimajorax`` [m] and ``eccentricity`` [].
    config : Config
        Configuration options

    Returns
    -------
    tuple[float, float]
        Semi-major axis [m] and eccentricity [].
    """

    # Initial parameters from config
    eccentricity = config.orbit.eccentricity
    migration = config.orbit.parameterized.migration
    t_mig = config.orbit.parameterized.time_migration
    tau_mig = config.orbit.parameterized.tau_migration

    if migration not in ('none', 'instant', 'sigmoid', 'high_ecc'):
        # Defensive: the config validator already restricts migration to the
        # four names above, so this is reachable only through a stub config.
        raise ValueError(
            f'Unknown migration option: {migration!r}. '
            'Expected "none", "instant", "sigmoid" or "high_ecc".'
        )
    sma_i, sma_f = track_endpoints(config)

    # Time step
    current_time = float(hf_row['Time'])

    # Validated above, so an early return here cannot hide a bad config.
    if migration == 'none' or current_time < t_mig:
        return hf_row['semimajorax'], hf_row['eccentricity']

    if migration == 'instant':  # instant migration
        hf_row['semimajorax'] = instant_migration(
            t=current_time, sma_init=sma_i, sma_final=sma_f, time_migration=t_mig
        )
    elif migration == 'sigmoid':  # sigmoid migration
        hf_row['semimajorax'] = sigmoid_migration(
            t=current_time,
            sma_init=sma_i,
            sma_final=sma_f,
            time_migration=t_mig,
            tau_mig=tau_mig,
        )
    else:  # high-eccentricity migration, which sets both elements
        hf_row['semimajorax'], hf_row['eccentricity'] = high_eccentricity_migration(
            t=current_time,
            ecc=eccentricity,
            sma_init=sma_i,
            sma_final=sma_f,
            time_migration=t_mig,
            tau_mig=tau_mig,
        )

    return hf_row['semimajorax'], hf_row['eccentricity']
