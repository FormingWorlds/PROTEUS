# Analytical giant-impact accretion module
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from proteus.accretion.common import ImpactEvent, validate_timeline
from proteus.utils.constants import AU, M_earth, M_sun, const_G
from proteus.utils.structure_estimate import iron_fractions, nl20_planet_radius_km

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def _body_radius(config: Config, mass: float) -> float:
    """Radius of a rocky body of the given mass [m].

    Uses the Noack & Lasbleis (2020) mass-radius scaling, the same
    parameterization the dummy interior structure uses, evaluated at the
    planet's configured core fraction so an impactor and its target share a
    composition.

    Parameters
    ----------
    config : Config
        Model configuration; read for the core fraction.
    mass : float
        Body mass [kg].

    Returns
    -------
    radius : float
        Body radius [m].
    """
    m_ratio = mass / M_earth
    _, x_fe, _ = iron_fractions(
        config.interior_struct.core_frac,
        config.interior_struct.core_frac_mode,
        mass_tot_M_earth=m_ratio,
    )
    return nl20_planet_radius_km(x_fe, m_ratio) * 1.0e3


def _impact_masses(config: Config) -> list[float]:
    """Mass each impact delivers [kg], in time order.

    The planet approaches its asymptotic mass exponentially, so the mass
    accreted between two times is the difference of the law evaluated at them.
    With impacts spaced evenly in time the increments therefore decay, and the
    first impact is the largest. The increments are then rescaled to sum to the
    configured total, which makes the delivered mass exactly what was asked for
    while leaving the law in charge of the distribution.

    Parameters
    ----------
    config : Config
        Model configuration.

    Returns
    -------
    masses : list of float
        Impactor masses [kg], one per impact, in time order.
    """
    dummy = config.accretion.dummy
    n_impacts = int(dummy.num_impacts)
    tau = float(dummy.timescale)

    times = _impact_times(config)
    edges = [0.0, *times]
    weights = [
        math.exp(-edges[k] / tau) - math.exp(-edges[k + 1] / tau) for k in range(n_impacts)
    ]

    # A timescale far from the impact spacing makes the law unusable in one of
    # two ways, and both have to be caught here rather than surfacing later as
    # a zero-mass impactor. Too short and the law completes inside the first
    # interval, so every later weight underflows to zero; too long and the
    # accreted fraction over the whole timeline underflows, so they all do.
    total = sum(weights)
    if total <= 0.0 or min(weights) <= 0.0:
        raise ValueError(
            f'accretion.dummy.timescale = {tau:.3e} yr cannot distribute mass over '
            f'{n_impacts} impacts ending at time_last = {float(dummy.time_last):.3e} yr: '
            'the accretion law is either finished or has barely begun by the time the '
            'impacts are spaced, leaving at least one of them with no mass to deliver. '
            'Bring timescale closer to the impact spacing, '
            f'{float(dummy.time_last) / n_impacts:.3e} yr.'
        )

    delivered = float(dummy.mass_accreted) * M_earth
    return [delivered * w / total for w in weights]


def _impact_times(config: Config) -> list[float]:
    """Time of each impact [yr], evenly spaced up to the configured last one."""
    dummy = config.accretion.dummy
    n_impacts = int(dummy.num_impacts)
    t_last = float(dummy.time_last)
    return [(k + 1) * t_last / n_impacts for k in range(n_impacts)]


def _merged_orbit(
    m_target: float, m_impactor: float, a_target: float, eccentricity: float, m_star: float
) -> tuple[float, float, float]:
    """Orbit and encounter velocity produced by a perfect merger.

    A collision conserves linear momentum, not energy, so the merged body
    leaves the collision point with the mass-weighted mean of the two
    velocities, and its orbit follows from that velocity at that radius.

    The geometry is coplanar and fully determined by one parameter: the target
    is on a circular orbit of radius ``a_target``, and the impactor is on an
    orbit of the same semi-major axis with eccentricity ``eccentricity``,
    evaluated where it crosses the target. At that radius the impactor's speed
    equals the circular speed while its velocity is tilted, which is what
    supplies the relative velocity at contact. In the small-eccentricity limit
    that relative velocity reduces to ``eccentricity * v_kep``.

    Parameters
    ----------
    m_target, m_impactor : float
        Masses of the two bodies [kg].
    a_target : float
        Semi-major axis of the target's circular orbit [m].
    eccentricity : float
        Eccentricity of the impactor's orbit [1].
    m_star : float
        Mass of the host star [kg].

    Returns
    -------
    a_after : float
        Semi-major axis of the merged body [m].
    e_after : float
        Eccentricity of the merged body [1].
    v_encounter : float
        Relative velocity of the two bodies at contact [m s-1], before the
        gravitational focusing that the mutual escape velocity adds.
    """
    mu = const_G * m_star
    v_kep = math.sqrt(mu / a_target)

    # Velocity components at the crossing radius, (radial, tangential). The
    # target is circular, so it is purely tangential. The impactor shares the
    # semi-major axis, so it shares the speed, but carries the angular momentum
    # of an eccentric orbit and makes up the rest radially.
    v_target = (0.0, v_kep)
    v_impactor = (v_kep * eccentricity, v_kep * math.sqrt(1.0 - eccentricity**2))

    v_encounter = math.hypot(
        v_impactor[0] - v_target[0],
        v_impactor[1] - v_target[1],
    )

    m_merged = m_target + m_impactor
    v_merged = (
        (m_target * v_target[0] + m_impactor * v_impactor[0]) / m_merged,
        (m_target * v_target[1] + m_impactor * v_impactor[1]) / m_merged,
    )

    # Vis-viva at the collision radius, then the angular momentum fixes the
    # eccentricity. Averaging two bound velocities at one radius can only lower
    # the specific energy, so the merged orbit is always bound and interior to
    # the target's.
    speed_sq = v_merged[0] ** 2 + v_merged[1] ** 2
    a_after = 1.0 / (2.0 / a_target - speed_sq / mu)

    h = a_target * v_merged[1]
    e_after = math.sqrt(max(0.0, 1.0 - h * h / (mu * a_after)))

    return a_after, e_after, v_encounter


def get_timeline(config: Config) -> list[ImpactEvent]:
    """Build an impact timeline from the analytical accretion law.

    Grows the configured planet by the configured mass through a chain of
    perfect mergers, deriving each impact's masses, radii, velocities and orbit
    change from scaling laws rather than from a dynamical model. The result is
    deterministic: the same configuration always produces the same history.

    Parameters
    ----------
    config : Config
        Model configuration.

    Returns
    -------
    events : list of ImpactEvent
        Impacts to apply during the run, in time order.
    """
    dummy = config.accretion.dummy

    times = _impact_times(config)
    masses = _impact_masses(config)

    m_star = float(config.star.mass) * M_sun
    eccentricity = float(dummy.eccentricity)
    impact_parameter = float(dummy.impact_parameter)
    offset = float(config.accretion.time_offset)

    m_target = float(config.planet.mass_tot) * M_earth
    a_target = float(config.orbit.semimajoraxis) * AU

    events = []
    for index, (time, m_impactor) in enumerate(zip(times, masses)):
        m_merged = m_target + m_impactor

        r_target = _body_radius(config, m_target)
        r_impactor = _body_radius(config, m_impactor)

        a_after, e_after, v_encounter = _merged_orbit(
            m_target, m_impactor, a_target, eccentricity, m_star
        )

        # Contact speed: the encounter velocity, focused by the pair's mutual
        # gravity. This is the convention the collision erosion law expects and
        # it puts the collision velocity at or above the escape velocity for
        # any encounter, including a strictly circular one.
        v_esc = math.sqrt(2.0 * const_G * m_merged / (r_target + r_impactor))
        v_impact = math.hypot(v_encounter, v_esc)

        events.append(
            ImpactEvent(
                time=time + offset,
                M_target_before=m_target,
                M_impactor=m_impactor,
                M_merged_after=m_merged,
                v_impact=v_impact,
                v_esc=v_esc,
                impact_parameter=impact_parameter,
                R_target_before=r_target,
                R_impactor=r_impactor,
                rho_target=m_target / (4.0 / 3.0 * math.pi * r_target**3),
                rho_impactor=m_impactor / (4.0 / 3.0 * math.pi * r_impactor**3),
                a_before=a_target,
                a_after=a_after,
                e_after=e_after,
                id_target=0,
                id_impactor=index + 1,
            )
        )

        m_target = m_merged
        a_target = a_after

    validate_timeline(events)

    log.info(
        'Generated %d impacts: %.4f -> %.4f M_earth over %.3e yr',
        len(events),
        float(config.planet.mass_tot),
        m_target / M_earth,
        times[-1],
    )
    return events
