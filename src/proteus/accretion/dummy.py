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

# Zero-pressure densities of the two components a rocky body is treated as a
# mixture of [kg m-3]. They set the floor the mass-radius scaling is capped
# against, so a small impactor cannot come out less dense than its own minerals.
_RHO_IRON = 7870.0
_RHO_SILICATE = 3300.0

# Smallest impact worth applying, as a fraction of the total accreted mass. The
# growth law's increments decay geometrically, so a timescale far shorter than
# the impact spacing drives the later ones toward zero; each would still re-melt
# the whole mantle and reset the orbit, which a boulder cannot do. Rejecting
# them names the configuration error rather than letting the run apply it.
_MIN_IMPACT_MASS_FRAC = 1.0e-4

# Smallest impactor worth applying, as a fraction of the target it strikes. Every
# impact re-melts the whole mantle, strips atmosphere and moves the orbit, so a
# body far below this cannot be one: the Moon-forming impactor is of order a
# tenth of Earth, and a thousandth is already three orders below that.
_MIN_IMPACTOR_TARGET_RATIO = 1.0e-3


def _body_radius(config: Config, mass: float) -> float:
    """Radius of a rocky body of the given mass [m].

    Uses the Noack & Lasbleis (2020) mass-radius scaling, the same
    parameterization the dummy interior structure uses, evaluated at the
    planet's configured core fraction so an impactor and its target share a
    composition.

    That scaling is calibrated for planets and its radius grows as
    ``M**0.282``, so the bulk density it implies falls without bound as the
    mass does. Extrapolated to an impactor a hundred times lighter than Earth
    it returns a body less dense than its own uncompressed minerals, which is
    impossible and which would propagate into the collision speed and the
    erosion law. The radius is therefore capped so the bulk density never falls
    below the zero-pressure value of an iron and silicate mixture at the same
    iron fraction. Above roughly a tenth of an Earth mass the cap is inactive
    and the scaling governs; below it the body is treated as uncompressed,
    which is the correct limit for a small body.

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
    radius = nl20_planet_radius_km(x_fe, m_ratio) * 1.0e3

    rho_uncompressed = 1.0 / (x_fe / _RHO_IRON + (1.0 - x_fe) / _RHO_SILICATE)
    radius_uncompressed = (3.0 * mass / (4.0 * math.pi * rho_uncompressed)) ** (1.0 / 3.0)

    return min(radius, radius_uncompressed)


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

    # A timescale far from the impact spacing makes the law unusable. Too short
    # and it completes inside the first interval, leaving the later impacts with
    # a vanishing share; too long and the accreted fraction over the whole
    # timeline underflows, leaving all of them with one. The test is on the
    # delivered masses rather than on the weights, because a weight can be
    # positive and still describe an impact too small to be a giant impact,
    # which would nonetheless re-melt the mantle and reset the orbit.
    total = sum(weights)
    if total <= 0.0:
        smallest = 0.0
    else:
        smallest = min(weights) / total

    if smallest < _MIN_IMPACT_MASS_FRAC:
        raise ValueError(
            f'accretion.dummy.timescale = {tau:.3e} yr cannot distribute mass over '
            f'{n_impacts} impacts ending at time_last = {float(dummy.time_last):.3e} yr: '
            f'the smallest would carry {smallest:.3e} of the accreted mass, below the '
            f'{_MIN_IMPACT_MASS_FRAC:.0e} floor, which is not a giant impact but would '
            'still re-melt the mantle and reset the orbit. Bring timescale closer to the '
            f'impact spacing, {float(dummy.time_last) / n_impacts:.3e} yr, or ask for '
            'fewer impacts.'
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
    m_target: float,
    m_impactor: float,
    a_target: float,
    e_target: float,
    e_impactor: float,
    m_star: float,
) -> tuple[float, float, float]:
    """Orbit and encounter velocity produced by a perfect merger.

    A collision conserves linear momentum, not energy, so the merged body
    leaves the collision point with the mass-weighted mean of the two
    velocities, and its orbit follows from that velocity at that radius.

    The geometry is coplanar and co-orbital: both bodies share the semi-major
    axis ``a_target`` and are evaluated where they cross that radius, each with
    its own eccentricity. At that radius a body's speed equals the circular
    speed whatever its eccentricity, while its velocity is tilted out of the
    tangential direction by an amount the eccentricity sets, and that tilt is
    what supplies the relative velocity at contact. The two are taken to cross
    in opposite radial directions, one rising and one falling, which is the
    configuration that brings them together. For a circular target and small
    impactor eccentricity the relative velocity reduces to
    ``e_impactor * v_kep``.

    Parameters
    ----------
    m_target, m_impactor : float
        Masses of the two bodies [kg].
    a_target : float
        Shared semi-major axis [m].
    e_target, e_impactor : float
        Eccentricities of the two bodies' orbits [1].
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

    # Velocity components at the crossing radius, (radial, tangential). A body
    # sharing the semi-major axis shares the speed, but an eccentric one carries
    # less angular momentum and makes up the difference radially. The two cross
    # in opposite radial senses, so their radial components have opposite signs.
    v_target = (-v_kep * e_target, v_kep * math.sqrt(1.0 - e_target**2))
    v_impactor = (v_kep * e_impactor, v_kep * math.sqrt(1.0 - e_impactor**2))

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
    # eccentricity. Averaging two velocities of equal magnitude can only lower
    # the speed, so under this co-orbital geometry the merged orbit is always
    # bound and never wider than the one the bodies shared. That is a property
    # of the shared semi-major axis, not a general result for mergers.
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
    e_target = float(config.orbit.eccentricity)

    events = []
    for index, (time, m_impactor) in enumerate(zip(times, masses)):
        m_merged = m_target + m_impactor

        if m_impactor > m_target:
            raise ValueError(
                f'Impact {index} would strike a target lighter than the impactor '
                f'({m_impactor / M_earth:.4f} onto {m_target / M_earth:.4f} M_earth). '
                'Everything downstream treats the target as the surviving body: it is '
                'the target whose mantle re-melts and whose atmosphere is stripped, so '
                'the roles cannot be reversed. Lower accretion.dummy.mass_accreted or '
                'raise planet.mass_tot.'
            )

        # Whether an impact is a giant impact is a statement about the two bodies,
        # not about how the delivered mass happens to be divided up. The share of
        # the budget is bounded elsewhere, but a large budget spread over many
        # impacts onto a heavy planet can still schedule collisions far too small
        # to melt a mantle or reset an orbit, which is what each one goes on to do.
        if m_impactor < _MIN_IMPACTOR_TARGET_RATIO * m_target:
            raise ValueError(
                f'Impact {index} carries {m_impactor / m_target:.3e} of its target '
                f'mass ({m_impactor / M_earth:.4e} onto {m_target / M_earth:.4f} '
                f'M_earth), below the {_MIN_IMPACTOR_TARGET_RATIO:.0e} floor. An '
                'impact that small is not a giant impact, yet it would still re-melt '
                'the whole mantle, strip the atmosphere and move the orbit. Raise '
                'accretion.dummy.mass_accreted or ask for fewer impacts.'
            )

        r_target = _body_radius(config, m_target)
        r_impactor = _body_radius(config, m_impactor)

        a_after, e_after, v_encounter = _merged_orbit(
            m_target, m_impactor, a_target, e_target, eccentricity, m_star
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
                e_before=e_target,
                e_after=e_after,
                id_target=0,
                id_impactor=index + 1,
            )
        )

        m_target = m_merged
        a_target = a_after
        e_target = e_after

    validate_timeline(events)

    log.info(
        'Generated %d impacts: %.4f -> %.4f M_earth over %.3e yr',
        len(events),
        float(config.planet.mass_tot),
        m_target / M_earth,
        times[-1],
    )
    return events
