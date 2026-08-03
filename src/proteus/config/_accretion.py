from __future__ import annotations

from attr.validators import ge, gt, in_, le, lt
from attrs import define, field

from ._converters import none_if_none

SELECTORS = ('match_config', 'mass', 'semimajoraxis', 'id')


def valid_morrigan(instance, attribute, value):
    if instance.module != 'morrigan':
        return

    mor = instance.morrigan

    if mor.masses and len(mor.masses) != mor.num_planets:
        raise ValueError(
            f'`accretion.morrigan.masses` has {len(mor.masses)} entries but '
            f'num_planets = {mor.num_planets}; they must match'
        )

    if any(m <= 0 for m in mor.masses):
        raise ValueError('All `accretion.morrigan.masses` entries must be > 0')

    if mor.selector == 'semimajoraxis' and mor.selector_value is None:
        raise ValueError(
            '`accretion.morrigan.selector_value` must be set (target orbit in AU) '
            "when selector = 'semimajoraxis'"
        )

    if mor.selector == 'id' and mor.selector_value is None:
        raise ValueError(
            "`accretion.morrigan.selector_value` must be set (planet id) when selector = 'id'"
        )


@define
class Morrigan:
    """Parameters for the Morrigan giant-impact module.

    Morrigan evolves a system of embryos after disk dispersal, following
    Kimura et al. (2025), and reports the impacts experienced by one
    selected survivor. The stellar mass is taken from ``star.mass`` rather
    than repeated here, so the dynamical model and the rest of PROTEUS
    cannot disagree about the host star.

    Attributes
    ----------
    seed: int
        Random seed for the Monte Carlo. Fixing it makes an impact
        history reproducible; sweeping it samples the outcome distribution.
    num_planets: int
        Number of embryos the system starts with.
    masses: list of float
        Initial embryo masses [M_earth], one per embryo. An empty list
        starts every embryo at ``mass_equal``.
    mass_equal: float
        Initial mass of every embryo [M_earth], used when ``masses`` is empty.
    eccentricity_init: float
        Initial eccentricity shared by all embryos.
    inner_edge: float
        Semi-major axis of the innermost embryo [AU].
    spacing: float
        Initial separation between adjacent embryos, in mutual Hill radii.
        Typical values are 5 to 15; beyond roughly 30 the system does not
        go unstable within any useful evolution time, so the run finishes
        with no impacts. Capped at 50 purely to catch an
        order-of-magnitude mistake at configuration load.

        The cap is not the physical limit and does not track it. The
        layout condition has a pole where the requested gap approaches
        the span it is measured across, and its position scales with the
        embryo masses and with the cube root of the stellar mass: near 74
        mutual Hill radii for a pair of ten-Earth-mass embryos around a
        solar-mass star, but near 34 for the same pair around a
        0.1-solar-mass host. A spacing this validator accepts can
        therefore still be too wide for a compact, low-mass-host system.
        The dynamical model applies the exact condition and refuses such
        a layout by name, so that check, not this cap, is what guarantees
        a valid layout.
    density: float
        Uniform bulk density used to convert embryo mass to radius [kg m-3].
    impact_angle: float
        Impact angle [deg]. The impact parameter is its sine.
    evolution_time: float
        Duration of the dynamical evolution [Gyr].
    inner_cutoff: float
        Perihelion inside which an embryo counts as lost to the star [AU].
    selector: str
        Which survivor's impact history PROTEUS follows. 'match_config'
        picks the survivor whose initial mass and orbit are closest to the
        PROTEUS configuration, 'mass' the most massive survivor,
        'semimajoraxis' the survivor whose final orbit is nearest
        ``selector_value`` [AU], and 'id' the embryo with index
        ``selector_value``.
    selector_value: float or None
        Target value for the 'semimajoraxis' and 'id' selectors. Ignored
        otherwise.
    """

    seed: int = field(default=1, validator=ge(0))

    num_planets: int = field(default=10, validator=ge(2))
    masses: list[float] = field(factory=list)
    mass_equal: float = field(default=0.5, validator=gt(0))
    eccentricity_init: float = field(default=0.01, validator=ge(0))

    inner_edge: float = field(default=0.1, validator=gt(0))
    spacing: float = field(default=10.0, validator=[gt(0), le(50.0)])
    density: float = field(default=5500.0, validator=gt(0))
    impact_angle: float = field(default=45.0, validator=ge(0))

    evolution_time: float = field(default=1.0, validator=gt(0))
    inner_cutoff: float = field(default=0.005, validator=gt(0))

    selector: str = field(default='match_config', validator=in_(SELECTORS))
    selector_value: float | str | None = field(default=None, converter=none_if_none)


def valid_accretiondummy(instance, attribute, value):
    """Refuse a timeline path aimed at the module that generates its own."""
    if instance.dummy.timeline_path is None:
        return

    raise ValueError(
        '`accretion.dummy.timeline_path` is not a parameter of the analytical '
        'accretion module, which derives its own impact history. To replay a '
        "timeline from a file, set `accretion.module = 'timeline'` and move the "
        'path to `accretion.timeline.timeline_path`.'
    )


def valid_accretiontimeline(instance, attribute, value):
    if instance.module != 'timeline':
        return

    if instance.timeline.timeline_path is None:
        raise ValueError(
            '`accretion.timeline.timeline_path` must point at an impact timeline file '
            "when accretion.module = 'timeline'"
        )


@define
class AccretionTimeline:
    """Impact timeline replayed from a file.

    Applies a pre-written sequence of impacts instead of deriving one from a
    dynamical model. Every impact consequence is computed exactly as it is for
    a model-derived timeline, so this reproduces a published impact history,
    drives PROTEUS from a history computed elsewhere, or applies a hand-written
    sequence for a controlled experiment.

    Attributes
    ----------
    timeline_path: str or None
        Path to the impact timeline file. Environment variables and ``~``
        are expanded.
    """

    timeline_path: str | None = field(default=None, converter=none_if_none)


@define
class AccretionDummy:
    """Analytical stand-in for a dynamical giant-impact model.

    Builds a self-consistent accretion history from scaling laws rather than
    integrating a system of embryos, in the same spirit as the other dummy
    modules in PROTEUS: fast, deterministic, and dependency-free, at the cost
    of the dynamics.

    The planet approaches an asymptotic mass exponentially, the standard
    picture of an accretion rate that decays as the feeding zone empties.
    Impacts are placed at evenly spaced times and each one delivers the mass
    the law accretes over its interval, so the increments decay with time and
    the largest impact is the first. Radii follow the Noack & Lasbleis (2020)
    mass-radius scaling, collision velocities combine the pair's mutual escape
    velocity with an encounter velocity set by ``eccentricity``, and each
    merged orbit follows from conserving linear momentum through the collision.

    Attributes
    ----------
    mass_accreted: float
        Total mass delivered over the whole timeline [M_earth]. The growth law
        sets how this is distributed in time and between impacts; the
        increments are scaled so they sum to exactly this value.
    num_impacts: int
        Number of impacts in the timeline.
    timescale: float
        E-folding time of the accretion law [yr]. Short compared with
        ``time_last`` concentrates the mass in the first impacts; long
        compared with it spreads the mass evenly.
    time_last: float
        Time of the final impact [yr]. Impacts are spaced evenly from
        ``time_last / num_impacts`` up to this time.
    eccentricity: float
        Encounter eccentricity [1], setting both the approach velocity that
        adds to the mutual escape velocity and the impactor's orbit.
    impact_parameter: float
        Impact parameter of every collision [1], the sine of the impact angle.
        Zero is head-on, one is grazing.
    timeline_path: str or None
        Not a parameter of this module. Present only to reject a configuration
        that sets it here, which asks to replay a file and would otherwise be
        served a generated timeline at default settings. Use
        ``accretion.module = "timeline"`` and ``accretion.timeline.timeline_path``.
    """

    timeline_path: str | None = field(default=None, converter=none_if_none)
    mass_accreted: float = field(default=0.1, validator=gt(0))
    num_impacts: int = field(default=3, validator=ge(1))
    timescale: float = field(default=1.0e6, validator=gt(0))
    time_last: float = field(default=5.0e6, validator=gt(0))
    eccentricity: float = field(default=0.05, validator=[ge(0), lt(1)])
    impact_parameter: float = field(default=0.5, validator=[ge(0), le(1)])


def valid_impactor_volatiles(instance, attribute, value):
    """Refuse ppmw budgets that the selected content mode would ignore."""
    if instance.impactor_volatiles == 'ppmw':
        return
    set_fields = [
        f'impactor_{e}_ppmw'
        for e in ('H', 'C', 'N', 'S', 'O')
        if getattr(instance, f'impactor_{e}_ppmw') > 0.0
    ]
    if set_fields:
        raise ValueError(
            f'`accretion.{"`, `accretion.".join(set_fields)}` set, but the ppmw '
            f"budgets are read only when accretion.impactor_volatiles = 'ppmw' "
            f"(currently '{instance.impactor_volatiles}'). Select the ppmw mode "
            'or remove the budgets.'
        )


def valid_impactor_budget_total(instance, attribute, value):
    """Refuse a volatile budget that exceeds the impactor's own mass.

    The budgets are fractions of the impactor mass, and the rock the impact
    adds to the interior anchor is what is left once they are taken out. A
    total at or above 1e6 ppmw leaves nothing or less than nothing, so the
    impact would shrink the anchor while still crediting the full volatile
    mass to the planet. Whole-planet mass stays self-consistent through that,
    so nothing downstream can detect it.
    """
    if instance.impactor_volatiles != 'ppmw':
        return
    budgets = {e: getattr(instance, f'impactor_{e}_ppmw') for e in ('H', 'C', 'N', 'S', 'O')}
    total = sum(budgets.values())
    if total >= 1.0e6:
        named = ', '.join(f'{e}={v:g}' for e, v in budgets.items() if v > 0.0)
        raise ValueError(
            f'The impactor volatile budgets sum to {total:g} ppmw ({total / 1.0e4:.3g}% '
            f'of the impactor mass), which leaves no rock to accrete ({named}). '
            'Each budget is a fraction of the impactor mass, so they must total '
            'below 1e6 ppmw.'
        )


@define
class Accretion:
    """Giant-impact accretion, delivery, and module selection.

    An impact grows the planet, delivers volatiles, re-melts the mantle,
    strips part of the atmosphere, and moves the orbit. The impactor
    volatile content is set by ``impactor_volatiles``: "dry" impactors
    (the default) add silicate and iron mass only, "match_planet"
    impactors carry the planet's own formation composition, and "ppmw"
    impactors carry the per-element budgets configured below.

    The mantle re-melt is a thermodynamic reset, not an energy deposition.
    It re-applies the run's ``planet.temperature_mode`` initial condition to
    the whole mantle, so the heat it injects is set by the mantle's own state
    and mass rather than by the energy the collision carried. The two agree in
    order of magnitude for a large impact onto a mantle that has cooled
    appreciably, which is the regime this coupling targets, and diverge outside
    it: a mantle already near the initial condition absorbs almost nothing, and
    a cool mantle struck by a small impactor absorbs far more than the impact
    supplied. On the Aragog interior, which resolves an entropy profile and so
    can quantify the injection, the run reports it as a fraction of the impact
    kinetic energy and warns when that fraction leaves the physically expected
    band. That report is the only check available, because the conservation
    residual adds the injection to both of its sides and so stays closed for
    any value. The scalar interiors reset a temperature rather than a profile
    and book no injection, so neither the report nor the warning applies there.
    Interpret the thermal response to an impact as a property of the chosen
    initial condition rather than of the collision.

    Attributes
    ----------
    module: str or None
        Accretion module to use. Choices: None, "dummy", "timeline", "morrigan".
    morrigan: Morrigan
        Parameters for the Morrigan giant-impact module.
    dummy: AccretionDummy
        Parameters for the analytical dummy module.
    timeline: AccretionTimeline
        Parameters for replaying an impact timeline from file.
    time_offset: float
        Offset applied to every impact time when mapping the timeline onto
        the PROTEUS time axis [yr]. A dynamical model measures time from
        disk dispersal, while PROTEUS measures it from the start of its
        own evolution. Impacts that still land at or before the start of
        the run are discarded with a warning, and their mass is not
        applied anywhere: the configured planet mass and orbit define the
        initial state on their own.
    impactor_volatiles: str
        Where each impactor's volatile content comes from. Choices:
        "dry" (impactors carry rock and iron only), "match_planet" (every
        impactor carries the planet's own initial fractional volatile
        abundances, scaled to the impactor mass, on the assumption that
        all embryos co-formed from the same disk material), "ppmw" (the
        per-element ``impactor_<e>_ppmw`` budgets below). The content is
        split into an atmospheric and a dissolved part by mirroring the
        planet's own partitioning at impact time; the atmospheric part
        loses the same collision fraction that strips the target's
        atmosphere and the remainder of the content is delivered.
    impactor_H_ppmw: float
        Hydrogen carried by each impactor [ppmw of impactor mass].
    impactor_C_ppmw: float
        Carbon carried by each impactor [ppmw of impactor mass].
    impactor_N_ppmw: float
        Nitrogen carried by each impactor [ppmw of impactor mass].
    impactor_S_ppmw: float
        Sulfur carried by each impactor [ppmw of impactor mass].
    impactor_O_ppmw: float
        Oxygen carried by each impactor [ppmw of impactor mass].
    atmloss_module: str or None
        How impact atmosphere loss is computed. Choices: None (no impact
        atmosphere loss at all: the target keeps its atmosphere and a
        volatile-bearing impactor delivers its whole content), "constant"
        (the fixed fraction below), "zephyrus" (the giant-impact erosion
        scaling law of Kegerreis et al. 2020, evaluated by
        ``zephyrus.collision.mass_loss`` from each impact's collision
        parameters). One fraction governs both bodies at each impact: the
        target loses that fraction of its atmosphere, and a
        volatile-bearing impactor loses the same fraction of its
        atmospheric part and delivers the remainder. PROTEUS itself ships
        no impact loss physics.
    atmloss_frac: float
        Fraction of the atmosphere removed by each impact when
        ``atmloss_module = "constant"`` [0-1]. Applies to the target's
        atmosphere and to the impactor's atmospheric part alike.
    """

    module: str | None = field(
        default='none',
        validator=in_((None, 'dummy', 'timeline', 'morrigan')),
        converter=none_if_none,
    )

    morrigan: Morrigan = field(factory=Morrigan, validator=valid_morrigan)
    dummy: AccretionDummy = field(factory=AccretionDummy, validator=valid_accretiondummy)
    timeline: AccretionTimeline = field(
        factory=AccretionTimeline, validator=valid_accretiontimeline
    )

    time_offset: float = field(default=0.0)

    # Impactor volatile content source. 'dry' impactors add silicate and
    # iron mass only, so the planet's bulk volatile concentration falls by
    # dilution as it grows; 'match_planet' scales the planet's initial
    # fractional abundances to the impactor; 'ppmw' uses the fields below.
    impactor_volatiles: str = field(
        default='dry',
        validator=in_(('dry', 'match_planet', 'ppmw')),
    )

    # Per-element impactor content, read when impactor_volatiles = 'ppmw'.
    impactor_H_ppmw: float = field(default=0.0, validator=ge(0))
    impactor_C_ppmw: float = field(default=0.0, validator=ge(0))
    impactor_N_ppmw: float = field(default=0.0, validator=ge(0))
    impactor_S_ppmw: float = field(default=0.0, validator=ge(0))
    # The cross-field check rides on the LAST ppmw field: attrs runs field
    # validators in definition order, so only here are the mode selector and
    # every budget it guards populated.
    impactor_O_ppmw: float = field(
        default=0.0,
        validator=[ge(0), valid_impactor_volatiles, valid_impactor_budget_total],
    )

    # Impact atmosphere loss. Disabled by default; the constant module
    # applies a fixed fraction, the zephyrus module the Kegerreis et al.
    # (2020) scaling law from each impact's collision parameters.
    atmloss_module: str | None = field(
        default='none',
        validator=in_((None, 'constant', 'zephyrus')),
        converter=none_if_none,
    )
    atmloss_frac: float = field(default=0.0, validator=[ge(0), le(1)])

    @property
    def delivers_volatiles(self) -> bool:
        """Can an impactor carry any volatile mass under the selected mode?"""
        if self.impactor_volatiles == 'dry':
            return False
        if self.impactor_volatiles == 'match_planet':
            return True
        return any(getattr(self, f'impactor_{e}_ppmw') > 0.0 for e in ('H', 'C', 'N', 'S', 'O'))
