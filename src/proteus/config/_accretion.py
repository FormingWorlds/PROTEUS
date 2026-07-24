from __future__ import annotations

from attr.validators import ge, gt, in_, le
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
    spacing: float = field(default=10.0, validator=gt(0))
    density: float = field(default=5500.0, validator=gt(0))
    impact_angle: float = field(default=45.0, validator=ge(0))

    evolution_time: float = field(default=1.0, validator=gt(0))
    inner_cutoff: float = field(default=0.005, validator=gt(0))

    selector: str = field(default='match_config', validator=in_(SELECTORS))
    selector_value: float | str | None = field(default=None, converter=none_if_none)


def valid_accretiondummy(instance, attribute, value):
    if instance.module != 'dummy':
        return

    if instance.dummy.timeline_path is None:
        raise ValueError(
            '`accretion.dummy.timeline_path` must point at an impact timeline file '
            "when accretion.module = 'dummy'"
        )


@define
class AccretionDummy:
    """Dummy accretion module, driven by a pre-written impact timeline.

    Reads a timeline file instead of running a dynamical model, so impact
    consequences can be exercised against a known event sequence.

    Attributes
    ----------
    timeline_path: str or None
        Path to the impact timeline file. Environment variables and ``~``
        are expanded.
    """

    timeline_path: str | None = field(default=None, converter=none_if_none)


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


@define
class Accretion:
    """Giant-impact accretion, delivery, and module selection.

    An impact grows the planet, delivers volatiles, re-melts the mantle,
    strips part of the atmosphere, and moves the orbit. The impactor
    volatile content is set by ``impactor_volatiles``: "dry" impactors
    (the default) add silicate and iron mass only, "match_planet"
    impactors carry the planet's own formation composition, and "ppmw"
    impactors carry the per-element budgets configured below.

    Attributes
    ----------
    module: str or None
        Accretion module to use. Choices: None, "dummy", "morrigan".
    morrigan: Morrigan
        Parameters for the Morrigan giant-impact module.
    dummy: AccretionDummy
        Parameters for the timeline-driven dummy module.
    time_offset: float
        Offset applied to every impact time when mapping the timeline onto
        the PROTEUS time axis [yr]. A dynamical model measures time from
        disk dispersal, while PROTEUS measures it from the start of its
        own evolution. Impacts landing before the start of the run are
        folded into the initial condition.
    impactor_volatiles: str
        Where each impactor's volatile content comes from. Choices:
        "dry" (impactors carry rock and iron only), "match_planet" (every
        impactor carries the planet's own initial fractional volatile
        abundances, scaled to the impactor mass, on the assumption that
        all embryos co-formed from the same disk material), "ppmw" (the
        per-element ``impactor_<e>_ppmw`` budgets below). The content is
        split into an atmospheric and a dissolved part by mirroring the
        planet's own partitioning at impact time; with impact atmosphere
        loss active the atmospheric part is lost with the collision and
        only the dissolved part is delivered.
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
        (the fixed target fraction below). Whenever a loss module is
        active, the impactor's own atmospheric volatiles are fully lost
        with the collision, independently of the fraction below. A
        ZEPHYRUS collision-loss law will become a further choice when
        available; PROTEUS itself ships no impact loss physics.
    atmloss_frac: float
        Fraction of the TARGET's atmosphere removed by each impact when
        ``atmloss_module = "constant"`` [0-1]. Does not scale the
        impactor-side loss.
    """

    module: str | None = field(
        default='none',
        validator=in_((None, 'dummy', 'morrigan')),
        converter=none_if_none,
    )

    morrigan: Morrigan = field(factory=Morrigan, validator=valid_morrigan)
    dummy: AccretionDummy = field(factory=AccretionDummy, validator=valid_accretiondummy)

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
    impactor_O_ppmw: float = field(default=0.0, validator=[ge(0), valid_impactor_volatiles])

    # Impact atmosphere loss. Disabled by default; the constant module is a
    # placeholder with the call shape of the coming ZEPHYRUS collision law.
    atmloss_module: str | None = field(
        default='none',
        validator=in_((None, 'constant')),
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
