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


@define
class Accretion:
    """Giant-impact accretion, delivery, and module selection.

    An impact grows the planet, delivers volatiles, re-melts the mantle,
    strips part of the atmosphere, and moves the orbit. The impactor
    composition below sets how much volatile mass each impactor carries;
    it defaults to zero, so impactors are dry unless delivery is
    requested.

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
        How the fraction of atmosphere lost to each impact is computed.
        Choices: None (no impact atmosphere loss), "constant" (the fixed
        fraction below). A ZEPHYRUS collision-loss law will become a
        further choice when available; PROTEUS itself ships no impact
        loss physics.
    atmloss_frac: float
        Fraction of the atmosphere removed by each impact when
        ``atmloss_module = "constant"`` [0-1].
    """

    module: str | None = field(
        default='none',
        validator=in_((None, 'dummy', 'morrigan')),
        converter=none_if_none,
    )

    morrigan: Morrigan = field(factory=Morrigan, validator=valid_morrigan)
    dummy: AccretionDummy = field(factory=AccretionDummy, validator=valid_accretiondummy)

    time_offset: float = field(default=0.0)

    # Impactor volatile content, applied to every impact. Zero means the
    # impactor adds silicate and iron mass only, so the planet's bulk
    # volatile concentration falls by dilution as it grows.
    impactor_H_ppmw: float = field(default=0.0, validator=ge(0))
    impactor_C_ppmw: float = field(default=0.0, validator=ge(0))
    impactor_N_ppmw: float = field(default=0.0, validator=ge(0))
    impactor_S_ppmw: float = field(default=0.0, validator=ge(0))
    impactor_O_ppmw: float = field(default=0.0, validator=ge(0))

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
        """Does any impactor volatile budget exceed zero?"""
        return any(getattr(self, f'impactor_{e}_ppmw') > 0.0 for e in ('H', 'C', 'N', 'S', 'O'))
