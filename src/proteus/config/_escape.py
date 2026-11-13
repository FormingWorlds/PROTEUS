from __future__ import annotations

from attrs import define, field
from attrs.validators import ge, gt, in_, le

from ._converters import none_if_none, zero_if_none


def valid_zephyrus(instance, attribute, value):
    """Validate ZEPHYRUS settings.

    Pxuv must lie in (0, 10] bar and the efficiency in [0, 1]. Under the
    dispatcher scheme the full settings block is checked by ZEPHYRUS itself,
    so a bad value fails at load with the name of the field.
    """
    if instance.module != 'zephyrus':
        return

    Pxuv = instance.zephyrus.Pxuv
    if Pxuv is None or Pxuv <= 0 or Pxuv > 10:
        raise ValueError('`zephyrus.Pxuv` must be >0 and <= 10 bar')

    efficiency = instance.zephyrus.efficiency
    if efficiency is None or efficiency < 0 or efficiency > 1:
        raise ValueError('`zephyrus.efficiency` must be >=0 and <=1')

    if getattr(instance.zephyrus, 'scheme', 'el') == 'dispatcher':
        from proteus.escape.dispatcher import dispatch_settings

        try:
            dispatch_settings(instance.zephyrus).validate()
        except ValueError as exc:
            raise ValueError(f'escape.zephyrus: {exc}') from exc


def valid_reservoir(instance, attribute, value):
    """Check the escape reservoir against the escape model.

    ``pxuv`` takes the escaping composition from the wind base, which only the
    ZEPHYRUS dispatcher computes. The dispatcher sets that composition itself,
    so a mass-fraction reservoir (``bulk`` or ``outgas``) cannot be combined
    with it. Unset resolves to ``pxuv`` under the dispatcher and ``outgas``
    otherwise.
    """
    ress = ('bulk', 'outgas', 'pxuv')
    if value is not None and value not in ress:
        raise ValueError(f'Escape reservoir must be one of: {ress}')
    dispatcher = (
        instance.module == 'zephyrus'
        and getattr(instance.zephyrus, 'scheme', 'el') == 'dispatcher'
    )
    if value == 'pxuv' and not dispatcher:
        raise ValueError(
            'escape.reservoir = "pxuv" requires escape.module = "zephyrus" with '
            'escape.zephyrus.scheme = "dispatcher", which sets the escaping composition '
            'at the wind base.'
        )
    if dispatcher and value in ('bulk', 'outgas'):
        raise ValueError(
            f'escape.reservoir = "{value}" cannot be combined with the ZEPHYRUS '
            'dispatcher, which sets the escaping composition itself. Leave '
            'escape.reservoir unset (or "pxuv"), or set escape.zephyrus.scheme = "el".'
        )


@define
class Zephyrus:
    """Parameters for Zephyrus module.

    Attributes
    ----------
    scheme: str
        Escape prescription. "dispatcher" selects the escape regime per state
        (boil-off, hydrodynamic wind, hydrostatic Jeans escape, Roche-lobe
        overflow) and returns a rate per element; "el" is energy-limited
        escape alone, split over the elements by mass fraction.
    Pxuv: float
        Pressure at which XUV radiation becomes opaque [bar]. Used by "el",
        and by the dispatcher when ``base_method = "fixed_pressure"``.
    efficiency: float
        Escape efficiency factor.
    tidal: bool | None
        Tidal reduction of the escape barrier. Unset resolves to on under the
        dispatcher and off under "el".
    base_method: str
        Dispatcher wind base: "lopez" (where the base is optically thick to
        XUV, Lopez 2017) or "fixed_pressure" (at ``Pxuv``).
    base_out_of_range: str
        Dispatcher wind base above the top of the atmosphere profile: "clamp"
        to the top level, or "extend" the profile above it with the Bates upper
        structure of the hydrostatic branch.
    P_photo: float
        Photospheric level of the dispatcher [bar]: the radius of the
        energy-limited rate and the boil-off launch level.
    kappa_photo: float
        Grey photospheric opacity [m2 kg-1]; the boil-off rate scales as its
        inverse.
    kn_crit: float
        Sonic-point Knudsen number above which the wind is rarefied and the
        hydrostatic branch applies.
    kn_hysteresis: float
        Factor by which the previous step's side of the Knudsen switch widens
        the threshold, so an evolving track does not alternate between
        branches. 1 disables it.
    gate: str
        Escape temperature that tests exosphere stability: "neutral" or "plasma".
    efficiency_mode: str
        "fixed" uses ``efficiency``; "caldiroli" uses the fit of Caldiroli et
        al. (2022).
    T_exo_mode: str
        Exobase temperature of the hydrostatic branch: "prescribed" uses
        ``T_exo``, "thermostat" estimates it from heating against cooling.
    T_exo: float
        Prescribed exobase temperature [K].
    cool_atomic: bool
        Atomic line cooling in the wind thermostat.
    cool_co2_band: bool
        CO2 15 micron band cooling in the wind thermostat.
    cool_o_finestructure: bool
        Atomic oxygen fine-structure cooling in the wind thermostat.
    cool_recombination: bool
        Recombination cooling in the wind thermostat.
    fractionate: bool
        Split the bulk rate over the elements with the multi-species
        diffusion closure; off splits by the abundance at the wind base.
    nozzle_temperature: str
        Temperature of the L1 overflow rate: "photospheric" or "wind".
    residual_mode: str
        "off" keeps the interior-luminosity-capped bolometric rate out of the
        comparison past the boil-off threshold; "luminosity_capped" admits it.
    photon_limit: bool
        Cap the XUV wind at one escaping ion per absorbed photon.
    recombination_limit: bool
        Admit the recombination-limited wind rate.
    lambda_crit: float
        Restricted Jeans parameter below which boil-off applies.
    kzz: float
        Eddy diffusion coefficient [m2 s-1] of the hydrostatic branch.
    T_anchor_decades: float
        Pressure decades at the top of the atmosphere column over which the
        temperature the hydrostatic thermosphere starts from is averaged. The
        optically thin top layers have a poorly constrained temperature. 0
        takes the topmost level.
    hydrostatic_levels_min: int
        First quadrature grid of the diffusion-limited supply integrals.
    hydrostatic_levels_max: int
        Refinement ceiling of those integrals, at least four times the minimum.
    hydrostatic_rtol: float
        Relative change in the rate at which the refinement stops.
    """

    scheme: str = field(default='dispatcher', validator=in_(('dispatcher', 'el')))
    Pxuv: float = field(default=5e-5, validator=ge(0))
    efficiency: float = field(default=0.1, validator=(ge(0), le(1)))
    tidal: bool | None = field(default=None, converter=none_if_none)

    base_method: str = field(default='lopez', validator=in_(('lopez', 'fixed_pressure')))
    base_out_of_range: str = field(default='clamp', validator=in_(('clamp', 'extend')))
    P_photo: float = field(default=0.02, validator=gt(0))
    kappa_photo: float = field(default=1e-2, validator=gt(0))
    kn_crit: float = field(default=1.0, validator=gt(0))
    kn_hysteresis: float = field(default=1.5, validator=ge(1))
    gate: str = field(default='neutral', validator=in_(('neutral', 'plasma')))
    efficiency_mode: str = field(default='fixed', validator=in_(('fixed', 'caldiroli')))
    T_exo_mode: str = field(default='prescribed', validator=in_(('prescribed', 'thermostat')))
    T_exo: float = field(default=1000.0, validator=gt(0))
    cool_atomic: bool = field(default=True)
    cool_co2_band: bool = field(default=True)
    cool_o_finestructure: bool = field(default=True)
    cool_recombination: bool = field(default=True)
    fractionate: bool = field(default=True)
    nozzle_temperature: str = field(
        default='photospheric', validator=in_(('photospheric', 'wind'))
    )
    residual_mode: str = field(default='off', validator=in_(('off', 'luminosity_capped')))
    photon_limit: bool = field(default=True)
    recombination_limit: bool = field(default=True)
    lambda_crit: float = field(default=20.0, validator=gt(0))
    kzz: float = field(default=3.0e2, validator=gt(0))
    T_anchor_decades: float = field(default=3.0, validator=ge(0))
    hydrostatic_levels_min: int = field(default=200, validator=ge(2))
    hydrostatic_levels_max: int = field(default=3200, validator=ge(8))
    hydrostatic_rtol: float = field(default=1.0e-2, validator=gt(0))


def valid_escapedummy(instance, attribute, value):
    """Dummy escape requires a non-negative escape rate."""
    if instance.module != 'dummy':
        return

    rate = instance.dummy.rate
    if rate is None or rate < 0:
        raise ValueError('`escape.dummy.rate` must be >= 0')


@define
class EscapeDummy:
    """Dummy module.

    Attributes
    ----------
    rate: float
        Bulk unfractionated escape rate [kg s-1]
    """

    rate: float = field(default=0.0, validator=ge(0), converter=zero_if_none)


def valid_escapeboreas(instance, attribute, value):
    """Hook for BOREAS cross-field checks; no active constraints at present."""
    if instance.module != 'boreas':
        return


@define
class EscapeBoreas:
    """BOREAS escape module.

    Attributes
    ----------
    fractionate: bool
        Enable elemental fractionation in outflow?
    efficiency: float
        Energy efficiency factor.
    sigma_H: float
        Absorption cross-section of H in XUV [cm2]
    sigma_O: float
        Absorption cross-section of O in XUV [cm2]
    sigma_C: float
        Absorption cross-section of C in XUV [cm2]
    sigma_N: float
        Absorption cross-section of N in XUV [cm2]
    sigma_S: float
        Absorption cross-section of S in XUV [cm2]
    kappa_H2O: float
        Grey H2O opacity in IR [cm2 g-1]
    kappa_H2: float
        Grey H2 opacity in IR [cm2 g-1]
    kappa_O2: float
        Grey O2 opacity in IR [cm2 g-1]
    kappa_CO2: float
        Grey CO2 opacity in IR [cm2 g-1]
    kappa_CO: float
        Grey CO opacity in IR [cm2 g-1]
    kappa_CH4: float
        Grey CH4 opacity in IR [cm2 g-1]
    kappa_N2: float
        Grey N2 opacity in IR [cm2 g-1]
    kappa_NH3: float
        Grey NH3 opacity in IR [cm2 g-1]
    kappa_H2S: float
        Grey H2S opacity in IR [cm2 g-1]
    kappa_SO2: float
        Grey SO2 opacity in IR [cm2 g-1]
    kappa_S2: float
        Grey S2 opacity in IR [cm2 g-1]
    """

    fractionate: bool = field(default=True)
    efficiency: float = field(default=0.1, validator=(ge(0), le(1)))

    sigma_H: float = field(default=1.89e-18, validator=ge(0))
    sigma_O: float = field(default=2.00e-18, validator=ge(0))
    sigma_C: float = field(default=2.50e-18, validator=ge(0))
    sigma_N: float = field(default=3.00e-18, validator=ge(0))
    sigma_S: float = field(default=6.00e-18, validator=ge(0))

    kappa_H2: float = field(default=1e-2, validator=ge(0))
    kappa_H2O: float = field(default=1e-0, validator=ge(0))
    kappa_O2: float = field(default=1e-0, validator=ge(0))
    kappa_CO2: float = field(default=1e-0, validator=ge(0))
    kappa_CO: float = field(default=1e-0, validator=ge(0))
    kappa_CH4: float = field(default=1e-0, validator=ge(0))
    kappa_N2: float = field(default=1e-0, validator=ge(0))
    kappa_NH3: float = field(default=1e-0, validator=ge(0))
    kappa_H2S: float = field(default=1e-0, validator=ge(0))
    kappa_SO2: float = field(default=1e-0, validator=ge(0))
    kappa_S2: float = field(default=1e-0, validator=ge(0))


@define
class Escape:
    """Escape parameters and module selection.

    Attributes
    ----------
    reservoir: str | None
        Composition of the escaping gas: "bulk" (whole-planet element mass
        fractions), "outgas" (atmospheric element mass fractions), or "pxuv"
        (the per-element rates of the ZEPHYRUS dispatcher, set at the wind
        base). Unset resolves to "pxuv" under the dispatcher and "outgas"
        otherwise.
    module: str | None
        Escape module to use. Choices: None, "dummy", "zephyrus", "boreas".
    zephyrus: Zephyrus
        Parameters for zephyrus module.
    dummy: EscapeDummy
        Parameters for dummy escape module.
    boreas: EscapeBoreas
        Parameters for BOREAS escape module.
    hill_clamp: bool | None
        Limit the XUV level to the Hill radius. Gas beyond the Hill radius is
        not bound to the planet, so an XUV radius outside it sizes the escape
        cross-section with material the planet does not hold. Unset resolves
        to off under the ZEPHYRUS dispatcher, whose Roche-lobe screen and L1
        overflow rate treat that geometry, and on otherwise.
    hill_clamp_frac: float
        Fraction of the Hill radius used as that limit, when hill_clamp is
        enabled.
    step_max_frac: float
        Largest share of the escapable reservoir a single step may remove. The
        bulk rate is sized without reference to how much mass remains, so over
        a long step it can ask for many times the reservoir.

        The default guards reduced compositions rather than limiting the grid.
        Measured on production cases, every one at IW-5 asks for more than it at
        some step and the worst for eighty times the reservoir, while cases from
        IW-1 to IW+5 stay under a thousandth of it and never meet the cap at
        all. A larger value lets one step take more of what is left, so a
        reduced case draws its reservoir down in fewer and coarser steps. The
        cap keeps binding at every setting in this range, because an unchanged
        rate settles at ``max_frac / (1 - max_frac)``, which is above the cap
        for every value it accepts.
    step_dt_floor_frac: float
        How far below ``params.dt.minimum`` a capped step may shorten the next
        one. The floor otherwise overrides the shortened step on 94 % of the
        steps the cap binds on, leaving the reduction inert where it is needed.
    """

    module: str | None = field(
        default='zephyrus',
        validator=in_((None, 'dummy', 'zephyrus', 'boreas')),
        converter=none_if_none,
    )

    zephyrus: Zephyrus = field(factory=Zephyrus, validator=valid_zephyrus)
    dummy: EscapeDummy = field(factory=EscapeDummy, validator=valid_escapedummy)
    boreas: EscapeBoreas = field(factory=EscapeBoreas, validator=valid_escapeboreas)

    reservoir: str | None = field(
        default=None, validator=valid_reservoir, converter=none_if_none
    )

    hill_clamp: bool | None = field(default=None, converter=none_if_none)
    hill_clamp_frac: float = field(default=1.0, validator=(gt(0.0), le(1.0)))

    step_max_frac: float = field(default=0.25, validator=(gt(0.0), le(1.0)))
    step_dt_floor_frac: float = field(default=1.0e-3, validator=(gt(0.0), le(1.0)))

    def __attrs_post_init__(self):
        """Resolve the settings whose default depends on the escape scheme."""
        dispatcher = self.uses_dispatcher
        if self.reservoir is None:
            self.reservoir = 'pxuv' if dispatcher else 'outgas'
        if self.hill_clamp is None:
            self.hill_clamp = not dispatcher
        if self.zephyrus.tidal is None:
            self.zephyrus.tidal = dispatcher

    @property
    def uses_dispatcher(self) -> bool:
        """Is escape computed by the ZEPHYRUS regime dispatcher?"""
        return self.module == 'zephyrus' and self.zephyrus.scheme == 'dispatcher'

    @property
    def xuv_defined_by_radius(self) -> bool:
        """Does Rxuv define the escape level?

        If the escape level is defined by constant Pxuv, then return False. This depends
        on the escape module used. BOREAS calculates both Pxuv and Rxuv, while the default
        assumes that Pxuv is constant, which is used to find Rxuv from the r(p) profile.
        """
        return self.module == 'boreas'
