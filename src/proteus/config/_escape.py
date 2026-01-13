from __future__ import annotations

from attrs import define, field
from attrs.validators import ge, in_, le

from ._converters import none_if_none, zero_if_none


def valid_zephyrus(instance, attribute, value):
    if instance.module != "zephyrus":
        return

    Pxuv = instance.zephyrus.Pxuv
    if (not Pxuv) or (Pxuv < 0) or (Pxuv > 10):
        raise ValueError("`zephyrus.Pxuv` must be >0 and < 10 bar")

    efficiency = instance.zephyrus.efficiency
    if (not efficiency) or (efficiency < 0) or (efficiency > 1):
        raise ValueError("`zephyrus.efficiency` must be >=0 and <=1")

@define
class Zephyrus:
    """Parameters for Zephyrus module.

    Attributes
    ----------
    Pxuv: float
        Pressure at which XUV radiation become opaque in the planetary atmosphere [bar]
    efficiency: float
        Escape efficiency factor
    tidal: bool
        Tidal contribution enabled
    """
    Pxuv: float       = field(default=5e-5, validator=ge(0))
    efficiency: float = field(default=0.1,  validator=(ge(0), le(1)))
    tidal: bool       = field(default=False)

def valid_escapedummy(instance, attribute, value):
    if instance.module != "dummy":
        return

    rate = instance.dummy.rate
    if (not rate) or (rate < 0) :
        raise ValueError("`escape.dummy.rate` must be >0")

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
    if instance.module != "boreas":
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
    efficiency: float = field(default=0.1,      validator=(ge(0), le(1)))

    sigma_H: float    = field(default=1.89e-18, validator=ge(0))
    sigma_O: float    = field(default=2.00e-18, validator=ge(0))
    sigma_C: float    = field(default=2.50e-18, validator=ge(0))
    sigma_N: float    = field(default=3.00e-18, validator=ge(0))
    sigma_S: float    = field(default=6.00e-18, validator=ge(0))

    kappa_H2:  float  = field(default=1e-2,     validator=ge(0))
    kappa_H2O: float  = field(default=1e-0,     validator=ge(0))
    kappa_O2:  float  = field(default=1e-0,     validator=ge(0))
    kappa_CO2: float  = field(default=1e-0,     validator=ge(0))
    kappa_CO:  float  = field(default=1e-0,     validator=ge(0))
    kappa_CH4: float  = field(default=1e-0,     validator=ge(0))
    kappa_N2:  float  = field(default=1e-0,     validator=ge(0))
    kappa_NH3: float  = field(default=1e-0,     validator=ge(0))
    kappa_H2S: float  = field(default=1e-0,     validator=ge(0))
    kappa_SO2: float  = field(default=1e-0,     validator=ge(0))
    kappa_S2: float   = field(default=1e-0,     validator=ge(0))


def valid_reservoir(instance, attribute, value):
    ress = ('bulk','outgas')
    if instance.reservoir not in ress:
        raise ValueError(f"Escape reservoir must be one of: {ress}")

@define
class Escape:
    """Escape parameters and module selection.

    Attributes
    ----------
    reservoir: str
        Escaping composition when not doing fractionation. Choices: bulk, outgas.
    module: str | None
        Escape module to use. Choices: None, "dummy", "zephyrus", "boreas".
    zephyrus: Zephyrus
        Parameters for zephyrus module.
    dummy: EscapeDummy
        Parameters for dummy escape module.
    boreas: EscapeBoreas
        Parameters for BOREAS escape module.
    """

    module: str | None = field(
        validator=in_((None, 'dummy', 'zephyrus', 'boreas')), converter=none_if_none
        )

    zephyrus: Zephyrus   = field(factory=Zephyrus,      validator=valid_zephyrus)
    dummy: EscapeDummy   = field(factory=EscapeDummy,   validator=valid_escapedummy)
    boreas: EscapeBoreas = field(factory=EscapeBoreas,  validator=valid_escapeboreas)

    reservoir: str = field(default='outgas', validator=valid_reservoir)

    @property
    def xuv_defined_by_radius(self) -> bool:
        """Does Rxuv define the escape level?

        If the escape level is defined by constant Pxuv, then return False. This depends
        on the escape module used. BOREAS calculates both Pxuv and Rxuv, while the default
        assumes that Pxuv is constant, which is used to find Rxuv from the r(p) profile.
        """
        if self.module == 'boreas':
            return True
        else:
            return False
