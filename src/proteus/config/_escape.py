from __future__ import annotations

from attrs import define, field
from attrs.validators import ge, in_, le

from ._converters import none_if_none


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
    rate: float = field(default=0.0, validator=ge(0))

def valid_escapeboreas(instance, attribute, value):
    if instance.module != "boreas":
        return

@define
class EscapeBoreas:
    """BOREAS escape module.

    Attributes
    ----------
    efficiency: float
        Energy efficiency factor.
    alpha_rec: float
        Recombination coefficient [cm3 s-1]
    sigma_XUV: float
        Absorption cross-section in XUV [cm2 molecule-1]
    kappa_H2O: float
        H2O opacity in XUV [cm2 g-1]
    kappa_H2: float
        H2 opacity in XUV [cm2 g-1]
    kappa_O2: float
        O2 opacity in XUV [cm2 g-1]
    kappa_CO2: float
        CO2 opacity in XUV [cm2 g-1]
    kappa_CO: float
        CO opacity in XUV [cm2 g-1]
    kappa_CH4: float
        CH4 opacity in XUV [cm2 g-1]
    kappa_N2: float
        N2 opacity in XUV [cm2 g-1]
    kappa_NH3: float
        NH3 opacity in XUV [cm2 g-1]
    """
    efficiency: float = field(default=0.1,      validator=(ge(0), le(1)))
    alpha_rec: float  = field(default=2.6e-13,  validator=ge(0))
    sigma_XUV: float  = field(default=1.89e-18, validator=ge(0))
    kappa_H2:  float  = field(default=1e-2,   validator=ge(0))
    kappa_H2O: float  = field(default=1e-0,   validator=ge(0))
    kappa_O2:  float  = field(default=1e-0,   validator=ge(0))
    kappa_CO2: float  = field(default=1e-0,   validator=ge(0))
    kappa_CO:  float  = field(default=1e-0,   validator=ge(0))
    kappa_CH4: float  = field(default=1e-0,   validator=ge(0))
    kappa_N2:  float  = field(default=1e-0,   validator=ge(0))
    kappa_NH3: float  = field(default=1e-0,   validator=ge(0))

def valid_reservoir(instance, attribute, value):

    ress = ('bulk','outgas', 'pxuv')
    if instance.reservoir not in ress:
        raise ValueError(f"Escape reservoir must be one of: {ress}")

    if (instance.module == "boreas") and (instance.reservoir != "pxuv"):
        raise ValueError("Escape reservoir must be 'pxuv' when using module 'boreas'")

    if (instance.module != "boreas") and (instance.reservoir == "pxuv"):
        raise ValueError("Escape reservoir cannot be 'pxuv' unless using module 'boreas'")

@define
class Escape:
    """Escape parameters and module selection.

    Attributes
    ----------
    reservoir: str
        Reservoir representing the escaping composition. Choices: bulk, outgas.
    module: str | None
        Escape module to use. Choices: "none", "dummy", "zephyrus".
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
    def xuv_defined_by_radius(self) -> int:
        """Does Rxuv define the escape level?

        If it does, return True.
        If the escape level is instead set by Pxuv, then return False.
        """
        if self.module == 'boreas':
            return True
        else:
            return False
