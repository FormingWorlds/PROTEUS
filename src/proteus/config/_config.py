from __future__ import annotations

import logging

import tomlkit
from attrs import asdict, define, field

from ._accretion import Accretion
from ._atmos_chem import AtmosChem
from ._atmos_clim import AtmosClim
from ._converters import dict_replace_none
from ._escape import Escape
from ._interior import Interior
from ._observe import Observe
from ._orbit import Orbit
from ._outgas import Outgas
from ._params import Params
from ._planet import Planet
from ._star import Star
from ._struct import Struct

log = logging.getLogger('fwl.' + __name__)


def spada_zephyrus(instance, attribute, value):
    # using zephyrus
    #     zephyrus requires MORS + Spada
    if (instance.escape.module == 'zephyrus') and not (
        (instance.star.module == 'mors') and (instance.star.mors.tracks == 'spada')
    ):
        raise ValueError('ZEPHYRUS must be used with MORS and the Spada evolution tracks')


def instmethod_dummy(instance, attribute, value):
    # Instellation method 'inst' only support for dummy star module
    if (instance.orbit.instellation_method == 'inst') and not (instance.star.module == 'dummy'):
        raise ValueError("Instellation method can only be 'inst' when star.module=dummy ")


def instmethod_evolve(instance, attribute, value):
    # Orbital evolution not supported when installation_method is 'inst'
    if (instance.orbit.instellation_method == 'inst') and instance.orbit.evolve:
        raise ValueError(
            "Planet orbital evolution not supported for `instellation_method='inst'`"
        )


def satellite_evolve(instance, attribute, value):
    # Planetary orbital evolution not supported when also modelling satellite
    if instance.orbit.satellite and instance.orbit.evolve:
        raise ValueError(
            'Planet orbital evolution cannot be used simultaneously with a satellite'
        )


def tides_enabled_orbit(instance, attribute, value):
    # Tides in interior requires orbit module to not be None
    if (instance.interior_energetics.heat_tidal) and (instance.orbit.module is None):
        raise ValueError('Interior tidal heating requires an orbit module to be enabled')


CURRENT_CONFIG_VERSION = '3.0'


def valid_config_version(instance, attribute, value):
    if value != CURRENT_CONFIG_VERSION:
        raise ValueError(
            f'Config file version "{value}" is not compatible with this version of PROTEUS '
            f'(requires config_version = "{CURRENT_CONFIG_VERSION}"). '
            f'Please update your configuration file to match the current format. '
            f'See input/all_options.toml for the full reference.'
        )


def check_module_dependencies(instance, attribute, value):
    """Check that required external packages are importable for the selected modules."""
    import importlib

    checks = {
        'calliope': (
            instance.outgas.module == 'calliope',
            'calliope',
            'outgas.module = "calliope" requires the calliope package. '
            'Install with: git clone git@github.com:FormingWorlds/CALLIOPE && pip install -e CALLIOPE/.',
        ),
        'atmodeller': (
            instance.outgas.module == 'atmodeller',
            'atmodeller',
            'outgas.module = "atmodeller" requires the optional atmodeller package '
            '(GPL-3.0 licensed). The standard outgassing backend is calliope and '
            'needs no extra install. '
            'Install atmodeller with: pip install "fwl-proteus[atmodeller]" '
            '(or pip install "atmodeller>=1.0.0").',
        ),
        'vulcan': (
            instance.atmos_chem.module == 'vulcan',
            'vulcan',
            'atmos_chem.module = "vulcan" requires the optional VULCAN package '
            '(GPL-3.0 licensed). VULCAN is not needed for a standard PROTEUS run. '
            'Install it with: pip install "fwl-proteus[vulcan]" '
            '(or bash tools/get_vulcan.sh for an editable checkout).',
        ),
        'boreas': (
            instance.escape.module == 'boreas',
            'boreas',
            'escape.module = "boreas" requires the optional boreas package. '
            'Install it with: bash tools/get_boreas.sh',
        ),
    }

    for name, (needed, pkg, msg) in checks.items():
        if needed:
            try:
                importlib.import_module(pkg)
            except ImportError as e:
                raise ImportError(f'{msg}\n  Original error: {e}') from e


def boreas_requires_atmosphere(instance, attribute, value):
    """BOREAS escape requires a radiative atmosphere (not dummy)."""
    if (instance.escape.module == 'boreas') and (instance.atmos_clim.module == 'dummy'):
        raise ValueError(
            'escape.module = "boreas" requires a radiative atmosphere model (agni or janus), '
            'not atmos_clim.module = "dummy". BOREAS needs per-level T/P/composition profiles.'
        )


def observe_resolved_atmosphere(instance, attribute, value):
    # Synthetic observations require a spatially resolved atmosphere profile
    if (instance.observe.synthesis is not None) and (instance.atmos_clim.module == 'dummy'):
        raise ValueError('Observational synthesis requires that atmos_clim != dummy')


def janus_escape_atmosphere(instance, attribute, value):
    # Using escape.zephyrus with JANUS requires params.stop.escape to be True
    if (
        (instance.escape.module == 'zephyrus')
        and (instance.atmos_clim.module == 'janus')
        and (not instance.params.stop.escape.enabled)
    ):
        raise ValueError(
            'When using escape.zephyrus with JANUS, params.stop.escape must be True.'
        )


def planet_mass_valid(instance, attribute, value):
    """Validate that mass_tot is within range."""
    mass_tot = instance.planet.mass_tot
    if mass_tot <= 0:
        raise ValueError('The total planet mass must be > 0')
    if mass_tot > 20:
        raise ValueError('The total planet mass must be < 20 M_earth')


def planet_oxygen_mode_explicit(instance, attribute, value):
    """Validate O_mode in planet.elements.

    Whole-planet oxygen accounting requires every config to declare how
    the IC O budget is interpreted. Valid modes: 'ic_chemistry' (default,
    defer to CALLIOPE equilibrium), 'ppmw', 'kg', 'FeO_mantle_wt_pct'.
    """
    # O_mode is now validated by the attrs in_() validator on the field
    # itself. This function remains as a hook for cross-field checks
    # (e.g. fO2_source compatibility) that reference O_mode.
    pass


def planet_fO2_source_compat(instance, attribute, value):
    """Validate planet.fO2_source against O_mode, volatile_mode, and
    against availability.

    Rejection rules:

    1. ``fO2_source = "from_mantle_redox"`` is a reserved enum value for
       the radial Fe3+/Fe2+ fO2 framework (issue #653, Schaefer et al.
       2024). The runtime path for it does not exist yet; reject the
       config so users do not silently fall through to the default behaviour.

    2. ``fO2_source = "from_O_budget"`` requires the O budget to be
       authoritative. ``O_mode = "ic_chemistry"`` defers the O inventory
       to the chemistry solver, so there is nothing to invert against.
       Require a concrete O budget
       ("ppmw" / "kg" / "FeO_mantle_wt_pct") instead.

    3. ``fO2_source = "from_O_budget"`` requires the volatile budget to
       be set element-wise. ``volatile_mode = "gas_prs"`` supplies
       partial pressures directly and makes ``planet.elements.O_mode``
       inoperative, so there is nothing to invert against. Switch
       ``volatile_mode`` to ``"elements"`` or pick a different fO2_source.

    4. ``fO2_source = "from_O_budget"`` requires an outgassing backend
       with an authoritative-O implementation: CALLIOPE
       (``equilibrium_atmosphere_authoritative_O``) and atmodeller
       (native mass-constraint API) both qualify; the ``dummy`` backend
       does not. The runtime dispatch echoes this rejection too, but
       failing at config-load saves the user from burning interior IC
       and structure setup before hitting the wall.

    ``fO2_source = "user_constant"`` (default) accepts every O_mode and
    every volatile_mode.

    Warning rule:

    - ``fO2_source = "from_O_budget"`` with a non-default
      ``outgas.fO2_shift_IW`` emits a UserWarning. With this source the
      buffer offset is *derived*, so a user-supplied value is silently
      ignored. The warning surfaces the misconfiguration without
      blocking the run.
    """
    fO2_source = instance.planet.fO2_source
    O_mode = instance.planet.elements.O_mode
    volatile_mode = instance.planet.volatile_mode

    if fO2_source == 'from_mantle_redox':
        raise ValueError(
            'planet.fO2_source = "from_mantle_redox" is reserved for the '
            'radial Fe3+/Fe2+ tracking framework (issue #653) and is not '
            'yet wired into the runtime. Use "user_constant" (fO2 '
            'buffered by outgas.fO2_shift_IW) or "from_O_budget" '
            '(authoritative O budget, fO2 derived) instead.'
        )

    if fO2_source == 'from_O_budget' and O_mode == 'ic_chemistry':
        raise ValueError(
            'planet.fO2_source = "from_O_budget" requires an authoritative '
            'O budget but planet.elements.O_mode = "ic_chemistry" defers '
            'the O inventory to the chemistry solver. Pick an explicit O '
            'budget mode ("ppmw", "kg", or "FeO_mantle_wt_pct") or switch '
            'fO2_source back to "user_constant".'
        )

    if fO2_source == 'from_O_budget' and volatile_mode == 'gas_prs':
        raise ValueError(
            'planet.fO2_source = "from_O_budget" requires '
            'planet.volatile_mode = "elements" so the O inventory is '
            'derived from planet.elements.O_budget. Under '
            'volatile_mode = "gas_prs" the user supplies partial '
            'pressures directly and the element budgets are inert, so '
            'there is no O target to invert against. Either switch '
            'volatile_mode to "elements" or set fO2_source back to '
            '"user_constant".'
        )

    if fO2_source == 'from_O_budget':
        outgas_module = getattr(instance.outgas, 'module', None)
        if outgas_module == 'dummy':
            raise ValueError(
                'planet.fO2_source = "from_O_budget" requires an '
                'outgassing backend with an authoritative-O '
                'implementation. outgas.module = "dummy" has no '
                'chemistry to invert against. Switch outgas.module to '
                '"calliope" or '
                '"atmodeller", or set fO2_source back to '
                '"user_constant".'
            )

    if fO2_source == 'from_O_budget':
        import warnings

        fO2_shift_IW = getattr(instance.outgas, 'fO2_shift_IW', None)
        if fO2_shift_IW is not None and fO2_shift_IW != 0.0:
            warnings.warn(
                'outgas.fO2_shift_IW = %.3f is set but '
                'planet.fO2_source = "from_O_budget" derives the buffer '
                'offset from the O budget. The configured value is used '
                'only as the solver initial fO2 guess, not as the buffered '
                'offset.' % fO2_shift_IW,
                UserWarning,
                stacklevel=2,
            )


def boundary_requires_fixed_surface_state(instance, attribute, value):
    """Boundary backend assumes a fixed surface state coupling."""
    if (instance.interior_energetics.module == 'boundary') and (
        instance.atmos_clim.surf_state != 'fixed'
    ):
        raise ValueError(
            "Must set atmos_clim.surf_state='fixed' when interior_energetics.module='boundary'"
        )


def boundary_zalmoxis_incompatible(instance, attribute, value):
    """Boundary backend is not yet wired up to the Zalmoxis structure refresh."""
    if (instance.interior_energetics.module == 'boundary') and (
        instance.interior_struct.module == 'zalmoxis'
    ):
        raise ValueError(
            'Boundary interior module cannot currently be used with the '
            'zalmoxis structure module'
        )


@define
class Config:
    """Root config parameters.

    Attributes
    ----------
    config_version: str
        Version of the configuration file format.
    params: Params
        Parameters for code execution, output files, time-stepping, convergence.
    star: Star
        Stellar parameters, model selection.
    orbit: Orbit
        Orbital and star-system parameters.
    planet: Planet
        Bulk planet properties (mass, initial volatile inventory).
    interior_struct: Struct
        Planetary structure calculation (radius, composition, Zalmoxis).
    interior_energetics: Interior
        Magma ocean / mantle energetics model parameters, model selection.
    outgas: Outgas
        Outgassing parameters (fO2, etc) and included volatiles.
    atmos_clim: AtmosClim
        Planetary atmosphere climate parameters, model selection.
    atmos_chem: AtmosChem
        Planetary atmosphere chemistry parameters, model selection.
    escape: Escape
        Atmospheric escape parameters, model selection.
    accretion: Accretion
        Late accretion / delivery model selection.
    observe: Observe
        Synthetic observations.
    """

    params: Params = field(factory=Params)
    star: Star = field(factory=Star)
    orbit: Orbit = field(
        factory=Orbit, validator=(instmethod_dummy, instmethod_evolve, satellite_evolve)
    )
    planet: Planet = field(
        factory=Planet,
        validator=(
            planet_mass_valid,
            planet_oxygen_mode_explicit,
            planet_fO2_source_compat,
        ),
    )
    interior_struct: Struct = field(factory=Struct)
    interior_energetics: Interior = field(
        factory=Interior,
        validator=(
            tides_enabled_orbit,
            boundary_requires_fixed_surface_state,
            boundary_zalmoxis_incompatible,
        ),
    )
    outgas: Outgas = field(factory=Outgas)
    atmos_clim: AtmosClim = field(factory=AtmosClim)
    atmos_chem: AtmosChem = field(factory=AtmosChem)
    escape: Escape = field(
        factory=Escape,
        validator=(spada_zephyrus, janus_escape_atmosphere, boreas_requires_atmosphere),
    )
    accretion: Accretion = field(factory=Accretion)
    observe: Observe = field(factory=Observe, validator=(observe_resolved_atmosphere,))

    config_version: str = field(
        default='3.0',
        validator=(valid_config_version, check_module_dependencies),
    )

    def write(self, out: str):
        """
        Write configuration to a new TOML file.
        """

        # Convert to dictionary
        cfg = dict(asdict(self))

        # Replace None with "none"
        cfg = dict_replace_none(cfg)

        # Write to TOML file
        with open(out, 'w') as hdl:
            tomlkit.dump(cfg, hdl)
