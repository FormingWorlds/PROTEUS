# Zalmoxis interior module
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import platformdirs
from scipy.interpolate import interp1d
from zalmoxis.solver import main

from proteus.config import Config
from proteus.utils.constants import (
    M_earth,
    R_earth,
    element_list,
)
from proteus.utils.data import get_zalmoxis_eos_dir, get_zalmoxis_melting_curves

FWL_DATA_DIR = Path(os.environ.get('FWL_DATA', platformdirs.user_data_dir('fwl_data')))

# Set up logging
logger = logging.getLogger('fwl.' + __name__)

# Module-level cache for density seeding between Zalmoxis calls.
# Stores the last successful density profile so the next call can
# use it as a starting point for the Picard iteration.
_density_cache = {'density': None, 'radii': None}

# Mapping from PROTEUS volatile species to Zalmoxis EOS component names.
# Only species with Zalmoxis EOS tables are included.
_VOLATILE_EOS_MAP = {
    'H2O': 'PALEOS:H2O',
    'H2': 'Chabrier:H',
}


def _make_derived_solidus(liquidus_func, mushy_zone_factor: float):
    """Create a solidus function as T_sol(P) = T_liq(P) * mushy_zone_factor.

    Parameters
    ----------
    liquidus_func : callable
        P [Pa] -> T_liquidus [K].
    mushy_zone_factor : float
        Cryoscopic depression factor in [0.7, 1.0].

    Returns
    -------
    callable
        P [Pa] -> T_solidus [K].
    """

    def solidus_func(P):
        return liquidus_func(P) * mushy_zone_factor

    return solidus_func


def get_zalmoxis_output_filepath(outdir: str):
    """Returns the output file path for Zalmoxis data.
    Args:
        outdir (str): Output directory.
    Returns:
        str: Path to the output file.
    """
    return os.path.join(outdir, 'data', 'zalmoxis_output.dat')


def build_volatile_profile(hf_row: dict, mantle_eos: str):
    """Build a VolatileProfile from helpfile volatile masses.

    Computes per-phase (liquid/solid) mass fractions for dissolved volatiles
    that have Zalmoxis EOS tables. Returns None if no volatiles are dissolved
    or if the mantle liquid/solid masses are unavailable.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row with volatile mass keys (e.g. ``H2O_kg_liquid``,
        ``H2O_kg_solid``) and mantle mass keys (``M_mantle_liquid``,
        ``M_mantle_solid``).
    mantle_eos : str
        Primary mantle EOS identifier (e.g. ``'PALEOS:MgSiO3'``).

    Returns
    -------
    VolatileProfile or None
        Profile with per-phase fractions, or None if not applicable.
    """
    from zalmoxis.mixing import VolatileProfile

    M_liq = float(hf_row.get('M_mantle_liquid', 0.0))
    M_sol = float(hf_row.get('M_mantle_solid', 0.0))

    # Need mantle mass data to compute fractions
    if M_liq + M_sol <= 0:
        return None

    w_liquid = {}
    w_solid = {}
    has_nonzero = False

    for species, eos_name in _VOLATILE_EOS_MAP.items():
        kg_liq = float(hf_row.get(f'{species}_kg_liquid', 0.0))
        kg_sol = float(hf_row.get(f'{species}_kg_solid', 0.0))

        # Only include species with meaningful dissolved mass
        if kg_liq + kg_sol <= 0:
            continue

        # Mass fraction in liquid phase
        w_l = kg_liq / M_liq if M_liq > 0 else 0.0
        # Mass fraction in solid phase
        w_s = kg_sol / M_sol if M_sol > 0 else 0.0

        w_liquid[eos_name] = w_l
        w_solid[eos_name] = w_s
        has_nonzero = True

    if not has_nonzero:
        return None

    # Normalize: total volatile fraction in each phase must not exceed 0.95
    # (at least 5% silicate). Clamp proportionally if sum exceeds the limit.
    for w_dict in (w_liquid, w_solid):
        total = sum(w_dict.values())
        max_volatile_frac = 0.95
        if total > max_volatile_frac:
            scale = max_volatile_frac / total
            for k in w_dict:
                w_dict[k] *= scale

    logger.info(
        'Built VolatileProfile: liquid=%s, solid=%s',
        {k: f'{v:.4f}' for k, v in w_liquid.items()},
        {k: f'{v:.4f}' for k, v in w_solid.items()},
    )

    return VolatileProfile(
        w_liquid=w_liquid,
        w_solid=w_solid,
        primary_component=mantle_eos,
    )


def extend_mantle_eos_with_volatiles(mantle_eos: str, volatile_profile) -> str:
    """Extend a single-component mantle EOS string with volatile components.

    If a VolatileProfile is provided and the mantle EOS is a single component,
    this adds the volatile EOS components with small placeholder fractions.
    The actual fractions are overridden at each ODE step by the VolatileProfile.

    Parameters
    ----------
    mantle_eos : str
        Base mantle EOS string (e.g. ``'PALEOS:MgSiO3'``).
    volatile_profile : VolatileProfile or None
        Profile containing volatile EOS component names.

    Returns
    -------
    str
        Extended EOS string (e.g.
        ``'PALEOS:MgSiO3:0.98+PALEOS:H2O:0.01+Chabrier:H:0.01'``),
        or the original string if no extension needed.
    """
    if volatile_profile is None:
        return mantle_eos

    # Don't modify if already multi-component
    if '+' in mantle_eos:
        return mantle_eos

    # Collect all volatile EOS components from the profile
    all_vol_components = set()
    for d in (volatile_profile.w_liquid, volatile_profile.w_solid):
        all_vol_components.update(d.keys())

    if not all_vol_components:
        return mantle_eos

    # Build extended string with small placeholder fractions
    # (actual fractions set by VolatileProfile at each radius)
    n_vol = len(all_vol_components)
    placeholder = 0.01  # 1% each
    primary_frac = max(0.5, 1.0 - n_vol * placeholder)
    parts = [f'{mantle_eos}:{primary_frac:.4f}']
    for comp in sorted(all_vol_components):
        parts.append(f'{comp}:{placeholder:.4f}')

    extended = '+'.join(parts)
    logger.info('Extended mantle EOS: %s -> %s', mantle_eos, extended)
    return extended


def _get_target_surface_pressure(config: Config, hf_row: dict) -> float:
    """Determine the surface pressure boundary condition for Zalmoxis.

    Parameters
    ----------
    config : Config
        PROTEUS configuration object.
    hf_row : dict
        Current helpfile row.

    Returns
    -------
    float
        Target surface pressure in Pa.
    """
    # After outgassing has run, use the atmospheric surface pressure
    p_surf_bar = hf_row.get('P_surf', 0)
    if np.isfinite(p_surf_bar) and p_surf_bar > 0:
        return p_surf_bar * 1e5  # bar -> Pa

    # First call, before outgassing. Estimate from initial volatile
    # partial pressures specified in the config.
    _SPECIES = ('H2O', 'CO2', 'N2', 'S2', 'SO2', 'H2S', 'NH3', 'H2', 'CH4', 'CO')
    try:
        gas_prs = config.planet.gas_prs
        p_init_bar = sum(float(getattr(gas_prs, s, 0)) for s in _SPECIES)
        p_init_pa = p_init_bar * 1e5  # bar -> Pa
    except (TypeError, ValueError, AttributeError):
        p_init_pa = 0.0

    # Floor at 1 atm (bare rock), ceiling at 1 GPa
    return max(101325.0, min(p_init_pa, 1e9))


def load_zalmoxis_configuration(
    config: Config,
    hf_row: dict,
    temperature_mode_override: str | None = None,
):
    """Loads the model configuration for Zalmoxis and calculates the dry mass of the planet based on the total mass and the mass of volatiles.
    Args:
        config (Config): The configuration object containing the Zalmoxis parameters.
        hf_row (dict): A dictionary containing the mass of volatiles and other parameters.
        temperature_mode_override: Optional local override for
            ``config.planet.temperature_mode``. Lets callers force a different
            structure-solve mode (e.g. 'adiabatic' for SPIDER coupling with
            T-dependent mantle EOS) without mutating the shared Config object.
            When None, falls back to ``config.planet.temperature_mode``.
    Returns:
        dict: A dictionary containing the Zalmoxis configuration parameters.
    """

    # Setup target planet mass (input parameter) as the total mass of the planet (dry mass + volatiles) [kg]
    total_planet_mass = config.planet.mass_tot * M_earth

    logger.info(
        'Total target planet mass (dry mass + volatiles): %s kg '
        'with EOS: core=%s, mantle=%s, ice=%s',
        total_planet_mass,
        config.interior_struct.zalmoxis.core_eos,
        config.interior_struct.zalmoxis.mantle_eos,
        config.interior_struct.zalmoxis.ice_layer_eos or 'none',
    )

    # Calculate the total mass of 'wet' elements in the planet
    M_volatiles = 0.0
    for e in element_list:
        if e == 'O':  # Oxygen is set by fO2, so we skip it here (const_fO2)
            continue
        M_volatiles += hf_row[e + '_kg_total']

    logger.info(f'Volatile mass: {M_volatiles} kg')

    # Calculate the target planet mass (dry mass) by subtracting the mass of volatiles from the total planet mass
    planet_mass = total_planet_mass - M_volatiles

    logger.info(f'Target planet mass (dry mass): {planet_mass} kg ')

    # Build per-layer EOS config dict from PROTEUS config fields
    layer_eos_config = {
        'core': config.interior_struct.zalmoxis.core_eos,
        'mantle': config.interior_struct.zalmoxis.mantle_eos,
    }
    if config.interior_struct.zalmoxis.ice_layer_eos is not None:
        layer_eos_config['ice_layer'] = config.interior_struct.zalmoxis.ice_layer_eos

    # Mushy zone factor: controls width of partially molten region in PALEOS
    # unified EOS. Applied as T_solidus = T_liquidus * mushy_zone_factor.
    mzf = config.interior_struct.zalmoxis.mushy_zone_factor
    mushy_zone_factors = {
        'PALEOS:iron': mzf,
        'PALEOS:MgSiO3': mzf,
        'PALEOS:H2O': mzf,
    }

    # Core fraction: value from config, interpretation from core_frac_mode.
    # When mode is "mass", this is the true mass fraction.
    # When mode is "radius", Zalmoxis converts radius fraction to mass fraction internally.
    return {
        'planet_mass': planet_mass,
        'core_mass_fraction': config.interior_struct.core_frac,
        'core_frac_mode': config.interior_struct.core_frac_mode,
        'mantle_mass_fraction': config.interior_struct.zalmoxis.mantle_mass_fraction,
        # For the structure solve, 'accretion' and 'isentropic' both reduce
        # to 'adiabatic' inside Zalmoxis. Zalmoxis only solves the structure
        # (M-R via hydrostatic + EOS); the entropy IC for Aragog/SPIDER is
        # set independently from config.planet.ini_entropy. 'accretion'
        # delays White+Li T-profile until after structure converges;
        # 'isentropic' (CHILI protocol) means the energetics solver
        # consumes ini_entropy, not the Zalmoxis T-profile, so the choice
        # of structure-solve T is decoupled from the IC. 'adiabatic_from_cmb'
        # is passed through to Zalmoxis with the CMB-anchor temperature so
        # the structure-side T(r) integrates upward from T_cmb at R_cmb,
        # matching the entropy that the energetics solver receives via
        # compute_initial_entropy. temperature_mode_override lets SPIDER
        # coupling force adiabatic without mutating the shared Config
        # object (see proteus rules §"Config mutability").
        'temperature_mode': (
            'adiabatic'
            if (temperature_mode_override or config.planet.temperature_mode)
            in ('accretion', 'isentropic')
            else (temperature_mode_override or config.planet.temperature_mode)
        ),
        'surface_temperature': config.planet.tsurf_init,
        'cmb_temperature': config.planet.tcmb_init,
        'center_temperature': config.planet.tcenter_init,
        'temp_profile_file': None,
        'layer_eos_config': layer_eos_config,
        'mushy_zone_factor': mzf,
        'mushy_zone_factors': mushy_zone_factors,
        'num_layers': config.interior_struct.zalmoxis.num_levels,
        'target_surface_pressure': _get_target_surface_pressure(config, hf_row),
        # Solver tolerances and iteration limits
        'tolerance_outer': config.interior_struct.zalmoxis.solver_tol_outer,
        'tolerance_inner': config.interior_struct.zalmoxis.solver_tol_inner,
        'max_iterations_outer': config.interior_struct.zalmoxis.solver_max_iter_outer,
        'max_iterations_inner': config.interior_struct.zalmoxis.solver_max_iter_inner,
    }


def load_zalmoxis_material_dictionaries():
    """Build an EOS registry dict with file paths pointing to FWL_DATA.

    Returns the same dict format as Zalmoxis ``EOS_REGISTRY``, but with
    every ``eos_file`` path resolved under ``FWL_DATA/zalmoxis_eos/``
    instead of ``ZALMOXIS_ROOT/data/``.  This ensures that Zalmoxis,
    when called from PROTEUS, reads EOS data from the central FWL_DATA
    location managed by ``download_zalmoxis_eos()``.

    Returns
    -------
    dict
        Flat dict keyed by EOS identifier string (e.g.
        ``"Seager2007:iron"``, ``"PALEOS:MgSiO3"``, ``"Chabrier:H"``).
    """
    eos_base = get_zalmoxis_eos_dir()

    # Seager2007 paths (also in legacy EOS_material_properties location)
    seager_dir = eos_base / 'EOS_Seager2007'
    if not seager_dir.exists():
        seager_dir = FWL_DATA_DIR / 'EOS_material_properties' / 'EOS_Seager2007'

    _seager_iron = {'eos_file': str(seager_dir / 'eos_seager07_iron.txt')}
    _seager_silicate = {'eos_file': str(seager_dir / 'eos_seager07_silicate.txt')}
    _seager_water = {'eos_file': str(seager_dir / 'eos_seager07_water.txt')}

    # Wolf & Bower 2018
    wb_dir = eos_base / 'EOS_WolfBower2018_1TPa'
    _wb_melted = {
        'eos_file': str(wb_dir / 'density_melt.dat'),
        'adiabat_grad_file': str(wb_dir / 'adiabat_temp_grad_melt.dat'),
    }
    _wb_solid = {'eos_file': str(wb_dir / 'density_solid.dat')}

    # RTPress 100 TPa
    rt_dir = eos_base / 'EOS_RTPress_melt_100TPa'
    _rt_melted = {
        'eos_file': str(rt_dir / 'density_melt.dat'),
        'adiabat_grad_file': str(rt_dir / 'adiabat_temp_grad_melt.dat'),
    }

    # PALEOS 2-phase MgSiO3 (separate solid/liquid)
    paleos2ph_dir = eos_base / 'EOS_PALEOS_MgSiO3'
    _paleos2ph_melted = {
        'eos_file': str(paleos2ph_dir / 'paleos_mgsio3_tables_pt_proteus_liquid.dat'),
        'format': 'paleos',
    }
    _paleos2ph_solid = {
        'eos_file': str(paleos2ph_dir / 'paleos_mgsio3_tables_pt_proteus_solid.dat'),
        'format': 'paleos',
    }

    # PALEOS unified tables
    _paleos_iron = {
        'eos_file': str(eos_base / 'EOS_PALEOS_iron' / 'paleos_iron_eos_table_pt.dat'),
        'format': 'paleos_unified',
    }
    _paleos_mgsio3 = {
        'eos_file': str(
            eos_base / 'EOS_PALEOS_MgSiO3_unified' / 'paleos_mgsio3_eos_table_pt.dat'
        ),
        'format': 'paleos_unified',
    }
    _paleos_h2o = {
        'eos_file': str(eos_base / 'EOS_PALEOS_H2O' / 'paleos_water_eos_table_pt.dat'),
        'format': 'paleos_unified',
    }

    # Chabrier H/He
    _chabrier_h = {
        'eos_file': str(eos_base / 'EOS_Chabrier2021_HHe' / 'chabrier2021_H.dat'),
        'format': 'paleos_unified',
    }

    # PALEOS-API live tabulation entries. These carry only a GridSpec at
    # build time; the dispatch layer (``zalmoxis.eos.paleos_api_cache``)
    # mutates them in place on first density query to populate ``eos_file``
    # and rewrite ``format`` to the downstream value (``paleos_unified`` or
    # ``paleos``). Cache keys are SHA+grid-hash under
    # ``$ZALMOXIS_ROOT/data/EOS_PALEOS_API/``; cold-cache cost is a one-time
    # generator run (see ``zalmoxis.eos.paleos_api``).
    from zalmoxis.eos.paleos_api import (
        make_default_grid_h2o,
        make_default_grid_iron,
        make_default_grid_mgsio3,
    )
    _paleos_api_iron = {
        'format': 'paleos_api',
        'material': 'iron',
        'grid_spec': make_default_grid_iron(),
    }
    _paleos_api_mgsio3 = {
        'format': 'paleos_api',
        'material': 'mgsio3',
        'grid_spec': make_default_grid_mgsio3(),
    }
    _paleos_api_h2o = {
        'format': 'paleos_api',
        'material': 'h2o',
        'grid_spec': make_default_grid_h2o(),
        'h2o_table_path': None,
    }
    _paleos_api_2ph_mgsio3_melted = {
        'format': 'paleos_api_2phase',
        'material': 'mgsio3',
        'side': 'liquid',
        'grid_spec': make_default_grid_mgsio3(),
    }
    _paleos_api_2ph_mgsio3_solid = {
        'format': 'paleos_api_2phase',
        'material': 'mgsio3',
        'side': 'solid',
        'grid_spec': make_default_grid_mgsio3(),
    }

    return {
        # Seager2007 static
        'Seager2007:iron': {'core': _seager_iron},
        'Seager2007:MgSiO3': {'mantle': _seager_silicate},
        'Seager2007:H2O': {'ice_layer': _seager_water},
        # Wolf & Bower 2018 T-dependent
        'WolfBower2018:MgSiO3': {
            'core': _seager_iron,
            'melted_mantle': _wb_melted,
            'solid_mantle': _wb_solid,
        },
        # RTPress 100 TPa extended melt + WB2018 solid
        'RTPress100TPa:MgSiO3': {
            'core': _seager_iron,
            'melted_mantle': _rt_melted,
            'solid_mantle': _wb_solid,
        },
        # PALEOS 2-phase MgSiO3
        'PALEOS-2phase:MgSiO3': {
            'core': _seager_iron,
            'melted_mantle': _paleos2ph_melted,
            'solid_mantle': _paleos2ph_solid,
        },
        # PALEOS unified
        'PALEOS:iron': _paleos_iron,
        'PALEOS:MgSiO3': _paleos_mgsio3,
        'PALEOS:H2O': _paleos_h2o,
        # PALEOS-API live-tabulated (dispatch populates eos_file on demand)
        'PALEOS-API:iron': _paleos_api_iron,
        'PALEOS-API:MgSiO3': _paleos_api_mgsio3,
        'PALEOS-API:H2O': _paleos_api_h2o,
        'PALEOS-API-2phase:MgSiO3': {
            'core': _seager_iron,
            'melted_mantle': _paleos_api_2ph_mgsio3_melted,
            'solid_mantle': _paleos_api_2ph_mgsio3_solid,
        },
        # Chabrier H/He
        'Chabrier:H': _chabrier_h,
    }


def resolve_2phase_mgsio3_paths(mantle_eos: str, mat_dicts: dict):
    """Return (solid_eos_path, liquid_eos_path) for the 2-phase MgSiO3 tables.

    Selects the API key (``PALEOS-API-2phase:MgSiO3``) when ``mantle_eos``
    is from the PALEOS-API family, otherwise the shipped key
    (``PALEOS-2phase:MgSiO3``). For API entries, calls
    :func:`zalmoxis.eos.paleos_api_cache.resolve_registry_entry` to
    materialise cached ``.dat`` paths in place; this is required because
    :func:`load_zalmoxis_material_dictionaries` rebuilds a fresh registry
    each call, so any earlier in-place mutation is lost.

    Returns
    -------
    tuple[str | None, str | None]
        Absolute filesystem paths if both tables exist, ``(None, None)``
        otherwise. Caller is responsible for treating ``None`` as "no
        2-phase tables available, fall back".
    """
    use_api = mantle_eos.startswith(('PALEOS-API:', 'PALEOS-API-2phase:'))
    twophase_key = 'PALEOS-API-2phase:MgSiO3' if use_api else 'PALEOS-2phase:MgSiO3'
    twophase = mat_dicts.get(twophase_key, {})
    if not twophase:
        logger.warning(
            'resolve_2phase_mgsio3_paths: registry has no entry for %s '
            '(mantle_eos=%s); 2-phase fallback disabled. This is a '
            'silent-wrong landmine if the caller continues with empty paths.',
            twophase_key,
            mantle_eos,
        )
        return None, None
    if use_api:
        try:
            from zalmoxis.eos.paleos_api_cache import resolve_registry_entry
            resolve_registry_entry(twophase)
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(
                'resolve_2phase_mgsio3_paths: PALEOS-API resolver unavailable '
                '(%s); cannot materialise %s tables.', e, twophase_key,
            )
            return None, None
    solid_eos = twophase.get('solid_mantle', {}).get('eos_file', '')
    liquid_eos = twophase.get('melted_mantle', {}).get('eos_file', '')
    solid_eos = solid_eos if solid_eos and os.path.isfile(solid_eos) else None
    liquid_eos = liquid_eos if liquid_eos and os.path.isfile(liquid_eos) else None
    return solid_eos, liquid_eos


def load_zalmoxis_solidus_liquidus_functions(mantle_eos: str, config: Config):
    """Loads the solidus and liquidus functions for Zalmoxis based on the mantle EOS.

    Melting curves are needed for two purposes:
    1. Temperature-dependent density in the mushy zone (WolfBower2018, RTPress).
    2. phi(r) blending in VolatileProfile (any EOS with dissolved volatiles).

    For WolfBower2018/RTPress100TPa, loads SPIDER-format P-T files from FWL_DATA.
    For PALEOS unified and PALEOS-2phase, the liquidus comes from the analytic
    Belonoshko+2005 / Fei+2021 curve (Zalmoxis ``'PALEOS-liquidus'``) which is
    the basis Zalmoxis uses for MgSiO3 phase separation, and the solidus is
    derived as T_sol = T_liq * mushy_zone_factor. This keeps the curves used
    for phi-blending and 2-phase nabla_ad consistent with the unified PALEOS
    density-interpolation phase boundaries.

    Parameters
    ----------
    mantle_eos : str
        Mantle EOS string (e.g. ``"WolfBower2018:MgSiO3"``, ``"PALEOS:MgSiO3"``,
        ``"PALEOS-2phase:MgSiO3"``).
    config : Config
        PROTEUS configuration object.

    Returns
    -------
    tuple or None
        (solidus_func, liquidus_func) callable P [Pa] -> T [K], or None.
    """
    _TDEP_PREFIXES = ('WolfBower2018', 'RTPress100TPa')
    if mantle_eos.startswith(_TDEP_PREFIXES):
        return get_zalmoxis_melting_curves(config)

    # PALEOS unified and PALEOS-2phase: both use the same analytic Belonoshko+2005 /
    # Fei+2021 melting curve (`PALEOS-liquidus`) as the basis for MgSiO3 phase
    # separation. The unified path uses it for in-table density interpolation;
    # the 2-phase path uses it inside `_compute_paleos_dtdp` to weight nabla_ad
    # across the solid/liquid blend (mixing.py:_compute_paleos_dtdp). The solidus
    # is derived as `liquidus * mushy_zone_factor` so the mushy band lines up with
    # the unified PALEOS density interpolation. Without these curves, the
    # 2-phase nabla_ad call fails and Zalmoxis structure solve diverges; the
    # unified path falls back to phi=0.5 everywhere in VolatileProfile.
    if mantle_eos.startswith(
        ('PALEOS:', 'PALEOS-2phase:', 'PALEOS-API:', 'PALEOS-API-2phase:')
    ):
        try:
            from zalmoxis.melting_curves import get_solidus_liquidus_functions

            _, liquidus_func = get_solidus_liquidus_functions(
                solidus_id='Stixrude14-solidus',  # unused, but API requires it
                liquidus_id='PALEOS-liquidus',
            )
            mzf = config.interior_struct.zalmoxis.mushy_zone_factor
            solidus_func = _make_derived_solidus(liquidus_func, mzf)
            logger.info(
                'PALEOS melting curves (%s): liquidus from PALEOS, '
                'solidus = liquidus * %.2f (mushy_zone_factor)',
                mantle_eos,
                mzf,
            )
            return solidus_func, liquidus_func
        except Exception as e:
            logger.warning('Could not load PALEOS melting curves: %s', e)
            return None

    return None


def scale_temperature_profile_for_aragog(
    config: Config, mantle_radii: np.ndarray, mantle_temperature_profile: np.ndarray
):
    """Scales the temperature profile obtained from Zalmoxis to match the number of levels required by Aragog.
    Args:
        config (Config): The configuration object containing the configuration parameters.
        mantle_radii (np.ndarray): The radial positions of the mantle layers from Zalmoxis.
        mantle_temperature_profile (np.ndarray): The temperature profile of the mantle layers from Zalmoxis.
    Returns:
        np.ndarray: The scaled temperature profile matching the number of levels in Aragog.
    """

    # Number of levels in Aragog mesh
    mesh_grid_size = config.interior_energetics.num_levels - 1

    # Create new evenly spaced radial positions for Aragog
    radii_to_interpolate = np.linspace(mantle_radii[0], mantle_radii[-1], mesh_grid_size)

    # Cubic interpolation onto the Aragog radial mesh
    cubic_interp = interp1d(mantle_radii, mantle_temperature_profile, kind='cubic')
    return cubic_interp(radii_to_interpolate)


def write_spider_mesh_file(
    outdir: str,
    mantle_radii: np.ndarray,
    mantle_pressure: np.ndarray,
    mantle_density: np.ndarray,
    mantle_gravity: np.ndarray,
    num_basic: int,
) -> str:
    """Write an external mesh file for SPIDER from Zalmoxis mantle profiles.

    Interpolates the Zalmoxis mantle arrays onto uniformly-spaced SPIDER
    basic and staggered nodes, then writes the mesh file in the format
    expected by SPIDER's ``SetMeshFromExternalFile()``.

    Parameters
    ----------
    outdir : str
        PROTEUS output directory (file is written to ``outdir/data/``).
    mantle_radii : np.ndarray
        Radial positions from CMB to surface, ascending [m].
    mantle_pressure : np.ndarray
        Pressure at each radius [Pa].
    mantle_density : np.ndarray
        Density at each radius [kg/m^3].
    mantle_gravity : np.ndarray
        Gravity magnitude at each radius [m/s^2] (positive).
    num_basic : int
        Number of SPIDER basic nodes (shell boundaries).

    Returns
    -------
    str
        Path to the written mesh file.
    """
    num_staggered = num_basic - 1
    R_surf = float(mantle_radii[-1])
    R_cmb = float(mantle_radii[0])

    # Basic nodes: uniform spacing from surface to CMB (descending r)
    r_b = np.linspace(R_surf, R_cmb, num_basic)
    # Staggered nodes: midpoints between consecutive basic nodes
    r_s = 0.5 * (r_b[:-1] + r_b[1:])

    # Interpolate Zalmoxis profiles onto node positions
    # mantle_radii is ascending, np.interp requires ascending xp
    P_b = np.interp(r_b, mantle_radii, mantle_pressure)
    rho_b = np.interp(r_b, mantle_radii, mantle_density)
    g_b = np.interp(r_b, mantle_radii, mantle_gravity)

    P_s = np.interp(r_s, mantle_radii, mantle_pressure)
    rho_s = np.interp(r_s, mantle_radii, mantle_density)
    g_s = np.interp(r_s, mantle_radii, mantle_gravity)

    # Negate gravity for SPIDER convention (inward-pointing, negative)
    g_b = -np.abs(g_b)
    g_s = -np.abs(g_s)

    # Write mesh file
    mesh_path = os.path.join(outdir, 'data', 'spider_mesh.dat')
    with open(mesh_path, 'w') as f:
        f.write(f'# {num_basic} {num_staggered}\n')
        for i in range(num_basic):
            f.write(f'{r_b[i]:.15e} {P_b[i]:.15e} {rho_b[i]:.15e} {g_b[i]:.15e}\n')
        for i in range(num_staggered):
            f.write(f'{r_s[i]:.15e} {P_s[i]:.15e} {rho_s[i]:.15e} {g_s[i]:.15e}\n')

    logger.info(
        'Wrote SPIDER mesh file: %s (%d basic + %d staggered nodes)',
        mesh_path,
        num_basic,
        num_staggered,
    )

    return mesh_path


def generate_spider_tables(config: Config, outdir: str):
    """Generate P-S EOS tables and phase boundaries from PALEOS data.

    Produces P-S lookup tables for density, temperature, heat capacity,
    thermal expansion, and adiabatic gradient, plus solidus/liquidus phase
    boundaries in S(P) format. These are consumed by the entropy-IC verify
    in Aragog (and by SPIDER if the structure module is SPIDER).

    Supports two PALEOS layouts:

    1. ``paleos_unified`` (e.g. ``PALEOS:MgSiO3``): single P-T table covering
       both phases plus mushy zone. Solidus is derived from
       ``mushy_zone_factor * liquidus``.
    2. ``PALEOS-2phase:<solid>`` (e.g. ``PALEOS-2phase:MgSiO3``): separate
       solid + liquid PALEOS tables. Phase boundaries are sampled at the
       PALEOS-liquidus temperature from each phase table directly. The
       ``mushy_zone_factor`` config value is ignored (no analytic mushy
       zone exists for 2-phase; the gap between solid-table-top and
       liquid-table-bottom defines the latent heat).

    For non-PALEOS EOS types (WolfBower2018, RTPress100TPa), returns None
    and the caller is expected to fall back on pre-existing SPIDER tables.

    Parameters
    ----------
    config : Config
        Configuration object with struct.zalmoxis settings.
    outdir : str
        Output directory. Tables are written to ``outdir/data/spider_eos/``.

    Returns
    -------
    dict or None
        Keys ``'eos_dir'``, ``'solidus_path'``, ``'liquidus_path'`` with
        absolute paths. Returns None if the mantle EOS is not PALEOS.
    """
    from zalmoxis.eos_export import generate_spider_eos_tables, generate_spider_phase_boundaries
    from zalmoxis.melting_curves import get_solidus_liquidus_functions

    mantle_eos = config.interior_struct.zalmoxis.mantle_eos

    # Use FWL_DATA paths (not ZALMOXIS_ROOT) for EOS file lookup
    mat_dicts = load_zalmoxis_material_dictionaries()
    eos_entry = mat_dicts.get(mantle_eos)

    if eos_entry is None:
        logger.info(
            'Mantle EOS %s not found in material dictionary; '
            'using pre-existing SPIDER tables.', mantle_eos,
        )
        return None

    # PALEOS-API live tabulation: materialise cached .dat paths in place so the
    # downstream format / eos_file lookups see concrete paths. No-op for
    # non-PALEOS-API entries. First call on a cold cache triggers generation.
    from zalmoxis.eos.dispatch import _is_paleos_api
    if _is_paleos_api(eos_entry):
        from zalmoxis.eos.paleos_api_cache import resolve_registry_entry
        logger.info(
            'PALEOS-API live tabulation: resolving cached tables for %s '
            '(cold-cache build may take up to ~1 h at 600 pts/decade)',
            mantle_eos,
        )
        resolve_registry_entry(eos_entry)

    # Detect format: paleos_unified vs PALEOS-2phase (nested dict).
    is_unified = eos_entry.get('format') == 'paleos_unified'
    is_twophase = (
        'melted_mantle' in eos_entry and 'solid_mantle' in eos_entry
        and isinstance(eos_entry.get('melted_mantle'), dict)
        and isinstance(eos_entry.get('solid_mantle'), dict)
    )

    if not (is_unified or is_twophase):
        logger.info(
            'Mantle EOS %s is neither PALEOS unified nor PALEOS-2phase; '
            'using pre-existing SPIDER tables.', mantle_eos,
        )
        return None

    # Resolve unified file (if present) and 2-phase files (if present).
    eos_file = eos_entry.get('eos_file', '')
    eos_file = eos_file if eos_file and os.path.isfile(eos_file) else None

    if is_twophase:
        solid_eos = eos_entry['solid_mantle'].get('eos_file', '')
        liquid_eos = eos_entry['melted_mantle'].get('eos_file', '')
    else:
        # Unified mantle: also look for sibling 2-phase tables to harden
        # the property surfaces (avoids interpolation across the melting
        # curve discontinuity in the unified table). Use the API-aware
        # helper so PALEOS-API unified runs pull API 2-phase tables
        # rather than silently pulling shipped Zenodo ones.
        solid_eos, liquid_eos = resolve_2phase_mgsio3_paths(mantle_eos, mat_dicts)

    solid_eos = solid_eos if solid_eos and os.path.isfile(solid_eos) else None
    liquid_eos = liquid_eos if liquid_eos and os.path.isfile(liquid_eos) else None

    # For the 2-phase path, the downstream Zalmoxis functions still require
    # an `eos_file` positional (used only to seed default property
    # interpolators that are immediately overridden by the 2-phase ones).
    # Pass the solid table as a sentinel: any valid PALEOS table works.
    if eos_file is None:
        if solid_eos is not None:
            eos_file = solid_eos
        else:
            logger.warning(
                'No PALEOS EOS file available for %s '
                '(unified missing and 2-phase incomplete); skipping table gen.',
                mantle_eos,
            )
            return None

    if is_twophase and not (solid_eos and liquid_eos):
        logger.warning(
            'PALEOS-2phase entry %s missing solid or liquid file; skipping.',
            mantle_eos,
        )
        return None

    # Phase boundaries: PALEOS-liquidus is the analytic Belonoshko+2005 /
    # Fei+2021 Simon-Glatzel curve. For PALEOS-2phase, mushy_zone_factor=1.0
    # collapses solidus = liquidus and the latent-heat gap is supplied by
    # the entropy difference between solid_table.s(P, T_liq) and
    # liquid_table.s(P, T_liq).
    _, liquidus_func = get_solidus_liquidus_functions(
        solidus_id='Stixrude14-solidus',  # unused, but API requires it
        liquidus_id='PALEOS-liquidus',
    )
    mzf = config.interior_struct.zalmoxis.mushy_zone_factor
    solidus_func = _make_derived_solidus(liquidus_func, mzf)
    if is_twophase:
        logger.info(
            'PALEOS-2phase phase boundaries: solidus T = liquidus T '
            '(mushy_zone_factor=%.2f, latent heat from 2-phase tables)',
            mzf,
        )
    else:
        logger.info(
            'PALEOS unified phase boundaries: solidus = liquidus * %.2f '
            '(mushy_zone_factor)', mzf,
        )

    spider_eos_dir = os.path.join(outdir, 'data', 'spider_eos')

    # Determine pressure range from planet mass (higher mass needs wider range)
    mass_tot = config.planet.mass_tot or 1.0
    # P_max for the SPIDER P-S lookup grid. Must cover the actual P_cmb
    # of the planet; original 200 GPa cap was too low for mass_tot > ~2.
    # Cap raised to 10 TPa to cover very massive rocky planets without
    # hitting the table edge. See interior_energetics/aragog.py for the
    # matching bump and the comment on EOS / melting-curve calibration
    # ranges.
    P_max = min(1.0e13, 150e9 * mass_tot + 200e9)

    if solid_eos and liquid_eos:
        logger.info('Using PALEOS-2phase tables for entropy-IC table generation')

    # Table resolution from config
    nP = config.interior_struct.zalmoxis.lookup_nP
    nS = config.interior_struct.zalmoxis.lookup_nS

    # Cache check: skip regeneration if tables exist and pressure range unchanged.
    # The pressure range depends on planet mass, which doesn't change during evolution.
    cache_marker = os.path.join(spider_eos_dir, '.cache_info.txt')
    layout = '2phase' if is_twophase else 'unified'
    cache_key = f'P_max={P_max:.6e}_nP={nP}_nS={nS}_mzf={mzf}_layout={layout}'
    if os.path.isfile(cache_marker):
        with open(cache_marker) as f:
            existing_key = f.read().strip()
        if existing_key == cache_key:
            # Tables are up to date
            solidus_path = os.path.join(spider_eos_dir, 'solidus.dat')
            liquidus_path = os.path.join(spider_eos_dir, 'liquidus.dat')
            if os.path.isfile(solidus_path) and os.path.isfile(liquidus_path):
                logger.info(
                    'Reusing cached PALEOS-derived P-S entropy tables '
                    '(P_max=%.2e, %dx%d)', P_max, nP, nS,
                )
                return {
                    'eos_dir': spider_eos_dir,
                    'solidus_path': solidus_path,
                    'liquidus_path': liquidus_path,
                }

    # Generate phase boundaries
    logger.info(
        'Generating PALEOS-derived P-S phase boundaries (%d P points)...', nP,
    )
    pb_result = generate_spider_phase_boundaries(
        solidus_func=solidus_func,
        liquidus_func=liquidus_func,
        eos_file=eos_file,
        P_range=(1e5, P_max),
        n_P=nP,
        output_dir=spider_eos_dir,
        solid_eos_file=solid_eos,
        liquid_eos_file=liquid_eos,
    )

    # Generate full EOS tables
    logger.info(
        'Generating PALEOS-derived P-S EOS tables (%d x %d)...', nP, nS,
    )
    generate_spider_eos_tables(
        eos_file=eos_file,
        solidus_func=solidus_func,
        liquidus_func=liquidus_func,
        P_range=(1e5, P_max),
        n_P=nP,
        n_S=nS,
        output_dir=spider_eos_dir,
        solid_eos_file=solid_eos,
        liquid_eos_file=liquid_eos,
    )

    # Write cache marker so subsequent calls skip regeneration
    try:
        with open(cache_marker, 'w') as f:
            f.write(cache_key)
    except OSError:
        pass

    return {
        'eos_dir': spider_eos_dir,
        'solidus_path': pb_result['solidus_path'],
        'liquidus_path': pb_result['liquidus_path'],
    }


def zalmoxis_solver(
    config: Config,
    outdir: str,
    hf_row: dict,
    num_spider_nodes: int = 0,
    temperature_function=None,
    temperature_mode_override: str | None = None,
):
    """Run the Zalmoxis solver to compute the interior structure of a planet.

    Parameters
    ----------
    config : Config
        Configuration object.
    outdir : str
        Output directory where results will be saved.
    hf_row : dict
        Dictionary containing volatile masses and other parameters.
    num_spider_nodes : int
        Number of SPIDER basic nodes. If > 0, writes a SPIDER mesh file
        and returns its path as the second element of the return tuple.
    temperature_function : callable or None, optional
        External temperature function ``f(r, P) -> T`` in (m, Pa, K).
        When provided, bypasses Zalmoxis's internal temperature mode
        dispatch. Used to pass SPIDER/Aragog T(r) profiles in memory.
    temperature_mode_override : str or None, optional
        Local override for ``config.planet.temperature_mode``. Lets callers
        force a different structure-solve mode without mutating the shared
        Config object (see proteus rules §"Config mutability"). When None,
        the Config value is used. Currently used by
        ``determine_interior_radius_with_zalmoxis`` to force ``adiabatic``
        for SPIDER coupling with a T-dependent mantle EOS.

    Returns
    -------
    cmb_radius : float
        Core-mantle boundary radius [m].
    spider_mesh_file : str or None
        Path to the SPIDER mesh file, or None if ``num_spider_nodes == 0``.
    """

    # Load the Zalmoxis configuration parameters
    config_params = load_zalmoxis_configuration(
        config, hf_row, temperature_mode_override=temperature_mode_override
    )

    # Build volatile profile from dissolved volatile masses (if available).
    # This enables phi(r)-weighted volatile blending inside the Zalmoxis ODE.
    # Skipped when dry_mantle=True (Stage 1 phase-aware coupling): the
    # structure solver then uses only the canonical mantle EOS tables.
    mantle_eos = config.interior_struct.zalmoxis.mantle_eos
    if config.interior_struct.zalmoxis.dry_mantle:
        volatile_profile = None
        logger.info(
            'Structure solver: dry_mantle=True, skipping VolatileProfile '
            '(mantle EOS uses %s tables only).', mantle_eos,
        )
    else:
        volatile_profile = build_volatile_profile(hf_row, mantle_eos)

    # Configure global miscibility if enabled
    if config.interior_struct.zalmoxis.global_miscibility and volatile_profile is not None:
        volatile_profile.global_miscibility = True
        # Initialize x_interior from current dissolved masses
        M_mantle = float(hf_row.get('M_mantle', 0.0))
        if M_mantle > 0:
            H2_kg_liquid = float(hf_row.get('H2_kg_liquid', 0.0))
            if H2_kg_liquid > 0:
                volatile_profile.x_interior['Chabrier:H'] = H2_kg_liquid / (
                    M_mantle + H2_kg_liquid
                )
            H2O_kg_liquid = float(hf_row.get('H2O_kg_liquid', 0.0))
            if H2O_kg_liquid > 0:
                volatile_profile.x_interior['PALEOS:H2O'] = H2O_kg_liquid / (
                    M_mantle + H2O_kg_liquid
                )

    # Extend mantle EOS string with volatile components so the LayerMixture
    # includes them (VolatileProfile overrides fractions at each radius).
    if volatile_profile is not None:
        config_params['layer_eos_config']['mantle'] = extend_mantle_eos_with_volatiles(
            config_params['layer_eos_config']['mantle'], volatile_profile
        )

    # Get the output location for Zalmoxis output and create the file if it does not exist
    output_zalmoxis = get_zalmoxis_output_filepath(outdir)
    open(output_zalmoxis, 'a').close()

    # Run structure solve: use miscibility wrapper when enabled
    mat_dicts = load_zalmoxis_material_dictionaries()
    melt_funcs = load_zalmoxis_solidus_liquidus_functions(mantle_eos, config)
    input_data_dir = os.path.join(outdir, 'data')

    if config.interior_struct.zalmoxis.global_miscibility:
        from zalmoxis.solver import solve_miscible_interior

        # Build H2 mass targets from current volatile inventories
        h2_mass_targets = {}
        H2_kg_total = float(hf_row.get('H2_kg_total', 0.0))
        H2_kg_atm = float(hf_row.get('H2_kg_atm', 0.0))
        H2_kg_dissolved = H2_kg_total - H2_kg_atm
        if H2_kg_dissolved > 0:
            h2_mass_targets['Chabrier:H'] = H2_kg_dissolved

        H2O_kg_liquid = float(hf_row.get('H2O_kg_liquid', 0.0))
        if H2O_kg_liquid > 0:
            h2_mass_targets['PALEOS:H2O'] = H2O_kg_liquid

        model_results = solve_miscible_interior(
            config_params,
            material_dictionaries=mat_dicts,
            melting_curves_functions=melt_funcs,
            input_dir=input_data_dir,
            volatile_profile=volatile_profile,
            temperature_function=temperature_function,
            h2_mass_targets=h2_mass_targets,
            max_iterations=config.interior_struct.zalmoxis.miscibility_max_iter,
            mass_tolerance=config.interior_struct.zalmoxis.miscibility_tol,
        )

        # Write solvus info to hf_row
        if model_results.get('solvus_radius') is not None:
            hf_row['R_solvus'] = model_results['solvus_radius']
            hf_row['T_solvus'] = model_results['solvus_temperature']
            hf_row['P_solvus'] = model_results['solvus_pressure']
        hf_row['X_H2_int'] = model_results.get('x_interior_converged', {}).get(
            'Chabrier:H', 0.0
        )

        logger.info(
            'Global miscibility: solvus R=%.2e m, T=%.0f K, P=%.2e Pa, '
            'X_H2_int=%.4f, converged=%s (%d iters)',
            hf_row.get('R_solvus', 0.0),
            hf_row.get('T_solvus', 0.0),
            hf_row.get('P_solvus', 0.0),
            hf_row.get('X_H2_int', 0.0),
            model_results.get('miscibility_converged', False),
            model_results.get('miscibility_iterations', 0),
        )
    else:
        model_results = main(
            config_params,
            material_dictionaries=mat_dicts,
            melting_curves_functions=melt_funcs,
            input_dir=input_data_dir,
            volatile_profile=volatile_profile,
            temperature_function=temperature_function,
            p_center_hint=hf_row.get('P_center'),
            initial_density=_density_cache.get('density'),
            initial_radii=_density_cache.get('radii'),
        )

    # Extract results from the model
    radii = model_results['radii']
    density = model_results['density']
    gravity = model_results['gravity']
    pressure = model_results['pressure']
    temperature = model_results['temperature']
    mass_enclosed = model_results['mass_enclosed']
    cmb_mass = model_results['cmb_mass']
    core_mantle_mass = model_results['core_mantle_mass']
    converged = model_results['converged']
    converged_pressure = model_results['converged_pressure']
    converged_density = model_results['converged_density']
    converged_mass = model_results['converged_mass']

    # Adaptive retry: if mass convergence failed, retry with relaxed tolerances
    if not converged and converged_pressure and converged_density and not converged_mass:
        retry_tol = config_params.get('tolerance_outer', 3e-3) * 3
        retry_iter = int(config_params.get('max_iterations_outer', 100) * 2)
        logger.warning(
            'Zalmoxis mass convergence failed; retrying with relaxed tolerance '
            '(tol_outer=%.1e, max_iter=%d)',
            retry_tol,
            retry_iter,
        )
        config_params_retry = dict(config_params)
        config_params_retry['tolerance_outer'] = retry_tol
        config_params_retry['max_iterations_outer'] = retry_iter

        model_results = main(
            config_params_retry,
            material_dictionaries=mat_dicts,
            melting_curves_functions=melt_funcs,
            input_dir=input_data_dir,
            volatile_profile=volatile_profile,
            temperature_function=temperature_function,
        )

        radii = model_results['radii']
        pressure = model_results['pressure']
        temperature = model_results['temperature']
        mass_enclosed = model_results['mass_enclosed']
        cmb_mass = model_results['cmb_mass']
        core_mantle_mass = model_results['core_mantle_mass']
        converged = model_results['converged']
        converged_pressure = model_results['converged_pressure']
        converged_density = model_results['converged_density']
        converged_mass = model_results['converged_mass']

        if converged:
            logger.info('Zalmoxis converged on retry with relaxed tolerances')

    # Check convergence before proceeding. Non-converged solutions
    # (e.g. when EOS table range is exceeded) produce garbage values
    # that would corrupt the simulation state.
    if not converged:
        diag = (
            f'Zalmoxis did not converge: '
            f'pressure={converged_pressure}, density={converged_density}, '
            f'mass={converged_mass}. '
            f'Final M={mass_enclosed[-1]:.2e} kg, R={radii[-1]:.2e} m. '
            f'EOS: core={config.interior_struct.zalmoxis.core_eos}, '
            f'mantle={config.interior_struct.zalmoxis.mantle_eos}.'
        )
        logger.error(diag)
        raise RuntimeError(diag)

    # Extract the index of the core-mantle boundary mass in the mass array
    cmb_index = np.argmax(mass_enclosed >= cmb_mass)

    # Extract the planet radius and core-mantle boundary radius
    planet_radius = radii[-1]
    cmb_radius = radii[cmb_index]

    # Calculate the average density of the planet using the calculated mass and radius
    average_density = mass_enclosed[-1] / (4 / 3 * np.pi * radii[-1] ** 3)

    # Cache density for next call's Picard seeding
    _density_cache['density'] = model_results['density'].copy()
    _density_cache['radii'] = model_results['radii'].copy()

    # Final results of the Zalmoxis interior model
    logger.info('Found solution for interior structure with Zalmoxis')
    logger.info(
        f'Interior (dry calculated mass) mass: {mass_enclosed[-1]} kg or approximately {mass_enclosed[-1] / M_earth:.2f} M_earth'
    )
    logger.info(
        f'Interior radius: {planet_radius:.2e} m or {planet_radius / R_earth:.2f} R_earth'
    )
    logger.info(f'Core radius: {cmb_radius:.2e} or {cmb_radius / R_earth:.2f} R_earth')
    logger.info(f'Core-mantle boundary mass: {mass_enclosed[cmb_index]:.2e} kg')
    logger.info(f'Mantle density at the core-mantle boundary: {density[cmb_index]:.2e} kg/m^3')
    logger.info(
        f'Core density at the core-mantle boundary: {density[cmb_index - 1]:.2e} kg/m^3'
    )
    logger.info(f'Pressure at the core-mantle boundary: {pressure[cmb_index]:.2e} Pa')
    logger.info(f'Pressure at the center: {pressure[0]:.2e} Pa')
    logger.info(f'Average density: {average_density:.2e} kg/m^3')
    logger.info(
        f'Core-mantle boundary mass fraction: {mass_enclosed[cmb_index] / mass_enclosed[-1]:.3f}'
    )
    logger.info(f'Core radius fraction: {cmb_radius / planet_radius:.4f}')
    logger.info(
        f'Inner mantle radius fraction: {radii[np.argmax(mass_enclosed >= core_mantle_mass)] / planet_radius:.4f}'
    )
    logger.info(
        f'Overall Convergence Status: {converged} with Pressure: {converged_pressure}, Density: {converged_density}, Mass: {converged_mass}'
    )

    # Self-consistent initial thermal state (White+Li 2025, Boujibar+2020).
    # Honor temperature_mode_override here as well: the accretion thermal
    # state only runs when the user actually wants accretion mode, not
    # when SPIDER has forced a local adiabatic override.
    _effective_temp_mode = temperature_mode_override or config.planet.temperature_mode
    if _effective_temp_mode == 'accretion':
        from zalmoxis.energetics import initial_thermal_state

        cmf = config.interior_struct.core_frac
        mantle_eos = config.interior_struct.zalmoxis.mantle_eos

        # Build PALEOS-derived nabla_ad and C_p when PALEOS EOS is configured.
        # This uses the actual EOS tables for the adiabatic gradient and heat
        # capacities instead of the constant defaults (Gruneisen adiabat,
        # Dulong-Petit C_Fe=450, C_sil=1250 J/kg/K from White+Li 2025).
        nabla_ad_func = None
        cp_iron_func = None
        cp_silicate_func = None
        C_iron = 450.0
        C_silicate = 1250.0

        if 'PALEOS' in mantle_eos:
            try:
                import math

                from scipy.interpolate import LinearNDInterpolator
                from zalmoxis.eos.interpolation import load_paleos_unified_table

                # Get EOS file paths from the material dictionaries
                mat_dicts = load_zalmoxis_material_dictionaries()
                mantle_mat = mat_dicts.get(mantle_eos, {})
                core_mat = mat_dicts.get(config.interior_struct.zalmoxis.core_eos, {})

                # Build nabla_ad(P, T) from PALEOS MgSiO3 unified table
                mantle_file = mantle_mat.get('eos_file', '')
                if mantle_file and os.path.isfile(mantle_file):
                    _cache = load_paleos_unified_table(mantle_file)

                    def _paleos_nabla_ad(P_Pa, T_K, _c=_cache):
                        if P_Pa <= 0 or T_K <= 0:
                            return 0.3
                        lp = max(_c['logp_min'], min(math.log10(P_Pa), _c['logp_max']))
                        lt = max(_c['logt_min'], min(math.log10(T_K), _c['logt_max']))
                        try:
                            v = float(_c['nabla_ad_interp']([[lp, lt]])[0])
                            if np.isfinite(v) and v > 0:
                                return v
                        except Exception:
                            pass
                        return 0.3

                    nabla_ad_func = _paleos_nabla_ad
                    logger.info('Using PALEOS nabla_ad(P,T) for initial thermal state adiabat')

                # Build C_p(P, T) interpolators from PALEOS tables for
                # mass-weighted integration over the radial structure
                def _build_cp_func(eos_file, fallback_cp):
                    """Build a C_p(P, T) interpolator from a PALEOS table."""
                    if not eos_file or not os.path.isfile(eos_file):
                        return None
                    _data = np.genfromtxt(eos_file, usecols=range(9), comments='#')
                    _P, _T, _cp = _data[:, 0], _data[:, 1], _data[:, 5]
                    _valid = (_P > 0) & np.isfinite(_cp) & (_cp > 0) & (_cp < 5000)
                    if np.sum(_valid) < 10:
                        return None
                    _lp = np.log10(_P[_valid])
                    _lt = np.log10(_T[_valid])
                    _interp = LinearNDInterpolator(list(zip(_lp, _lt)), _cp[_valid])

                    def _cp_func(P_Pa, T_K, _i=_interp, _fb=fallback_cp):
                        if P_Pa <= 0 or T_K <= 0:
                            return _fb
                        v = float(_i(math.log10(P_Pa), math.log10(T_K)))
                        if np.isfinite(v) and 0 < v < 5000:
                            return v
                        return _fb

                    return _cp_func

                core_file = core_mat.get('eos_file', '')
                cp_iron_func = _build_cp_func(core_file, C_iron)
                cp_silicate_func = _build_cp_func(mantle_file, C_silicate)

                if cp_iron_func is not None:
                    logger.info('Using PALEOS C_p(P,T) for iron (mass-weighted integration)')
                if cp_silicate_func is not None:
                    logger.info(
                        'Using PALEOS C_p(P,T) for silicate (mass-weighted integration)'
                    )

            except Exception as e:
                logger.warning(
                    'Could not build PALEOS thermal properties: %s. Using constants.', e
                )

        thermal = initial_thermal_state(
            model_results,
            core_mass_fraction=cmf,
            T_radiative_eq=hf_row.get('T_eqm', 255.0),
            f_accretion=config.planet.f_accretion,
            f_differentiation=config.planet.f_differentiation,
            C_iron=C_iron,
            C_silicate=C_silicate,
            nabla_ad_func=nabla_ad_func,
            cp_iron_func=cp_iron_func,
            cp_silicate_func=cp_silicate_func,
        )
        hf_row['T_cmb_initial'] = thermal['T_cmb']
        hf_row['T_surf_accr'] = thermal['T_surf_accr']
        # Key consumed by Aragog setup_solver and _set_entropy_ic
        hf_row['T_surface_initial'] = thermal['T_surf_accr']
        hf_row['U_grav_diff'] = thermal['U_differentiated']
        hf_row['U_grav_undiff'] = thermal['U_undifferentiated']
        hf_row['DeltaT_accretion'] = thermal['Delta_T_accretion']
        hf_row['DeltaT_differentiation'] = thermal['Delta_T_differentiation']
        hf_row['DeltaT_adiabat'] = thermal['Delta_T_adiabat']
        hf_row['core_state_initial'] = thermal['core_state']

        # Store the adiabatic T(r) profile for interior solver initialization.
        # SPIDER/Aragog use this to set the initial temperature/entropy profile.
        hf_row['_initial_T_profile'] = thermal['T_profile']
        hf_row['_initial_T_radii'] = thermal['radii']
        hf_row['_initial_T_pressure'] = thermal['pressure']

        logger.info(
            'Initial thermal state (White+Li 2025): T_CMB=%.0f K, '
            'T_surf_accr=%.0f K, DeltaT_G=%.0f K, DeltaT_D=%.0f K, '
            'DeltaT_ad=%.0f K, core=%s',
            thermal['T_cmb'],
            thermal['T_surf_accr'],
            thermal['Delta_T_accretion'],
            thermal['Delta_T_differentiation'],
            thermal['Delta_T_adiabat'],
            thermal['core_state'],
        )

    # Update the surface radius, interior radius, and mass in the hf_row
    hf_row['R_int'] = planet_radius
    hf_row['R_core'] = cmb_radius
    hf_row['M_int'] = mass_enclosed[-1]
    hf_row['M_core'] = mass_enclosed[cmb_index]
    hf_row['gravity'] = gravity[-1]
    hf_row['P_center'] = model_results.get('p_center')
    hf_row['P_cmb'] = float(pressure[cmb_index])

    # Self-consistent core density from Zalmoxis structure
    if cmb_radius > 0:
        hf_row['core_density'] = mass_enclosed[cmb_index] / (4.0 / 3.0 * np.pi * cmb_radius**3)
    else:
        hf_row['core_density'] = 0.0

    # Core heat capacity: when 'self', use Dulong-Petit for iron (~450 J/kg/K).
    # When numeric, use the config value directly.
    cfg_heatcap = config.interior_struct.core_heatcap
    hf_row['core_heatcap'] = 450.0 if cfg_heatcap == 'self' else float(cfg_heatcap)

    logger.info(f'Saving Zalmoxis output to {output_zalmoxis}')

    # Select mantle arrays (to match the mesh needed for Aragog)
    mantle_radii = radii[cmb_index:]
    mantle_pressure = pressure[cmb_index:]
    mantle_density = density[cmb_index:]
    mantle_gravity = gravity[cmb_index:]
    mantle_temperature = temperature[cmb_index:]

    # Scale mantle temperature to match Aragog temperature profile format
    mantle_temperature_scaled = scale_temperature_profile_for_aragog(
        config, mantle_radii, mantle_temperature
    )

    # Write temperature profile to a separate file for Aragog to read
    np.savetxt(
        os.path.join(outdir, 'data', 'zalmoxis_output_temp.txt'), mantle_temperature_scaled
    )

    # Stage 1c.4 scalar-g control knob: when
    # ``interior_energetics.aragog.scalar_gravity_override`` is True,
    # collapse the radial gravity array into a uniform scalar (the
    # surface value from hf_row['gravity']) for the files Aragog and
    # SPIDER both read. Aragog's per-node path at solver.reset() then
    # interpolates to that constant everywhere, which is functionally
    # identical to the pre-1c.1 scalar-g code path. Avoids having to
    # pin the aragog submodule back to 8fc5072 for a scalar-g control
    # run.
    scalar_g_override = getattr(
        config.interior_energetics.aragog, 'scalar_gravity_override', False
    )
    if scalar_g_override:
        g_scalar = float(hf_row.get('gravity', 9.81))
        mantle_gravity_out = np.full_like(mantle_gravity, g_scalar)
        logger.info(
            'scalar_gravity_override=True: collapsing zalmoxis_output.dat + '
            'spider_mesh.dat gravity column to uniform %.4f m/s^2 '
            '(Stage 1c.4 control run)',
            g_scalar,
        )
    else:
        mantle_gravity_out = mantle_gravity

    # Save final grids to the output file for the mantle for Aragog
    with open(output_zalmoxis, 'w') as f:
        for i in range(len(mantle_radii)):
            f.write(
                f'{mantle_radii[i]:.17e} {mantle_pressure[i]:.17e} {mantle_density[i]:.17e} {mantle_gravity_out[i]:.17e} {mantle_temperature[i]:.17e}\n'
            )

    # Determine SPIDER domain: [R_cmb, R_solvus] when global_miscibility is
    # enabled, otherwise [R_cmb, R_surface] (standard).
    spider_radii = mantle_radii
    spider_pressure = mantle_pressure
    spider_density = mantle_density
    spider_gravity = mantle_gravity

    if config.interior_struct.zalmoxis.global_miscibility:
        R_solvus = hf_row.get('R_solvus')
        if R_solvus is not None and R_solvus < planet_radius:
            # Truncate arrays at the solvus: SPIDER only evolves the
            # miscible interior below the binodal surface
            solvus_mask = mantle_radii <= R_solvus * 1.001  # small tolerance
            if np.any(solvus_mask):
                spider_radii = mantle_radii[solvus_mask]
                spider_pressure = mantle_pressure[solvus_mask]
                spider_density = mantle_density[solvus_mask]
                spider_gravity = mantle_gravity[solvus_mask]
                logger.info(
                    'SPIDER domain truncated at solvus: R_solvus=%.3e m '
                    '(%.2f R_earth), %d of %d shells',
                    R_solvus,
                    R_solvus / R_earth,
                    len(spider_radii),
                    len(mantle_radii),
                )

    # Write SPIDER mesh file if requested. Re-uses the possibly-collapsed
    # gravity array so the SPIDER path gets the same scalar-g override
    # behaviour when the flag is on.
    spider_mesh_file = None
    if num_spider_nodes > 0:
        if scalar_g_override:
            spider_gravity_out = np.full_like(spider_gravity, float(hf_row.get('gravity', 9.81)))
        else:
            spider_gravity_out = spider_gravity
        spider_mesh_file = write_spider_mesh_file(
            outdir,
            spider_radii,
            spider_pressure,
            spider_density,
            spider_gravity_out,
            num_spider_nodes,
        )

    return cmb_radius, spider_mesh_file
