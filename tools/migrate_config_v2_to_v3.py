#!/usr/bin/env python3
"""Translate a PROTEUS configuration file from format version 2.0 to 3.0.

The 2.0 to 3.0 change is a structural refactor, not a value tweak: the interior
splits into ``[interior_struct]`` and ``[interior_energetics]``, the volatile
inventory moves under ``[planet]``, several fields are renamed, and a number of
schema defaults change. The config loader ignores unknown keys, so a naive
rename pass silently reverts moved or omitted fields to a 3.0 default that may
differ from the 2.0 value, producing a file that loads cleanly but simulates a
different planet.

This tool avoids that failure mode with a materialise-map-emit engine:

1. Materialise the full 2.0 configuration by overlaying the user's explicit
   values on the 2.0 schema defaults (``migrate_data/v2_defaults.json``), so no
   field is left implicit.
2. Map every materialised field to its 3.0 location, applying renames and value
   transforms. Inactive-module sub-blocks are dropped (they never affect the
   run); any active-module field with no 3.0 home raises a warning so nothing is
   lost silently.
3. Emit only the values that deviate from the 3.0 default, plus the fields the
   user set explicitly and the small set of new-in-3.0 fields whose default
   changes behaviour relative to 2.0 (for example ``kappah_floor``). Every
   divergent default, whether at an identical path or a renamed one, is therefore
   pinned automatically.

The result is validated by structuring it through the 3.0 ``Config`` schema; a
structural error is raised rather than written. The unit test
(``tests/tools/test_migrate_config_v2_to_v3.py``) checks map completeness,
new-field classification, and per-field regression. A separate developer harness
resolves each 2.0 input through main's loader and the 3.0 output through this
branch's loader and asserts field-by-field equivalence across the map; because
it needs both schema revisions importable from two checkouts, it is run manually
against a checkout of main rather than in the unit suite.

Usage
-----
    python tools/migrate_config_v2_to_v3.py old_v2.toml -o new_v3.toml
    python tools/migrate_config_v2_to_v3.py old_v2.toml          # writes old_v2.v3.toml
    python tools/migrate_config_v2_to_v3.py old.grid.toml --grid -o new.grid.toml

Notes
-----
The interior structure module ``zalmoxis`` and the energetics module ``aragog``
were substantially redesigned between 2.0 and 3.0. When a config selects one of
those as its active module, the tool maps the fields it can and warns on the
rest; an exact reproduction of those two backends is not guaranteed and the
warning says so. The common 2.0 stack (SPIDER-internal structure, SPIDER or
dummy energetics, AGNI or dummy atmosphere, CALLIOPE outgassing) maps faithfully.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

_DATA = Path(__file__).resolve().parent / 'migrate_data'

# ----------------------------------------------------------------------------
# Value transforms
# ----------------------------------------------------------------------------


def _struct_module(v):
    """Map the 2.0 structure module name to its 3.0 equivalent.

    The 2.0 ``"self"`` module (SPIDER computes its own radial structure) is
    named ``"spider"`` in 3.0; ``zalmoxis`` and ``dummy`` are unchanged.
    """
    return {'self': 'spider'}.get(v, v)


def _mixing_length(v):
    """Map the 2.0 integer mixing-length code to the 3.0 string.

    SPIDER uses ``1`` for the nearest-boundary scheme and ``2`` for the
    constant scheme; 3.0 expresses these as ``"nearest"`` and ``"constant"``.
    """
    return {1: 'nearest', 2: 'constant'}.get(v, v)


def _volatile_mode(v):
    """Map the 2.0 ``delivery.initial`` value to the 3.0 ``volatile_mode``."""
    return {'volatiles': 'gas_prs', 'elements': 'elements'}.get(v, v)


def _to_bool(v):
    """Coerce a possibly-float 2.0 flag (``0.0``/``1.0``) to bool."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return v


# Earth radius [m]. 2.0 struct.radius_int is in Earth radii (consumed as
# ``radius_int * R_earth``); 3.0 planet.R_int_override is in metres. Hard-coded
# to keep the value transform independent of an importable proteus at call time;
# verified against proteus.utils.constants.R_earth.
_R_EARTH_M = 6335439.0


def _radius_int_to_metres(v):
    """Convert a 2.0 ``radius_int`` (Earth radii) to 3.0 metres.

    The 'none' sentinel (no fixed radius) passes through unchanged so the keep
    filter handles it as an unset value.
    """
    if v in (None, 'none') or not isinstance(v, (int, float)):
        return v
    return float(v) * _R_EARTH_M


# ----------------------------------------------------------------------------
# Static field map (only the deltas; identical paths are copied automatically)
# ----------------------------------------------------------------------------

# Simple relocations and renames: 2.0 dotted path -> 3.0 dotted path.
# Value transforms (if any) are applied via TRANSFORMS keyed on the 2.0 path.
RENAMES = {
    'version': 'config_version',
    # structure
    'struct.mass_tot': 'planet.mass_tot',
    'struct.corefrac': 'interior_struct.core_frac',
    'struct.module': 'interior_struct.module',
    'struct.core_density': 'interior_struct.core_density',
    'struct.core_heatcap': 'interior_struct.core_heatcap',
    'struct.radius_int': 'planet.R_int_override',
    'struct.mesh_convergence_interval': 'interior_struct.zalmoxis.mesh_convergence_interval',
    'struct.mesh_max_shift': 'interior_struct.zalmoxis.mesh_max_shift',
    'struct.update_dphi_abs': 'interior_struct.zalmoxis.update_dphi_abs',
    'struct.update_dtmagma_frac': 'interior_struct.zalmoxis.update_dtmagma_frac',
    'struct.update_interval': 'interior_struct.zalmoxis.update_interval',
    'struct.update_min_interval': 'interior_struct.zalmoxis.update_min_interval',
    # zalmoxis structure block: overlapping fields (mapped only when zalmoxis is
    # the active structure module; otherwise dropped as inactive). The 3.0
    # zalmoxis (PALEOS/Newton) differs from 2.0 (Seager2007), so the Seager-only
    # solver internals have no 3.0 home and are dropped under the summary warning.
    'struct.zalmoxis.core_eos': 'interior_struct.zalmoxis.core_eos',
    'struct.zalmoxis.mantle_eos': 'interior_struct.zalmoxis.mantle_eos',
    'struct.zalmoxis.ice_layer_eos': 'interior_struct.zalmoxis.ice_layer_eos',
    'struct.zalmoxis.mantle_mass_fraction': 'interior_struct.zalmoxis.mantle_mass_fraction',
    'struct.zalmoxis.num_levels': 'interior_struct.zalmoxis.num_levels',
    'struct.zalmoxis.tolerance_outer': 'interior_struct.zalmoxis.solver_tol_outer',
    'struct.zalmoxis.tolerance_inner': 'interior_struct.zalmoxis.solver_tol_inner',
    'struct.zalmoxis.max_iterations_outer': 'interior_struct.zalmoxis.solver_max_iter_outer',
    'struct.zalmoxis.max_iterations_inner': 'interior_struct.zalmoxis.solver_max_iter_inner',
    # interior, shared top-level
    'interior.F_initial': 'interior_energetics.flux_guess',
    'interior.grain_size': 'interior_energetics.grain_size',
    'interior.melting_dir': 'interior_struct.melting_dir',
    'interior.eos_dir': 'interior_struct.eos_dir',
    'interior.module': 'interior_energetics.module',
    'interior.radiogenic_heat': 'interior_energetics.heat_radiogenic',
    'interior.tidal_heat': 'interior_energetics.heat_tidal',
    'interior.rheo_phi_loc': 'interior_energetics.rfront_loc',
    'interior.rheo_phi_wid': 'interior_energetics.rfront_wid',
    # delivery, radiogenic + accretion + volatile mode
    'delivery.radio_K': 'interior_energetics.radio_K',
    'delivery.radio_Th': 'interior_energetics.radio_Th',
    'delivery.radio_U': 'interior_energetics.radio_U',
    'delivery.radio_tref': 'interior_energetics.radio_tref',
    'delivery.module': 'accretion.module',
    'delivery.initial': 'planet.volatile_mode',
    'delivery.elements.use_metallicity': 'planet.elements.use_metallicity',
    'delivery.elements.metallicity': 'planet.elements.metallicity',
    # outgas
    'outgas.calliope.T_floor': 'outgas.T_floor',
    'outgas.calliope.rtol': 'outgas.solver_rtol',
    'outgas.calliope.xtol': 'outgas.solver_atol',
    # atmosphere top-level relocations
    'atmos_clim.prevent_warming': 'planet.prevent_warming',
    'atmos_clim.tmp_maximum': 'atmos_clim.janus.tmp_maximum',
    # time-stepping flatten
    'params.dt.proportional.propconst': 'params.dt.propconst',
    'params.dt.adaptive.atol': 'params.dt.atol',
    'params.dt.adaptive.rtol': 'params.dt.rtol',
    'params.dt.adaptive.scale_incr': 'params.dt.scale_incr',
    'params.dt.adaptive.scale_decr': 'params.dt.scale_decr',
    'params.dt.adaptive.window': 'params.dt.window',
}

# Per-active-interior-module renames (the SPIDER / Aragog / dummy sub-block
# fields that feed the shared [interior_energetics] axis or move elsewhere).
INTERIOR_RENAMES = {
    'spider': {
        'interior.spider.conduction': 'interior_energetics.trans_conduction',
        'interior.spider.convection': 'interior_energetics.trans_convection',
        'interior.spider.gravitational_separation': 'interior_energetics.trans_grav_sep',
        'interior.spider.mixing': 'interior_energetics.trans_mixing',
        'interior.spider.num_levels': 'interior_energetics.num_levels',
        'interior.spider.tolerance': 'interior_energetics.atol',
        'interior.spider.tolerance_rel': 'interior_energetics.rtol',
        'interior.spider.tsurf_atol': 'interior_energetics.tmagma_atol',
        'interior.spider.tsurf_rtol': 'interior_energetics.tmagma_rtol',
        'interior.spider.mixing_length': 'interior_energetics.mixing_length',
        'interior.spider.solver_type': 'interior_energetics.spider.solver_type',
        'interior.spider.matprop_smooth_width': 'interior_energetics.spider.matprop_smooth_width',
        'interior.spider.tolerance_struct': 'interior_energetics.spider.tolerance_struct',
        'interior.spider.log_output': 'interior_energetics.spider.log_output',
    },
    'aragog': {
        'interior.aragog.conduction': 'interior_energetics.trans_conduction',
        'interior.aragog.convection': 'interior_energetics.trans_convection',
        'interior.aragog.gravitational_separation': 'interior_energetics.trans_grav_sep',
        'interior.aragog.mixing': 'interior_energetics.trans_mixing',
        'interior.aragog.num_levels': 'interior_energetics.num_levels',
        'interior.aragog.tolerance': 'interior_energetics.rtol',
        'interior.aragog.mass_coordinates': 'interior_energetics.aragog.mass_coordinates',
    },
    'dummy': {
        'interior.dummy.mantle_cp': 'interior_energetics.dummy.mantle_cp',
        'interior.dummy.mantle_rho': 'interior_energetics.dummy.mantle_rho',
        'interior.dummy.mantle_tliq': 'interior_energetics.dummy.mantle_tliq',
        'interior.dummy.mantle_tsol': 'interior_energetics.dummy.mantle_tsol',
        'interior.dummy.H_radio': 'interior_energetics.dummy.heat_internal',
        'interior.dummy.tmagma_atol': 'interior_energetics.tmagma_atol',
        'interior.dummy.tmagma_rtol': 'interior_energetics.tmagma_rtol',
    },
}

# Value transforms keyed on the 2.0 path.
TRANSFORMS = {
    'version': lambda v: '3.0',
    'struct.module': _struct_module,
    'interior.spider.mixing_length': _mixing_length,
    'delivery.initial': _volatile_mode,
    'delivery.elements.use_metallicity': _to_bool,
    'orbit.instellation_method': lambda v: {'sma': 'distance'}.get(v, v),
    # 2.0 radius_int is in Earth radii; 3.0 R_int_override is in metres.
    'struct.radius_int': _radius_int_to_metres,
    # 2.0 used an empty string for "no ice layer"; 3.0 uses 'none'.
    'struct.zalmoxis.ice_layer_eos': lambda v: 'none' if v in ('', None) else v,
}

# 2.0 paths intentionally dropped (removed features / placeholders). Dropping a
# field the user set explicitly emits a warning; dropping a defaulted field is
# silent.
REMOVED = {
    'outgas.atmodeller.some_parameter',  # 2.0 placeholder, no 3.0 analogue
}

# New-in-3.0 fields whose 3.0 default changes behaviour relative to a 2.0 run
# that never had the field. Pinned to the value that reproduces 2.0, but only
# when the user did not set the field explicitly (an explicit choice wins).
OVERRIDES = {
    # 2.0 never passed -kappah_floor to SPIDER; the 3.0 default 10.0 would add a
    # near-solidus eddy-diffusivity floor that 2.0 runs did not apply.
    'interior_energetics.kappah_floor': 0.0,
    # 2.0 capped the time-step strictly at dt.maximum (timestep.py min(dt, max)).
    # 3.0 adds dt.maximum + maximum_rel * Time with a default maximum_rel of 1.0,
    # a time-growing cap 2.0 lacked. Pin 0.0 to recover the strict 2.0 cap.
    'params.dt.maximum_rel': 0.0,
}

# Element-budget fields consumed by the element handler (not mapped directly).
_ELEMENT_FIELDS = {
    'delivery.elements.H_ppmw',
    'delivery.elements.H_kg',
    'delivery.elements.H_oceans',
    'delivery.elements.CH_ratio',
    'delivery.elements.C_ppmw',
    'delivery.elements.C_kg',
    'delivery.elements.NH_ratio',
    'delivery.elements.N_ppmw',
    'delivery.elements.N_kg',
    'delivery.elements.SH_ratio',
    'delivery.elements.S_ppmw',
    'delivery.elements.S_kg',
}

# Initial-condition fields consumed by the temperature-mode handler.
_IC_FIELDS = {
    'interior.spider.ini_entropy',
    'interior.spider.ini_dsdr',
    'interior.aragog.ini_tmagma',
    'interior.dummy.ini_tmagma',
}

# Volatiles partial-pressure block: delivery.volatiles.X -> planet.gas_prs.X.
_VOLATILE_SPECIES = ('H2O', 'CO2', 'N2', 'S2', 'SO2', 'H2S', 'NH3', 'H2', 'CH4', 'CO')

# Grid/spectral fields that moved from the per-module atmosphere sub-block up to
# the shared [atmos_clim] level in 3.0. Hoisted from the active module's block.
_ATMOS_SHARED = (
    'spectral_group',
    'spectral_bands',
    'num_levels',
    'p_top',
    'p_obs',
    'overlap_method',
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _flatten(d, prefix=''):
    """Flatten a nested dict into ``{dotted.path: value}``."""
    out = {}
    for k, v in d.items():
        p = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, p))
        else:
            out[p] = v
    return out


def _unflatten(flat):
    """Rebuild a nested dict from ``{dotted.path: value}``."""
    root = {}
    for path, val in flat.items():
        parts = path.split('.')
        node = root
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = val
    return root


def _normalise(v):
    """Normalise a value for default-equality comparison.

    Treats the string ``"none"`` and ``None`` as equal, and compares numbers by
    value so ``1`` and ``1.0`` match.
    """
    if v is None or (isinstance(v, str) and v.lower() == 'none'):
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    return v


class MigrationReport:
    """Record of what the translation did, for the user-facing summary."""

    def __init__(self):
        self.renamed = []
        self.pinned_divergent = []
        self.overridden = []
        self.dropped_inactive = []
        self.dropped_removed = set()
        self.warnings = []

    def text(self):
        lines = []
        if self.warnings:
            lines.append('Warnings:')
            lines += [f'  ! {w}' for w in self.warnings]
        lines.append(f'Renamed/moved fields: {len(self.renamed)}')
        if self.pinned_divergent:
            lines.append('Pinned (2.0 value differs from 3.0 default):')
            lines += [f'  - {p} = {v!r}' for p, v in sorted(self.pinned_divergent)]
        if self.overridden:
            lines.append('Backwards-compat overrides for new 3.0 fields:')
            lines += [f'  - {p} = {v!r}' for p, v in sorted(self.overridden)]
        if self.dropped_removed:
            lines.append(f'Dropped (removed in 3.0): {sorted(self.dropped_removed)}')
        if self.dropped_inactive:
            lines.append(f'Dropped inactive-module fields: {len(self.dropped_inactive)}')
        return '\n'.join(lines)


# ----------------------------------------------------------------------------
# Core translation
# ----------------------------------------------------------------------------


def _load_v2_defaults():
    with open(_DATA / 'v2_defaults.json') as f:
        return json.load(f)


def _v3_defaults():
    """Return the live 3.0 schema defaults as a flat dict (stub-import safe)."""
    import types

    if 'proteus.proteus' not in sys.modules:
        stub = types.ModuleType('proteus.proteus')
        stub.Proteus = object
        sys.modules['proteus.proteus'] = stub
    import typing

    import attr

    from proteus.config._config import Config

    def sub(f):
        fac = getattr(f.default, 'factory', None)
        if fac is not None and attr.has(fac):
            return fac
        t = f.type
        if isinstance(t, str) or t is None:
            return None
        if attr.has(t):
            return t
        for a in typing.get_args(t) or ():
            if attr.has(a):
                return a
        return None

    none_ok = set()

    def accepts_none(f):
        """True if the field maps the string 'none' to None (none_if_none)."""
        conv = f.converter
        if conv is None:
            return False
        try:
            return conv('none') is None
        except Exception:
            return False

    def walk(cls, prefix='', seen=frozenset()):
        try:
            attr.resolve_types(cls)
        except Exception:
            pass
        out = {}
        for f in attr.fields(cls):
            p = f'{prefix}.{f.name}' if prefix else f.name
            s = sub(f)
            if s is not None and s not in seen:
                out.update(walk(s, p, seen | {s}))
                continue
            if accepts_none(f):
                none_ok.add(p)
            d = f.default
            fac = getattr(d, 'factory', None)
            if d is attr.NOTHING:
                out[p] = '<REQUIRED>'
            elif fac is not None:
                try:
                    out[p] = fac()
                except Exception:
                    out[p] = '<FACTORY>'
            else:
                out[p] = d
        return out

    return walk(Config), none_ok


def _active_modules(eff_v2):
    return {
        'struct': _struct_module(eff_v2.get('struct.module')),
        'interior': eff_v2.get('interior.module'),
        'atmos_clim': eff_v2.get('atmos_clim.module'),
        'outgas': eff_v2.get('outgas.module'),
        'escape': eff_v2.get('escape.module'),
        'star': eff_v2.get('star.module'),
    }


def _inactive_prefixes(active):
    """Sub-block prefixes to drop because their module is not active."""
    drop = []
    # structure: zalmoxis block + zalmoxis-only struct knobs unless zalmoxis active
    if active['struct'] != 'zalmoxis':
        drop += ['struct.zalmoxis.', 'struct.mesh_', 'struct.update_']
    # interior energetics: the non-active solver sub-blocks
    for mod in ('spider', 'aragog', 'dummy', 'boundary'):
        if active['interior'] != mod:
            drop.append(f'interior.{mod}.')
    # atmosphere: the two non-active sub-blocks
    for mod in ('agni', 'janus', 'dummy'):
        if active['atmos_clim'] != mod:
            drop.append(f'atmos_clim.{mod}.')
    # star / escape / outgas inactive sub-blocks
    for mod in ('mors', 'dummy'):
        if active['star'] != mod:
            drop.append(f'star.{mod}.')
    for mod in ('zephyrus', 'dummy', 'boreas'):
        if active['escape'] != mod:
            drop.append(f'escape.{mod}.')
    for mod in ('calliope', 'atmodeller'):
        if active['outgas'] != mod:
            drop.append(f'outgas.{mod}.')
    return tuple(drop)


def _handle_elements(eff_v2, explicit, v3, report):
    """Derive planet.elements.* mode/budget pairs from the 2.0 element block."""

    def g(name):
        return eff_v2.get(f'delivery.elements.{name}', 0.0) or 0.0

    # Hydrogen: oceans/kg are absolute (additive in 2.0), ppmw is relative.
    h_oceans, h_kg, h_ppmw = g('H_oceans'), g('H_kg'), g('H_ppmw')
    if h_ppmw and (h_oceans or h_kg):
        report.warnings.append(
            'H set by both ppmw and absolute (oceans/kg) in 2.0; using ppmw. '
            'Check planet.elements.H_*.'
        )
    if h_ppmw:
        v3['planet.elements.H_mode'], v3['planet.elements.H_budget'] = 'ppmw', h_ppmw
    elif h_oceans and h_kg:
        report.warnings.append(
            'H set by both oceans and kg in 2.0 (additive); 3.0 supports one '
            'mode. Using oceans; fold kg into it if needed.'
        )
        v3['planet.elements.H_mode'], v3['planet.elements.H_budget'] = 'oceans', h_oceans
    elif h_kg:
        v3['planet.elements.H_mode'], v3['planet.elements.H_budget'] = 'kg', h_kg
    elif h_oceans:
        v3['planet.elements.H_mode'], v3['planet.elements.H_budget'] = 'oceans', h_oceans
    else:
        v3['planet.elements.H_mode'], v3['planet.elements.H_budget'] = 'oceans', 0.0

    # C / N / S: ratio-to-H takes precedence, then ppmw, then kg.
    for el, ratio_key, ratio_mode in (
        ('C', 'CH_ratio', 'C/H'),
        ('N', 'NH_ratio', 'N/H'),
        ('S', 'SH_ratio', 'S/H'),
    ):
        ratio, ppmw, kg = g(ratio_key), g(f'{el}_ppmw'), g(f'{el}_kg')
        # 2.0 sums the ppmw (relative) and kg (absolute) terms; 3.0 carries a
        # single mode, so a config that set both cannot be reproduced exactly.
        if ppmw and kg:
            report.warnings.append(
                f'{el} set by both ppmw and kg in 2.0 (which sums them); 3.0 '
                f'supports one mode. Using ppmw={ppmw}; fold the kg term in by '
                f'hand if it matters.'
            )
        if ratio:
            mode, budget = ratio_mode, ratio
        elif ppmw:
            mode, budget = 'ppmw', ppmw
        elif kg:
            mode, budget = 'kg', kg
        else:
            mode, budget = ratio_mode, 0.0
        v3[f'planet.elements.{el}_mode'] = mode
        v3[f'planet.elements.{el}_budget'] = budget

    # Oxygen: no 2.0 field. ic_chemistry reproduces the buffered 2.0 behaviour.
    v3['planet.elements.O_mode'] = 'ic_chemistry'
    v3['planet.elements.O_budget'] = 0.0

    # use_metallicity: 2.0 scaled N/S from metallicity only when NH_ratio/SH_ratio
    # were > 0; 3.0 scales them unconditionally. The gating field is gone, so a
    # 2.0 config that relied on it cannot be reproduced exactly.
    if _to_bool(eff_v2.get('delivery.elements.use_metallicity', False)):
        report.warnings.append(
            'use_metallicity = true: 3.0 scales N and S from metallicity '
            'unconditionally (2.0 gated this on NH_ratio/SH_ratio > 0). Verify '
            'the intended N and S inclusion in the 3.0 run.'
        )


def _handle_atmos_hoist(eff_v2, explicit, v3, active, report):
    """Hoist the active atmosphere module's grid/spectral fields to atmos_clim.

    In 2.0 spectral_group/spectral_bands/num_levels/p_top/p_obs/overlap_method
    lived under [atmos_clim.agni] and [atmos_clim.janus] separately; in 3.0 they
    are shared fields on [atmos_clim], taken from the active module.
    """
    mod = active['atmos_clim']
    if mod not in ('agni', 'janus'):
        return
    for field in _ATMOS_SHARED:
        src = f'atmos_clim.{mod}.{field}'
        if src in eff_v2:
            v3[f'atmos_clim.{field}'] = eff_v2[src]


def _handle_temperature_mode(eff_v2, explicit, v3, active, report):
    """Infer planet.temperature_mode and the initial-condition fields."""
    mod = active['interior']
    if mod == 'spider':
        ini_s = eff_v2.get('interior.spider.ini_entropy')
        if ini_s not in (None, 'none'):
            v3['planet.temperature_mode'] = 'isentropic'
            v3['planet.ini_entropy'] = ini_s
            dsdr = eff_v2.get('interior.spider.ini_dsdr')
            if dsdr not in (None, 'none'):
                v3['planet.ini_dsdr'] = dsdr
        else:
            report.warnings.append(
                'SPIDER interior without ini_entropy; temperature_mode left at '
                'the 3.0 default. Set planet.temperature_mode explicitly.'
            )
    elif mod in ('aragog', 'dummy'):
        ini_t = eff_v2.get(f'interior.{mod}.ini_tmagma')
        if ini_t not in (None, 'none'):
            v3['planet.temperature_mode'] = 'isothermal'
            v3['planet.tsurf_init'] = ini_t
        if mod == 'aragog':
            report.warnings.append(
                'Aragog energetics was redesigned in 3.0. The initial condition '
                'is mapped as isothermal from ini_tmagma; review '
                'planet.temperature_mode and the [interior_energetics] block.'
            )


def translate(v2_toml: dict):
    """Translate a parsed 2.0 config dict into a validated 3.0 config dict.

    Parameters
    ----------
    v2_toml : dict
        The parsed 2.0 TOML (``tomllib.load`` output).

    Returns
    -------
    nested_v3 : dict
        The 3.0 config as a nested dict, validated through the 3.0 schema.
    report : MigrationReport
        What the translation renamed, pinned, overrode, dropped, and warned on.
    """
    report = MigrationReport()
    v2_defaults = _load_v2_defaults()
    v3_defaults, none_ok = _v3_defaults()

    explicit = set(_flatten(v2_toml))
    # Materialise: 2.0 defaults overlaid with explicit user values. User fields
    # absent from the snapshot (real 2.0 configs may carry fields beyond the
    # snapshot's schema revision) are still included so the mapper can place
    # them; if they have no 3.0 home the mapper warns rather than silently drops.
    eff_v2 = dict(v2_defaults)
    eff_v2.update(_flatten(v2_toml))

    active = _active_modules(eff_v2)
    if active['interior'] == 'aragog':
        report.warnings.append(
            'Active interior module is aragog; 3.0 aragog differs from 2.0. '
            'Faithful reproduction is not guaranteed; review the result.'
        )
    if active['struct'] == 'zalmoxis':
        report.warnings.append(
            'Active structure module is zalmoxis; 3.0 zalmoxis (PALEOS/Newton) '
            'differs from 2.0 (Seager2007). Review interior_struct.zalmoxis.'
        )

    inactive = _inactive_prefixes(active)
    inactive_bare = {p[:-1] for p in inactive}  # bare block leaves (Optional=None)
    interior_renames = INTERIOR_RENAMES.get(active['interior'], {})

    v3 = {}  # flat 3.0 path -> value
    provenance_explicit = set()  # 3.0 paths originating from explicit 2.0 fields

    def emit(v3_path, value, src_path):
        v3[v3_path] = value
        if src_path in explicit:
            provenance_explicit.add(v3_path)

    atmos_hoist_src = {f'atmos_clim.{active["atmos_clim"]}.{field}' for field in _ATMOS_SHARED}
    for v2_path, val in eff_v2.items():
        # consumed by special handlers
        if v2_path in _ELEMENT_FIELDS or v2_path in _IC_FIELDS or v2_path in atmos_hoist_src:
            continue
        if v2_path.startswith('delivery.volatiles.'):
            sp = v2_path.split('.')[-1]
            if sp in _VOLATILE_SPECIES:
                emit(f'planet.gas_prs.{sp}', val, v2_path)
            continue
        if v2_path in REMOVED:
            if v2_path in explicit:
                report.dropped_removed.add(v2_path)
            continue
        if v2_path.startswith(inactive) or v2_path in inactive_bare:
            if v2_path in explicit:
                report.dropped_inactive.append(v2_path)
            continue
        # transform value if a transform is registered
        tval = TRANSFORMS[v2_path](val) if v2_path in TRANSFORMS else val
        # destination
        if v2_path in interior_renames:
            dst = interior_renames[v2_path]
        elif v2_path in RENAMES:
            dst = RENAMES[v2_path]
        elif v2_path in v3_defaults:  # identical path in 3.0
            dst = v2_path
        else:
            # No 3.0 home. Warn only when the field was set by the user AND is
            # part of main's 2.0 schema (so it actually affected the 2.0 run) AND
            # is not in a redesigned block (zalmoxis/aragog get one summary
            # warning instead of per-field noise). A field absent from main's
            # schema was ignored by the 2.0 loader, so dropping it is faithful.
            redesigned = v2_path.startswith(('struct.zalmoxis.', 'interior.aragog.'))
            if v2_path in explicit and v2_path in v2_defaults and not redesigned:
                report.warnings.append(f'Unmapped 2.0 field (left out): {v2_path}')
            continue
        emit(dst, tval, v2_path)
        if v2_path != dst:
            report.renamed.append((v2_path, dst))

    # special handlers
    _handle_elements(eff_v2, explicit, v3, report)
    _handle_atmos_hoist(eff_v2, explicit, v3, active, report)
    _handle_temperature_mode(eff_v2, explicit, v3, active, report)

    # derived fields with no 2.0 source
    if v3.get('interior_struct.module') == 'spider':
        # 2.0 "self" structure used corefrac as a radius fraction, and the 3.0
        # spider structure module only supports radius mode.
        v3['interior_struct.core_frac_mode'] = 'radius'
    # mark element/IC outputs that came from explicit user fields
    for src in _ELEMENT_FIELDS | _IC_FIELDS:
        if src in explicit:
            provenance_explicit |= {
                p
                for p in v3
                if p.startswith(
                    (
                        'planet.elements.',
                        'planet.ini_',
                        'planet.tsurf',
                        'planet.temperature_mode',
                        'planet.gas_prs.',
                    )
                )
            }

    # new-field behavioural overrides (skip when the user set the field
    # explicitly: an explicit 2.0 choice wins over the backwards-compat pin)
    for path, value in OVERRIDES.items():
        if path in provenance_explicit:
            continue
        v3[path] = value
        report.overridden.append((path, value))

    # emit only deviations-from-3.0-default, explicit user choices, and overrides
    keep = {}
    for path, val in v3.items():
        v3def = v3_defaults.get(path, '<MISSING>')
        nval, ndef = _normalise(val), _normalise(v3def)
        if nval is None:
            if path in none_ok:
                # The 3.0 field accepts 'none'. Preserve the 2.0 none only when
                # the 3.0 default is a concrete value (so the default would not
                # reproduce the 2.0 unset state); otherwise it matches the
                # default and is written only if the user set it explicitly.
                if ndef is None:
                    if path in provenance_explicit:
                        keep[path] = 'none'
                else:
                    keep[path] = 'none'
                    if path not in provenance_explicit and path not in OVERRIDES:
                        report.pinned_divergent.append((path, 'none'))
                continue
            # Field cannot hold 'none' (e.g. a positive-float quantity): the 2.0
            # 'none' meant unset/auto, so let the 3.0 default apply.
            if path == 'planet.mass_tot':
                report.warnings.append(
                    'Structure was radius-specified in 2.0 (mass_tot unset). '
                    'planet.mass_tot is left at the 3.0 default and '
                    'planet.R_int_override carries the radius; verify the '
                    'intended planet mass.'
                )
            continue
        deviates = nval != ndef
        if deviates or path in provenance_explicit or path in OVERRIDES:
            keep[path] = val
            if deviates and path not in provenance_explicit and path not in OVERRIDES:
                report.pinned_divergent.append((path, val))
    keep['config_version'] = '3.0'

    nested = _unflatten(keep)
    _validate(nested)
    return nested, report


def _validate(nested_v3):
    """Structure the nested 3.0 dict through the schema; raise on failure."""
    import types

    if 'proteus.proteus' not in sys.modules:
        stub = types.ModuleType('proteus.proteus')
        stub.Proteus = object
        sys.modules['proteus.proteus'] = stub
    import cattrs

    from proteus.config._config import Config

    cattrs.structure(nested_v3, Config)


# ----------------------------------------------------------------------------
# Grid-config translation
# ----------------------------------------------------------------------------

_GRID_AXIS_RENAMES = {
    'delivery.elements.H_ppmw': 'planet.elements.H_budget',
    'delivery.elements.H_kg': 'planet.elements.H_budget',
    'delivery.elements.H_oceans': 'planet.elements.H_budget',
    'delivery.elements.CH_ratio': 'planet.elements.C_budget',
    'delivery.elements.C_ppmw': 'planet.elements.C_budget',
    'delivery.elements.C_kg': 'planet.elements.C_budget',
    'delivery.elements.NH_ratio': 'planet.elements.N_budget',
    'delivery.elements.SH_ratio': 'planet.elements.S_budget',
    'struct.corefrac': 'interior_struct.core_frac',
    'struct.mass_tot': 'planet.mass_tot',
    'interior.F_initial': 'interior_energetics.flux_guess',
}


def translate_grid(grid_toml: dict):
    """Translate a 2.0 grid-config dict to 3.0.

    Renames axis table names to 3.0 Config paths and adds ``config_version``.
    The base config referenced by ``ref_config`` must be migrated separately.
    """
    report = MigrationReport()
    out = {'config_version': '3.0'}
    for key, val in grid_toml.items():
        if key in ('version', 'config_version'):
            continue
        if '.' not in key:  # header field (output, ref_config, use_slurm, ...)
            out[key] = val
            continue
        # axis table
        new_key = _GRID_AXIS_RENAMES.get(key, key)
        out[new_key] = val
        if new_key != key:
            report.renamed.append((key, new_key))
            if new_key in (
                'planet.elements.H_budget',
                'planet.elements.C_budget',
                'planet.elements.N_budget',
                'planet.elements.S_budget',
            ):
                mode = {'H': 'ppmw', 'C': 'C/H', 'N': 'N/H', 'S': 'S/H'}[
                    new_key.split('.')[-1][0]
                ]
                report.warnings.append(
                    f'Axis {new_key} requires the base config to set '
                    f'{new_key.rsplit(".", 1)[0]}.{new_key.split(".")[-1][0]}_mode = '
                    f'"{mode}".'
                )
    return out, report


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _dump_toml(nested, path):
    import tomli_w

    with open(path, 'wb') as f:
        tomli_w.dump(_replace_none(nested), f)


def _replace_none(d):
    """Replace None with the string 'none' so the result round-trips as TOML."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _replace_none(v)
        elif v is None:
            out[k] = 'none'
        else:
            out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='Migrate a PROTEUS config from 2.0 to 3.0.')
    ap.add_argument('input', type=Path, help='2.0 TOML config (or grid config with --grid)')
    ap.add_argument('-o', '--output', type=Path, default=None, help='output path')
    ap.add_argument('--grid', action='store_true', help='input is a grid config')
    args = ap.parse_args(argv)

    with open(args.input, 'rb') as f:
        data = tomllib.load(f)

    if args.grid:
        nested, report = translate_grid(data)
    else:
        nested, report = translate(data)

    out = args.output or args.input.with_suffix('.v3.toml')
    _dump_toml(nested, out)
    print(f'Wrote {out}')
    print(report.text())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
