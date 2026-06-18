"""Tests for the 2.0 -> 3.0 config migration tool.

Three guarantees are checked without any external dependency, using the baked
2.0 default snapshot and the live 3.0 schema:

* map completeness: every 2.0 field is either copied (identical path), renamed,
  consumed by a special handler, removed deliberately, or droppable as an
  inactive-module sub-block. A 2.0 field matching none of these would be lost
  silently, so the test fails if one appears.
* new-field classification: the set of 3.0-only fields equals the reviewed
  union of OVERRIDES (pinned for backwards-compatibility) and a hard-coded
  reviewed-neutral allowlist. A field added to the 3.0 schema later lands in
  neither set and fails the test, forcing a conscious classification.
* regression: representative 2.0 inputs translate to a validated 3.0 config with
  the expected field placements, units, and values.

A separate developer harness (tools/migrate_data is the data; the harness lives
under output_files and is not committed) resolves each 2.0 input through main's
loader and the 3.0 output through the PR loader and asserts field-by-field
equivalence; it needs both schema revisions importable from two checkouts, so it
is run manually rather than in this unit suite.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_TOOLS = Path(__file__).resolve().parents[2] / 'tools'
sys.path.insert(0, str(_TOOLS))

# The tool imports proteus.config; stub the heavy pipeline submodule so importing
# it never initialises the Julia-backed atmosphere code.
if 'proteus.proteus' not in sys.modules:
    _stub = types.ModuleType('proteus.proteus')
    _stub.Proteus = object
    sys.modules['proteus.proteus'] = _stub

import migrate_config_v2_to_v3 as mig  # noqa: E402


def _v3():
    defaults, none_ok = mig._v3_defaults()
    return defaults, none_ok


# Sub-block prefixes the tool drops when their module is not the active one.
# delivery.volatiles is deliberately NOT here: those fields are mapped to
# planet.gas_prs.*, not dropped, and are checked separately.
_DROP_PREFIXES = (
    'struct.zalmoxis.',
    'struct.mesh_',
    'struct.update_',
    'interior.spider.',
    'interior.aragog.',
    'interior.dummy.',
    'interior.boundary.',
    'atmos_clim.agni.',
    'atmos_clim.janus.',
    'atmos_clim.dummy.',
    'star.mors.',
    'star.dummy.',
    'escape.zephyrus.',
    'escape.dummy.',
    'escape.boreas.',
    'outgas.calliope.',
    'outgas.atmodeller.',
)

# Hard-coded list of 3.0-only fields whose 3.0 default reproduces the 2.0
# behaviour (no field present), reviewed once by hand. A new 3.0 field not in
# this set and not in OVERRIDES fails test_new_field_classified, forcing a
# decision: pin it (OVERRIDES) or certify it neutral (add it here).
_REVIEWED_NEUTRAL = frozenset(
    {
        'atmos_clim.aerosols_enabled',
        'atmos_clim.agni.grey_opacity_lw',
        'atmos_clim.agni.grey_opacity_sw',
        'atmos_clim.agni.spectral_file',
        'atmos_clim.dummy.fixed_flux',
        'atmos_clim.janus.cloud_alpha',
        'interior_energetics.adams_williamson_beta',
        'interior_energetics.adams_williamson_rhos',
        'interior_energetics.adiabatic_bulk_modulus',
        'interior_energetics.aragog.atol_temperature_equivalent',
        'interior_energetics.aragog.backend',
        'interior_energetics.aragog.core_bc',
        'interior_energetics.aragog.phase_smoothing',
        'interior_energetics.aragog.phi_step_cap',
        'interior_energetics.aragog.scalar_gravity_override',
        'interior_energetics.aragog.solver_method',
        'interior_energetics.aragog.tolerance_struct',
        'interior_energetics.boundary.T_liquidus',
        'interior_energetics.boundary.T_p_0',
        'interior_energetics.boundary.T_solidus',
        'interior_energetics.boundary.Tsurf_event_change',
        'interior_energetics.boundary.activation_energy',
        'interior_energetics.boundary.atm_heat_capacity',
        'interior_energetics.boundary.atm_heat_capacity_const',
        'interior_energetics.boundary.atol',
        'interior_energetics.boundary.creep_parameter',
        'interior_energetics.boundary.critical_rayleigh_number',
        'interior_energetics.boundary.dynamic_viscosity',
        'interior_energetics.boundary.eta_constant',
        'interior_energetics.boundary.eta_melt_const',
        'interior_energetics.boundary.eta_solid_const',
        'interior_energetics.boundary.heat_fusion_silicate',
        'interior_energetics.boundary.logging',
        'interior_energetics.boundary.nusselt_exponent',
        'interior_energetics.boundary.rtol',
        'interior_energetics.boundary.silicate_density',
        'interior_energetics.boundary.silicate_heat_capacity',
        'interior_energetics.boundary.thermal_conductivity',
        'interior_energetics.boundary.thermal_diffusivity',
        'interior_energetics.boundary.thermal_expansivity',
        'interior_energetics.boundary.viscosity_activation_temp',
        'interior_energetics.boundary.viscosity_model',
        'interior_energetics.boundary.viscosity_prefactor',
        'interior_energetics.const_Cp',
        'interior_energetics.const_S_ref',
        'interior_energetics.const_T_ref',
        'interior_energetics.const_alpha',
        'interior_energetics.const_cond',
        'interior_energetics.const_log10visc',
        'interior_energetics.const_properties',
        'interior_energetics.const_rho',
        'interior_energetics.core_tfac_avg',
        'interior_energetics.eddy_diffusivity_chemical',
        'interior_energetics.eddy_diffusivity_thermal',
        'interior_energetics.latent_heat_of_fusion',
        'interior_energetics.melt_cond',
        'interior_energetics.melt_log10visc',
        'interior_energetics.num_tolerance',
        'interior_energetics.param_utbl',
        'interior_energetics.param_utbl_const',
        'interior_energetics.phase_transition_width',
        'interior_energetics.radio_Al',
        'interior_energetics.radio_Fe',
        'interior_energetics.solid_cond',
        'interior_energetics.solid_log10visc',
        'interior_energetics.spider.log_output',
        'interior_energetics.spider.tolerance_rel',
        'interior_energetics.spider.tolerance_struct',
        'interior_energetics.surface_bc_mode',
        'interior_energetics.write_flux_diagnostics',
        'interior_struct.zalmoxis.dry_mantle',
        'interior_struct.zalmoxis.equilibrate_init',
        'interior_struct.zalmoxis.equilibrate_max_iter',
        'interior_struct.zalmoxis.equilibrate_tol',
        'interior_struct.zalmoxis.global_miscibility',
        'interior_struct.zalmoxis.lookup_nP',
        'interior_struct.zalmoxis.lookup_nS',
        'interior_struct.zalmoxis.miscibility_max_iter',
        'interior_struct.zalmoxis.miscibility_tol',
        'interior_struct.zalmoxis.mushy_zone_factor',
        'interior_struct.zalmoxis.newton_absolute_tolerance',
        'interior_struct.zalmoxis.newton_max_iter',
        'interior_struct.zalmoxis.newton_relative_tolerance',
        'interior_struct.zalmoxis.newton_tol',
        'interior_struct.zalmoxis.outer_solver',
        'interior_struct.zalmoxis.update_dw_comp_abs',
        'interior_struct.zalmoxis.update_stale_ceiling',
        'interior_struct.zalmoxis.use_anderson',
        'interior_struct.zalmoxis.use_jax',
        'observe.clip_vmr',
        'observe.module',
        'observe.petitRADTRANS.include_cia',
        'observe.petitRADTRANS.include_rayleigh',
        'observe.petitRADTRANS.input_data_path',
        'observe.petitRADTRANS.line_opacity_mode',
        'observe.reference_pressure',
        'outgas.atmodeller.eos_CH4',
        'outgas.atmodeller.eos_CO',
        'outgas.atmodeller.eos_CO2',
        'outgas.atmodeller.eos_H2',
        'outgas.atmodeller.eos_H2O',
        'outgas.atmodeller.include_condensates',
        'outgas.atmodeller.solubility_CH4',
        'outgas.atmodeller.solubility_CO',
        'outgas.atmodeller.solubility_CO2',
        'outgas.atmodeller.solubility_H2',
        'outgas.atmodeller.solubility_H2O',
        'outgas.atmodeller.solubility_N2',
        'outgas.atmodeller.solubility_S2',
        'outgas.atmodeller.solver_max_steps',
        'outgas.atmodeller.solver_mode',
        'outgas.atmodeller.solver_multistart',
        'outgas.calliope.nguess',
        'outgas.calliope.nsolve',
        'outgas.calliope.p_guess_max',
        'outgas.h2_binodal',
        'params.dt.hysteresis_iters',
        'params.dt.hysteresis_sfinc',
        'params.dt.max_growth_factor',
        'params.dt.mushy_maximum',
        'params.dt.mushy_upper',
        'params.dt.scale_decr',
        'params.dt.scale_incr',
        'params.dt.window',
        'params.out.dt_write_rel',
        'params.stop.solid.freeze_volatiles',
        'planet.delta_T_super',
        'planet.fO2_source',
        'planet.f_accretion',
        'planet.f_differentiation',
        'planet.tcenter_init',
        'planet.tcmb_init',
        'planet.volatile_reservoir',
    }
)


def test_map_completeness():
    """Every 2.0 field has a destination, a handler, a removal, or a drop rule."""
    v3_defaults, _ = _v3()
    v2 = mig._load_v2_defaults()
    interior_targets = set()
    for d in mig.INTERIOR_RENAMES.values():
        interior_targets |= set(d)
    atmos_shared_src = {
        f'atmos_clim.{m}.{f}' for m in ('agni', 'janus') for f in mig._ATMOS_SHARED
    }
    unhandled = []
    for path in v2:
        if path == 'version':
            continue
        if path in v3_defaults:  # identical path, copied
            continue
        if path in mig.RENAMES or path in interior_targets:
            continue
        if path in mig._ELEMENT_FIELDS or path in mig._IC_FIELDS:
            continue
        if path in atmos_shared_src:
            continue
        if path.startswith('delivery.volatiles.'):  # mapped to planet.gas_prs.*
            continue
        if path in mig.REMOVED:
            continue
        if path == 'observe.synthesis':
            continue
        if path.startswith(_DROP_PREFIXES) or path == 'struct.zalmoxis':
            continue
        unhandled.append(path)
    assert not unhandled, f'2.0 fields with no handling rule: {sorted(unhandled)}'


def _compute_pm(v3_defaults):
    """3.0-only field paths (not reachable from any 2.0 field)."""
    v2 = mig._load_v2_defaults()
    interior_all = {}
    for d in mig.INTERIOR_RENAMES.values():
        interior_all.update(d)
    landed = set()
    for path in v2:
        if path in mig.RENAMES:
            landed.add(mig.RENAMES[path])
        elif path in interior_all:
            landed.add(interior_all[path])
        elif path in v3_defaults:
            landed.add(path)
    landed |= {
        p
        for p in v3_defaults
        if p.startswith(
            (
                'planet.elements.',
                'planet.gas_prs.',
                'planet.ini_',
                'atmos_clim.spectral',
                'atmos_clim.num_levels',
                'atmos_clim.p_',
                'atmos_clim.overlap_method',
            )
        )
    }
    landed |= {
        'planet.temperature_mode',
        'planet.tsurf_init',
        'planet.mass_tot',
        'planet.volatile_mode',
        'planet.R_int_override',
        'interior_struct.core_frac_mode',
        'config_version',
    }
    return set(v3_defaults) - landed


def test_new_field_classified():
    """Every 3.0-only field is an override or in the reviewed-neutral allowlist.

    Exact set equality (both directions): a new 3.0 schema field lands in neither
    set and fails here, and a removed field leaves a stale allowlist entry that
    also fails, so the classification cannot silently drift.
    """
    v3_defaults, _ = _v3()
    pm = _compute_pm(v3_defaults)
    overridden = set(mig.OVERRIDES)
    assert (pm - overridden) == _REVIEWED_NEUTRAL, (
        '3.0-only fields changed; classify the difference as override or neutral. '
        f'new (unclassified): {sorted((pm - overridden) - _REVIEWED_NEUTRAL)}; '
        f'stale allowlist entries: {sorted(_REVIEWED_NEUTRAL - (pm - overridden))}'
    )
    # The two known behaviour-changing new fields are pinned, not neutral.
    assert 'interior_energetics.kappah_floor' in overridden
    assert 'params.dt.maximum_rel' in overridden


def test_kappah_floor_and_maximum_rel_overrides():
    """kappah_floor pins 0.0 (no floor) and maximum_rel pins 0.0 (strict dt cap)."""
    assert mig.OVERRIDES['interior_energetics.kappah_floor'] == 0.0
    assert mig.OVERRIDES['params.dt.maximum_rel'] == 0.0


def _translate(v2_dict):
    nested, report = mig.translate(v2_dict)
    return mig._flatten(nested), report


def _minimal_spider_v2():
    """A minimal but complete 2.0 SPIDER config (required fields set)."""
    return {
        'version': '2.0',
        'star': {
            'module': 'mors',
            'mass': 1.0,
            'age_ini': 0.1,
            'mors': {
                'tracks': 'spada',
                'age_now': 4.5,
                'rot_pcntle': 50.0,
                'spectrum_source': 'phoenix',
            },
        },
        'orbit': {
            'module': 'none',
            'semimajoraxis': 1.0,
            'eccentricity': 0.0,
            'zenith_angle': 48.19,
            's0_factor': 0.375,
            'instellation_method': 'sma',
        },
        'struct': {
            'module': 'self',
            'corefrac': 0.55,
            'mass_tot': 1.0,
            'core_density': 10738.33,
            'core_heatcap': 880.0,
        },
        'atmos_clim': {
            'module': 'agni',
            'surf_state': 'skin',
            'agni': {
                'spectral_group': 'Honeyside',
                'spectral_bands': '48',
                'num_levels': 40,
                'p_top': 1e-5,
            },
        },
        'escape': {'module': 'none'},
        'interior': {
            'module': 'spider',
            'melting_dir': 'Monteux-600',
            'eos_dir': 'WolfBower2018_MgSiO3',
            'F_initial': 1000.0,
            'tidal_heat': False,
            # Distinct tolerances so a tolerance/tolerance_rel -> atol/rtol swap
            # is detectable (they differ by orders of magnitude).
            'spider': {
                'ini_entropy': 3300.0,
                'ini_dsdr': -4.698e-6,
                'num_levels': 190,
                'mixing_length': 2,
                'tolerance': 2e-9,
                'tolerance_rel': 7e-11,
            },
        },
        'outgas': {
            'module': 'calliope',
            'fO2_shift_IW': 2.0,
            'calliope': {'T_floor': 700.0, 'rtol': 1e-4, 'xtol': 1e-6},
        },
        'delivery': {
            'module': 'none',
            'initial': 'elements',
            'elements': {'H_ppmw': 109.0, 'CH_ratio': 1.0, 'NH_ratio': 0.5, 'SH_ratio': 2.0},
        },
        'observe': {'synthesis': 'none'},
        'atmos_chem': {'module': 'none'},
    }


def test_spider_interior_split_and_renames():
    """The interior splits cleanly and the SPIDER renames map correctly."""
    flat, _ = _translate(_minimal_spider_v2())
    assert flat['config_version'] == '3.0'
    assert flat['interior_struct.module'] == 'spider'
    assert flat['interior_struct.core_frac_mode'] == 'radius'
    assert flat['interior_struct.core_frac'] == pytest.approx(0.55)
    assert flat['interior_struct.melting_dir'] == 'Monteux-600'
    assert flat['interior_energetics.module'] == 'spider'
    assert flat['interior_energetics.num_levels'] == 190
    assert flat['interior_energetics.mixing_length'] == 'constant'  # 2 -> "constant"
    assert flat['interior_energetics.flux_guess'] == pytest.approx(1000.0)


def test_tolerance_atol_rtol_not_swapped():
    """tolerance -> atol and tolerance_rel -> rtol (distinct, so a swap fails)."""
    flat, _ = _translate(_minimal_spider_v2())
    # 2e-9 to atol, 7e-11 to rtol; a swap would put 7e-11 in atol (> tol apart).
    assert flat['interior_energetics.atol'] == pytest.approx(2e-9)
    assert flat['interior_energetics.rtol'] == pytest.approx(7e-11)


def test_isentropic_ic_and_kappah_floor():
    """A SPIDER ini_entropy maps to isentropic; kappah_floor is pinned to 0."""
    flat, _ = _translate(_minimal_spider_v2())
    assert flat['planet.temperature_mode'] == 'isentropic'
    assert flat['planet.ini_entropy'] == pytest.approx(3300.0)
    assert flat['interior_energetics.kappah_floor'] == pytest.approx(0.0)


def test_maximum_rel_pinned_when_unset():
    """An omitted maximum_rel pins 0.0 (strict 2.0 dt cap), not the 3.0 1.0."""
    flat, _ = _translate(_minimal_spider_v2())
    assert flat['params.dt.maximum_rel'] == pytest.approx(0.0)


def test_explicit_maximum_rel_is_respected():
    """An explicit 2.0 maximum_rel wins over the backwards-compat override."""
    v2 = _minimal_spider_v2()
    v2['params'] = {'dt': {'maximum_rel': 1.0}}
    flat, _ = _translate(v2)
    assert flat['params.dt.maximum_rel'] == pytest.approx(1.0)


def test_element_modes_from_ratios():
    """H ppmw, C/H, N/H, S/H element budgets map to mode+budget pairs."""
    flat, _ = _translate(_minimal_spider_v2())
    assert flat['planet.elements.H_mode'] == 'ppmw'
    assert flat['planet.elements.H_budget'] == pytest.approx(109.0)
    assert flat['planet.elements.C_mode'] == 'C/H'
    assert flat['planet.elements.C_budget'] == pytest.approx(1.0)
    assert flat['planet.elements.S_mode'] == 'S/H'
    assert flat['planet.elements.S_budget'] == pytest.approx(2.0)
    assert flat['planet.elements.O_mode'] == 'ic_chemistry'


def test_additive_element_budget_warns():
    """C set by both ppmw and kg (2.0 sums them) cannot be one mode: warn."""
    v2 = _minimal_spider_v2()
    v2['delivery']['elements'] = {'H_ppmw': 100.0, 'C_ppmw': 200.0, 'C_kg': 5e19}
    _, report = _translate(v2)
    assert any('ppmw and kg' in w for w in report.warnings)


def test_instellation_method_sma_to_distance():
    """The orbit instellation method renames from sma to distance."""
    flat, _ = _translate(_minimal_spider_v2())
    assert flat['orbit.instellation_method'] == 'distance'


def test_outgas_solver_tolerances_renamed():
    """calliope rtol/xtol hoist to outgas.solver_rtol/solver_atol (distinct)."""
    flat, _ = _translate(_minimal_spider_v2())
    assert flat['outgas.solver_rtol'] == pytest.approx(1e-4)
    assert flat['outgas.solver_atol'] == pytest.approx(1e-6)  # xtol -> solver_atol
    assert flat['outgas.T_floor'] == pytest.approx(700.0)


def test_unset_F_initial_pins_main_default():
    """A 2.0 config that omits F_initial pins the main default 1000.0, not -1.

    The 3.0 flux_guess default is -1 (auto sigma*T^4) but main's 2.0 default was
    1000.0; omitting would silently switch the initial-flux behaviour.
    """
    v2 = _minimal_spider_v2()
    del v2['interior']['F_initial']
    flat, _ = _translate(v2)
    assert flat['interior_energetics.flux_guess'] == pytest.approx(1000.0)


def test_radius_int_converts_earth_radii_to_metres():
    """A radius-specified 2.0 config converts radius_int (R_earth) to metres.

    2.0 consumes radius_int as radius_int * R_earth; 3.0 R_int_override is in
    metres. A verbatim copy would be a ~6.3e6x radius error.
    """
    v2 = _minimal_spider_v2()
    del v2['struct']['mass_tot']  # radius-specified instead of mass-specified
    v2['struct']['radius_int'] = 1.0  # 1 Earth radius
    flat, report = _translate(v2)
    assert flat['planet.R_int_override'] == pytest.approx(6335439.0)
    # Discrimination guard: a verbatim (un-converted) copy would be 1.0, which is
    # six orders of magnitude away from the correct metre value.
    assert flat['planet.R_int_override'] > 1e6
    # planet.mass_tot is left at the 3.0 default with a warning.
    assert any('radius-specified' in w for w in report.warnings)


def test_clean_spider_config_warns_only_about_legacy_observe_synthesis():
    """A clean SPIDER config only warns about the legacy observe.synthesis field."""
    _, report = _translate(_minimal_spider_v2())
    assert report.warnings == ['Unmapped 2.0 field (left out): observe.synthesis'], (
        report.warnings
    )


def test_grid_axis_renames():
    """Grid axis table names map to 3.0 Config paths and config_version is set."""
    grid = {
        'version': '2.0',
        'output': 'g/',
        'ref_config': 'input/base.toml',
        'use_slurm': False,
        'max_jobs': 4,
        'max_days': 1,
        'max_mem': 3,
        'outgas.fO2_shift_IW': {'method': 'direct', 'values': [-2, 0, 2]},
        'delivery.elements.H_ppmw': {
            'method': 'arange',
            'start': 5e3,
            'stop': 2e4,
            'step': 5e3,
        },
        'delivery.elements.SH_ratio': {'method': 'direct', 'values': [1, 2]},
    }
    out, report = mig.translate_grid(grid)
    assert out['config_version'] == '3.0'
    assert 'planet.elements.H_budget' in out
    assert 'planet.elements.S_budget' in out
    assert 'outgas.fO2_shift_IW' in out  # unchanged axis
    assert out['ref_config'] == 'input/base.toml'  # header preserved
    assert out['planet.elements.H_budget']['step'] == pytest.approx(5e3)
    assert any('H_mode' in w for w in report.warnings)
