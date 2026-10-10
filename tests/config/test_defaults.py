"""
Unit tests for configuration dataclass defaults.

This module verifies that all configuration dataclasses (Params, Interior, etc.)
initialize with safe, physically valid default values when no arguments are provided.
This ensures that the simulation defaults to a known state (usually Earth-like or safe dummy)
without preventing manual configuration.

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

import pytest

from proteus.config._interior import Aragog, Interior, InteriorDummy, Spider
from proteus.config._params import (
    OutputParams,
    Params,
    StopDisint,
    StopEscape,
    StopIters,
    StopParams,
    StopRadeqm,
    StopSolid,
    StopTime,
    TimeStepParams,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_output_params_defaults():
    """
    Test verification of OutputParams defaults.

    Verifies that output parameters default to safe, standard values:
    - plotting format: png (universally supported)
    - logging: INFO (standard verbosity)
    - write/plot intervals: reasonable defaults to avoid spamming disk I/O
    """
    out = OutputParams(path='test_path')
    assert out.path == 'test_path'  # Path is mandatory, no default
    assert out.logging == 'INFO'
    assert out.plot_fmt == 'png'
    assert out.write_mod == 1  # Write every step (safe for short runs)
    assert out.plot_mod == 5  # Plot every 10 steps
    assert out.archive_mod is None  # Archiving disabled by default
    assert out.remove_sf is False  # Keep spectral files by default for debugging


@pytest.mark.unit
def test_dt_params_defaults():
    """
    Test verification of TimeStepParams and sub-configs defaults.

    Verifies time-stepping defaults are set for stable integration:
    - method: adaptive (safest for general use)
    - limits: 3e2 yr to 1e7 yr (covers typical geological timescales)
    - adaptive tolerance: 10% change per step (standard stability/speed trade-off)
    """
    dt = TimeStepParams()
    assert dt.method == 'adaptive'
    assert dt.minimum == pytest.approx(1e4, rel=1e-12)  # Minimum step 300 years
    assert dt.minimum_rel == pytest.approx(1e-5, rel=1e-12)  # Relative minimum precision
    assert dt.maximum == pytest.approx(1e7, rel=1e-12)  # Maximum step 10 Myr
    assert dt.initial == pytest.approx(3e1, rel=1e-12)  # Start with 1000 years

    # Proportional and adaptive parameters (flattened)
    assert dt.propconst == pytest.approx(52.0, rel=1e-12)
    assert dt.atol == pytest.approx(0.02, rel=1e-12)
    assert dt.rtol == pytest.approx(0.10, rel=1e-12)

    # Evection dt-cap trio: opt-in, disabled by default via None rather
    # than a numeric 0 (which would be behaviourally indistinguishable
    # from disabled but pass the >0 validator's exclusion silently).
    assert dt.evection_maximum is None
    assert dt.evection_growth_factor is None
    assert dt.evection_cooldown_iters is None


@pytest.mark.unit
def test_dt_params_evection_trio_accepts_none_string_and_rejects_non_positive():
    """The evection dt-cap trio (evection_maximum/evection_growth_factor/
    evection_cooldown_iters) must accept the TOML string sentinel
    ``'none'`` (structured to Python ``None`` by the ``none_if_none``
    converter, the same mechanism ``rot_period``/``phoenix_radius`` use)
    and a strictly positive value, but reject both a zero and a negative
    value -- 0 is deliberately NOT a valid opt-out spelling any more (it
    was, before this test), only ``None``/``'none'`` disables the
    mechanism.
    """
    for field_name in ('evection_maximum', 'evection_growth_factor', 'evection_cooldown_iters'):
        # 'none' string (TOML spelling) structures to Python None.
        assert getattr(TimeStepParams(**{field_name: 'none'}), field_name) is None
        # A genuine positive value is accepted and passed through untouched.
        assert getattr(TimeStepParams(**{field_name: 5}), field_name) == 5

        # Discrimination: 0 (the OLD opt-out spelling) and a negative
        # value must both now be rejected, not silently accepted as
        # another way to disable the mechanism.
        with pytest.raises(ValueError):
            TimeStepParams(**{field_name: 0})
        with pytest.raises(ValueError):
            TimeStepParams(**{field_name: -1})


@pytest.mark.unit
def test_stop_params_defaults():
    """
    Test verification of StopParams and sub-configs defaults.

    Verifies termination criteria defaults:
    - Time: 6 Gyr (Solar System age + margin)
    - Solidification: 1% melt fraction (rheological transition)
    - strict: False (allows faster termination)
    """
    stop = StopParams()
    assert stop.strict is False  # Strict mode requires double-check of conditions

    # Iters
    assert isinstance(stop.iters, StopIters)
    assert stop.iters.enabled is True
    assert stop.iters.minimum == 5
    assert stop.iters.maximum == 9000

    # Time
    assert isinstance(stop.time, StopTime)
    assert stop.time.enabled is True
    assert stop.time.maximum == pytest.approx(6e9, rel=1e-12)

    # Solid
    assert isinstance(stop.solid, StopSolid)
    assert stop.solid.enabled is True
    assert stop.solid.phi_crit == pytest.approx(0.01, rel=1e-12)

    # Radeqm
    assert isinstance(stop.radeqm, StopRadeqm)
    assert stop.radeqm.enabled is True
    assert stop.radeqm.atol == pytest.approx(1.0, rel=1e-12)

    # Escape
    assert isinstance(stop.escape, StopEscape)
    assert stop.escape.enabled is True
    assert stop.escape.p_stop == pytest.approx(3.0, rel=1e-12)

    # Disint (defaults to disabled)
    assert isinstance(stop.disint, StopDisint)
    assert stop.disint.enabled is False
    assert stop.disint.roche_enabled is True
    assert stop.disint.spin_enabled is True


@pytest.mark.unit
def test_params_defaults():
    """
    Test verification of root Params object defaults.

    Verifies that the top-level configuration object instantiates correctly
    with all sub-components (Output, TimeStep, Stop).
    """
    # Params default factory for 'out' fails because OutputParams requires path
    # So we must provide it
    out = OutputParams(path='test')
    p = Params(out=out)
    assert p.out == out
    assert isinstance(p.dt, TimeStepParams)
    assert isinstance(p.stop, StopParams)
    assert p.resume is False  # Default to fresh start
    assert p.offline is False  # Default to online (allow data downloads)


@pytest.mark.unit
def test_interior_defaults():
    """
    Test verification of Interior and sub-module defaults.

    Checks default physics settings for the interior layer:
    - radiogenic/tidal heating: Enabled by default (energy sources)
    - grain size: 1e-3 m (standard crystal size)
    - Initial flux: 1000 W/m^2 (hot start)
    """
    # Interior requires module argument
    # If module='spider', we must provide a valid spider config
    spider_cfg = Spider()
    i = Interior(module='spider', spider=spider_cfg)
    assert i.module == 'spider'
    assert i.spider == spider_cfg
    assert i.heat_radiogenic is True  # Heating terms on
    assert i.heat_tidal is False
    assert i.grain_size == pytest.approx(1e-3, rel=1e-12)  # 1 mm crystals
    assert i.flux_guess == -1  # Auto-detect

    # Sub-modules defaults
    assert isinstance(i.aragog, Aragog)
    assert i.num_levels == 80  # num_levels is on Interior, not Aragog

    assert isinstance(i.dummy, InteriorDummy)

    # Test Aragog module selection
    aragog_cfg = Aragog()
    i2 = Interior(module='aragog', aragog=aragog_cfg)
    assert i2.module == 'aragog'
    assert i2.aragog == aragog_cfg

    # Test Dummy module selection
    dummy_cfg = InteriorDummy()
    i3 = Interior(module='dummy', dummy=dummy_cfg)
    assert i3.module == 'dummy'
    assert i3.dummy == dummy_cfg


@pytest.mark.unit
def test_spider_defaults():
    """
    Test verification of Spider specific defaults.

    History:
    - Tier 4 (2026-04-08) promoted ``tolerance_rel`` and
      ``matprop_smooth_width`` from Spider to the top-level Interior
      class, leaving only ``solver_type`` as SPIDER-specific.
    - 2026-04-09 reverted ``matprop_smooth_width`` back to Spider as
      a real field (default 1e-2) after Aragog's Jgrav smoothing was
      replaced with a parameter-free cubic Hermite polynomial that
      does not need a width knob.

    ``tolerance_rel`` remains a deprecation-aliased sentinel field
    (default -1.0 = "not set"; a positive value is copied to
    ``Interior.rtol`` with a DeprecationWarning).
    """
    s = Spider()
    assert s.solver_type == 'bdf'
    # Deprecation alias sentinel (not set)
    assert s.tolerance_rel == pytest.approx(-1.0, abs=1e-12)
    # Real SPIDER-only field (post 2026-04-09)
    assert s.matprop_smooth_width == pytest.approx(1e-2)


@pytest.mark.unit
def test_aragog_defaults():
    """
    Test verification of Aragog specific defaults.

    Verifies ARAGOG (Python-based interior module) defaults.
    """
    a = Aragog()
    assert a.mass_coordinates is True
    assert a.backend == 'jax'
    assert a.separation_viscosity == 'mixture'
    assert not hasattr(a, 'jax')
    assert not hasattr(a, 'use_jax_jacobian')
    assert not hasattr(a, 'dilatation'), (
        'dilatation slot must be removed; existing TOMLs setting '
        'this field should now fail to load.'
    )
    # Per-call step caps: schema defaults 0.0, which the wrapper reads as off
    # (no cap) on every interior. -1.0 is the single off sentinel; any other
    # negative, NaN, or infinity is rejected at load.
    assert a.phi_step_cap == pytest.approx(0.0)
    assert a.temperature_step_cap == pytest.approx(0.0)
    assert a.entropy_step_cap == pytest.approx(0.0)
    # The -1.0 sentinel is admitted and round-trips unchanged.
    assert Aragog(phi_step_cap=-1.0).phi_step_cap == pytest.approx(-1.0)
    assert Aragog(temperature_step_cap=-1.0).temperature_step_cap == pytest.approx(-1.0)
    assert Aragog(entropy_step_cap=-1.0).entropy_step_cap == pytest.approx(-1.0)
    # A non-sentinel negative (differs from -1.0 by well over any tolerance) is
    # rejected on every step-cap field, so a malformed value cannot silently
    # disable the guard.
    with pytest.raises(ValueError):
        Aragog(phi_step_cap=-0.01)
    with pytest.raises(ValueError):
        Aragog(temperature_step_cap=-5.0)
    with pytest.raises(ValueError):
        Aragog(entropy_step_cap=-5.0)
    # Positive values persist.
    assert Aragog(phi_step_cap=0.05).phi_step_cap == pytest.approx(0.05)
    assert Aragog(temperature_step_cap=150.0).temperature_step_cap == pytest.approx(150.0)
    assert Aragog(entropy_step_cap=80.0).entropy_step_cap == pytest.approx(80.0)
    # Phase-boundary entropy margin: a positive-float proximity band whose
    # default 200.0 matches Aragog's own default, so a config that omits the
    # key is bit-identical to current behaviour. Unlike the step caps (which
    # admit the -1.0 off sentinel and 0.0) this uses gt(0): a proximity band
    # has no meaningful disabled state, so both 0.0 and negatives are rejected.
    # Pinning the exact 200.0 band is itself the discrimination guard: it
    # rejects a 0.0 that would silently switch off the near-boundary max_step
    # tightening this band controls.
    assert a.phase_boundary_entropy_margin == pytest.approx(200.0)
    with pytest.raises(ValueError):
        Aragog(phase_boundary_entropy_margin=0.0)
    with pytest.raises(ValueError):
        Aragog(phase_boundary_entropy_margin=-50.0)
    assert Aragog(
        phase_boundary_entropy_margin=350.0
    ).phase_boundary_entropy_margin == pytest.approx(350.0)
    import pytest as _pt

    with _pt.raises(ValueError):
        Aragog(backend='diffrax')
    # The deleted dilatation kwarg must now raise: an unknown attrs
    # kwarg is the user-facing signal that an old TOML carries a stale
    # field. ``TypeError`` from attrs' ``__init__``.
    with _pt.raises(TypeError):
        Aragog(dilatation=True)


def test_aragog_core_module_defaults_and_mode_validator():
    """Verify core-module sub-config defaults and validator bounds."""
    from proteus.config._interior import AragogCoreModule

    aragog = Aragog()
    cm = aragog.core_module
    assert not hasattr(cm, 'rho_cen')
    assert not hasattr(cm, 'length_scale')
    with pytest.raises(TypeError):
        AragogCoreModule(rho_cen=12500.0)
    with pytest.raises(TypeError):
        AragogCoreModule(length_scale=7272e3)
    assert cm.melting_curve == 'iron'
    assert cm.light_element_fraction == pytest.approx(0.0)  # pure iron
    assert cm.q_radio == pytest.approx(0.0)
    assert cm.ds_fusion == pytest.approx(172.8, rel=1e-6)  # 1.16 k_B/atom of Fe

    assert Aragog(core_bc='core_module').core_bc == 'core_module'
    with pytest.raises(ValueError):
        Aragog(core_bc='core_reservoir')
    with pytest.raises(ValueError):
        AragogCoreModule(light_element_fraction=1.0)  # bound is exclusive
    with pytest.raises(ValueError):
        AragogCoreModule(melting_curve='cubic')
    with pytest.raises(ValueError):
        AragogCoreModule(icn_width=0.0)

    # Diagnostics fields: Nimmo (2015) conductivity default, ohmic
    # fraction bounded on (0, 1], and the two printed CHR09 geometries.
    assert cm.k_core == pytest.approx(130.0)
    assert cm.f_ohm == pytest.approx(1.0)
    assert cm.flux_geometry == 'const_flux'
    with pytest.raises(ValueError):
        AragogCoreModule(k_core=0.0)
    with pytest.raises(ValueError):
        AragogCoreModule(f_ohm=1.5)  # bound is inclusive at 1
    with pytest.raises(ValueError):
        AragogCoreModule(f_ohm=0.0)  # bound is exclusive at 0
    with pytest.raises(ValueError):
        AragogCoreModule(flux_geometry='spherical_cow')

    # CMB boundary-layer critical Rayleigh number: Thiriet et al. (2019) Table 2 default.
    assert cm.ra_crit_cmb == pytest.approx(450.0)
    assert AragogCoreModule(ra_crit_cmb=1000.0).ra_crit_cmb == pytest.approx(1000.0)
    for bad in (0.0, float('inf'), float('nan')):
        with pytest.raises(ValueError):
            AragogCoreModule(ra_crit_cmb=bad)


def test_aragog_phase_boundary_cap_defaults_to_rate_and_rejects_other_values():
    """The schema default is 'rate'; 'fixed' is kept; anything else is rejected at load."""
    assert Aragog().phase_boundary_cap == 'rate'
    assert Aragog(phase_boundary_cap='fixed').phase_boundary_cap == 'fixed'
    for bad in ('Rate', '', 'adaptive'):
        with pytest.raises(ValueError, match='phase_boundary_cap'):
            Aragog(phase_boundary_cap=bad)


@pytest.mark.parametrize(
    ('module', 'expected'),
    [('aragog', 1e-8), ('spider', 1e-10), ('dummy', 1e-10), ('boundary', 1e-10)],
)
def test_unset_interior_rtol_resolves_per_module(module, expected):
    """An unset rtol resolves to 1e-8 for Aragog and 1e-10 otherwise; an explicit value is kept."""
    assert Interior(module=module).rtol == pytest.approx(expected)
    assert Interior(module=module, rtol=3e-9).rtol == pytest.approx(3e-9)


def test_deprecated_rtol_alias_overrides_the_aragog_default():
    """num_tolerance still copies into rtol for Aragog, with its deprecation warning."""
    with pytest.warns(DeprecationWarning, match='num_tolerance'):
        assert Interior(module='aragog', num_tolerance=1e-6).rtol == pytest.approx(1e-6)


@pytest.mark.parametrize(
    ('struct', 'core_bc', 'stratified', 'solver', 'refused'),
    [
        ('zalmoxis', 'core_module', False, 'cvode', None),
        ('zalmoxis', 'core_module', True, 'bdf', None),
        ('dummy', 'energy_balance', True, 'radau', None),
        ('dummy', 'core_module', False, 'cvode', "needs interior_struct.module = 'zalmoxis'"),
        ('spider', 'core_module', False, 'cvode', "not 'spider'"),
        ('zalmoxis', 'core_module', True, 'radau', "not 'radau'"),
    ],
)
def test_core_module_requirements_are_checked_at_config_load(
    struct, core_bc, stratified, solver, refused
):
    """The core_module core needs the Zalmoxis structure, which provides the core mass and
    the central pressure of its profile fit, and its resolved shell does not run on Radau;
    another core boundary is not checked."""
    from types import SimpleNamespace

    from proteus.config._config import check_core_module_requirements

    instance = SimpleNamespace(
        interior_struct=SimpleNamespace(module=struct),
        interior_energetics=SimpleNamespace(
            module='aragog',
            aragog=SimpleNamespace(
                core_bc=core_bc,
                solver_method=solver,
                core_module=SimpleNamespace(stratification=stratified),
            ),
        ),
    )
    if refused:
        with pytest.raises(ValueError, match=refused):
            check_core_module_requirements(instance, None, None)
    else:
        assert check_core_module_requirements(instance, None, None) is None
    instance.interior_energetics.module = 'spider'
    assert check_core_module_requirements(instance, None, None) is None


def test_a_core_module_core_without_the_zalmoxis_structure_is_refused_when_built():
    """Building a configuration runs the core_module check: the dummy structure is refused,
    the Zalmoxis structure builds, and a light-element mass fraction of 1 is out of range."""
    import attrs
    from helpers import PROTEUS_ROOT

    from proteus.config import read_config_object
    from proteus.config._interior import AragogCoreModule

    cfg = read_config_object(PROTEUS_ROOT / 'input' / 'all_options.toml')
    aragog = attrs.evolve(cfg.interior_energetics.aragog, core_bc='core_module')
    interior = attrs.evolve(cfg.interior_energetics, module='aragog', aragog=aragog)
    built = attrs.evolve(cfg, interior_energetics=interior)
    assert built.interior_struct.module == 'zalmoxis'
    with pytest.raises(ValueError, match="needs interior_struct.module = 'zalmoxis'"):
        dummy = attrs.evolve(cfg.interior_struct, module='dummy', melting_dir='Monteux-600')
        attrs.evolve(built, interior_struct=dummy)
    assert AragogCoreModule(c_light=0.046).c_light == pytest.approx(0.046)
    with pytest.raises(ValueError):
        AragogCoreModule(c_light=1.0)
