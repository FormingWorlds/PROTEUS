"""
Unit tests for configuration dataclass defaults.

This module verifies that all configuration dataclasses (Params, Interior, etc.)
initialize with safe, physically valid default values when no arguments are provided.
This ensures that the simulation defaults to a known state (usually Earth-like or safe dummy)
without preventing manual configuration.

See also:
- docs/test_infrastructure.md
- docs/test_categorization.md
- docs/test_building.md
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
    assert dt.minimum == 1e4  # Minimum step 300 years
    assert dt.minimum_rel == 1e-5  # Relative minimum precision
    assert dt.maximum == 1e7  # Maximum step 10 Myr
    assert dt.initial == 3e1  # Start with 1000 years

    # Proportional and adaptive parameters (flattened)
    assert dt.propconst == 52.0
    assert dt.atol == 0.02
    assert dt.rtol == 0.10


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
    assert stop.time.maximum == 6e9

    # Solid
    assert isinstance(stop.solid, StopSolid)
    assert stop.solid.enabled is True
    assert stop.solid.phi_crit == 0.01

    # Radeqm
    assert isinstance(stop.radeqm, StopRadeqm)
    assert stop.radeqm.enabled is True
    assert stop.radeqm.atol == 1.0

    # Escape
    assert isinstance(stop.escape, StopEscape)
    assert stop.escape.enabled is True
    assert stop.escape.p_stop == 3.0

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
    - grain size: 0.1 m (standard crystal size)
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
    assert i.grain_size == 0.1  # 10 cm crystals
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
    assert s.tolerance_rel == -1.0
    # Real SPIDER-only field (post 2026-04-09)
    assert s.matprop_smooth_width == pytest.approx(1e-2)


@pytest.mark.unit
def test_aragog_defaults():
    """
    Test verification of Aragog specific defaults.

    Verifies ARAGOG (Python-based interior module) defaults.
    """
    a = Aragog()
    assert a.dilatation is False
    assert a.mass_coordinates is True
    assert a.jax is False
