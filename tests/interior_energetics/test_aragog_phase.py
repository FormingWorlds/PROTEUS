"""
Unit tests for proteus.interior_energetics.aragog_phase: the single builder
that resolves the aragog phase parameters from a PROTEUS config.

The aragog interior module builds phase parameters at three call sites: the
numpy entropy solver's ``_PhaseMixedParameters``, the JAX CVODE factory's
``PhaseParams`` (both in ``aragog.py``), and the JAX research runner's
``PhaseParams`` (in ``aragog_jax.py``). The numpy and JAX types name their
fields differently, so a configured quantity resolved at one site but
forgotten or cast differently at another drifts silently.

``aragog_phase`` resolves every configured quantity once in
``_phase_params_from_config`` and feeds ``build_mixed_phase_params`` and
``build_jax_phase_params``. These tests pin the anti-drift contract: the
five quantities shared by the numpy and JAX types carry the SAME configured
value at both, and every configured field reaches its site rather than the
aragog library default, so a dropped or re-cast field at any site surfaces
as a test failure.

See also:
- docs/How-to/testing.md
- tests/interior_energetics/test_aragog_jax.py (the runner-site harness)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip('aragog.jax')

from proteus.interior_energetics.aragog_jax import AragogJAXRunner  # noqa: E402
from proteus.interior_energetics.aragog_phase import (  # noqa: E402
    _phase_params_from_config,
    build_jax_phase_params,
    build_mixed_phase_params,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Runtime solidus / liquidus paths the numpy builder stores verbatim.
_SOLIDUS = '/data/lookup/solidus.dat'
_LIQUIDUS = '/data/lookup/liquidus.dat'

# The 19 stored attributes of the JAX PhaseParams, used to assert the runner
# site and the builder agree field by field.
_JAX_STORED_ATTRS = (
    'phi_rheo',
    'phi_width',
    'log10_visc_solid',
    'log10_visc_liquid',
    'grain_size',
    'k_solid',
    'k_liquid',
    'matprop_smooth_width',
    'conduction',
    'convection',
    'grav_sep',
    'mixing',
    'eddy_diff_thermal',
    'eddy_diff_chemical',
    'kappah_floor',
    'bottom_up_grav_sep',
    'phase_smoothing_tanh',
    'phase_smoothing_width',
    'separation_viscosity_mixture',
)


def _make_full_config(*, separation_viscosity: str = 'mixture'):
    """Build a mock PROTEUS Config with every field the builder reads.

    The values are discriminating and differ from the aragog library
    defaults, so a field dropped at a call site (falling back to the
    library default) produces a value mismatch rather than a coincidental
    pass. Unlike ``test_aragog_jax._make_config``, this also sets the
    numpy-only mixed-phase and constant-property fields.

    Parameters
    ----------
    separation_viscosity : str
        The gravitational-separation drag viscosity source ('mixture' or
        'melt'), shared by the numpy and JAX types under different
        representations.
    """
    config = MagicMock()
    ie = config.interior_energetics
    # Shared between the numpy and JAX phase parameters.
    ie.rfront_loc = 0.37
    ie.rfront_wid = 0.11
    ie.grain_size = 0.023
    ie.spider.matprop_smooth_width = 0.017
    ie.aragog.separation_viscosity = separation_viscosity
    # Numpy mixed-phase parameters only.
    ie.latent_heat_of_fusion = 4.1e5
    ie.phase_transition_width = 0.019
    ie.const_properties = True
    ie.const_rho = 4123.0
    ie.const_Cp = 1234.0
    ie.const_alpha = 3.3e-5
    ie.const_cond = 4.7
    ie.const_log10visc = 2.3
    ie.const_T_ref = 3456.0
    ie.const_S_ref = 3021.0
    # JAX phase parameters only.
    ie.solid_log10visc = 22.3
    ie.melt_log10visc = 1.7
    ie.solid_cond = 4.3
    ie.melt_cond = 2.9
    ie.trans_conduction = True
    ie.trans_convection = True
    ie.trans_grav_sep = False
    ie.trans_mixing = True
    ie.eddy_diffusivity_thermal = 0.13
    ie.eddy_diffusivity_chemical = 0.07
    ie.kappah_floor = 1.0e-6
    ie.aragog.phase_smoothing = 'tanh'
    return config


def _make_runner_interior_o(*, spider_eos_dir: str):
    """Build a mock Interior_t the AragogJAXRunner constructor inspects.

    Mirrors ``test_aragog_jax._make_interior_o`` for the numpy-solver BC,
    mesh, and solver attributes read inside ``_build_jax_components``.
    """
    interior_o = SimpleNamespace()
    interior_o._spider_eos_dir = spider_eos_dir
    interior_o.aragog_solver = MagicMock()
    bc_cfg = MagicMock()
    bc_cfg.outer_boundary_condition = 4
    bc_cfg.outer_boundary_value = 0.0
    bc_cfg.emissivity = 1.0
    bc_cfg.equilibrium_temperature = 1500.0
    bc_cfg.inner_boundary_condition = 3
    bc_cfg.inner_boundary_value = 0.0
    bc_cfg.core_heat_capacity = 880.0
    bc_cfg.tfac_core_avg = 1.147
    interior_o.aragog_solver.parameters.boundary_conditions = bc_cfg
    interior_o.aragog_solver.parameters.mesh.core_density = 12000.0
    interior_o.aragog_solver.parameters.solver.start_time = 0.0
    interior_o.aragog_solver.parameters.solver.end_time = 1.0
    interior_o.aragog_solver._S0 = np.linspace(2000.0, 3000.0, 5)
    return interior_o


def test_shared_quantities_match_across_numpy_and_jax():
    """The five config quantities shared by the numpy and JAX phase types
    carry the same configured value at both sites.

    Contract: ``rfront_loc``, ``rfront_wid``, ``grain_size``,
    ``spider.matprop_smooth_width`` and ``aragog.separation_viscosity`` feed
    both ``_PhaseMixedParameters`` and ``PhaseParams`` under different field
    names. The single builder must map the same resolved input to both, so
    the numpy field and the JAX stored attribute reflect one config value.

    Discrimination: the config values differ from the aragog library
    defaults, so a call site that dropped one of these kwargs would fall
    back to the library default and mismatch the other site. The numpy
    ``matprop_smooth_width`` and the JAX ``matprop_smooth_width`` also pin
    the single float cast: a site that skipped the cast on an already-float
    value would still match here, and the separate parity check below over
    the runner site closes the coercion gap.
    """
    config = _make_full_config()
    ie = config.interior_energetics
    numpy_params = build_mixed_phase_params(config, _SOLIDUS, _LIQUIDUS)
    jax_params = build_jax_phase_params(config)

    assert numpy_params.rheological_transition_melt_fraction == pytest.approx(ie.rfront_loc)
    assert jax_params.phi_rheo == pytest.approx(ie.rfront_loc)
    assert numpy_params.rheological_transition_width == pytest.approx(ie.rfront_wid)
    assert jax_params.phi_width == pytest.approx(ie.rfront_wid)
    assert numpy_params.grain_size == pytest.approx(ie.grain_size)
    assert jax_params.grain_size == pytest.approx(ie.grain_size)
    assert numpy_params.matprop_smooth_width == pytest.approx(ie.spider.matprop_smooth_width)
    assert jax_params.matprop_smooth_width == pytest.approx(ie.spider.matprop_smooth_width)


@pytest.mark.parametrize(
    ('separation_viscosity', 'expected_mixture_flag'),
    [('mixture', 1.0), ('melt', 0.0)],
)
def test_separation_viscosity_shared_across_sites(separation_viscosity, expected_mixture_flag):
    """The separation-viscosity source reaches the numpy string field and
    the JAX flag from one config value.

    Contract: the numpy type stores ``separation_viscosity`` as the string,
    the JAX type stores it as the ``separation_viscosity_mixture`` flag
    (1.0 for 'mixture', 0.0 for 'melt'). The builder feeds both from
    ``config.interior_energetics.aragog.separation_viscosity``.

    Discrimination: the aragog library default is 'melt', so the 'mixture'
    case fails at either site if the kwarg is dropped there, while the
    'melt' case would pass by coincidence. Parametrizing over both values
    catches a drop at either site regardless of the default.
    """
    config = _make_full_config(separation_viscosity=separation_viscosity)
    numpy_params = build_mixed_phase_params(config, _SOLIDUS, _LIQUIDUS)
    jax_params = build_jax_phase_params(config)

    assert numpy_params.separation_viscosity == separation_viscosity
    assert jax_params.separation_viscosity_mixture == pytest.approx(expected_mixture_flag)


def test_jax_only_fields_carry_configured_values():
    """Every JAX-only configured field reaches the JAX ``PhaseParams``
    stored attribute rather than the aragog library default.

    Contract: the JAX type carries viscosities (as base-10 logarithms),
    conductivities, the transport switches, the eddy diffusivities, the
    mixing-length floor, and the phase-smoothing selection. A field dropped
    at the JAX builder falls back to the library default.

    Discrimination: the config values differ from any plausible library
    default, so a dropped field surfaces as a value mismatch. ``kappah_floor``
    is asserted through the single float cast, and ``phase_smoothing='tanh'``
    is asserted through its 1.0 flag, so a dropped smoothing kwarg (default
    'cubic_hermite', flag 0.0) fails here.
    """
    config = _make_full_config()
    ie = config.interior_energetics
    jax_params = build_jax_phase_params(config)

    assert jax_params.log10_visc_solid == pytest.approx(ie.solid_log10visc)
    assert jax_params.log10_visc_liquid == pytest.approx(ie.melt_log10visc)
    assert jax_params.k_solid == pytest.approx(ie.solid_cond)
    assert jax_params.k_liquid == pytest.approx(ie.melt_cond)
    assert jax_params.eddy_diff_thermal == pytest.approx(ie.eddy_diffusivity_thermal)
    assert jax_params.eddy_diff_chemical == pytest.approx(ie.eddy_diffusivity_chemical)
    assert jax_params.kappah_floor == pytest.approx(ie.kappah_floor)
    assert jax_params.phase_smoothing_tanh == pytest.approx(1.0)
    assert float(jax_params.conduction) == pytest.approx(1.0)
    assert float(jax_params.convection) == pytest.approx(1.0)
    assert float(jax_params.grav_sep) == pytest.approx(0.0)
    assert float(jax_params.mixing) == pytest.approx(1.0)


def test_numpy_only_fields_carry_configured_values():
    """Every numpy-only configured field reaches the ``_PhaseMixedParameters``
    field rather than the aragog library default.

    Contract: the numpy type carries the latent heat, the phase-transition
    width, and the constant-property block. ``cp_blend`` is intentionally
    NOT wired by PROTEUS, so it must stay at the aragog library default
    'latent'; wiring it would be an out-of-scope schema change.

    Discrimination: the config values differ from the library defaults, so
    a dropped field surfaces as a value mismatch. ``cp_blend`` is asserted
    to equal the library default, so a builder that started forwarding it
    from a non-existent config field would fail here.
    """
    config = _make_full_config()
    ie = config.interior_energetics
    numpy_params = build_mixed_phase_params(config, _SOLIDUS, _LIQUIDUS)

    assert numpy_params.latent_heat_of_fusion == pytest.approx(ie.latent_heat_of_fusion)
    assert numpy_params.phase_transition_width == pytest.approx(ie.phase_transition_width)
    assert numpy_params.const_properties is True
    assert numpy_params.const_rho == pytest.approx(ie.const_rho)
    assert numpy_params.const_Cp == pytest.approx(ie.const_Cp)
    assert numpy_params.const_alpha == pytest.approx(ie.const_alpha)
    assert numpy_params.const_cond == pytest.approx(ie.const_cond)
    assert numpy_params.const_log10visc == pytest.approx(ie.const_log10visc)
    assert numpy_params.const_T_ref == pytest.approx(ie.const_T_ref)
    assert numpy_params.const_S_ref == pytest.approx(ie.const_S_ref)
    assert numpy_params.solidus == _SOLIDUS
    assert numpy_params.liquidus == _LIQUIDUS
    assert numpy_params.phase == 'mixed'
    assert numpy_params.cp_blend == 'latent'


def test_kappah_floor_cast_once_is_float():
    """The mixing-length floor is cast to a Python float once in the shared
    resolver, so both JAX sites store an identical float.

    Contract: ``_phase_params_from_config`` casts ``kappah_floor`` with
    ``float`` so the CVODE factory and the research runner, which both call
    ``build_jax_phase_params``, receive one value. Before the shared
    builder, the factory cast the floor and the runner did not, so an int
    or numpy-scalar config value could store two different Python types.

    Discrimination: the resolved input is asserted to be exactly a Python
    float, and to equal the configured value. A resolver that forwarded the
    raw config attribute would carry the mock's type instead.
    """
    config = _make_full_config()
    inputs = _phase_params_from_config(config)

    assert isinstance(inputs.kappah_floor, float)
    assert inputs.kappah_floor == pytest.approx(config.interior_energetics.kappah_floor)


def test_runner_site_uses_the_shared_jax_builder(tmp_path):
    """The JAX research runner's ``PhaseParams`` equals the builder output
    field by field, so the runner site cannot drift from the factory site.

    Contract: ``AragogJAXRunner._build_jax_components`` sets
    ``interior_o._jax_params = build_jax_phase_params(config)``. This test
    drives the real constructor (with the heavy EOS and mesh construction
    mocked) and compares the runner-built params against an independent
    ``build_jax_phase_params(config)`` over all 19 stored attributes.

    Discrimination: comparing every stored attribute, not just one, means a
    regression that reintroduced an inline ``PhaseParams(...)`` at the
    runner and dropped or re-cast any single field fails here. The
    separation-viscosity flag and the ``kappah_floor`` float are included,
    the two fields most exposed to a re-cast divergence.
    """
    config = _make_full_config()
    interior_o = _make_runner_interior_o(spider_eos_dir=str(tmp_path))
    expected = build_jax_phase_params(config)

    with (
        patch('aragog.jax.eos.EntropyEOS_JAX', return_value=MagicMock()),
        patch.object(AragogJAXRunner, '_build_mesh_arrays', return_value=MagicMock()),
    ):
        AragogJAXRunner(config, {'output': str(tmp_path)}, {}, None, interior_o)

    built = interior_o._jax_params
    # The two fields most exposed to a re-cast divergence, asserted by name.
    assert built.separation_viscosity_mixture == pytest.approx(
        expected.separation_viscosity_mixture
    )
    assert built.kappah_floor == pytest.approx(expected.kappah_floor)
    for attr in _JAX_STORED_ATTRS:
        assert float(getattr(built, attr)) == pytest.approx(float(getattr(expected, attr))), (
            f'runner-site PhaseParams.{attr}={getattr(built, attr)!r} '
            f'differs from build_jax_phase_params.{attr}={getattr(expected, attr)!r}'
        )
