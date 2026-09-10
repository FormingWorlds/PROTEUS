"""
Unit tests for proteus.interior_energetics.aragog module: Zalmoxis integration paths.

Tests the Zalmoxis-specific branches in AragogRunner.setup_solver() that set
inner_radius from zalmoxis_solver and configure temperature-dependent initial
conditions.

Testing standards and documentation:
- docs/How-to/testing.md: Running, writing, and marking tests; coverage and CI
- docs/Explanations/test_framework.md: Test tiers, physics invariants, and quality rules

Functions tested:
- AragogRunner.setup_solver(): Zalmoxis branches for inner_radius, EOS fallback
"""

from __future__ import annotations

from unittest.mock import MagicMock, create_autospec, patch

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _make_aragog_config(*, struct_module='spider', mantle_eos='Seager2007:silicate'):
    """Create a mock config for AragogRunner.setup_solver tests."""
    config = MagicMock()
    config.interior_struct.module = struct_module
    config.interior_struct.core_frac = 0.55
    config.interior_struct.zalmoxis.mantle_eos = mantle_eos
    config.interior_struct.core_density = 12500.0
    config.interior_struct.core_heatcap = 880.0
    config.interior_energetics.num_levels = 20
    config.interior_energetics.aragog.mass_coordinates = False
    config.interior_energetics.trans_conduction = True
    config.interior_energetics.trans_convection = True
    config.interior_energetics.trans_grav_sep = False
    config.interior_energetics.trans_mixing = True
    config.interior_energetics.aragog.atol_temperature_equivalent = 0.01
    config.interior_energetics.aragog.core_bc = 'energy_balance'
    config.interior_energetics.aragog.phase_smoothing = 'tanh'
    config.interior_energetics.aragog.solver_method = 'radau'
    config.interior_energetics.aragog.backend = 'numpy'
    config.interior_energetics.aragog.scalar_gravity_override = False
    config.interior_energetics.aragog.phi_step_cap = 0.0
    config.interior_energetics.aragog.temperature_step_cap = 0.0
    config.interior_energetics.aragog.entropy_step_cap = 0.0
    config.interior_energetics.aragog.phase_boundary_entropy_margin = 200.0
    config.interior_energetics.aragog.separation_viscosity = 'mixture'
    config.interior_energetics.spider.matprop_smooth_width = 0.0
    config.interior_energetics.const_properties = False
    config.interior_energetics.heat_radiogenic = False
    config.interior_energetics.heat_tidal = False
    config.planet.tsurf_init = 4000.0
    # Unified tolerance fields (rtol/atol at top level)
    config.interior_energetics.rtol = 1e-4
    config.interior_energetics.atol = 1e-4
    config.interior_energetics.tmagma_atol = 100.0
    config.interior_energetics.tmagma_rtol = 0.02
    # Physics-constant fields shared across Aragog and SPIDER
    config.interior_energetics.adams_williamson_rhos = 4078.95095544
    config.interior_energetics.adiabatic_bulk_modulus = 260e9
    config.interior_energetics.melt_log10visc = 2.0
    config.interior_energetics.solid_log10visc = 22.0
    config.interior_energetics.melt_cond = 4.0
    config.interior_energetics.solid_cond = 4.0
    config.interior_energetics.latent_heat_of_fusion = 4e6
    config.interior_energetics.phase_transition_width = 0.1
    config.interior_energetics.core_tfac_avg = 1.147
    config.params.out.logging = 'WARNING'
    config.interior_struct.eos_dir = 'WolfBower2018_MgSiO3'
    config.interior_struct.melting_dir = 'Wolf_Bower+2018'
    return config


@pytest.mark.unit
def test_setup_solver_zalmoxis_inner_radius(tmp_path):
    """setup_solver reads R_core from hf_row when struct.module='zalmoxis'."""
    from proteus.interior_energetics.aragog import AragogRunner

    outdir = str(tmp_path)
    config = _make_aragog_config(struct_module='zalmoxis')

    R_core_expected = 3.48e6
    hf_row = {
        'R_int': 6.371e6,
        'R_core': R_core_expected,
        'gravity': 9.81,
        'T_magma': 3000.0,
        'T_eqm': 255.0,
        'F_atm': 100.0,
    }
    interior_o = MagicMock()
    interior_o.tides = np.zeros(20)
    spider_eos_dir = tmp_path / 'spider_eos'
    spider_eos_dir.mkdir(parents=True)
    interior_o._spider_eos_dir = str(spider_eos_dir)

    # Create EOS dir
    eos_dir = (
        tmp_path / 'interior_lookup_tables' / 'EOS' / 'dynamic' / 'WolfBower2018_MgSiO3' / 'P-T'
    )
    eos_dir.mkdir(parents=True)
    (eos_dir / 'heat_capacity_melt.dat').write_text('dummy')
    mc_dir = tmp_path / 'interior_lookup_tables' / 'Melting_curves'
    mc_dir.mkdir(parents=True)

    with (
        patch('proteus.interior_energetics.aragog.FWL_DATA_DIR', tmp_path),
        patch('proteus.interior_energetics.aragog.Parameters') as mock_params,
        patch('proteus.interior_energetics.aragog.EntropySolver'),
        patch('proteus.interior_energetics.aragog.EntropyEOS'),
    ):
        AragogRunner.setup_solver(config, hf_row, interior_o, outdir)

    # Verify inner_radius was set from hf_row['R_core']
    call_kwargs = mock_params.call_args
    mesh_arg = call_kwargs.kwargs.get('mesh') or call_kwargs[1].get('mesh')
    assert mesh_arg.inner_radius == pytest.approx(R_core_expected)
    # Discriminator: a regression that read R_core from the wrong field
    # (e.g. config.interior_struct.core_frac * R_int = 0.55 * 6.371e6 =
    # 3.504e6) would still pass an approx pin on the right order of
    # magnitude. The fallback value is ~24 km away from R_core_expected;
    # require the gap to be smaller than that.
    R_core_fallback = 0.55 * 6.371e6
    assert abs(mesh_arg.inner_radius - R_core_expected) < abs(R_core_fallback - R_core_expected)
    # Bounded mesh discriminator (Section 3 boundedness): inner_radius
    # must lie strictly inside (0, R_int) regardless of source field.
    assert 0.0 < mesh_arg.inner_radius < 6.371e6


@pytest.mark.unit
def test_setup_solver_zalmoxis_wolfbower_temp(tmp_path):
    """setup_solver uses Zalmoxis T-profile for WolfBower2018 EOS (initial_condition=2)."""
    from proteus.interior_energetics.aragog import AragogRunner

    outdir = str(tmp_path)
    config = _make_aragog_config(struct_module='zalmoxis', mantle_eos='WolfBower2018:MgSiO3')

    hf_row = {
        'R_int': 6.371e6,
        'gravity': 9.81,
        'T_magma': 3000.0,
        'T_eqm': 255.0,
        'F_atm': 100.0,
    }
    interior_o = MagicMock()
    interior_o.tides = np.zeros(20)
    spider_eos_dir = tmp_path / 'spider_eos'
    spider_eos_dir.mkdir(parents=True)
    interior_o._spider_eos_dir = str(spider_eos_dir)

    eos_dir = (
        tmp_path / 'interior_lookup_tables' / 'EOS' / 'dynamic' / 'WolfBower2018_MgSiO3' / 'P-T'
    )
    eos_dir.mkdir(parents=True)
    (eos_dir / 'heat_capacity_melt.dat').write_text('dummy')
    mc_dir = tmp_path / 'interior_lookup_tables' / 'Melting_curves'
    mc_dir.mkdir(parents=True)

    with (
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            return_value=(3.48e6, None),
        ),
        patch('proteus.interior_energetics.aragog.FWL_DATA_DIR', tmp_path),
        patch('proteus.interior_energetics.aragog.Parameters'),
        patch('proteus.interior_energetics.aragog.EntropySolver'),
        patch('proteus.interior_energetics.aragog.EntropyEOS'),
        patch('proteus.interior_energetics.aragog._InitialConditionParameters') as mock_ic,
    ):
        AragogRunner.setup_solver(config, hf_row, interior_o, outdir)

    # WolfBower2018 should set initial_condition=2 with zalmoxis_output_temp.txt
    assert mock_ic.called
    call_kwargs = mock_ic.call_args[1]
    assert call_kwargs['initial_condition'] == 2
    assert 'zalmoxis_output_temp.txt' in call_kwargs['init_file']


@pytest.mark.unit
def test_setup_solver_eos_fallback(tmp_path):
    """setup_solver falls back to legacy EOS path when unified path is missing."""
    from proteus.interior_energetics.aragog import AragogRunner

    outdir = str(tmp_path)
    config = _make_aragog_config(struct_module='spider')

    hf_row = {
        'R_int': 6.371e6,
        'gravity': 9.81,
        'T_magma': 3000.0,
        'T_eqm': 255.0,
        'F_atm': 100.0,
    }
    interior_o = MagicMock()
    interior_o.tides = np.zeros(20)
    spider_eos_dir = tmp_path / 'spider_eos'
    spider_eos_dir.mkdir(parents=True)
    interior_o._spider_eos_dir = str(spider_eos_dir)

    # Only create legacy path, NOT unified path
    legacy_dir = (
        tmp_path
        / 'interior_lookup_tables'
        / '1TPa-dK09-elec-free'
        / 'MgSiO3_Wolf_Bower_2018_1TPa'
    )
    legacy_dir.mkdir(parents=True)
    (legacy_dir / 'heat_capacity_melt.dat').write_text('dummy')
    mc_dir = tmp_path / 'interior_lookup_tables' / 'Melting_curves'
    mc_dir.mkdir(parents=True)

    with (
        patch('proteus.interior_energetics.aragog.FWL_DATA_DIR', tmp_path),
        patch('proteus.interior_energetics.aragog.Parameters'),
        patch('proteus.interior_energetics.aragog.EntropySolver') as mock_solver,
        patch('proteus.interior_energetics.aragog.EntropyEOS'),
    ):
        AragogRunner.setup_solver(config, hf_row, interior_o, outdir)

    assert mock_solver.called
    # Fallback-path discriminator: the solver must have been instantiated
    # exactly once (the fallback path runs the setup body to completion;
    # a regression that retried after the unified-path miss could call
    # the solver more than once or zero times via a swallowed exception).
    assert mock_solver.call_count == 1


@pytest.mark.unit
def test_setup_solver_eos_not_found(tmp_path):
    """setup_solver raises FileNotFoundError when EOS data is missing."""
    from proteus.interior_energetics.aragog import AragogRunner

    outdir = str(tmp_path)
    config = _make_aragog_config(struct_module='spider')
    config.interior_struct.eos_dir = 'NonexistentEOS'

    hf_row = {
        'R_int': 6.371e6,
        'gravity': 9.81,
        'T_magma': 3000.0,
        'T_eqm': 255.0,
        'F_atm': 100.0,
    }
    interior_o = MagicMock()
    interior_o.tides = np.zeros(20)

    with (
        patch('proteus.interior_energetics.aragog.FWL_DATA_DIR', tmp_path),
        patch('proteus.interior_energetics.aragog.EntropySolver') as mock_solver,
        pytest.raises(FileNotFoundError, match='Aragog lookup data not found'),
    ):
        AragogRunner.setup_solver(config, hf_row, interior_o, outdir)

    # No-side-effect discriminator: the EOS-path check raises before the
    # solver is instantiated. A regression that downgraded the missing
    # data to a warning and proceeded with a stale path would have
    # called EntropySolver at least once.
    assert not mock_solver.called


@pytest.mark.unit
class TestUpdateStructureZalmoxisRefresh:
    """Verify that when the structure module is Zalmoxis and Zalmoxis
    re-solves mid-run, Aragog's inner_radius tracks R_core from hf_row
    on every coupling step, not just at init time.
    """

    def _make_solver(self, outer=6.4e6, inner=3.6e6, gravity=7.9):
        solver = MagicMock()
        solver.parameters.mesh.outer_radius = outer
        solver.parameters.mesh.inner_radius = inner
        solver.parameters.mesh.gravitational_acceleration = gravity
        interior_o = MagicMock()
        interior_o.aragog_solver = solver
        return solver, interior_o

    def test_zalmoxis_refreshes_inner_radius(self):
        """R_core that shifts between two coupling steps must land in
        solver.parameters.mesh.inner_radius."""
        from proteus.interior_energetics.aragog import AragogRunner

        solver, interior_o = self._make_solver(inner=3.4e6)
        config = _make_aragog_config(struct_module='zalmoxis')
        hf_row = {
            'R_int': 6.4e6,
            'R_core': 3.6e6,
            'gravity': 8.1,
            'Time': 1.0e5,
        }
        AragogRunner.update_structure(config, hf_row, interior_o)
        assert solver.parameters.mesh.outer_radius == pytest.approx(6.4e6)
        assert solver.parameters.mesh.inner_radius == pytest.approx(3.6e6)
        assert solver.parameters.mesh.gravitational_acceleration == pytest.approx(8.1)

    def test_zalmoxis_inner_radius_falls_back_to_core_frac(self):
        """Missing or non-positive R_core falls back to
        config.interior_struct.core_frac * R_int."""
        from proteus.interior_energetics.aragog import AragogRunner

        solver, interior_o = self._make_solver(inner=3.4e6)
        config = _make_aragog_config(struct_module='zalmoxis')
        config.interior_struct.core_frac = 0.50
        hf_row = {
            'R_int': 6.4e6,
            'R_core': 0.0,  # unset / not populated yet
            'gravity': 8.1,
            'Time': 0.0,
        }
        AragogRunner.update_structure(config, hf_row, interior_o)
        assert solver.parameters.mesh.inner_radius == pytest.approx(3.2e6)
        # Discriminator: 3.2e6 (= 0.50 * 6.4e6) is the fallback value;
        # a regression that read R_core=0.0 verbatim would set
        # inner_radius to 0, while a regression that used the old
        # init-time inner_radius (3.4e6) would land 200 km away.
        assert solver.parameters.mesh.inner_radius > 0.0
        assert abs(solver.parameters.mesh.inner_radius - 3.4e6) > 100e3

    def test_zalmoxis_rejects_negative_r_core(self):
        """A negative R_core (corrupt / failed solve) triggers the
        core_frac fallback rather than propagating a nonsensical mesh."""
        from proteus.interior_energetics.aragog import AragogRunner

        solver, interior_o = self._make_solver(inner=3.4e6)
        config = _make_aragog_config(struct_module='zalmoxis')
        config.interior_struct.core_frac = 0.40
        hf_row = {
            'R_int': 6.4e6,
            'R_core': -1.0,
            'gravity': 8.1,
            'Time': 0.0,
        }
        AragogRunner.update_structure(config, hf_row, interior_o)
        assert solver.parameters.mesh.inner_radius == pytest.approx(2.56e6)
        # Positivity discriminator (Section 3): the fallback exists
        # precisely so a nonsensical negative R_core never reaches the
        # mesh. A regression that propagated -1.0 verbatim would land
        # at a negative inner_radius and trigger this guard.
        assert solver.parameters.mesh.inner_radius > 0.0
        # Bounded discriminator: 2.56e6 = 0.40 * 6.4e6 must lie strictly
        # inside (0, R_int) and must differ from R_int.
        assert solver.parameters.mesh.inner_radius < 6.4e6

    def test_spider_branch_unchanged(self):
        """The existing spider / dummy branch continues to refresh
        inner_radius from hf_row['R_core']."""
        from proteus.interior_energetics.aragog import AragogRunner

        solver, interior_o = self._make_solver(inner=3.2e6)
        config = _make_aragog_config(struct_module='spider')
        hf_row = {
            'R_int': 6.4e6,
            'R_core': 3.5e6,
            'gravity': 9.81,
            'Time': 0.0,
        }
        AragogRunner.update_structure(config, hf_row, interior_o)
        assert solver.parameters.mesh.inner_radius == pytest.approx(3.5e6)
        # Discriminator: on the spider branch the inner_radius must
        # track hf_row['R_core'] directly, not the core_frac fallback
        # (= 0.55 * 6.4e6 = 3.52e6) and not the init-time inner_radius
        # (3.2e6). The pin above already distinguishes 3.52e6 (within
        # 20 km) but 3.2e6 is 300 km away; the explicit lower-bound
        # check makes the failure mode loud.
        assert abs(solver.parameters.mesh.inner_radius - 3.2e6) > 100e3


@pytest.mark.unit
def test_effective_step_caps_default_off_on_every_interior():
    """An unset step cap resolves to off (0.0) on every interior, zalmoxis included.

    Each cap is a SUNDIALS root function that returns control from the interior
    sub-solve when a cell's per-step change reaches the cap. On a benign
    freezing-front crossing it slices the coupled step into many small ones and
    drives the reported CMB heat flux briefly negative (a controlled comparison
    gives +48% outer coupled steps for the same interior work), so the coupled
    default is off and no interior arms a cap without the user opting in. This
    pins all three caps to 0.0 for the unset schema default across the zalmoxis,
    spider, and dummy interiors, so a regression that re-arms a positive default
    for any interior is caught. A distinct positive value resolves verbatim in
    the same call, so a resolver that masked the check by clamping every cap to
    0.0 is caught too.
    """
    from proteus.interior_energetics.aragog import (
        _effective_entropy_step_cap,
        _effective_phi_step_cap,
        _effective_temperature_step_cap,
    )

    def cfg(module, phi, t_cap, s_cap):
        c = MagicMock()
        c.interior_struct.module = module
        c.interior_energetics.aragog.phi_step_cap = phi
        c.interior_energetics.aragog.temperature_step_cap = t_cap
        c.interior_energetics.aragog.entropy_step_cap = s_cap
        return c

    # Unset schema default (0.0) resolves to off (0.0) on every interior.
    for module in ('zalmoxis', 'spider', 'dummy'):
        unset = cfg(module, 0.0, 0.0, 0.0)
        assert _effective_phi_step_cap(unset) == 0.0
        assert _effective_temperature_step_cap(unset) == 0.0
        assert _effective_entropy_step_cap(unset) == 0.0

    # Discrimination: a distinct positive value resolves verbatim on zalmoxis,
    # so the off result above is a real resolution, not a constant-0.0 stub.
    on = cfg('zalmoxis', 0.05, 250.0, 75.0)
    assert _effective_phi_step_cap(on) == pytest.approx(0.05)
    assert _effective_temperature_step_cap(on) == pytest.approx(250.0)
    assert _effective_entropy_step_cap(on) == pytest.approx(75.0)


@pytest.mark.unit
def test_effective_step_caps_pass_positive_verbatim_and_off_sentinel_disables():
    """A positive cap passes through verbatim; the -1.0 sentinel resolves to off.

    The cap mechanism stays available as an explicit opt-in for debugging a
    pathological config, so a positive value reaches Aragog unchanged on any
    interior. The -1.0 off sentinel, and defensively any negative, resolves to
    exactly 0.0 so no literal negative cap ever reaches the solver.
    Discrimination: the positive values are distinct per cap, so a resolver
    that swapped or dropped a field is caught, and the sentinel result is
    contrasted against the positive one so an off switch that leaked the
    negative through is caught.
    """
    from proteus.interior_energetics.aragog import (
        _effective_entropy_step_cap,
        _effective_phi_step_cap,
        _effective_temperature_step_cap,
    )

    def cfg(module, phi, t_cap, s_cap):
        c = MagicMock()
        c.interior_struct.module = module
        c.interior_energetics.aragog.phi_step_cap = phi
        c.interior_energetics.aragog.temperature_step_cap = t_cap
        c.interior_energetics.aragog.entropy_step_cap = s_cap
        return c

    # Positive values pass through verbatim on zalmoxis and on a non-zalmoxis
    # interior alike; distinct values pin each cap to its own field.
    on_z = cfg('zalmoxis', 0.05, 250.0, 75.0)
    assert _effective_phi_step_cap(on_z) == pytest.approx(0.05)
    assert _effective_temperature_step_cap(on_z) == pytest.approx(250.0)
    assert _effective_entropy_step_cap(on_z) == pytest.approx(75.0)
    on_s = cfg('spider', 0.2, 300.0, 90.0)
    assert _effective_phi_step_cap(on_s) == pytest.approx(0.2)
    assert _effective_temperature_step_cap(on_s) == pytest.approx(300.0)
    assert _effective_entropy_step_cap(on_s) == pytest.approx(90.0)

    # The -1.0 sentinel, and any negative, resolves to exactly 0.0, never a
    # literal negative reaching the solver.
    for module in ('zalmoxis', 'spider'):
        off = cfg(module, -1.0, -1.0, -1.0)
        assert _effective_phi_step_cap(off) == 0.0
        assert _effective_temperature_step_cap(off) == 0.0
        assert _effective_entropy_step_cap(off) == 0.0


@pytest.mark.unit
def test_aragog_schema_admits_off_sentinel_rejects_other_negatives():
    """Only -1.0 disables the step caps; any other negative, NaN, or inf raises.

    The resolver reads a single canonical off value (-1.0), so the schema admits
    exactly that among the negatives and round-trips it unchanged while rejecting
    every other out-of-range value loudly. A malformed cap that silently disabled
    the crystallisation-onset guard is the failure this prevents. The proximity
    band phase_boundary_entropy_margin keeps its positive-only contract, so its
    0.0 and negatives still raise; the paired assertions pin both sides so a
    regression that loosens either is caught.
    """
    from proteus.config._interior import Aragog

    # -1.0 is the one admitted negative and round-trips unchanged.
    caps = Aragog(phi_step_cap=-1.0, temperature_step_cap=-1.0, entropy_step_cap=-1.0)
    assert caps.phi_step_cap == pytest.approx(-1.0)
    assert caps.temperature_step_cap == pytest.approx(-1.0)
    assert caps.entropy_step_cap == pytest.approx(-1.0)
    # zero remains valid (resolves to off) and positive is verbatim.
    assert Aragog(phi_step_cap=0.0).phi_step_cap == 0.0
    assert Aragog(temperature_step_cap=150.0).temperature_step_cap == pytest.approx(150.0)
    # The sentinel is an exact match, so even a near-miss like -1.0001 raises;
    # that is what makes -1.0 a single canonical off value rather than a band.
    for bad in (-2.0, -0.5, -1.0001):
        with pytest.raises(ValueError):
            Aragog(phi_step_cap=bad)
        with pytest.raises(ValueError):
            Aragog(temperature_step_cap=bad)
        with pytest.raises(ValueError):
            Aragog(entropy_step_cap=bad)
    # NaN and the infinities are malformed caps and raise on every field.
    for bad in (float('nan'), float('inf'), float('-inf')):
        with pytest.raises(ValueError):
            Aragog(phi_step_cap=bad)
        with pytest.raises(ValueError):
            Aragog(temperature_step_cap=bad)
        with pytest.raises(ValueError):
            Aragog(entropy_step_cap=bad)
    # the proximity band keeps its positive-only contract
    with pytest.raises(ValueError):
        Aragog(phase_boundary_entropy_margin=-1.0)
    with pytest.raises(ValueError):
        Aragog(phase_boundary_entropy_margin=0.0)


# ---------------------------------------------------------------------------
# Phase-boundary entropy margin: wrapper threading and version-skew guard.
#
# These stand-in signatures stay independent of the installed Aragog so the
# assertions hold on CI, where the pip-installed Aragog may predate the field.
# create_autospec copies each stub's signature, which is what the wrapper's
# version-skew guard inspects to decide whether the field is supported.
# ---------------------------------------------------------------------------


def _paired_energy_stub(
    *,
    temperature_step_cap=None,
    entropy_step_cap=None,
    phase_boundary_entropy_margin=None,
    **rest,
):
    """Signature of a paired Aragog: it accepts all three managed stepping
    controls, so the guard treats phase_boundary_entropy_margin as supported
    and threads the value straight through."""
    return MagicMock()


def _caps_only_energy_stub(
    *,
    temperature_step_cap=None,
    entropy_step_cap=None,
    **rest,
):
    """Signature of an Aragog that accepts the step caps but predates
    phase_boundary_entropy_margin, so the guard must drop only the margin."""
    return MagicMock()


def _spider_fallback_scaffold(tmp_path):
    """Build the (hf_row, interior_o) inputs and the legacy EOS/melting dirs a
    spider-stack setup_solver needs to reach the _EnergyParameters call."""
    hf_row = {
        'R_int': 6.371e6,
        'gravity': 9.81,
        'T_magma': 3000.0,
        'T_eqm': 255.0,
        'F_atm': 100.0,
    }
    interior_o = MagicMock()
    interior_o.tides = np.zeros(20)
    spider_eos_dir = tmp_path / 'spider_eos'
    spider_eos_dir.mkdir(parents=True)
    interior_o._spider_eos_dir = str(spider_eos_dir)

    legacy_dir = (
        tmp_path
        / 'interior_lookup_tables'
        / '1TPa-dK09-elec-free'
        / 'MgSiO3_Wolf_Bower_2018_1TPa'
    )
    legacy_dir.mkdir(parents=True)
    (legacy_dir / 'heat_capacity_melt.dat').write_text('dummy')
    (tmp_path / 'interior_lookup_tables' / 'Melting_curves').mkdir(parents=True)
    return hf_row, interior_o


@pytest.mark.unit
def test_setup_solver_threads_phase_boundary_margin(tmp_path):
    """setup_solver passes phase_boundary_entropy_margin into _EnergyParameters
    verbatim when the installed Aragog accepts it.

    This pins the passthrough the release gate depends on: the omitted/default
    knob must reach Aragog as 200.0 (bit-identical to the previously hard-coded
    band), and a user override must arrive unchanged rather than being clamped
    or ignored. A supported field must never trigger the version-skew warning.
    """
    from proteus.interior_energetics.aragog import AragogRunner

    outdir = str(tmp_path)
    threaded = {}
    for requested in (200.0, 350.0):
        config = _make_aragog_config(struct_module='spider')
        config.interior_energetics.aragog.phase_boundary_entropy_margin = requested
        hf_row, interior_o = _spider_fallback_scaffold(tmp_path / f'run_{requested}')
        mock_ep = create_autospec(_paired_energy_stub)
        with (
            patch(
                'proteus.interior_energetics.aragog.FWL_DATA_DIR', tmp_path / f'run_{requested}'
            ),
            patch('proteus.interior_energetics.aragog.Parameters'),
            patch('proteus.interior_energetics.aragog.EntropySolver'),
            patch('proteus.interior_energetics.aragog.EntropyEOS'),
            patch('proteus.interior_energetics.aragog._EnergyParameters', mock_ep),
            patch('proteus.interior_energetics.aragog.log') as mock_log,
        ):
            AragogRunner.setup_solver(config, hf_row, interior_o, outdir)

        assert mock_ep.called
        threaded[requested] = mock_ep.call_args.kwargs['phase_boundary_entropy_margin']
        # A supported field is never dropped, so the guard warning must stay
        # silent even for a non-default value.
        assert not any(
            'phase_boundary_entropy_margin' in str(c) for c in mock_log.warning.call_args_list
        )

    # The value arrives unchanged at both the default and an override.
    assert threaded[200.0] == pytest.approx(200.0)
    assert threaded[350.0] == pytest.approx(350.0)
    # Discrimination: a wrapper that hard-coded or ignored the knob would send
    # the same number twice; the two requests must remain distinct.
    assert threaded[350.0] != pytest.approx(threaded[200.0])


@pytest.mark.unit
def test_setup_solver_threads_resolved_step_caps(tmp_path):
    """setup_solver threads the RESOLVED step caps into _EnergyParameters, not
    the raw config values.

    Each cap reaches Aragog as a SUNDIALS root function, so the off sentinel
    must be resolved before it is passed: a -1.0 in the config must arrive as
    0.0 (no cap), never as a literal -1.0 that would arm the root function at a
    negative threshold. A positive value must arrive verbatim. This pins the
    resolver on the wrapper path (phi is always threaded; the T and S caps are
    threaded when the installed Aragog accepts them), so a regression that
    forwarded config.*_step_cap directly is caught.
    """
    from proteus.interior_energetics.aragog import AragogRunner

    outdir = str(tmp_path)
    # config caps -> expected resolved caps at the _EnergyParameters call.
    # -1.0 is the off sentinel and must resolve to 0.0; a positive value passes
    # through unchanged.
    cases = {
        'sentinel': ((-1.0, -1.0, -1.0), (0.0, 0.0, 0.0)),
        'positive': ((0.2, 150.0, 175.0), (0.2, 150.0, 175.0)),
    }
    for name, (raw, expected) in cases.items():
        config = _make_aragog_config(struct_module='spider')
        config.interior_energetics.aragog.phi_step_cap = raw[0]
        config.interior_energetics.aragog.temperature_step_cap = raw[1]
        config.interior_energetics.aragog.entropy_step_cap = raw[2]
        hf_row, interior_o = _spider_fallback_scaffold(tmp_path / name)
        mock_ep = create_autospec(_paired_energy_stub)
        with (
            patch('proteus.interior_energetics.aragog.FWL_DATA_DIR', tmp_path / name),
            patch('proteus.interior_energetics.aragog.Parameters'),
            patch('proteus.interior_energetics.aragog.EntropySolver'),
            patch('proteus.interior_energetics.aragog.EntropyEOS'),
            patch('proteus.interior_energetics.aragog._EnergyParameters', mock_ep),
            patch('proteus.interior_energetics.aragog.log'),
        ):
            AragogRunner.setup_solver(config, hf_row, interior_o, outdir)

        assert mock_ep.called
        kwargs = mock_ep.call_args.kwargs
        assert kwargs['phi_step_cap'] == pytest.approx(expected[0])
        assert kwargs['temperature_step_cap'] == pytest.approx(expected[1])
        assert kwargs['entropy_step_cap'] == pytest.approx(expected[2])


@pytest.mark.unit
def test_setup_solver_drops_margin_on_old_aragog(tmp_path):
    """The version-skew guard drops phase_boundary_entropy_margin, and only it,
    when the installed Aragog predates the field.

    An Aragog that still accepts the step caps but lacks the margin must not
    crash on an unexpected keyword: the wrapper pops the margin from the kwargs.
    It warns only when the user set a non-default band (a dropped 200.0 is a
    silent no-op because Aragog's built-in default is also 200.0), so a default
    config degrades quietly while a meaningful override is surfaced once.
    """
    from proteus.interior_energetics.aragog import AragogRunner

    outdir = str(tmp_path)

    # Non-default margin on an old Aragog: dropped AND warned about.
    config = _make_aragog_config(struct_module='spider')
    config.interior_energetics.aragog.phase_boundary_entropy_margin = 350.0
    hf_row, interior_o = _spider_fallback_scaffold(tmp_path / 'nondefault')
    mock_ep = create_autospec(_caps_only_energy_stub)
    with (
        patch('proteus.interior_energetics.aragog.FWL_DATA_DIR', tmp_path / 'nondefault'),
        patch('proteus.interior_energetics.aragog.Parameters'),
        patch('proteus.interior_energetics.aragog.EntropySolver'),
        patch('proteus.interior_energetics.aragog.EntropyEOS'),
        patch('proteus.interior_energetics.aragog._EnergyParameters', mock_ep),
        patch('proteus.interior_energetics.aragog.log') as mock_log,
    ):
        AragogRunner.setup_solver(config, hf_row, interior_o, outdir)

    kwargs = mock_ep.call_args.kwargs
    # The unsupported margin is removed so construction cannot raise; the caps
    # the old Aragog does accept are left in place.
    assert 'phase_boundary_entropy_margin' not in kwargs
    assert 'temperature_step_cap' in kwargs
    assert 'entropy_step_cap' in kwargs
    # A non-default band that was silently discarded is surfaced exactly once.
    margin_warnings = [
        c for c in mock_log.warning.call_args_list if 'phase_boundary_entropy_margin' in str(c)
    ]
    assert len(margin_warnings) == 1

    # Default margin on the same old Aragog: still dropped, but no warning,
    # because Aragog's built-in 200.0 reproduces the requested band.
    config = _make_aragog_config(struct_module='spider')
    config.interior_energetics.aragog.phase_boundary_entropy_margin = 200.0
    hf_row, interior_o = _spider_fallback_scaffold(tmp_path / 'default')
    mock_ep = create_autospec(_caps_only_energy_stub)
    with (
        patch('proteus.interior_energetics.aragog.FWL_DATA_DIR', tmp_path / 'default'),
        patch('proteus.interior_energetics.aragog.Parameters'),
        patch('proteus.interior_energetics.aragog.EntropySolver'),
        patch('proteus.interior_energetics.aragog.EntropyEOS'),
        patch('proteus.interior_energetics.aragog._EnergyParameters', mock_ep),
        patch('proteus.interior_energetics.aragog.log') as mock_log,
    ):
        AragogRunner.setup_solver(config, hf_row, interior_o, outdir)

    assert 'phase_boundary_entropy_margin' not in mock_ep.call_args.kwargs
    assert not any(
        'phase_boundary_entropy_margin' in str(c) for c in mock_log.warning.call_args_list
    )


def test_setup_or_update_solver_tracks_stale_structure_steps():
    """The stale-structure counter increments while stale and resets when fresh.

    On the update branch (solver already built) ``setup_or_update_solver`` reads
    ``interior_o.structure_stale`` to run a consecutive-stale-step counter: each
    call on a stale Zalmoxis mesh increments it and any call on a fresh mesh
    resets it to zero. That counter feeds the operator-visible stale-mesh
    window, so a regression that fails to increment (silent stale drift) or
    fails to reset (a false persistent-stale reading after recovery) must be
    caught. The structure/solver updates are mocked so only the counter branch
    is exercised.
    """
    from proteus.interior_energetics.aragog import AragogRunner

    interior_o = MagicMock()
    interior_o.aragog_solver = MagicMock()  # non-None: take the update branch
    interior_o.ic = 0  # else-branch: update_structure + update_solver, both mocked
    interior_o._last_entropy = None  # skip the entropy-restore tail
    interior_o._stale_struct_steps = 0
    config = MagicMock()
    hf_row = {'R_int': 1.0e6, 'M_int': 1.0e23}
    # The update branch never dereferences dirs for a non-IC step; a plain
    # placeholder keeps the call signature satisfied without touching disk.
    dirs = {'output': 'unused'}

    with (
        patch.object(AragogRunner, 'update_structure'),
        patch.object(AragogRunner, 'update_solver'),
    ):
        # Stale mesh: the counter climbs by one on each successive call.
        interior_o.structure_stale = True
        AragogRunner.setup_or_update_solver(config, hf_row, interior_o, 1.0, dirs)
        assert interior_o._stale_struct_steps == 1
        AragogRunner.setup_or_update_solver(config, hf_row, interior_o, 1.0, dirs)
        assert interior_o._stale_struct_steps == 2
        # Fresh mesh: the counter resets to zero, not merely decrements.
        interior_o.structure_stale = False
        AragogRunner.setup_or_update_solver(config, hf_row, interior_o, 1.0, dirs)
        assert interior_o._stale_struct_steps == 0
        # Recovery then a new failure restarts the count from one, confirming the
        # reset is a true clear and the increment is not a running total.
        interior_o.structure_stale = True
        AragogRunner.setup_or_update_solver(config, hf_row, interior_o, 1.0, dirs)
        assert interior_o._stale_struct_steps == 1


@pytest.mark.unit
def test_solve_with_retry_ladder_exhaustion_names_the_solver_that_actually_ran(
    monkeypatch,
):
    """A retry-ladder exhaustion names the integrator that actually ran.

    ``solver_method`` can ask for CVODE and still run scipy: the wrapper is
    compiled against SUNDIALS and falls back silently on a build or ABI
    mismatch, so trusting the config name mislabels every scipy-fallback
    failure as a CVODE one. Covers CVODE available, CVODE unavailable
    (silent fallback to Radau), an explicit 'radau', and an explicit 'bdf',
    so a mutant that drops the solver_method check or collapses Radau/BDF
    into one label fails at least one branch. Each case also asserts
    ``solve()`` ran once per attempt, so a mutant that breaks the retry loop
    itself (wrong attempt count, early exit) fails alongside the label.
    """
    from proteus.interior_energetics.aragog import AragogRunner

    module_path = 'aragog.solver.entropy_solver'

    def _build_runner(status, solver_method='cvode'):
        runner = AragogRunner.__new__(AragogRunner)
        runner._config = MagicMock()
        runner._config.interior_energetics.aragog.solver_method = solver_method
        runner._config.planet.mass_tot = 1.0

        out = MagicMock()
        out.status = status
        out.T_core = 0.0

        solver = MagicMock()
        solver.parameters.solver.start_time = 0.0
        solver.parameters.solver.end_time = 1.0
        solver.get_current_dSdr_cmb.return_value = None
        solver._dSdr_cmb_init = None
        solver.get_state.return_value = out
        runner.aragog_solver = solver

        interior_o = MagicMock()
        interior_o._last_entropy = None

        hf_row = {'Time': 2.15e5, 'T_cmb': 0.0}
        return runner, interior_o, hf_row

    max_attempts = 6

    monkeypatch.setattr(f'{module_path}._CVODE_AVAILABLE', True)
    cvode_runner, cvode_interior_o, cvode_hf_row = _build_runner(status=-1)
    with pytest.raises(RuntimeError, match='CVODE status=-1') as cvode_info:
        cvode_runner._solve_with_retry(cvode_hf_row, cvode_interior_o)
    assert 'Radau status=' not in str(cvode_info.value)
    assert 'BDF status=' not in str(cvode_info.value)
    assert cvode_runner.aragog_solver.solve.call_count == max_attempts

    monkeypatch.setattr(f'{module_path}._CVODE_AVAILABLE', False)
    fallback_runner, fallback_interior_o, fallback_hf_row = _build_runner(status=-1)
    with pytest.raises(RuntimeError, match='Radau status=-1') as fallback_info:
        fallback_runner._solve_with_retry(fallback_hf_row, fallback_interior_o)
    assert 'CVODE status=' not in str(fallback_info.value)
    assert 'BDF status=' not in str(fallback_info.value)
    assert fallback_runner.aragog_solver.solve.call_count == max_attempts

    monkeypatch.setattr(f'{module_path}._CVODE_AVAILABLE', True)
    radau_runner, radau_interior_o, radau_hf_row = _build_runner(
        status=-1, solver_method='radau'
    )
    with pytest.raises(RuntimeError, match='Radau status=-1') as radau_info:
        radau_runner._solve_with_retry(radau_hf_row, radau_interior_o)
    assert 'CVODE status=' not in str(radau_info.value)
    assert 'BDF status=' not in str(radau_info.value)
    assert radau_runner.aragog_solver.solve.call_count == max_attempts

    monkeypatch.setattr(f'{module_path}._CVODE_AVAILABLE', True)
    bdf_runner, bdf_interior_o, bdf_hf_row = _build_runner(status=-1, solver_method='bdf')
    with pytest.raises(RuntimeError, match='BDF status=-1') as bdf_info:
        bdf_runner._solve_with_retry(bdf_hf_row, bdf_interior_o)
    assert 'CVODE status=' not in str(bdf_info.value)
    assert 'Radau status=' not in str(bdf_info.value)
    assert bdf_runner.aragog_solver.solve.call_count == max_attempts


@pytest.mark.unit
def test_active_solver_name_reports_unknown_when_cvode_probe_fails(monkeypatch):
    """A missing/renamed aragog CVODE flag yields an explicit unknown label.

    ``_aragog_cvode_available()`` reads aragog's private ``_CVODE_AVAILABLE``
    flag. aragog is a separate, actively developed package that owes that
    private name no stability guarantee. When that name is absent, the cvode
    branch must report the probe failure, not coerce to Radau, because a real
    CVODE run would then mislabel as scipy. The probe must never raise: the
    retry ladder relies on the intended ``RuntimeError``, not an uncaught
    ``ImportError``.
    """
    from proteus.interior_energetics.aragog import AragogRunner

    monkeypatch.delattr('aragog.solver.entropy_solver._CVODE_AVAILABLE')

    runner = AragogRunner.__new__(AragogRunner)
    runner._config = MagicMock()
    runner._config.interior_energetics.aragog.solver_method = 'cvode'
    name = runner._active_solver_name()
    assert 'unknown' in name.lower()
    assert name not in ('CVODE', 'Radau', 'BDF')

    # The missing flag must not leak into or corrupt the 'bdf' branch,
    # which never consults _CVODE_AVAILABLE in the first place.
    runner._config.interior_energetics.aragog.solver_method = 'bdf'
    assert runner._active_solver_name() == 'BDF'


@pytest.mark.unit
def test_aragog_still_exposes_cvode_availability_flag():
    """``_aragog_cvode_available`` depends on aragog's ``_CVODE_AVAILABLE``.

    aragog owes that private name no stability guarantee. If a later aragog
    renames or removes it while still satisfying the ``fwl-aragog>=26.07.04``
    floor, the CVODE label silently reverts to Radau in production. This test
    fails the moment the depended-on symbol disappears, so the drift is caught
    here instead of in a mislabelled run.
    """
    import aragog.solver.entropy_solver as entropy_solver

    assert hasattr(entropy_solver, '_CVODE_AVAILABLE'), (
        'aragog.solver.entropy_solver._CVODE_AVAILABLE is gone; '
        'AragogRunner._aragog_cvode_available can no longer detect CVODE'
    )


@pytest.mark.unit
def test_retry_exhaustion_labels_unknown_when_cvode_probe_fails(monkeypatch):
    """Exhaustion names the probe failure, not a wrong integrator.

    When the aragog CVODE flag is absent, the retry-ladder exhaustion message
    must name the probe failure rather than a specific integrator, so a real
    CVODE run does not mislabel as Radau. The path still raises the
    ``RuntimeError`` the retry ladder depends on, not an ``ImportError``.
    """
    from proteus.interior_energetics.aragog import AragogRunner

    monkeypatch.delattr('aragog.solver.entropy_solver._CVODE_AVAILABLE')

    runner = AragogRunner.__new__(AragogRunner)
    runner._config = MagicMock()
    runner._config.interior_energetics.aragog.solver_method = 'cvode'
    runner._config.planet.mass_tot = 1.0

    out = MagicMock()
    out.status = -1
    out.T_core = 0.0

    solver = MagicMock()
    solver.parameters.solver.start_time = 0.0
    solver.parameters.solver.end_time = 1.0
    solver.get_current_dSdr_cmb.return_value = None
    solver._dSdr_cmb_init = None
    solver.get_state.return_value = out
    runner.aragog_solver = solver

    interior_o = MagicMock()
    interior_o._last_entropy = None
    hf_row = {'Time': 2.15e5, 'T_cmb': 0.0}

    with pytest.raises(RuntimeError) as info:
        runner._solve_with_retry(hf_row, interior_o)
    msg = str(info.value)
    assert 'unknown' in msg.lower()
    assert 'Radau status=' not in msg
    assert 'CVODE status=' not in msg

    # The label is built only on the exhaustion branch, so confirm the ladder
    # ran the full six attempts rather than raising early.
    assert runner.aragog_solver.solve.call_count == 6


# --- Failure-mode-branched retry ladder ------------------------------------
#
# _solve_with_retry branches its recovery on the CVODE failure mode. A
# CV_TOO_MUCH_WORK stall (cvode_flag == -1) raises the step budget, then relaxes
# rtol (both bounded by caps), and only then halves dt, over eight attempts.
# Every other failure keeps the six-attempt dt-halving ladder. The helpers below
# script the solver so each solve() records the controls it ran under, which lets
# a test assert the exact per-attempt schedule and the finally-block restore.
#
# A MagicMock coerces int()/float()/str() of an auto-child to 1/1.0/a repr, so
# the mode-branch fields (cvode_flag, cvode_flag_name, tcore_change_max) and
# _max_steps must be set on each mock explicitly; attribute absence (an older
# aragog, or a solver without a cached step budget) is forced with del.

_RETRY_BASE_RTOL = 1.0e-6
_RETRY_BASE_MAX_STEPS = 1000
_RETRY_DT0 = 100.0


def _retry_out(
    status,
    *,
    cvode_flag=0,
    cvode_flag_name='',
    T_core=2000.0,
    has_tcore_change=True,
    tcore_change_max=0.0,
    e_res=None,
    e_state_cons=None,
):
    """Build one scripted SolverOutput stand-in with explicit mode fields.

    Pass e_res and e_state_cons to drive the conservation tripwire; both set
    the numeric residual fields explicitly, since an unset MagicMock child
    coerces to 1.0 and would not represent a real value.
    """
    out = MagicMock()
    out.status = status
    out.T_core = T_core
    out.cvode_flag = cvode_flag
    out.cvode_flag_name = cvode_flag_name
    if has_tcore_change:
        out.tcore_change_max = tcore_change_max
    else:
        # Older aragog returns no such attribute: force the getattr fallback.
        del out.tcore_change_max
    if e_res is not None:
        out.step_solver_residual_J = e_res
    if e_state_cons is not None:
        out.E_state_cons = e_state_cons
    return out


def _retry_solver(scripted_outs, *, with_max_steps=True):
    """Build a scripted solver mock that records the controls of each solve()."""
    calls = []
    solver = MagicMock()
    solver.parameters.solver.start_time = 0.0
    solver.parameters.solver.end_time = _RETRY_DT0
    solver.parameters.solver.rtol = _RETRY_BASE_RTOL
    solver.parameters.solver.max_steps = _RETRY_BASE_MAX_STEPS
    if with_max_steps:
        solver._max_steps = _RETRY_BASE_MAX_STEPS
    else:
        del solver._max_steps
    solver._atol_sf = 1.0
    solver.get_current_dSdr_cmb.return_value = None
    solver._dSdr_cmb_init = None

    def _record():
        s = solver.parameters.solver
        calls.append(
            {
                'dt': s.end_time - s.start_time,
                'rtol': s.rtol,
                'max_steps': getattr(solver, '_max_steps', None),
                'atol_sf': solver._atol_sf,
            }
        )

    solver.solve.side_effect = _record
    solver.get_state.side_effect = list(scripted_outs)
    return solver, calls


def _retry_runner(solver, monkeypatch, *, T_core_pre=2000.0, mass_tot=1.0):
    """Bind the scripted solver to an AragogRunner and return (runner, hf_row)."""
    from proteus.interior_energetics.aragog import AragogRunner

    monkeypatch.setattr('aragog.solver.entropy_solver._CVODE_AVAILABLE', True)
    runner = AragogRunner.__new__(AragogRunner)
    runner._config = MagicMock()
    runner._config.interior_energetics.aragog.solver_method = 'cvode'
    runner._config.planet.mass_tot = mass_tot
    runner.aragog_solver = solver
    interior_o = MagicMock()
    interior_o._last_entropy = None
    hf_row = {'Time': 1.0e6, 'T_cmb': T_core_pre}
    return runner, interior_o, hf_row


@pytest.mark.unit
def test_solve_with_retry_stiff_mode_ramps_maxsteps_and_rtol_then_halves_dt(monkeypatch):
    """A CV_TOO_MUCH_WORK stall relaxes step budget and rtol before halving dt.

    A stiffness stall is a step-budget problem, so halving dt alone wastes the
    ladder. The stiff branch instead runs eight attempts: attempts 2-4 hold dt
    and ramp max_steps to 2x/4x/8x(cap) and rtol to 2x/5x/10x(cap), attempts 5-8
    hold both caps and halve dt (0.5/0.25/0.125/0.0625). This pins the whole
    schedule, so a mutant that drops the rtol/max_steps ramp, off-by-ones the
    next_attempt index, freezes the range at six, or falls through the loop
    returning a failed SolverOutput instead of raising fails here. It also
    confirms the finally block restores the base rtol and step budget on the
    raising exit path, so a later coupling step never inherits a relaxed rtol.
    """
    stiff = [_retry_out(-1, cvode_flag=-1, cvode_flag_name='TOO_MUCH_WORK')] * 10
    solver, calls = _retry_solver(stiff)
    runner, interior_o, hf_row = _retry_runner(solver, monkeypatch)

    with pytest.raises(RuntimeError, match='TOO_MUCH_WORK') as info:
        runner._solve_with_retry(hf_row, interior_o)

    assert solver.solve.call_count == 8
    assert 'cvode_flag=-1' in str(info.value)

    expected = [
        (100.0, 1000, 1.0e-6),
        (100.0, 2000, 2.0e-6),
        (100.0, 4000, 5.0e-6),
        (100.0, 8000, 1.0e-5),
        (50.0, 8000, 1.0e-5),
        (25.0, 8000, 1.0e-5),
        (12.5, 8000, 1.0e-5),
        (6.25, 8000, 1.0e-5),
    ]
    assert len(calls) == 8
    for k, (dt, ms, rtol) in enumerate(expected):
        np.testing.assert_allclose(calls[k]['dt'], dt, err_msg=f'attempt {k + 1} dt')
        assert calls[k]['max_steps'] == ms, f'attempt {k + 1} max_steps'
        np.testing.assert_allclose(calls[k]['rtol'], rtol, err_msg=f'attempt {k + 1} rtol')

    # finally restores the base controls on the raising path.
    np.testing.assert_allclose(solver.parameters.solver.rtol, _RETRY_BASE_RTOL)
    assert solver._max_steps == _RETRY_BASE_MAX_STEPS


@pytest.mark.unit
def test_solve_with_retry_other_mode_caps_at_six_and_only_halves_dt(monkeypatch):
    """A non-stiff CVODE failure keeps the six-attempt dt-halving ladder.

    Any failure that is not CV_TOO_MUCH_WORK (cvode_flag != -1) must not widen
    the ladder to eight or touch rtol/max_steps: those levers belong to the
    stiff branch only. This confirms the exhaustion check reads the active
    six-attempt budget for the other mode, the dt halves each attempt
    (100/50/25/12.5/6.25/3.125), rtol and the step budget stay at their base
    values throughout, and the exhaustion message names the solver status
    without a stiff-mode flag suffix. A mutant that shares the stiff schedule
    across modes or miscounts the other-mode budget fails here.
    """
    other = [_retry_out(-1, cvode_flag=0, cvode_flag_name='')] * 8
    solver, calls = _retry_solver(other)
    runner, interior_o, hf_row = _retry_runner(solver, monkeypatch)

    with pytest.raises(RuntimeError, match='status=-1') as info:
        runner._solve_with_retry(hf_row, interior_o)

    assert solver.solve.call_count == 6
    assert 'TOO_MUCH_WORK' not in str(info.value)

    expected_dt = [100.0, 50.0, 25.0, 12.5, 6.25, 3.125]
    assert len(calls) == 6
    for k, dt in enumerate(expected_dt):
        np.testing.assert_allclose(calls[k]['dt'], dt, err_msg=f'attempt {k + 1} dt')
        assert calls[k]['max_steps'] == _RETRY_BASE_MAX_STEPS, f'attempt {k + 1} max_steps'
        np.testing.assert_allclose(
            calls[k]['rtol'], _RETRY_BASE_RTOL, err_msg=f'attempt {k + 1} rtol'
        )


@pytest.mark.unit
def test_solve_with_retry_stiff_success_returns_state_and_restores_controls(monkeypatch):
    """A stiff retry that succeeds returns the state and restores base controls.

    After four CV_TOO_MUCH_WORK failures the fifth attempt succeeds with a small
    CMB-temperature change, so the method must return that SolverOutput rather
    than raise, and the finally block must still restore the base rtol and step
    budget on the returning path. A mutant that restores only on the exception
    path, or continues past a success, fails here.
    """
    scripted = [_retry_out(-1, cvode_flag=-1, cvode_flag_name='TOO_MUCH_WORK')] * 4 + [
        _retry_out(0, T_core=2010.0, tcore_change_max=10.0)
    ]
    solver, calls = _retry_solver(scripted)
    runner, interior_o, hf_row = _retry_runner(solver, monkeypatch)

    out = runner._solve_with_retry(hf_row, interior_o)

    assert out is not None
    assert out.status == 0
    assert solver.solve.call_count == 5
    np.testing.assert_allclose(solver.parameters.solver.rtol, _RETRY_BASE_RTOL)
    assert solver._max_steps == _RETRY_BASE_MAX_STEPS


@pytest.mark.unit
def test_solve_with_retry_tcore_guard_trips_and_routes_to_dt_halving(monkeypatch):
    """A status=0 solve with an implausible T_core change is rejected as failure.

    tcore_change_max above the mass-scaled sanity threshold (3000 K at 1
    M_Earth) means the solver 'succeeded' with garbage, so the guard rejects it
    and continues the ladder. A guard trip carries cvode_flag == 0, so it must
    route to the six-attempt dt-halving branch, not the stiff branch, and the
    exhaustion message must name the T_core-jump reason rather than the
    misleading status=0. A mutant that treats a guard trip as stiff (eight
    attempts) or reports a bare status=0 fails here.
    """
    guard = [_retry_out(0, cvode_flag=0, T_core=9000.0, tcore_change_max=9000.0)] * 8
    solver, calls = _retry_solver(guard)
    runner, interior_o, hf_row = _retry_runner(solver, monkeypatch, T_core_pre=2000.0)

    with pytest.raises(RuntimeError, match='sanity threshold') as info:
        runner._solve_with_retry(hf_row, interior_o)

    assert solver.solve.call_count == 6
    assert 'status=0' in str(info.value)


@pytest.mark.unit
def test_solve_with_retry_tcore_guard_inactive_on_first_solve(monkeypatch):
    """The T_core-jump guard is inactive on the first solve (no prior T_core).

    On the very first coupling solve T_core_pre is 0, so no converged core
    temperature exists to compare against and the guard must be forced off even
    for a large tcore_change_max. The method must accept and return the solve on
    attempt 1. A mutant that keeps the guard active with T_core_pre == 0 would
    reject a legitimate first solve and fails here.
    """
    first = [_retry_out(0, T_core=9000.0, tcore_change_max=9000.0)] * 3
    solver, calls = _retry_solver(first)
    runner, interior_o, hf_row = _retry_runner(solver, monkeypatch, T_core_pre=0.0)

    out = runner._solve_with_retry(hf_row, interior_o)

    assert out is not None
    assert solver.solve.call_count == 1


@pytest.mark.unit
def test_solve_with_retry_tcore_guard_falls_back_to_endpoint_change(monkeypatch):
    """Without tcore_change_max the guard uses the endpoint change instead.

    An older aragog returns no tcore_change_max attribute, so the guard must
    fall back to the endpoint change abs(T_core - T_core_pre). A 7000 K endpoint
    jump (9000 from 2000) trips the 3000 K threshold and exhausts the ladder; a
    100 K jump (2100 from 2000) passes and returns on attempt 1. A mutant that
    drops the fallback (crashing on the missing attribute, or never checking)
    fails one of the two directions here.
    """
    tripping = [_retry_out(0, cvode_flag=0, T_core=9000.0, has_tcore_change=False)] * 8
    solver, calls = _retry_solver(tripping)
    runner, interior_o, hf_row = _retry_runner(solver, monkeypatch, T_core_pre=2000.0)
    with pytest.raises(RuntimeError, match='sanity threshold'):
        runner._solve_with_retry(hf_row, interior_o)
    assert solver.solve.call_count == 6

    passing = [_retry_out(0, T_core=2100.0, has_tcore_change=False)]
    solver2, _ = _retry_solver(passing)
    runner2, interior2, hf_row2 = _retry_runner(solver2, monkeypatch, T_core_pre=2000.0)
    out = runner2._solve_with_retry(hf_row2, interior2)
    assert out is not None
    assert solver2.solve.call_count == 1


@pytest.mark.unit
def test_solve_with_retry_restores_rtol_when_solver_has_no_cached_step_budget(monkeypatch):
    """A solver without a cached _max_steps still relaxes rtol and restores it.

    When the solver exposes no _max_steps attribute the base step budget comes
    from parameters.solver.max_steps and the stiff branch skips every _max_steps
    write, so no such attribute is ever created. The ladder must still run its
    eight stiff attempts, and the finally block must restore the base rtol
    without touching the absent step budget. A mutant that assumes _max_steps
    exists (crashing on capture or restore) fails here.
    """
    stiff = [_retry_out(-1, cvode_flag=-1, cvode_flag_name='TOO_MUCH_WORK')] * 10
    solver, calls = _retry_solver(stiff, with_max_steps=False)
    runner, interior_o, hf_row = _retry_runner(solver, monkeypatch)

    with pytest.raises(RuntimeError, match='TOO_MUCH_WORK'):
        runner._solve_with_retry(hf_row, interior_o)

    assert solver.solve.call_count == 8
    assert not hasattr(solver, '_max_steps')
    np.testing.assert_allclose(solver.parameters.solver.rtol, _RETRY_BASE_RTOL)
    assert all(c['max_steps'] is None for c in calls)


@pytest.mark.unit
def test_solve_with_retry_rejects_non_finite_tcore_change(monkeypatch):
    """A status=0 solve with a non-finite T_core change fails the sanity guard.

    A relaxed rtol can let CVODE return status=0 with a NaN core temperature or
    a NaN tcore_change_max. A bare 'dT > threshold' test passes such a solve,
    because 'NaN > 3000' is False, so the run would accept a corrupt state. The
    guard must reject a non-finite dT instead. This pins both the intra-solve
    path (tcore_change_max = NaN) and the endpoint fallback (T_core = NaN, no
    tcore_change_max); a mutant that drops the np.isfinite clause returns the
    NaN solve and fails here.
    """
    nan = float('nan')
    intra = [_retry_out(0, cvode_flag=0, T_core=2000.0, tcore_change_max=nan)] * 8
    solver, _ = _retry_solver(intra)
    runner, interior_o, hf_row = _retry_runner(solver, monkeypatch, T_core_pre=2000.0)
    with pytest.raises(RuntimeError, match='sanity threshold'):
        runner._solve_with_retry(hf_row, interior_o)
    assert solver.solve.call_count == 6


@pytest.mark.unit
def test_solve_with_retry_conservation_tripwire(monkeypatch):
    """A status=0 solve that breaks energy conservation grossly is rejected.

    The coarse tripwire rejects a solve whose residual |E_res/dE_cons| exceeds
    K=1e-9, where dE_cons is the enthalpy the step moved (end E_state_cons minus
    the pre-solve value carried in the helpfile row). This pins three cases: a
    gross ratio (1e-5) is rejected and, carrying cvode_flag == 0, routes to the
    six-attempt dt-halving branch with a conservation reason; a healthy ~2e-18
    ratio passes and returns on attempt 1; a NaN ratio is rejected, matching the
    T_core guard inversion 'not (ratio <= K)'. A mutant that drops the tripwire,
    or uses 'ratio > K' and so passes a NaN, fails here.
    """
    e_cons_pre = 1.0e30
    e_cons_end = 6.0e29  # dE_cons = -4.0e29 J, the enthalpy the step moved

    gross = [
        _retry_out(0, cvode_flag=0, e_res=1.0e25, e_state_cons=e_cons_end)
    ] * 8  # |E_res/dE_cons| = 2.5e-5 >> K
    solver, _ = _retry_solver(gross)
    runner, interior_o, hf_row = _retry_runner(solver, monkeypatch, T_core_pre=2000.0)
    hf_row['E_state_cons_J'] = e_cons_pre
    with pytest.raises(RuntimeError, match='conservation residual') as info:
        runner._solve_with_retry(hf_row, interior_o)
    assert solver.solve.call_count == 6
    assert 'status=0' in str(info.value)

    healthy = [
        _retry_out(0, e_res=-8.0e11, e_state_cons=e_cons_end)
    ]  # |E_res/dE_cons| = 2.0e-18 << K
    solver2, _ = _retry_solver(healthy)
    runner2, interior2, hf_row2 = _retry_runner(solver2, monkeypatch, T_core_pre=2000.0)
    hf_row2['E_state_cons_J'] = e_cons_pre
    out = runner2._solve_with_retry(hf_row2, interior2)
    assert out is not None
    assert solver2.solve.call_count == 1

    nan = float('nan')
    corrupt = [_retry_out(0, cvode_flag=0, e_res=nan, e_state_cons=e_cons_end)] * 8
    solver3, _ = _retry_solver(corrupt)
    runner3, interior3, hf_row3 = _retry_runner(solver3, monkeypatch, T_core_pre=2000.0)
    hf_row3['E_state_cons_J'] = e_cons_pre
    with pytest.raises(RuntimeError, match='conservation residual'):
        runner3._solve_with_retry(hf_row3, interior3)
    assert solver3.solve.call_count == 6

    endpoint = [_retry_out(0, cvode_flag=0, T_core=nan, has_tcore_change=False)] * 8
    solver2, _ = _retry_solver(endpoint)
    runner2, interior2, hf_row2 = _retry_runner(solver2, monkeypatch, T_core_pre=2000.0)
    with pytest.raises(RuntimeError, match='sanity threshold'):
        runner2._solve_with_retry(hf_row2, interior2)
    assert solver2.solve.call_count == 6


@pytest.mark.unit
def test_solve_with_retry_dt_never_increases_on_late_stiff_switch(monkeypatch):
    """A late switch to the stiff branch never retries at a coarser dt.

    When the first failures are non-stiff and a CV_TOO_MUCH_WORK stall appears
    only after dt has already shrunk, the stiff schedule (anchored to the
    absolute attempt index) would otherwise set a larger dt than the attempt
    that just failed. A retry must never run coarser than the step it retries,
    so the stiff dt is clamped to the failed dt. This scripts five non-stiff
    failures then a stall and asserts the dt sequence never increases; a mutant
    that drops the clamp jumps dt up at the switch and fails here.
    """
    scripted = [_retry_out(-1, cvode_flag=0, cvode_flag_name='')] * 5 + [
        _retry_out(-1, cvode_flag=-1, cvode_flag_name='TOO_MUCH_WORK')
    ] * 5
    solver, calls = _retry_solver(scripted)
    runner, interior_o, hf_row = _retry_runner(solver, monkeypatch)
    with pytest.raises(RuntimeError, match='TOO_MUCH_WORK'):
        runner._solve_with_retry(hf_row, interior_o)

    assert solver.solve.call_count == 8
    dts = [c['dt'] for c in calls]
    assert all(dts[k + 1] <= dts[k] + 1e-12 for k in range(len(dts) - 1)), (
        f'dt not monotone non-increasing: {dts}'
    )
    # the switch is at attempt 6 -> 7: the stiff retry holds dt, never raises it.
    assert calls[6]['dt'] <= calls[5]['dt']


@pytest.mark.unit
def test_solve_with_retry_restores_base_tolerance_on_reverse_switch(monkeypatch):
    """A switch from the stiff branch back to a non-stiff failure runs at base.

    If a CV_TOO_MUCH_WORK stall relaxes rtol and the step budget and then a
    later attempt fails for a different reason, the non-stiff branch must run at
    base tolerance, not inherit the stiff relaxation; its own purpose is more
    resolution, not looser tolerance. This scripts three stalls then non-stiff
    failures. Attempt 4 still runs the relaxation set by attempt 3's stall,
    because the switch is only seen when attempt 4 fails, so attempts 5-8 must
    be back at base rtol and base max_steps. A mutant that omits the base
    restore on the non-stiff branch leaves those attempts relaxed and fails here.
    """
    scripted = [_retry_out(-1, cvode_flag=-1, cvode_flag_name='TOO_MUCH_WORK')] * 3 + [
        _retry_out(-1, cvode_flag=0, cvode_flag_name='')
    ] * 6
    solver, calls = _retry_solver(scripted)
    runner, interior_o, hf_row = _retry_runner(solver, monkeypatch)
    with pytest.raises(RuntimeError):
        runner._solve_with_retry(hf_row, interior_o)

    assert solver.solve.call_count == 8
    for k in range(4, 8):
        np.testing.assert_allclose(
            calls[k]['rtol'], _RETRY_BASE_RTOL, err_msg=f'attempt {k + 1} rtol'
        )
        assert calls[k]['max_steps'] == _RETRY_BASE_MAX_STEPS, f'attempt {k + 1} max_steps'
