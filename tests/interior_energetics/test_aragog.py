"""
Unit tests for proteus.interior_energetics.aragog module: Zalmoxis integration paths.

Tests the Zalmoxis-specific branches in AragogRunner.setup_solver() that set
inner_radius from zalmoxis_solver and configure temperature-dependent initial
conditions.

Testing standards and documentation:
- docs/test_infrastructure.md: Test infrastructure overview
- docs/test_categorization.md: Test marker definitions
- docs/test_building.md: Best practices for test construction

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
def test_effective_phi_step_cap_auto_enables_for_zalmoxis():
    """The melt-fraction cap defaults ON for the zalmoxis interior stack.

    A zalmoxis run that leaves phi_step_cap at the disabled schema default
    (0.0) must be promoted to the non-zero coupled-stack default so the
    crystallisation-onset core-temperature discontinuity is guarded without
    the user having to opt in. A non-zalmoxis interior is left untouched, and
    an explicit positive value always wins. The discrimination guard pins all
    three branches so a regression that drops the auto-enable, fires it for
    the wrong module, or overrides the user value is caught.
    """
    from proteus.interior_energetics.aragog import (
        _ZALMOXIS_DEFAULT_PHI_STEP_CAP,
        _effective_phi_step_cap,
    )

    def cfg(module, cap):
        c = MagicMock()
        c.interior_struct.module = module
        c.interior_energetics.aragog.phi_step_cap = cap
        return c

    # zalmoxis + disabled default -> promoted to the non-zero default
    promoted = _effective_phi_step_cap(cfg('zalmoxis', 0.0))
    assert promoted == pytest.approx(_ZALMOXIS_DEFAULT_PHI_STEP_CAP)
    assert promoted > 0.0
    # non-zalmoxis interior keeps the disabled value (no auto-enable)
    assert _effective_phi_step_cap(cfg('spider', 0.0)) == 0.0
    assert _effective_phi_step_cap(cfg('dummy', 0.0)) == 0.0
    # explicit user value wins on every interior, even zalmoxis
    assert _effective_phi_step_cap(cfg('zalmoxis', 0.05)) == pytest.approx(0.05)
    assert _effective_phi_step_cap(cfg('spider', 0.2)) == pytest.approx(0.2)
    # the auto-enabled default must differ from the disabled value, else the
    # promotion would be a no-op
    assert _ZALMOXIS_DEFAULT_PHI_STEP_CAP != 0.0


@pytest.mark.unit
def test_effective_temperature_and_entropy_step_caps_auto_enable_for_zalmoxis():
    """The temperature and entropy step caps also default ON for zalmoxis.

    The melt-fraction cap cannot bound the core-temperature drop once a cell
    is fully solid, so the temperature and entropy caps must be auto-enabled
    alongside it for the zalmoxis stack. Same promotion contract as the
    melt-fraction cap: disabled schema default promoted for zalmoxis, left
    alone for other interiors, explicit positive value wins. Discrimination
    guards pin each branch and assert the auto-enabled defaults are non-zero.
    """
    from proteus.interior_energetics.aragog import (
        _ZALMOXIS_DEFAULT_ENTROPY_STEP_CAP,
        _ZALMOXIS_DEFAULT_TEMPERATURE_STEP_CAP,
        _effective_entropy_step_cap,
        _effective_temperature_step_cap,
    )

    def cfg(module, t_cap, s_cap):
        c = MagicMock()
        c.interior_struct.module = module
        c.interior_energetics.aragog.temperature_step_cap = t_cap
        c.interior_energetics.aragog.entropy_step_cap = s_cap
        return c

    # zalmoxis + disabled defaults -> promoted to the non-zero defaults
    assert _effective_temperature_step_cap(cfg('zalmoxis', 0.0, 0.0)) == pytest.approx(
        _ZALMOXIS_DEFAULT_TEMPERATURE_STEP_CAP
    )
    assert _effective_entropy_step_cap(cfg('zalmoxis', 0.0, 0.0)) == pytest.approx(
        _ZALMOXIS_DEFAULT_ENTROPY_STEP_CAP
    )
    # non-zalmoxis keeps disabled
    assert _effective_temperature_step_cap(cfg('spider', 0.0, 0.0)) == 0.0
    assert _effective_entropy_step_cap(cfg('dummy', 0.0, 0.0)) == 0.0
    # explicit values win, even on zalmoxis
    assert _effective_temperature_step_cap(cfg('zalmoxis', 250.0, 0.0)) == pytest.approx(250.0)
    assert _effective_entropy_step_cap(cfg('zalmoxis', 0.0, 75.0)) == pytest.approx(75.0)
    # auto-enabled defaults must aggressively suppress jumps (non-zero, finite)
    assert _ZALMOXIS_DEFAULT_TEMPERATURE_STEP_CAP > 0.0
    assert _ZALMOXIS_DEFAULT_ENTROPY_STEP_CAP > 0.0


@pytest.mark.unit
def test_negative_step_cap_disables_even_on_zalmoxis():
    """The -1.0 off sentinel beats the zalmoxis auto-enable and never leaks through.

    The zalmoxis auto-enable promotes the 0.0 schema default so a user who never
    touches the field is protected. A user who sets the -1.0 sentinel is opting
    out, and the resolver must honour that by returning 0.0 (no cap) even on
    zalmoxis, where the plain 0.0 default would instead be promoted. The sentinel
    must never reach Aragog as a literal negative cap, so every branch resolves
    to exactly 0.0. Discrimination guards contrast the disabled result against
    the auto-enabled default and against a positive override so a regression that
    lets the sentinel promote, leak through, or clamp to the default is caught.
    """
    from proteus.interior_energetics.aragog import (
        _ZALMOXIS_DEFAULT_ENTROPY_STEP_CAP,
        _ZALMOXIS_DEFAULT_PHI_STEP_CAP,
        _ZALMOXIS_DEFAULT_TEMPERATURE_STEP_CAP,
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

    # -1.0 sentinel on zalmoxis -> disabled (0.0), overriding the auto-enable
    off = cfg('zalmoxis', -1.0, -1.0, -1.0)
    assert _effective_phi_step_cap(off) == 0.0
    assert _effective_temperature_step_cap(off) == 0.0
    assert _effective_entropy_step_cap(off) == 0.0
    # the sentinel on a non-zalmoxis interior is also disabled, never negative
    off_spider = cfg('spider', -1.0, -1.0, -1.0)
    assert _effective_phi_step_cap(off_spider) == 0.0
    assert _effective_temperature_step_cap(off_spider) == 0.0
    assert _effective_entropy_step_cap(off_spider) == 0.0
    # discrimination: the disabled result differs from the value the same 0.0
    # default would have promoted to, so the off switch is not a silent no-op
    assert _ZALMOXIS_DEFAULT_PHI_STEP_CAP > 0.0
    assert _ZALMOXIS_DEFAULT_TEMPERATURE_STEP_CAP > 0.0
    assert _ZALMOXIS_DEFAULT_ENTROPY_STEP_CAP > 0.0
    # a positive override still wins over the auto-enable default
    on = cfg('zalmoxis', 0.05, 250.0, 75.0)
    assert _effective_phi_step_cap(on) == pytest.approx(0.05)
    assert _effective_temperature_step_cap(on) == pytest.approx(250.0)
    assert _effective_entropy_step_cap(on) == pytest.approx(75.0)


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
    # zero remains valid (the auto-enable default) and positive is verbatim.
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
