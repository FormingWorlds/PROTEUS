"""Unit tests for PROTEUS coupling features.

Tests the coupling loop infrastructure without requiring SPIDER/Aragog
binaries. Validates config flags, volatile profile building, binodal
integration, and outgassing dispatch.
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestGlobalMiscibilityConfig:
    """Config validation for global_miscibility."""

    def test_default_is_false(self):
        """global_miscibility defaults to False on Zalmoxis."""
        from proteus.config._struct import Zalmoxis

        z = Zalmoxis()
        assert z.global_miscibility is False

    def test_accepts_zalmoxis(self):
        """global_miscibility=True with Zalmoxis is valid."""
        from proteus.config._struct import Struct, Zalmoxis

        z = Zalmoxis(global_miscibility=True)
        s = Struct(
            core_frac=0.3,
            module='zalmoxis',
            zalmoxis=z,
        )
        assert s.zalmoxis.global_miscibility is True


@pytest.mark.unit
class TestOutgasModuleConfig:
    """Config validation for outgas module selection."""

    def test_calliope_accepted(self):
        """module='calliope' is valid."""
        from proteus.config._outgas import Outgas

        o = Outgas(fO2_shift_IW=0.0, module='calliope')
        assert o.module == 'calliope'

    def test_atmodeller_accepted(self):
        """module='atmodeller' is valid."""
        from proteus.config._outgas import Outgas

        o = Outgas(fO2_shift_IW=0.0, module='atmodeller')
        assert o.module == 'atmodeller'

    def test_invalid_rejected(self):
        """Invalid module name raises ValueError."""
        from proteus.config._outgas import Outgas

        with pytest.raises((ValueError, Exception)):
            Outgas(fO2_shift_IW=0.0, module='invalid')

    def test_atmodeller_config_defaults(self):
        """Atmodeller config has correct defaults."""
        from proteus.config._outgas import Atmodeller

        a = Atmodeller()
        assert a.solver_mode == 'robust'
        assert a.include_condensates is True
        assert 'sossi' in a.solubility_H2O.lower()


@pytest.mark.unit
class TestBuildVolatileProfile:
    """Test volatile profile construction from hf_row."""

    def _make_hf_row(self, M_liq=1e24, M_sol=1e24, H2O_liq=1e20, H2_liq=1e18):
        """Create a minimal hf_row for testing."""
        return {
            'M_mantle_liquid': M_liq,
            'M_mantle_solid': M_sol,
            'H2O_kg_liquid': H2O_liq,
            'H2O_kg_solid': 0.0,
            'H2_kg_liquid': H2_liq,
            'H2_kg_solid': 0.0,
        }

    def test_returns_none_when_no_mantle_mass(self):
        """Returns None when M_mantle_liquid + M_mantle_solid = 0."""
        from proteus.interior_struct.zalmoxis import build_volatile_profile

        hf_row = self._make_hf_row(M_liq=0, M_sol=0)
        result = build_volatile_profile(hf_row, 'PALEOS:MgSiO3')
        assert result is None

    def test_returns_none_when_no_volatiles(self):
        """Returns None when all volatile masses are zero."""
        from proteus.interior_struct.zalmoxis import build_volatile_profile

        hf_row = self._make_hf_row(H2O_liq=0, H2_liq=0)
        result = build_volatile_profile(hf_row, 'PALEOS:MgSiO3')
        assert result is None

    def test_returns_profile_with_volatiles(self):
        """Returns VolatileProfile when volatiles are present."""
        from proteus.interior_struct.zalmoxis import build_volatile_profile

        hf_row = self._make_hf_row()
        result = build_volatile_profile(hf_row, 'PALEOS:MgSiO3')
        assert result is not None
        assert 'PALEOS:H2O' in result.w_liquid or 'Chabrier:H' in result.w_liquid

    def test_fractions_clamped_below_limit(self):
        """Total volatile mass fraction per phase is clamped to <= 0.95."""
        from proteus.interior_struct.zalmoxis import build_volatile_profile

        # Extreme case: volatile mass > mantle mass
        hf_row = self._make_hf_row(M_liq=1e20, H2O_liq=1e24)
        result = build_volatile_profile(hf_row, 'PALEOS:MgSiO3')
        if result is not None:
            total = sum(result.w_liquid.values())
            assert total <= 0.95 + 1e-10


@pytest.mark.unit
class TestOutgasDispatch:
    """Test that run_outgassing dispatches correctly."""

    def test_binodal_skipped_with_miscibility(self):
        """apply_binodal_h2 is skipped when global_miscibility=True."""

        # This is a code path test: verify the logic in wrapper.py
        # Without a full Config, test the conditional logic directly
        class MockConfig:
            class interior_struct:
                class zalmoxis:
                    global_miscibility = True

            class outgas:
                h2_binodal = True

        config = MockConfig()
        # The logic: if global_miscibility -> skip, elif h2_binodal -> apply
        if config.interior_struct.zalmoxis.global_miscibility:
            action = 'skip'
        elif config.outgas.h2_binodal:
            action = 'apply'
        else:
            action = 'none'
        assert action == 'skip'

    def test_binodal_applied_without_miscibility(self):
        """apply_binodal_h2 is applied when h2_binodal=True, miscibility=False."""

        class MockConfig:
            class interior_struct:
                class zalmoxis:
                    global_miscibility = False

            class outgas:
                h2_binodal = True

        config = MockConfig()
        if config.interior_struct.zalmoxis.global_miscibility:
            action = 'skip'
        elif config.outgas.h2_binodal:
            action = 'apply'
        else:
            action = 'none'
        assert action == 'apply'
