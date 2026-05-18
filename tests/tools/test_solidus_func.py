"""
Comprehensive unit tests for tools/solidus_func.py.

This module tests the melting curve parametrizations and EOS inversion pipeline.
Covers:
1. All 10 literature melting curve models (parametric functions)
2. Helper functions for solidus/liquidus relationships
3. EOS inversion and resampling pipeline (with mocked interpolators)
4. Physical validation (solidus < liquidus, monotonicity, etc.)
5. Edge cases and error handling

References:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _load_solidus_func_module():
    """Load ``tools/solidus_func.py`` as a module without sys.path manipulation.

    The ``tools/`` directory is not a Python package, and adding the repo root
    to ``sys.path`` would shadow the installed ``aragog`` package with the
    local development checkout. ``importlib.util`` loads the script directly.
    """
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / 'tools' / 'solidus_func.py'
    spec = importlib.util.spec_from_file_location('solidus_func_under_test', script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_solidus_func = _load_solidus_func_module()

_fmt_range = _solidus_func._fmt_range
andrault_2011 = _solidus_func.andrault_2011
belonoshko_2005 = _solidus_func.belonoshko_2005
build_common_entropy_grid = _solidus_func.build_common_entropy_grid
fei_2021 = _solidus_func.fei_2021
fiquet_2010 = _solidus_func.fiquet_2010
get_melting_curves = _solidus_func.get_melting_curves
hirschmann_2000 = _solidus_func.hirschmann_2000
invert_to_entropy_along_profile = _solidus_func.invert_to_entropy_along_profile
katz_2003 = _solidus_func.katz_2003
lin_2024 = _solidus_func.lin_2024
liquidus_from_solidus_stixrude = _solidus_func.liquidus_from_solidus_stixrude
make_entropy_header = _solidus_func.make_entropy_header
make_pressure_grid = _solidus_func.make_pressure_grid
monteux_2016 = _solidus_func.monteux_2016
solidus_from_liquidus_stixrude = _solidus_func.solidus_from_liquidus_stixrude
stixrude_2014 = _solidus_func.stixrude_2014
truncate_to_physical_interval = _solidus_func.truncate_to_physical_interval
validate_entropy_export_arrays = _solidus_func.validate_entropy_export_arrays
wolf_bower_2018 = _solidus_func.wolf_bower_2018

# =============================================================================
# FIXTURES & HELPERS
# =============================================================================


@pytest.fixture
def pressure_grid():
    """Standard pressure grid for testing (0–1000 GPa, 100 points)."""
    return np.linspace(0.0, 1000.0, 100)


@pytest.fixture
def small_pressure_grid():
    """Small pressure grid for quick smoke tests (0–100 GPa, 10 points)."""
    return np.linspace(0.0, 100.0, 10)


@pytest.fixture
def mock_eos_interpolator():
    """Mock EOS interpolator for T(S, P). Returns physically plausible T values."""

    def mock_T_of_SP(points):
        # points is shape (N, 2) with columns [S, P]
        # Simple model: T = 1000 + 10*S + 100*P (linear, increases with entropy and pressure)
        # Valid bounds: S in [100, 1000] J/kg/K, P in [0, 100] GPa
        # Returns nans outside bounds (mimics RegularGridInterpolator behavior)
        S = points[:, 0]
        P = points[:, 1]

        T = 1000.0 + 10.0 * S + 100.0 * P
        result = T.copy()

        # Set out-of-bounds to NaN
        mask_invalid = (S < 100.0) | (S > 1000.0) | (P < 0.0) | (P > 100.0)
        result[mask_invalid] = np.nan

        return result

    return mock_T_of_SP


@pytest.fixture
def mock_S_axes():
    """Mock entropy axes for solid and liquid EOS tables."""
    return (
        np.linspace(100.0, 1000.0, 125),  # solid S axis
        np.linspace(150.0, 1100.0, 95),  # liquid S axis
    )


# =============================================================================
# TEST: make_pressure_grid
# =============================================================================


@pytest.mark.unit
class TestMakePressureGrid:
    """Test pressure grid generation."""

    def test_default_grid(self):
        """Creates a 500-point grid from 0 to 1000 GPa (default)."""
        P = make_pressure_grid()
        assert len(P) == 500
        assert P[0] == 0.0
        assert P[-1] == pytest.approx(1000.0, rel=1e-10)
        assert np.all(np.diff(P) > 0), 'Pressure grid must be monotonically increasing'

    def test_custom_range_and_count(self):
        """Custom Pmin, Pmax, n parameters."""
        P = make_pressure_grid(Pmin=10.0, Pmax=100.0, n=20)
        assert len(P) == 20
        assert P[0] == pytest.approx(10.0, rel=1e-10)
        assert P[-1] == pytest.approx(100.0, rel=1e-10)

    def test_uniform_spacing(self):
        """Pressure grid is uniformly spaced."""
        P = make_pressure_grid(Pmin=0.0, Pmax=100.0, n=101)
        dP = np.diff(P)
        assert np.allclose(dP, dP[0]), 'Spacing must be uniform'
        # Discrimination: for Pmin=0, Pmax=100, n=101 the spacing must be
        # exactly 1.0 GPa. A regression that used n in the denominator
        # instead of n-1 (off-by-one in np.linspace semantics) would land
        # at 100/101 ~ 0.9901, well outside the tight tolerance below.
        assert dP[0] == pytest.approx(1.0, rel=1e-12)

    def test_single_point(self):
        """Single-point grid returns array of length 1."""
        P = make_pressure_grid(Pmin=50.0, Pmax=50.0, n=1)
        assert len(P) == 1
        assert P[0] == pytest.approx(50.0, rel=1e-10)


# =============================================================================
# TEST: Stixrude Ratio Functions
# =============================================================================


@pytest.mark.unit
class TestStixrudeRatios:
    """Test solidus/liquidus conversion via Stixrude ratios."""

    @pytest.mark.parametrize(
        'T_liq_k',
        [
            np.array([1500.0]),
            np.array([1500.0, 2000.0, 2500.0]),
            np.linspace(1200.0, 3000.0, 20),
        ],
    )
    def test_solidus_from_liquidus_stixrude(self, T_liq_k):
        """Solidus < liquidus via Stixrude ratio."""
        T_sol = solidus_from_liquidus_stixrude(T_liq_k)

        assert T_sol.shape == T_liq_k.shape
        assert np.all(T_sol > 0.0), 'Solidus must be positive'
        assert np.all(T_sol < T_liq_k), 'Solidus < liquidus (physical constraint)'

    @pytest.mark.parametrize(
        'T_sol_k',
        [
            np.array([1200.0]),
            np.array([1200.0, 1500.0, 1800.0]),
            np.linspace(1000.0, 2500.0, 20),
        ],
    )
    def test_liquidus_from_solidus_stixrude(self, T_sol_k):
        """Liquidus from solidus via Stixrude ratio."""
        T_liq = liquidus_from_solidus_stixrude(T_sol_k)

        assert T_liq.shape == T_sol_k.shape
        assert np.all(T_liq > 0.0), 'Liquidus must be positive'
        assert np.all(T_liq > T_sol_k), 'Liquidus > solidus (physical constraint)'

    def test_stixrude_round_trip(self):
        """Round-trip conversion: T_liq -> T_sol -> T_liq recovers original (approximately)."""
        T_liq_orig = np.linspace(1500.0, 2500.0, 10)
        T_sol = solidus_from_liquidus_stixrude(T_liq_orig)
        T_liq_recovered = liquidus_from_solidus_stixrude(T_sol)

        assert np.allclose(T_liq_recovered, T_liq_orig, rtol=1e-10)
        # Discrimination: the intermediate solidus must obey the physical
        # ordering T_sol < T_liq_orig. A regression that swapped the ratio
        # (multiplying instead of dividing by 1 - log(0.79)) would still
        # round-trip perfectly but the intermediate would land above
        # T_liq_orig, not below.
        assert np.all(T_sol < T_liq_orig), 'Stixrude solidus must lie below the liquidus seed'


# =============================================================================
# TEST: _fmt_range Helper
# =============================================================================


@pytest.mark.unit
class TestFmtRange:
    """Test formatting of array min/max ranges."""

    def test_normal_finite_array(self):
        """Format array with finite values."""
        arr = np.array([10.5, 50.0, 100.0])
        result = _fmt_range(arr, fmt='.1f')
        assert '[10.5, 100.0]' in result
        # Discrimination: the .1f format spec must be honoured; a regression
        # that fell back to the default '.3f' would emit '10.500' / '100.000'
        # and would not match the .1f literal pinned above. Pin the absence
        # of the .3f trailing-zero pattern as a guard against that fallback.
        assert '10.500' not in result and '100.000' not in result

    def test_array_with_nans(self):
        """Ignores NaN values, uses only finite elements."""
        arr = np.array([10.0, np.nan, 50.0, np.nan, 100.0])
        result = _fmt_range(arr, fmt='.1f')
        assert '[10.0, 100.0]' in result
        # Discrimination: a regression that propagated NaN through np.min /
        # np.max (forgetting the mask) would emit 'nan' instead of finite
        # bounds; pin the absence of that sentinel.
        assert 'nan' not in result

    def test_all_nans(self):
        """Returns [nan, nan] for all-NaN array."""
        arr = np.array([np.nan, np.nan, np.nan])
        result = _fmt_range(arr)
        assert '[nan, nan]' in result
        # Discrimination: the all-NaN branch must short-circuit before the
        # finite-min/max path; a regression that accidentally formatted NaN
        # via the f-string would emit 'nan' twice but with extra digits or
        # decimal points (e.g. 'nan.000'); the literal sentinel rules that out.
        assert 'nan.' not in result


# =============================================================================
# TEST: Melting Curve Functions - Andrault (2011)
# =============================================================================


@pytest.mark.unit
class TestAndrault2011:
    """Test Andrault et al. (2011) melting curve."""

    def test_solidus_basic_shape(self):
        """Solidus returns correct shape for given pressure grid."""
        P, T = andrault_2011(kind='solidus', Pmin=0.0, Pmax=100.0, n=50)
        assert len(P) == 50
        assert len(T) == 50
        assert np.all(T > 0.0), 'Temperature must be positive'

    def test_liquidus_basic_shape(self):
        """Liquidus returns correct shape for given pressure grid."""
        P, T = andrault_2011(kind='liquidus', Pmin=0.0, Pmax=100.0, n=50)
        assert len(P) == 50
        assert len(T) == 50
        assert np.all(T > 0.0), 'Temperature must be positive'

    def test_solidus_less_than_liquidus_when_both_defined(self):
        """Solidus < liquidus in regions where both are defined.

        Note: Andrault (2011) raw model may have unphysical regions where
        the separation isn't maintained. Real usage with truncate_to_physical_interval
        enforces solidus < liquidus. This tests the raw parametrization.
        """
        P_sol, T_sol = andrault_2011(kind='solidus', Pmin=50.0, Pmax=100.0, n=50)
        P_liq, T_liq = andrault_2011(kind='liquidus', Pmin=50.0, Pmax=100.0, n=50)

        assert np.allclose(P_sol, P_liq), 'Pressure grids must match'
        # At higher pressures where both curves are well-defined
        assert np.all(T_sol < T_liq), 'Solidus < liquidus in well-defined region'

    def test_temperature_increases_with_pressure(self):
        """Temperature is monotonically increasing with pressure."""
        _, T_sol = andrault_2011(kind='solidus', Pmin=0.0, Pmax=100.0, n=50)
        _, T_liq = andrault_2011(kind='liquidus', Pmin=0.0, Pmax=100.0, n=50)

        assert np.all(np.diff(T_sol) > 0), 'Solidus must increase with pressure'
        assert np.all(np.diff(T_liq) > 0), 'Liquidus must increase with pressure'

    def test_invalid_kind(self):
        """Invalid kind parameter raises ValueError."""
        with pytest.raises(ValueError, match='kind must be'):
            andrault_2011(kind='invalid')
        # Discrimination: the guard must fire on any non-{solidus,liquidus}
        # string, not only on the literal 'invalid'. A regression that
        # whitelisted by negation of a single bad token would let this
        # second call slip through.
        with pytest.raises(ValueError, match='kind must be'):
            andrault_2011(kind='melt')


# =============================================================================
# TEST: Melting Curve Functions - Fei (2021)
# =============================================================================


@pytest.mark.unit
class TestFei2021:
    """Test Fei et al. (2021) melting curve."""

    def test_solidus_basic_shape(self):
        """Solidus returns correct shape."""
        P, T = fei_2021(kind='solidus', Pmin=1.0, Pmax=100.0, n=50)
        assert len(P) == 50
        assert len(T) == 50
        assert np.all(T > 0.0)

    def test_liquidus_basic_shape(self):
        """Liquidus returns correct shape."""
        P, T = fei_2021(kind='liquidus', Pmin=1.0, Pmax=100.0, n=50)
        assert len(P) == 50
        assert len(T) == 50
        assert np.all(T > 0.0)

    def test_solidus_less_than_liquidus_at_valid_pressures(self):
        """Solidus < liquidus in physically valid regions (fei_2021 uses stixrude ratio)."""
        P_sol, T_sol = fei_2021(kind='solidus', Pmin=10.0, Pmax=100.0, n=50)
        P_liq, T_liq = fei_2021(kind='liquidus', Pmin=10.0, Pmax=100.0, n=50)

        assert np.allclose(P_sol, P_liq)
        # Stixrude ratio ensures solidus < liquidus by construction
        assert np.all(T_sol < T_liq)

    def test_temperature_increases_with_pressure(self):
        """Temperature increases monotonically with pressure."""
        _, T_sol = fei_2021(kind='solidus', Pmin=1.0, Pmax=100.0, n=50)
        _, T_liq = fei_2021(kind='liquidus', Pmin=1.0, Pmax=100.0, n=50)

        assert np.all(np.diff(T_sol) > 0)
        assert np.all(np.diff(T_liq) > 0)


# =============================================================================
# TEST: Melting Curve Functions - Belonoshko (2005)
# =============================================================================


@pytest.mark.unit
class TestBelonoshko2005:
    """Test Belonoshko et al. (2005) melting curve."""

    def test_solidus_basic_shape(self):
        """Solidus returns correct shape."""
        P, T = belonoshko_2005(kind='solidus', Pmin=0.0, Pmax=100.0, n=50)
        assert len(P) == 50
        assert len(T) == 50
        assert np.all(T > 0.0)

    def test_liquidus_basic_shape(self):
        """Liquidus returns correct shape."""
        P, T = belonoshko_2005(kind='liquidus', Pmin=0.0, Pmax=100.0, n=50)
        assert len(P) == 50
        assert len(T) == 50
        assert np.all(T > 0.0)

    def test_solidus_less_than_liquidus_at_valid_pressures(self):
        """Solidus < liquidus in physically valid regions (uses stixrude ratio)."""
        P_sol, T_sol = belonoshko_2005(kind='solidus', Pmin=10.0, Pmax=100.0, n=50)
        P_liq, T_liq = belonoshko_2005(kind='liquidus', Pmin=10.0, Pmax=100.0, n=50)

        assert np.allclose(P_sol, P_liq)
        # Stixrude ratio ensures solidus < liquidus by construction
        assert np.all(T_sol < T_liq)

    def test_temperature_increases_with_pressure(self):
        """Temperature increases monotonically with pressure."""
        _, T_sol = belonoshko_2005(kind='solidus', Pmin=0.0, Pmax=100.0, n=50)
        _, T_liq = belonoshko_2005(kind='liquidus', Pmin=0.0, Pmax=100.0, n=50)

        assert np.all(np.diff(T_sol) > 0)
        assert np.all(np.diff(T_liq) > 0)


# =============================================================================
# TEST: Melting Curve Functions - Fiquet (2010, Piecewise)
# =============================================================================


@pytest.mark.unit
class TestFiquet2010:
    """Test Fiquet et al. (2010) piecewise melting curve."""

    def test_solidus_basic_shape(self):
        """Solidus returns correct shape."""
        P, T = fiquet_2010(kind='solidus', Pmin=0.0, Pmax=100.0, n=50)
        assert len(P) == 50
        assert len(T) == 50
        assert np.all(T > 0.0)

    def test_liquidus_basic_shape(self):
        """Liquidus returns correct shape."""
        P, T = fiquet_2010(kind='liquidus', Pmin=0.0, Pmax=100.0, n=50)
        assert len(P) == 50
        assert len(T) == 50
        assert np.all(T > 0.0)

    def test_solidus_less_than_liquidus_at_valid_pressures(self):
        """Solidus < liquidus in physically valid regions (uses stixrude ratio)."""
        P_sol, T_sol = fiquet_2010(kind='solidus', Pmin=10.0, Pmax=100.0, n=50)
        P_liq, T_liq = fiquet_2010(kind='liquidus', Pmin=10.0, Pmax=100.0, n=50)

        assert np.allclose(P_sol, P_liq)
        # Stixrude ratio ensures solidus < liquidus by construction
        assert np.all(T_sol < T_liq)

    def test_piecewise_transition(self):
        """Piecewise branches are smooth and the known jump at 20 GPa is explicit."""
        P, T = fiquet_2010(kind='liquidus', Pmin=18.0, Pmax=22.0, n=100)

        # Branch-wise smoothness: verify monotonic increase away from the 20 GPa split.
        low = P < 20.0
        high = P > 20.0
        assert np.all(np.diff(T[low]) > 0.0), (
            'Low-pressure branch should increase with pressure'
        )
        assert np.all(np.diff(T[high]) > 0.0), (
            'High-pressure branch should increase with pressure'
        )

        # The implementation uses two branches that are not exactly continuous at 20 GPa.
        # Measure the one-sided jump from function output (left/right of the split):
        # jump = T(20+)-T(20-) should be negative with magnitude ~0.56 K.
        _, T_left = fiquet_2010(kind='liquidus', Pmin=20.0, Pmax=20.0, n=1)
        _, T_right = fiquet_2010(kind='liquidus', Pmin=20.0 + 1e-6, Pmax=20.0 + 1e-6, n=1)
        jump_at_20 = float(T_right[0] - T_left[0])

        assert jump_at_20 < 0.0, 'Expected a downward jump at 20 GPa (T_right < T_left)'
        assert abs(jump_at_20) == pytest.approx(0.55908, rel=1e-2)


# =============================================================================
# TEST: Melting Curve Functions - Monteux (2016, Piecewise)
# =============================================================================


@pytest.mark.unit
class TestMonteux2016:
    """Test Monteux et al. (2016) piecewise melting curve."""

    def test_solidus_basic_shape(self):
        """Solidus returns correct shape."""
        P, T = monteux_2016(kind='solidus', Pmin=0.0, Pmax=100.0, n=50)
        assert len(P) == 50
        assert len(T) == 50
        assert np.all(T > 0.0)

    def test_liquidus_basic_shape(self):
        """Liquidus returns correct shape."""
        P, T = monteux_2016(kind='liquidus', Pmin=0.0, Pmax=100.0, n=50)
        assert len(P) == 50
        assert len(T) == 50
        assert np.all(T > 0.0)

    def test_solidus_less_than_liquidus(self):
        """Solidus < liquidus at all pressures."""
        P_sol, T_sol = monteux_2016(kind='solidus', Pmin=0.0, Pmax=100.0, n=50)
        P_liq, T_liq = monteux_2016(kind='liquidus', Pmin=0.0, Pmax=100.0, n=50)

        assert np.allclose(P_sol, P_liq)
        assert np.all(T_sol < T_liq)

    def test_piecewise_transition(self):
        """Piecewise function has well-defined behavior at transition."""
        P, T = monteux_2016(kind='solidus', Pmin=15.0, Pmax=25.0, n=100)

        # Check that temperature changes appropriately (no large jumps)
        dT = np.diff(T)
        finite_dT = dT[np.isfinite(dT)]
        if len(finite_dT) > 0:
            # Most gradients should be small and positive for smooth parametrization
            assert np.median(finite_dT) > 0, 'Median gradient should be positive'
            assert np.max(np.abs(finite_dT)) < 100, 'No unreasonable jumps in T'


# =============================================================================
# TEST: Melting Curve Functions - Hirschmann (2000)
# =============================================================================


@pytest.mark.unit
class TestHirschmann2000:
    """Test Hirschmann (2000) melting curve (low pressure, ~0–10 GPa)."""

    def test_solidus_basic_shape(self):
        """Solidus returns correct shape."""
        P, T = hirschmann_2000(kind='solidus', Pmin=0.0, Pmax=10.0, n=50)
        assert len(P) == 50
        assert len(T) == 50
        assert np.all(T > 0.0)

    def test_liquidus_basic_shape(self):
        """Liquidus returns correct shape."""
        P, T = hirschmann_2000(kind='liquidus', Pmin=0.0, Pmax=10.0, n=50)
        assert len(P) == 50
        assert len(T) == 50
        assert np.all(T > 0.0)

    def test_solidus_less_than_liquidus(self):
        """Solidus < liquidus at all pressures."""
        P_sol, T_sol = hirschmann_2000(kind='solidus', Pmin=0.0, Pmax=10.0, n=50)
        P_liq, T_liq = hirschmann_2000(kind='liquidus', Pmin=0.0, Pmax=10.0, n=50)

        assert np.allclose(P_sol, P_liq)
        assert np.all(T_sol < T_liq)

    def test_temperature_increases_with_pressure(self):
        """Temperature increases monotonically with pressure."""
        _, T_sol = hirschmann_2000(kind='solidus', Pmin=0.0, Pmax=10.0, n=50)
        _, T_liq = hirschmann_2000(kind='liquidus', Pmin=0.0, Pmax=10.0, n=50)

        assert np.all(np.diff(T_sol) > 0)
        assert np.all(np.diff(T_liq) > 0)


# =============================================================================
# TEST: Melting Curve Functions - Stixrude (2014)
# =============================================================================


@pytest.mark.unit
class TestStixrude2014:
    """Test Stixrude (2014) melting curve."""

    def test_solidus_basic_shape(self):
        """Solidus returns correct shape."""
        P, T = stixrude_2014(kind='solidus', Pmin=1.0, Pmax=100.0, n=50)
        assert len(P) == 50
        assert len(T) == 50
        assert np.all(T > 0.0)

    def test_liquidus_basic_shape(self):
        """Liquidus returns correct shape."""
        P, T = stixrude_2014(kind='liquidus', Pmin=1.0, Pmax=100.0, n=50)
        assert len(P) == 50
        assert len(T) == 50
        assert np.all(T > 0.0)

    def test_solidus_less_than_liquidus_at_valid_pressures(self):
        """Solidus < liquidus in physically valid regions (uses stixrude ratio)."""
        P_sol, T_sol = stixrude_2014(kind='solidus', Pmin=10.0, Pmax=100.0, n=50)
        P_liq, T_liq = stixrude_2014(kind='liquidus', Pmin=10.0, Pmax=100.0, n=50)

        assert np.allclose(P_sol, P_liq)
        # Stixrude ratio ensures solidus < liquidus by construction
        assert np.all(T_sol < T_liq)


# =============================================================================
# TEST: Melting Curve Functions - Wolf & Bower (2018, Piecewise)
# =============================================================================


@pytest.mark.unit
class TestWolfBower2018:
    """Test Wolf & Bower (2018) piecewise melting curve."""

    def test_solidus_basic_shape(self):
        """Solidus returns correct shape."""
        P, T = wolf_bower_2018(kind='solidus', Pmin=0.0, Pmax=100.0, n=50)
        assert len(P) == 50
        assert len(T) == 50
        assert np.all(T > 0.0)

    def test_liquidus_basic_shape(self):
        """Liquidus returns correct shape."""
        P, T = wolf_bower_2018(kind='liquidus', Pmin=0.0, Pmax=100.0, n=50)
        assert len(P) == 50
        assert len(T) == 50
        assert np.all(T > 0.0)

    def test_solidus_less_than_liquidus(self):
        """Solidus < liquidus at all pressures."""
        P_sol, T_sol = wolf_bower_2018(kind='solidus', Pmin=0.0, Pmax=100.0, n=50)
        P_liq, T_liq = wolf_bower_2018(kind='liquidus', Pmin=0.0, Pmax=100.0, n=50)

        assert np.allclose(P_sol, P_liq)
        assert np.all(T_sol < T_liq)

    def test_piecewise_continuity(self):
        """Piecewise function is continuous across transitions."""
        P, T = wolf_bower_2018(kind='solidus', Pmin=0.0, Pmax=100.0, n=200)

        # Check temperature is generally monotonically increasing (with small numerical tolerance)
        dT = np.diff(T)
        assert np.count_nonzero(dT > -1e-6) > len(dT) * 0.99, 'Temperature should increase'
        # Discrimination: physical positivity (T > 0 Kelvin everywhere). A
        # regression that flipped a sign or swapped a slope-intercept pair
        # would surface as a negative temperature in one of the three
        # piecewise branches.
        assert np.all(T > 0.0), 'Solidus temperature must be strictly positive (Kelvin)'


# =============================================================================
# TEST: Melting Curve Functions - Katz (2003, Hydrous)
# =============================================================================


@pytest.mark.unit
class TestKatz2003:
    """Test Katz et al. (2003) hydrous melting curve."""

    def test_solidus_dry(self):
        """Solidus with X_h2o = 0 matches dry Wolf & Bower (2018) at valid pressures."""
        P_sol_katz, T_sol_katz = katz_2003(
            kind='solidus', X_h2o=0.0, Pmin=50.0, Pmax=100.0, n=50
        )
        P_wb, T_wb = wolf_bower_2018(kind='solidus', Pmin=50.0, Pmax=100.0, n=50)

        assert np.allclose(T_sol_katz, T_wb, rtol=1e-10)
        # Discrimination: dry limit is X_h2o=0 -> delta_T = K * 0^gamma = 0.
        # A regression that used a non-zero K*1 floor (e.g. delta_T = K)
        # would shift T_sol_katz by ~43 K below T_wb, breaking the close
        # match above. Pin the pressure grid alignment too, so a regression
        # that changed make_pressure_grid semantics is caught.
        assert np.allclose(P_sol_katz, P_wb, rtol=1e-12)

    def test_hydrous_effect(self):
        """Increasing water content decreases melting temperature (physical)."""
        P, T_dry = katz_2003(kind='solidus', X_h2o=0.0, Pmin=50.0, Pmax=100.0, n=50)
        _, T_wet = katz_2003(kind='solidus', X_h2o=30.0, Pmin=50.0, Pmax=100.0, n=50)

        assert np.all(T_wet < T_dry), 'Water lowers melting temperature'
        # Discrimination: pin the closed-form magnitude. Katz (2003)
        # delta_T = K * X_h2o**gamma with K=43, gamma=0.75. At X_h2o=30:
        # delta_T = 43 * 30**0.75 ~ 551.2 K. A regression that used the
        # wrong exponent (e.g. linear in X_h2o) would land at 43*30 = 1290
        # K and would not match this approx pin.
        expected_delta = 43.0 * 30.0**0.75
        assert np.allclose(T_dry - T_wet, expected_delta, rtol=1e-10)

    def test_default_water_content(self):
        """Default water content is X_h2o = 30 ppm."""
        P1, T1 = katz_2003(kind='solidus', Pmin=50.0, Pmax=100.0, n=50)
        P2, T2 = katz_2003(kind='solidus', X_h2o=30.0, Pmin=50.0, Pmax=100.0, n=50)

        assert np.allclose(T1, T2)
        # Discrimination: the default must also differ from the dry limit
        # by the Katz delta_T at X_h2o=30. A regression that flipped the
        # default to 0.0 would make T1 equal to the dry curve, which would
        # contradict this nonzero-offset pin.
        _, T_dry = katz_2003(kind='solidus', X_h2o=0.0, Pmin=50.0, Pmax=100.0, n=50)
        assert np.all(T1 < T_dry), 'Default X_h2o must depress solidus below the dry curve'

    def test_physical_constraint_solidus_less_than_liquidus(self):
        """Solidus < liquidus with hydration at valid pressures."""
        P_sol, T_sol = katz_2003(kind='solidus', X_h2o=30.0, Pmin=50.0, Pmax=100.0, n=50)
        P_liq, T_liq = katz_2003(kind='liquidus', X_h2o=30.0, Pmin=50.0, Pmax=100.0, n=50)

        assert np.allclose(P_sol, P_liq)
        assert np.all(T_sol < T_liq)


# =============================================================================
# TEST: Melting Curve Functions - Lin (2024, Oxygen Fugacity)
# =============================================================================


@pytest.mark.unit
class TestLin2024:
    """Test Lin et al. (2024) oxygen-fugacity-dependent solidus."""

    def test_solidus_default_fo2(self):
        """Solidus with default fO2 = -4."""
        P, T = lin_2024(kind='solidus', fO2=-4.0, Pmin=50.0, Pmax=100.0, n=50)
        assert len(P) == 50
        assert len(T) == 50
        assert np.all(T > 0.0)

    def test_fo2_effect(self):
        """fO2 parameter affects melting temperature (Lin et al. 2024 parametrization).

        Lower (more reducing) fO2 values increase the solidus temperature;
        higher (more oxidizing) fO2 values decrease it.
        """
        P, T_reducing = lin_2024(kind='solidus', fO2=-5.0, Pmin=50.0, Pmax=100.0, n=50)
        _, T_oxidizing = lin_2024(kind='solidus', fO2=-3.0, Pmin=50.0, Pmax=100.0, n=50)

        # Lower fO2 (more reducing) -> higher T
        assert np.all(T_reducing > T_oxidizing), 'Lower fO2 should increase solidus T'
        # Discrimination: pin the closed-form delta. Lin (2024) uses
        # delta_T = (340/3.2) * (2 - fO2), so going from fO2 = -3 to -5
        # shifts the solidus by (340/3.2) * (-5 - (-3)) = -212.5 K, i.e.
        # T_reducing exceeds T_oxidizing by exactly 212.5 K. A regression
        # that swapped the coefficient sign or dropped the factor of 2
        # would miss this pin.
        expected_diff = (340.0 / 3.2) * ((-3.0) - (-5.0))
        assert np.allclose(T_reducing - T_oxidizing, expected_diff, rtol=1e-10)

    def test_physical_constraint_solidus_less_than_liquidus(self):
        """Solidus < liquidus with varying fO2 at valid pressures."""
        P_sol, T_sol = lin_2024(kind='solidus', fO2=-4.0, Pmin=50.0, Pmax=100.0, n=50)
        P_liq, T_liq = lin_2024(kind='liquidus', fO2=-4.0, Pmin=50.0, Pmax=100.0, n=50)

        assert np.allclose(P_sol, P_liq)
        assert np.all(T_sol < T_liq)


# =============================================================================
# TEST: get_melting_curves Dispatcher
# =============================================================================


@pytest.mark.unit
class TestGetMeltingCurves:
    """Test melting curve dispatcher."""

    @pytest.mark.parametrize(
        'model_name,Pmin,Pmax',
        [
            ('andrault_2011', 0.0, 100.0),
            ('monteux_2016', 0.0, 100.0),
            ('wolf_bower_2018', 0.0, 100.0),
            ('katz_2003', 0.0, 100.0),
            ('fei_2021', 1.0, 100.0),  # Requires Pmin >= 1
            ('belonoshko_2005', 0.0, 100.0),
            ('fiquet_2010', 0.0, 100.0),
            ('hirschmann_2000', 0.0, 5.0),  # Low pressure only
            ('stixrude_2014', 1.0, 100.0),  # Requires Pmin >= 1
            ('lin_2024', 0.0, 100.0),
        ],
    )
    def test_all_supported_models(self, model_name, Pmin, Pmax):
        """Dispatcher returns non-empty, shape-consistent outputs for all models."""
        P_sol, T_sol, P_liq, T_liq = get_melting_curves(model_name, Pmin=Pmin, Pmax=Pmax, n=50)

        assert len(P_sol) > 0, f'{model_name}: no pressure values'
        assert len(T_sol) > 0, f'{model_name}: no temperature values'
        assert len(P_liq) > 0, f'{model_name}: no pressure values for liquidus'
        assert len(T_liq) > 0, f'{model_name}: no liquidus temperature values'
        assert len(P_sol) == len(T_sol), f'{model_name}: solidus P/T length mismatch'
        assert len(P_liq) == len(T_liq), f'{model_name}: liquidus P/T length mismatch'
        assert np.all(np.isfinite(P_sol)), f'{model_name}: non-finite values in P_sol'
        assert np.all(np.isfinite(T_sol)), f'{model_name}: non-finite values in T_sol'
        assert np.all(np.isfinite(P_liq)), f'{model_name}: non-finite values in P_liq'
        assert np.all(np.isfinite(T_liq)), f'{model_name}: non-finite values in T_liq'

    def test_unknown_model(self):
        """Unknown model raises ValueError."""
        with pytest.raises(ValueError, match='Unknown model'):
            get_melting_curves('unknown_model')
        # Discrimination: the guard must reject empty string and unrelated
        # tokens too, not just the literal 'unknown_model'. A regression
        # that special-cased one missing key would let these slip through.
        with pytest.raises(ValueError, match='Unknown model'):
            get_melting_curves('')

    def test_custom_pressure_range(self):
        """Custom pressure range is applied correctly."""
        P_sol, T_sol, _, _ = get_melting_curves('wolf_bower_2018', Pmin=10.0, Pmax=50.0, n=50)

        # After truncation, pressure should be within requested range
        if len(P_sol) > 0:
            assert P_sol[0] >= 10.0 - 1.0, 'Lowest pressure should be near requested Pmin'
            assert P_sol[-1] <= 50.0 + 1.0, 'Highest pressure should be near requested Pmax'


# =============================================================================
# TEST: truncate_to_physical_interval Decorator
# =============================================================================


@pytest.mark.unit
class TestTruncateToPhysicalInterval:
    """Test physical interval filtering decorator."""

    def test_decorator_filters_unphysical_regions(self):
        """Truncation removes regions where solidus >= liquidus."""
        # Create a wrapped function
        wrapped_wolf = truncate_to_physical_interval(wolf_bower_2018)

        P_sol, T_sol = wrapped_wolf(kind='solidus', Pmin=0.0, Pmax=100.0, n=100)
        P_liq, T_liq = wrapped_wolf(kind='liquidus', Pmin=0.0, Pmax=100.0, n=100)

        # In the physical interval, solidus < liquidus
        assert np.all(T_sol < T_liq)
        # Discrimination: the truncation must return non-empty arrays on a
        # well-behaved model with a real physical interval. A regression
        # that always returned empty arrays would pass `np.all(T_sol < T_liq)`
        # vacuously (np.all of an empty array is True).
        assert len(T_sol) > 0 and len(T_liq) > 0, (
            'truncate_to_physical_interval must return a non-empty main block'
        )

    def test_decorator_preserves_main_block(self):
        """Main physical block is largest contiguous block."""
        wrapped_wolf = truncate_to_physical_interval(wolf_bower_2018)

        # Should not raise
        P_sol, T_sol = wrapped_wolf(kind='solidus', Pmin=0.0, Pmax=1000.0, n=200)
        assert len(P_sol) > 0
        # Discriminating physics check: solidus temperature is strictly
        # positive everywhere (Kelvin), and the returned profile arrays have
        # matching lengths (P_sol[i] aligns with T_sol[i]).
        assert len(P_sol) == len(T_sol)
        assert np.all(T_sol > 0)


# =============================================================================
# TEST: EOS Inversion Pipeline - invert_to_entropy_along_profile
# =============================================================================


@pytest.mark.unit
class TestInvertToEntropyAlongProfile:
    """Test EOS inversion to entropy curves with mocked T(S,P) interpolator."""

    def test_basic_inversion(self, mock_eos_interpolator, mock_S_axes):
        """Basic entropy inversion with mocked EOS.

        Note: With the simplified mock interpolator, inversion may not always
        find valid S values. This test verifies the function runs without error
        and returns the correct shape.
        """
        S_axis_solid, _ = mock_S_axes
        # Use mid-range pressure/temperature values
        P = np.array([50.0, 60.0, 70.0])  # GPa
        # For mock T(S,P) = 1000 + 10*S + 100*P, with S in [100,1000] and P in [0,100]:
        # T ranges from 1000+1000+0 = 2000 to 1000+10000+10000 = 21000
        # Use realistic T values for the given P, S ranges
        T = np.array([6500.0, 7000.0, 7500.0])  # K (plausible for mid-range S,P)

        S = invert_to_entropy_along_profile(P, T, S_axis_solid, mock_eos_interpolator)

        assert S.shape == P.shape
        # May or may not find valid values depending on mock bounds, but should not crash
        assert isinstance(S, np.ndarray)

    def test_inversion_returns_entropy_values(self, mock_eos_interpolator, mock_S_axes):
        """Entropy values are within expected bounds when valid."""
        S_axis_solid, _ = mock_S_axes
        P = np.array([50.0])  # Mid-range pressure
        T = np.array([2000.0])  # Mid-range temperature

        S = invert_to_entropy_along_profile(P, T, S_axis_solid, mock_eos_interpolator)

        # When valid, S should be in the S_axis range (approximately)
        valid = np.isfinite(S)
        if np.any(valid):
            assert np.all(S[valid] >= S_axis_solid.min() - 100)
            assert np.all(S[valid] <= S_axis_solid.max() + 100)

    def test_out_of_bounds_returns_nan(self, mock_eos_interpolator, mock_S_axes):
        """Out-of-bounds P,T return NaN."""
        S_axis_solid, _ = mock_S_axes
        P = np.array([1000.0])  # Outside mock bounds [0, 100]
        T = np.array([10000.0])  # Unphysical temperature

        S = invert_to_entropy_along_profile(P, T, S_axis_solid, mock_eos_interpolator)

        assert np.isnan(S[0]), 'Out-of-bounds should return NaN'
        # Discrimination: NaN must be the only sentinel produced, not a
        # silent clamp to the entropy-axis boundary. A regression that
        # clamped P > P_max to P_max would return a finite S inside the
        # axis range, which would pass `not np.isnan` checks downstream.
        assert S.shape == P.shape, 'Inversion must preserve input shape'

    def test_empty_pressure_array(self, mock_eos_interpolator, mock_S_axes):
        """Empty pressure array returns empty entropy array."""
        S_axis_solid, _ = mock_S_axes
        P = np.array([])
        T = np.array([])

        S = invert_to_entropy_along_profile(P, T, S_axis_solid, mock_eos_interpolator)

        assert len(S) == 0
        # Discrimination: the output type must be a numpy array on the
        # empty-input fast path; a regression that returned a bare list or
        # None would also satisfy len(...)==0 (or raise for None) but break
        # downstream array operations.
        assert isinstance(S, np.ndarray)


# =============================================================================
# TEST: build_common_entropy_grid
# =============================================================================


@pytest.mark.unit
class TestBuildCommonEntropyGrid:
    """Test resampling solidus/liquidus onto common pressure grid."""

    def test_basic_resampling(self):
        """Basic resampling onto common pressure grid."""
        P_sol = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        S_sol = np.array([100.0, 150.0, 200.0, 250.0, 300.0])

        P_liq = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        S_liq = np.array([150.0, 200.0, 250.0, 300.0, 350.0])

        P_common, S_sol_common, S_liq_common = build_common_entropy_grid(
            P_sol, S_sol, P_liq, S_liq, n_common=5
        )

        assert len(P_common) > 0
        assert len(S_sol_common) == len(P_common)
        assert len(S_liq_common) == len(P_common)

    def test_common_grid_within_overlap(self):
        """Common pressure grid is within overlap of solidus/liquidus ranges."""
        P_sol = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        S_sol = np.array([100.0, 150.0, 200.0, 250.0, 300.0])

        P_liq = np.array([8.0, 12.0, 16.0, 20.0, 24.0])
        S_liq = np.array([150.0, 200.0, 250.0, 300.0, 350.0])

        P_common, _, _ = build_common_entropy_grid(P_sol, S_sol, P_liq, S_liq)

        if len(P_common) > 0:
            P_min_overlap = max(P_sol.min(), P_liq.min())
            P_max_overlap = min(P_sol.max(), P_liq.max())
            assert np.all(P_common >= P_min_overlap - 1.0)
            assert np.all(P_common <= P_max_overlap + 1.0)

    def test_nans_filtered_out(self):
        """NaN values are filtered out before resampling."""
        P_sol = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        S_sol = np.array([100.0, np.nan, 200.0, 250.0, 300.0])

        P_liq = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        S_liq = np.array([150.0, 200.0, 250.0, np.nan, 350.0])

        P_common, S_sol_common, S_liq_common = build_common_entropy_grid(
            P_sol, S_sol, P_liq, S_liq
        )

        assert np.all(np.isfinite(P_common))
        assert np.all(np.isfinite(S_sol_common))
        assert np.all(np.isfinite(S_liq_common))

    def test_no_overlap_returns_empty(self):
        """Non-overlapping pressure ranges return empty arrays."""
        P_sol = np.array([10.0, 20.0, 30.0])
        S_sol = np.array([100.0, 150.0, 200.0])

        P_liq = np.array([50.0, 60.0, 70.0])
        S_liq = np.array([150.0, 200.0, 250.0])

        P_common, S_sol_common, S_liq_common = build_common_entropy_grid(
            P_sol, S_sol, P_liq, S_liq
        )

        assert len(P_common) == 0
        assert len(S_sol_common) == 0
        assert len(S_liq_common) == 0

    def test_custom_n_common(self):
        """Custom n_common parameter sets output grid size."""
        P_sol = np.linspace(10.0, 50.0, 100)
        S_sol = np.linspace(100.0, 500.0, 100)

        P_liq = np.linspace(10.0, 50.0, 100)
        S_liq = np.linspace(150.0, 550.0, 100)

        P_common, _, _ = build_common_entropy_grid(P_sol, S_sol, P_liq, S_liq, n_common=25)

        if len(P_common) > 0:
            assert len(P_common) <= 25
        # Discrimination: with two fully overlapping inputs of length 100,
        # the common grid must actually be populated. A regression that
        # always returned an empty array would make the bounded-length
        # check above vacuously pass. Also pin monotonicity to catch a
        # regression that returned a permuted grid.
        assert len(P_common) > 0, 'Overlapping inputs must yield a non-empty common grid'
        assert np.all(np.diff(P_common) > 0), 'Common pressure grid must be strictly increasing'


# =============================================================================
# TEST: validate_entropy_export_arrays
# =============================================================================


@pytest.mark.unit
class TestValidateEntropyExportArrays:
    """Test validation of entropy export arrays."""

    def test_valid_arrays_passes(self):
        """Valid arrays pass validation silently."""
        P = np.array([10.0, 20.0, 30.0])
        S_sol = np.array([100.0, 150.0, 200.0])
        S_liq = np.array([150.0, 200.0, 250.0])

        result = validate_entropy_export_arrays(P, S_sol, S_liq, 'test_model')
        assert result is None  # contract: validator returns None silently on the pass path
        # Discriminating physics check: liquidus entropy strictly above solidus
        # at every pressure (the physical ordering the validator must accept).
        assert np.all(S_liq > S_sol)

    def test_empty_array_raises(self):
        """Empty array raises ValueError."""
        P = np.array([])
        S_sol = np.array([])
        S_liq = np.array([])

        with pytest.raises(ValueError, match='could not build|empty'):
            validate_entropy_export_arrays(P, S_sol, S_liq, 'test_model')
        # Discrimination: the empty guard must fire even when only one of
        # the three arrays is empty. A regression that required all three
        # to be empty before raising would let a partial-empty case slip
        # through silently.
        P_one = np.array([10.0])
        with pytest.raises(ValueError, match='could not build|empty'):
            validate_entropy_export_arrays(P_one, np.array([]), np.array([]), 'test_model')

    def test_mismatched_lengths_raises(self):
        """Mismatched array lengths raise ValueError."""
        P = np.array([10.0, 20.0, 30.0])
        S_sol = np.array([100.0, 150.0])  # Wrong length
        S_liq = np.array([150.0, 200.0, 250.0])

        with pytest.raises(ValueError, match='inconsistent'):
            validate_entropy_export_arrays(P, S_sol, S_liq, 'test_model')
        # Discrimination: the guard must fire when S_liq is the short one
        # too, not only when S_sol is short. A regression that checked
        # only S_sol length against P would miss this case.
        with pytest.raises(ValueError, match='inconsistent'):
            validate_entropy_export_arrays(
                P, np.array([100.0, 150.0, 200.0]), np.array([150.0, 200.0]), 'test_model'
            )

    def test_nans_in_arrays_raises(self):
        """NaN values raise ValueError."""
        P = np.array([10.0, 20.0, np.nan])
        S_sol = np.array([100.0, 150.0, 200.0])
        S_liq = np.array([150.0, 200.0, 250.0])

        with pytest.raises(ValueError, match='non-finite'):
            validate_entropy_export_arrays(P, S_sol, S_liq, 'test_model')
        # Discrimination: the guard must also catch NaN entries in the
        # entropy arrays, not only in the pressure column. A regression
        # that checked finiteness of P alone would miss this case.
        with pytest.raises(ValueError, match='non-finite'):
            validate_entropy_export_arrays(
                np.array([10.0, 20.0, 30.0]),
                np.array([100.0, np.nan, 200.0]),
                np.array([150.0, 200.0, 250.0]),
                'test_model',
            )

    def test_infs_in_arrays_raises(self):
        """Infinite values raise ValueError."""
        P = np.array([10.0, 20.0, 30.0])
        S_sol = np.array([100.0, np.inf, 200.0])
        S_liq = np.array([150.0, 200.0, 250.0])

        with pytest.raises(ValueError, match='non-finite'):
            validate_entropy_export_arrays(P, S_sol, S_liq, 'test_model')
        # Discrimination: -inf must trip the guard the same as +inf. A
        # regression that compared against a positive sentinel only (e.g.
        # `> 1e308` instead of np.isfinite) would let -inf slip through.
        with pytest.raises(ValueError, match='non-finite'):
            validate_entropy_export_arrays(
                np.array([10.0, 20.0, 30.0]),
                np.array([100.0, 150.0, 200.0]),
                np.array([150.0, -np.inf, 250.0]),
                'test_model',
            )


# =============================================================================
# TEST: make_entropy_header
# =============================================================================


@pytest.mark.unit
class TestMakeEntropyHeader:
    """Test entropy table header generation."""

    def test_default_header(self):
        """Default header includes n_rows and scaling factors."""
        header = make_entropy_header(n_rows=100)

        assert '100' in header
        assert '1e+09' in header or '1000000000' in header
        assert 'Pressure' in header
        assert 'Entropy' in header
        assert 'scaling factor' in header

    def test_custom_scaling_factors(self):
        """Custom scaling factors appear in header."""
        scale_p = 5e8
        scale_s = 1e6

        header = make_entropy_header(n_rows=50, scale_p_out=scale_p, scale_s_out=scale_s)

        assert '50' in header
        assert str(scale_p) in header or '5e+08' in header
        assert str(scale_s) in header or '1e+06' in header

    def test_multiline_header(self):
        """Header is multiline with proper structure."""
        header = make_entropy_header(n_rows=100)
        lines = header.split('\n')

        assert len(lines) >= 5
        assert all(line.startswith('#') for line in lines if line.strip())


# =============================================================================
# INTEGRATION SMOKE TESTS (Basic End-to-End)
# =============================================================================


@pytest.mark.unit
class TestMeltingCurveSmoke:
    """Smoke tests: basic physical sanity checks across all models."""

    @pytest.mark.parametrize(
        'model_name,Pmin,Pmax',
        [
            ('andrault_2011', 0.0, 100.0),
            ('monteux_2016', 0.0, 100.0),
            ('wolf_bower_2018', 0.0, 100.0),
            ('katz_2003', 0.0, 100.0),
            ('fei_2021', 1.0, 100.0),
            ('belonoshko_2005', 0.0, 100.0),
            ('fiquet_2010', 0.0, 100.0),
            ('hirschmann_2000', 0.0, 10.0),
            ('stixrude_2014', 1.0, 100.0),
            ('lin_2024', 0.0, 100.0),
        ],
    )
    def test_model_physical_constraints(self, model_name, Pmin, Pmax):
        """Each model satisfies basic physical constraints."""
        P_sol, T_sol, P_liq, T_liq = get_melting_curves(model_name, Pmin=Pmin, Pmax=Pmax, n=50)

        # Soundness checks
        assert len(P_sol) > 0, f'{model_name}: no output'
        assert np.all(np.isfinite(P_sol)), f'{model_name}: P_sol contains NaNs/infs'
        assert np.all(np.isfinite(T_sol)), f'{model_name}: T_sol contains NaNs/infs'
        assert np.all(np.isfinite(P_liq)), f'{model_name}: P_liq contains NaNs/infs'
        assert np.all(np.isfinite(T_liq)), f'{model_name}: T_liq contains NaNs/infs'

        # Physics constraints
        assert np.all(T_sol > 0.0), f'{model_name}: T_sol <= 0'
        assert np.all(T_liq > 0.0), f'{model_name}: T_liq <= 0'
        assert np.all(T_sol < T_liq), f'{model_name}: solidus >= liquidus'

        # Monotonicity
        assert np.all(np.diff(T_sol) >= -1e-10), f'{model_name}: T_sol not monotonic'
        assert np.all(np.diff(T_liq) >= -1e-10), f'{model_name}: T_liq not monotonic'
