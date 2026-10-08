"""
Unit tests for proteus.atmos_chem.dummy module.

Parameterized atmospheric chemistry without a kinetics solver. Generates
vertical mixing ratio profiles for ~17 species with altitude-dependent
structure (cold trap, photolysis layer, well-mixed background).

Physics tested:
- Well-mixed parent species below photolysis level
- H2O cold trap via Clausius-Clapeyron
- Photolysis product increase toward TOA
- VMR normalization (sum to 1 at each level)
- CSV output format matches VULCAN convention
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from proteus.atmos_chem.dummy import _ALL_SPECIES, _build_profiles, run_dummy_chem

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _make_hf_row(
    T_magma=3000.0,
    P_surf=100.0,
    gravity=9.81,
    H2O_vmr=0.9,
    CO2_vmr=0.08,
    N2_vmr=0.01,
    H2_vmr=0.005,
    CH4_vmr=0.005,
):
    """Build a minimal hf_row for dummy chemistry tests."""
    hf_row = {
        'T_magma': T_magma,
        'T_surf': T_magma,
        'P_surf': P_surf,
        'gravity': gravity,
        'atm_kg_per_mol': 0.018,
        'Time': 1e6,
        'H2O_vmr': H2O_vmr,
        'CO2_vmr': CO2_vmr,
        'N2_vmr': N2_vmr,
        'H2_vmr': H2_vmr,
        'CO_vmr': 0.0,
        'CH4_vmr': CH4_vmr,
        'SO2_vmr': 0.0,
        'S2_vmr': 0.0,
        'H2S_vmr': 0.0,
        'NH3_vmr': 0.0,
        'O2_vmr': 0.0,
    }
    return hf_row


def _make_config():
    config = MagicMock()
    config.atmos_clim.p_top = 1e-6
    return config


@pytest.mark.unit
def test_dummy_chem_profile_shape():
    """Output has correct number of levels and species."""
    hf_row = _make_hf_row()
    config = _make_config()
    result = _build_profiles(hf_row, config, num_levels=50)

    assert result['tmp'].shape == (50,)
    assert result['p'].shape == (50,)
    assert result['z'].shape == (50,)
    assert result['Kzz'].shape == (50,)
    for sp in _ALL_SPECIES:
        assert sp in result, f'Missing species {sp}'
        assert result[sp].shape == (50,)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_dummy_chem_vmr_normalization():
    """VMRs sum to 1.0 at every level (mass-fraction closure)."""
    hf_row = _make_hf_row()
    config = _make_config()
    result = _build_profiles(hf_row, config, num_levels=50)

    total = np.zeros(50)
    for sp in _ALL_SPECIES:
        total += result[sp]

    np.testing.assert_allclose(total, 1.0, atol=1e-10)
    # Boundedness invariant: every species VMR must lie in [0, 1] at every
    # level. A regression that produced a normalised total of 1 via a
    # negative-and-positive cancellation (e.g. signed photolysis product)
    # would pass the closure check above but fail this bound.
    for sp in _ALL_SPECIES:
        assert np.all(result[sp] >= 0.0), f'{sp} went negative'
        assert np.all(result[sp] <= 1.0 + 1e-12), f'{sp} exceeded unit fraction'


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_dummy_chem_h2o_cold_trap():
    """H2O VMR limited by Clausius-Clapeyron in the cold stratosphere.

    Use low T_surf (300 K) so the stratosphere (~200 K) has P_sat
    much smaller than P_total at intermediate pressures, and the
    cold trap suppresses H2O by orders of magnitude.
    """
    hf_row = _make_hf_row(
        T_magma=300.0,
        P_surf=1.0,
        H2O_vmr=0.5,
        CO2_vmr=0.3,
        N2_vmr=0.1,
        H2_vmr=0.05,
        CH4_vmr=0.05,
    )
    config = _make_config()
    result = _build_profiles(hf_row, config, num_levels=50)

    h2o = result['H2O']
    # Find minimum H2O in the column (cold trap minimum, not at TOA/surface)
    h2o_min = np.min(h2o)
    h2o_surf = h2o[-1]
    # Cold trap should suppress H2O by at least 10x relative to surface
    assert h2o_min < 0.1 * h2o_surf, (
        f'Cold trap should suppress H2O: min={h2o_min:.4e} vs surface={h2o_surf:.4e}'
    )
    # Boundedness invariant: the minimum H2O must occur strictly above the
    # surface index (the cold trap is in the upper troposphere, not at
    # ground level). A regression that applied saturation TO the surface
    # (broken Clausius-Clapeyron threshold) would put the minimum at
    # index -1 and fail this discriminator.
    h2o_min_idx = int(np.argmin(h2o))
    assert h2o_min_idx < len(h2o) - 1, (
        f'Cold trap minimum at surface index ({h2o_min_idx}); should be above'
    )


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_dummy_chem_photolysis_products():
    """Photolysis products (O, OH, H) increase toward TOA. The exponential
    factor exp(-P/P_photo) damps to ~0 at the surface (high P) and rises
    toward 1 at TOA (low P), so the TOA-vs-surface ratio must be large.
    """
    hf_row = _make_hf_row()
    config = _make_config()
    result = _build_profiles(hf_row, config, num_levels=50)

    for product in ('O', 'OH', 'H'):
        vmr = result[product]
        # Should be higher at TOA than at surface
        assert vmr[0] > vmr[-1], f'{product} should increase toward TOA'
        # Positivity + scale invariant: the photolysis factor is
        # exp(-P_surf / P_photo) where P_surf >> P_photo, so the surface
        # VMR must be many orders of magnitude smaller than the TOA value.
        # A regression that swapped the sign of the exponent (decreasing
        # with altitude) would also satisfy vmr[0] > vmr[-1] only if the
        # parent surface VMR happened to dominate; pin a large ratio so
        # that bug class is rejected.
        assert vmr[0] > 0.0, f'{product} must be positive at TOA'
        assert vmr[0] > 100.0 * max(vmr[-1], 1e-300), (
            f'{product} TOA/surface ratio too small for exp(-P/P_photo) law'
        )


@pytest.mark.unit
def test_dummy_chem_pressure_ordering():
    """Pressure increases from TOA (index 0) to surface (index -1)."""
    hf_row = _make_hf_row()
    config = _make_config()
    result = _build_profiles(hf_row, config, num_levels=50)

    p = result['p']
    assert p[0] < p[-1], 'TOA pressure should be less than surface'
    assert np.all(np.diff(p) > 0), 'Pressure must increase monotonically'


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_dummy_chem_altitude_ordering():
    """Altitude decreases monotonically from TOA (index 0) to surface
    (index -1). Monotonicity is the discriminating property: an endpoint
    check alone would also pass for an oscillating altitude profile that
    happens to satisfy z[0] > z[-1].
    """
    hf_row = _make_hf_row()
    config = _make_config()
    result = _build_profiles(hf_row, config, num_levels=50)

    z = result['z']
    assert z[0] > z[-1], 'TOA altitude should be greater than surface'
    # Monotonicity invariant: hydrostatic integration must give z strictly
    # decreasing with index. A regression that reversed only the endpoints
    # (or that produced a non-monotonic profile from a sign error in
    # `dz = H * log(p[i+1]/p[i])`) would fail this stricter check.
    assert np.all(np.diff(z) < 0), 'altitude must decrease monotonically'


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_dummy_chem_no_negative_vmrs():
    """No species should have negative VMR at any level (positivity
    invariant on a volume mixing ratio).
    """
    hf_row = _make_hf_row()
    config = _make_config()
    result = _build_profiles(hf_row, config, num_levels=50)

    for sp in _ALL_SPECIES:
        assert np.all(result[sp] >= 0), f'{sp} has negative VMR'
    # Discrimination: a regression that returned an all-zero profile would
    # trivially satisfy the >= 0 bound; pin the sum-of-all-species at the
    # surface to ~1 so an empty-profile bug is caught here.
    surface_sum = sum(result[sp][-1] for sp in _ALL_SPECIES)
    assert surface_sum == pytest.approx(1.0, abs=1e-10), (
        f'normalization should give surface sum 1, got {surface_sum}'
    )


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_dummy_chem_zero_atmosphere():
    """Zero surface VMRs should produce zero profiles without crashing.

    Edge case: a divide-by-zero in the normalization step (total == 0) is
    the most likely regression. The function under test guards against
    that with `np.where(total > 0, ...)`, so the species profiles must
    be cleanly zero AND the temperature / pressure grids must still be
    populated at the requested resolution.
    """
    hf_row = _make_hf_row(H2O_vmr=0, CO2_vmr=0, N2_vmr=0, H2_vmr=0, CH4_vmr=0)
    config = _make_config()
    result = _build_profiles(hf_row, config, num_levels=30)

    for sp in _ALL_SPECIES:
        np.testing.assert_allclose(result[sp], 0.0, atol=1e-12, err_msg=f'{sp} should be zero')
    # Discrimination: a regression that returned early (or short-circuited
    # the whole function) on all-zero VMR input would also pass the
    # species-zero check above by producing empty arrays. Pin that the
    # T / P / z / Kzz grids are still populated at the requested length
    # and that T and P are strictly positive everywhere.
    assert result['tmp'].shape == (30,) and result['p'].shape == (30,)
    assert np.all(result['tmp'] > 0.0) and np.all(result['p'] > 0.0)


@pytest.mark.unit
def test_dummy_chem_writes_csv(tmp_path):
    """run_dummy_chem writes a CSV file in VULCAN-compatible format."""
    hf_row = _make_hf_row()
    config = _make_config()
    dirs = {'output': str(tmp_path)}

    success = run_dummy_chem(dirs, config, hf_row)
    assert success is True

    csv_path = tmp_path / 'offchem' / 'dummy.csv'
    assert csv_path.exists()

    # Read and check structure
    import pandas as pd

    df = pd.read_csv(csv_path, delimiter=r'\s+')
    assert 'tmp' in df.columns
    assert 'p' in df.columns
    assert 'H2O' in df.columns
    assert 'O' in df.columns
    assert len(df) == 50


@pytest.mark.unit
def test_dummy_chem_online_mode(tmp_path):
    """Online mode writes per-snapshot CSV files."""
    hf_row = _make_hf_row()
    hf_row['Time'] = 5000.0
    config = _make_config()
    dirs = {'output': str(tmp_path)}

    success = run_dummy_chem(dirs, config, hf_row, online=True)
    assert success is True

    csv_path = tmp_path / 'offchem' / 'dummy_5000.csv'
    assert csv_path.exists()
