"""Unit tests for the pure-Python helpers in ``proteus.star.wrapper``.

Targets the small inverse-square scaling helper and the spectrum-write
helper. Heavier dispatch functions that drive MORS / Spada tracks are
covered by integration tests in nightly tier.

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

import numpy as np
import pytest

import proteus.star.wrapper as star_wrapper
from proteus.utils.constants import AU

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# scale_spectrum_to_toa: inverse-square distance law
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_scale_spectrum_to_toa_at_one_au_is_identity():
    """At separation = 1 AU (in metres), the scaling factor is exactly
    1: the flux array passes through unchanged. Discrimination: a
    regression that returned a different power of (AU/sep) would change
    the magnitude at 1 AU away from unity; also verify the returned
    array length matches the input (rules out a slice/truncation bug).
    """
    fl = [1.0, 2.0, 3.0]
    scaled = star_wrapper.scale_spectrum_to_toa(fl, AU)
    np.testing.assert_allclose(scaled, fl, rtol=1e-12)
    assert len(scaled) == len(fl)


@pytest.mark.physics_invariant
def test_scale_spectrum_to_toa_inverse_square_at_two_au():
    """At separation = 2 AU, the inverse-square law predicts flux scales
    by (1/2)^2 = 1/4. A regression that used (1/r) or (1/r^3) would not
    match this ratio.
    """
    fl = np.array([100.0])
    scaled = star_wrapper.scale_spectrum_to_toa(fl, 2.0 * AU)
    # Inverse-square: flux at 2 AU is 1/4 of flux at 1 AU
    assert scaled[0] == pytest.approx(25.0, rel=1e-12)
    # Discrimination: a 1/r scaling would give 50.0; a 1/r^3 scaling
    # would give 12.5. The wrong-formula values differ by > 10x rtol.
    assert abs(scaled[0] - 50.0) > 1
    assert abs(scaled[0] - 12.5) > 1


@pytest.mark.physics_invariant
def test_scale_spectrum_to_toa_amplifies_at_close_separation():
    """At separation = 0.5 AU, the scaling factor is (1/0.5)^2 = 4: the
    flux is amplified. A regression that flipped the ratio (sep/AU)^2
    instead of (AU/sep)^2 would attenuate by 1/4 instead of amplify by 4.
    """
    fl = np.array([10.0])
    scaled = star_wrapper.scale_spectrum_to_toa(fl, 0.5 * AU)
    # Discrimination: amplification by 4x; a flipped ratio gives 2.5
    assert scaled[0] == pytest.approx(40.0, rel=1e-12)
    assert scaled[0] > 10.0  # Amplification guard
    # Wrong-ratio would give 2.5; pin the magnitude order
    assert scaled[0] > 20.0


def test_scale_spectrum_to_toa_handles_numpy_array_input():
    """A numpy array passes through ``np.array(fl_arr) * factor`` without
    type change. Discrimination: the result must be a numpy ndarray
    with the same shape as the input, not a list or scalar.
    """
    fl = np.array([1.0, 2.0, 3.0, 4.0])
    scaled = star_wrapper.scale_spectrum_to_toa(fl, AU)
    assert isinstance(scaled, np.ndarray)
    assert scaled.shape == fl.shape


@pytest.mark.physics_invariant
def test_scale_spectrum_to_toa_preserves_sign_for_zero_flux():
    """A zero-flux input array stays at zero regardless of separation:
    the inverse-square scaling factor only multiplies. Discrimination:
    if a regression added an additive offset, this test would catch it
    at every separation (here 2 AU and 0.5 AU; both must produce zero).
    """
    fl = np.array([0.0, 0.0])
    scaled_far = star_wrapper.scale_spectrum_to_toa(fl, 2.0 * AU)
    scaled_near = star_wrapper.scale_spectrum_to_toa(fl, 0.5 * AU)
    np.testing.assert_allclose(scaled_far, [0.0, 0.0], atol=1e-30)
    np.testing.assert_allclose(scaled_near, [0.0, 0.0], atol=1e-30)


# ---------------------------------------------------------------------------
# update_stellar_radius / temperature / instellation: dummy + MORS (Spada
# AND Baraffe) dispatch branches. Spada paths run on every CHILI nightly,
# so they are already covered; Baraffe paths are not exercised by the
# default smoke tier and so live entirely as unit-test mocks here.
# ---------------------------------------------------------------------------


def _make_mors_config(tracks: str = 'baraffe'):
    """Build a MagicMock config object shaped like the live tree, with
    Mors as the star module and the requested track type."""
    from unittest.mock import MagicMock

    config = MagicMock()
    config.star.module = 'mors'
    config.star.mors.tracks = tracks
    config.star.bol_scale = 1.0
    return config


def test_update_stellar_radius_baraffe_branch_calls_baraffestellarradius():
    """When the MORS track is Baraffe, the radius is read from
    ``stellar_track.BaraffeStellarRadius(age_yr)`` and stored in
    hf_row scaled by R_sun.

    Discrimination guard: the spada branch would call
    ``Value(age_Myr, 'Rstar')`` instead. Assert the Baraffe entry
    point was hit and ``Value`` was not.
    """
    from unittest.mock import MagicMock

    from proteus.star.wrapper import update_stellar_radius
    from proteus.utils.constants import R_sun

    config = _make_mors_config('baraffe')
    track = MagicMock()
    track.BaraffeStellarRadius.return_value = 0.42  # R_sun
    hf_row = {'age_star': 1.0e9}

    update_stellar_radius(hf_row, config, stellar_track=track)
    track.BaraffeStellarRadius.assert_called_once_with(1.0e9)
    assert track.Value.call_count == 0
    # SI conversion: R_sun multiplier.
    assert hf_row['R_star'] == pytest.approx(0.42 * R_sun, rel=1e-12)
    # Scale guard: half-Solar-radius star at 0.42 R_sun ~ 2.9e8 m;
    # a units bug returning R in m without the multiplication would
    # land at 0.42, eight orders of magnitude off.
    assert 1e8 < hf_row['R_star'] < 1e9


def test_update_stellar_temperature_baraffe_branch_calls_baraffestellarteff():
    """The Baraffe branch of update_stellar_temperature reads Teff
    from ``BaraffeStellarTeff(age_yr)`` and writes the value verbatim
    into hf_row.

    Discrimination: the spada path passes age in Myr and reads via
    ``Value``. Pin the Baraffe path's call signature and the resulting
    Teff value.
    """
    from unittest.mock import MagicMock

    from proteus.star.wrapper import update_stellar_temperature

    config = _make_mors_config('baraffe')
    track = MagicMock()
    track.BaraffeStellarTeff.return_value = 3200.0
    hf_row = {'age_star': 2.5e9}

    update_stellar_temperature(hf_row, config, stellar_track=track)
    track.BaraffeStellarTeff.assert_called_once_with(2.5e9)
    assert track.Value.call_count == 0
    assert hf_row['T_star'] == pytest.approx(3200.0, rel=1e-12)
    # Sign + physical-range guard.
    assert hf_row['T_star'] > 0
    assert 2000.0 < hf_row['T_star'] < 8000.0


@pytest.mark.physics_invariant
def test_update_instellation_baraffe_branch_uses_baraffesolarconstant_and_zeros_xuv():
    """The Baraffe branch of update_instellation calls
    ``stellar_track.BaraffeSolarConstant(age_yr, sep_in_AU)`` and
    sets F_xuv = 0 because Baraffe tracks do not provide XUV.

    Discrimination: the spada branch would use a different signature
    (Lbol/(4*pi*d^2) from Value('Lbol')) and would produce a finite
    F_xuv. Pin both: F_ins == BaraffeSolarConstant return, F_xuv == 0.
    """
    from unittest.mock import MagicMock

    from proteus.star.wrapper import update_instellation
    from proteus.utils.constants import AU

    config = _make_mors_config('baraffe')
    track = MagicMock()
    track.BaraffeSolarConstant.return_value = 1361.0
    hf_row = {'age_star': 4.567e9, 'separation': 1.0 * AU}

    update_instellation(hf_row, config, stellar_track=track)
    # Was passed as (age_yr, sep/AU); confirm sep/AU == 1.0.
    track.BaraffeSolarConstant.assert_called_once_with(4.567e9, 1.0)
    assert hf_row['F_ins'] == pytest.approx(1361.0, rel=1e-12)
    # XUV explicitly zero for Baraffe (no XUV in those tracks).
    assert hf_row['F_xuv'] == pytest.approx(0.0, abs=1e-12)


def test_update_instellation_dummy_branch_zeroes_fxuv_and_computes_finstellation():
    """The dummy-star path of update_instellation calls
    ``star.dummy.calc_instellation(Teff, R_star, sep)`` and explicitly
    sets F_xuv = 0 (dummy has no XUV model).

    Discrimination: confirm the dummy entry point was called with the
    Teff + R_star + sep passed in (no implicit unit conversion), and
    that F_xuv is exactly 0.0.
    """
    from unittest.mock import MagicMock, patch

    from proteus.star.wrapper import update_instellation

    config = MagicMock()
    config.star.module = 'dummy'
    config.star.dummy.Teff = 5778.0
    config.star.bol_scale = 1.0
    hf_row = {'R_star': 6.957e8, 'separation': 1.496e11}

    with patch('proteus.star.dummy.calc_instellation', return_value=1361.0) as mock_inst:
        update_instellation(hf_row, config)
    mock_inst.assert_called_once_with(5778.0, 6.957e8, 1.496e11)
    assert hf_row['F_ins'] == pytest.approx(1361.0, rel=1e-12)
    assert hf_row['F_xuv'] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_update_equilibrium_temperature_pins_stefan_boltzmann_closed_form():
    """T_eqm = ((1 - albedo) * S * s0_factor / sigma) ** 0.25.

    Discriminating: at S = 1361 W/m^2 and albedo = 0.3 the closed-form
    T_eqm is ~254 K. A regression to the wrong exponent (Stefan-
    Boltzmann is T^4, not T^3 or T^5) would land at ~1613 K or ~84 K
    respectively. Pin the value to 254 K with a clear tolerance and
    add explicit exponent guards.
    """
    from unittest.mock import MagicMock

    from proteus.star.wrapper import update_equilibrium_temperature
    from proteus.utils.constants import const_sigma

    config = MagicMock()
    config.orbit.s0_factor = 0.25  # disk-averaged absorption factor
    hf_row = {
        'F_ins': 1361.0,
        'albedo_pl': 0.3,
    }
    update_equilibrium_temperature(hf_row, config)
    F_asf = 1361.0 * 0.25 * (1 - 0.3)
    expected = (F_asf / const_sigma) ** 0.25
    assert hf_row['T_eqm'] == pytest.approx(expected, rel=1e-12)
    # Closed-form value pin: ~254 K for Earth-like albedo + insolation.
    assert hf_row['T_eqm'] == pytest.approx(254.0, abs=2.0)
    # Exponent guard: T^4 -> ~254 K. T^3 would give ~1613 K; T^5
    # would give ~84 K. Both are well outside any plausible tolerance.
    wrong_cube_root = (F_asf / const_sigma) ** (1.0 / 3.0)
    wrong_fifth_root = (F_asf / const_sigma) ** (1.0 / 5.0)
    assert abs(hf_row['T_eqm'] - wrong_cube_root) > 50
    assert abs(hf_row['T_eqm'] - wrong_fifth_root) > 50


@pytest.mark.physics_invariant
def test_update_stellar_radius_spada_branch_calls_value():
    """When the MORS track is Spada, the radius is read from
    ``stellar_track.Value(age_Myr, 'Rstar')`` (NOT the Baraffe path).

    Discrimination: the Baraffe branch calls BaraffeStellarRadius;
    assert Value was hit and BaraffeStellarRadius was not.
    """
    from unittest.mock import MagicMock

    from proteus.star.wrapper import update_stellar_radius
    from proteus.utils.constants import R_sun

    config = _make_mors_config('spada')
    track = MagicMock()
    track.Value.return_value = 0.85  # R_sun
    hf_row = {'age_star': 2.0e9}

    update_stellar_radius(hf_row, config, stellar_track=track)
    track.Value.assert_called_once_with(2.0e9 / 1e6, 'Rstar')
    assert track.BaraffeStellarRadius.call_count == 0
    assert hf_row['R_star'] == pytest.approx(0.85 * R_sun, rel=1e-12)
    # Scale guard
    assert 1e8 < hf_row['R_star'] < 2e9


@pytest.mark.physics_invariant
def test_update_stellar_temperature_spada_branch_calls_value():
    """Spada branch of update_stellar_temperature reads Teff from
    ``Value(age_Myr, 'Teff')``.

    Discrimination: pin the Spada call signature (age in Myr, key
    'Teff') and verify the Baraffe path was not called.
    """
    from unittest.mock import MagicMock

    from proteus.star.wrapper import update_stellar_temperature

    config = _make_mors_config('spada')
    track = MagicMock()
    track.Value.return_value = 5778.0
    hf_row = {'age_star': 4.567e9}

    update_stellar_temperature(hf_row, config, stellar_track=track)
    track.Value.assert_called_once_with(4.567e9 / 1e6, 'Teff')
    assert track.BaraffeStellarTeff.call_count == 0
    assert hf_row['T_star'] == pytest.approx(5778.0, rel=1e-12)
    assert hf_row['T_star'] > 0
