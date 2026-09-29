"""Unit tests for the pure-Python helpers in ``proteus.star.wrapper``.

Targets the small inverse-square scaling helper, the orbital average
behind the instellation, and the spectrum-write helper. Heavier dispatch
functions that drive MORS / Spada tracks are covered by integration tests
in nightly tier.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
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
    hf_row = {
        'age_star': 4.567e9,
        'separation': 1.0 * AU,
        'semimajorax': 1.0 * AU,
        'eccentricity': 0.0,
    }

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
    hf_row = {
        'R_star': 6.957e8,
        'separation': 1.496e11,
        'semimajorax': 1.496e11,
        'eccentricity': 0.0,
    }

    with patch('proteus.star.dummy.calc_instellation', return_value=1361.0) as mock_inst:
        update_instellation(hf_row, config)
    mock_inst.assert_called_once_with(5778.0, 6.957e8, 1.496e11)
    assert hf_row['F_ins'] == pytest.approx(1361.0, rel=1e-12)
    assert hf_row['F_xuv'] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# update_instellation: bolometric-scaling time window (bol_scale_start /
# bol_scale_duration). These gate the pre-existing `bol_scale` factor to a
# stellar-age window [start, start + duration] instead of applying it for
# the whole run.
# ---------------------------------------------------------------------------


def _dummy_config_for_bolscale(bol_scale, bol_scale_start, bol_scale_duration):
    """Dummy-star config with the three bolometric-scaling fields set."""
    from unittest.mock import MagicMock

    config = MagicMock()
    config.star.module = 'dummy'
    config.star.dummy.Teff = 5778.0
    config.star.bol_scale = bol_scale
    config.star.bol_scale_start = bol_scale_start
    config.star.bol_scale_duration = bol_scale_duration
    return config


def _run_update_instellation_dummy(config, age_star, s0_return=1361.0):
    from unittest.mock import patch

    from proteus.star.wrapper import update_instellation

    hf_row = {
        'R_star': 6.957e8,
        'separation': 1.496e11,
        'semimajorax': 1.496e11,
        'eccentricity': 0.0,
        'age_star': age_star,
    }
    with patch('proteus.star.dummy.calc_instellation', return_value=s0_return):
        update_instellation(hf_row, config)
    return hf_row


@pytest.mark.physics_invariant
def test_update_instellation_applies_scaling_inside_window():
    """Inside [bol_scale_start, bol_scale_start + bol_scale_duration], F_ins
    and F_xuv are multiplied by bol_scale and hf_row['bol_scale'] records
    the applied factor.

    Window: start=0.5 Gyr, duration=0.5 Gyr -> [0.5e9, 1.0e9] yr.
    age_star=0.7e9 yr sits inside it.
    """
    config = _dummy_config_for_bolscale(
        bol_scale=2.0, bol_scale_start=0.5, bol_scale_duration=0.5
    )
    hf_row = _run_update_instellation_dummy(config, age_star=0.7e9)
    assert hf_row['F_ins'] == pytest.approx(1361.0 * 2.0, rel=1e-12)
    assert hf_row['bol_scale'] == pytest.approx(2.0)


@pytest.mark.physics_invariant
def test_update_instellation_no_scaling_before_window():
    """age_star before bol_scale_start: scaling not yet active."""
    config = _dummy_config_for_bolscale(
        bol_scale=2.0, bol_scale_start=0.5, bol_scale_duration=0.5
    )
    hf_row = _run_update_instellation_dummy(config, age_star=0.4e9)
    assert hf_row['F_ins'] == pytest.approx(1361.0, rel=1e-12)
    assert hf_row['bol_scale'] == pytest.approx(1.0)


@pytest.mark.physics_invariant
def test_update_instellation_no_scaling_after_window():
    """age_star after bol_scale_start + bol_scale_duration: scaling has
    already ended."""
    config = _dummy_config_for_bolscale(
        bol_scale=2.0, bol_scale_start=0.5, bol_scale_duration=0.5
    )
    hf_row = _run_update_instellation_dummy(config, age_star=1.1e9)
    assert hf_row['F_ins'] == pytest.approx(1361.0, rel=1e-12)
    assert hf_row['bol_scale'] == pytest.approx(1.0)


@pytest.mark.physics_invariant
def test_update_instellation_window_is_half_open():
    """The window is half-open, [start, start + duration): age_star exactly
    equal to the start age applies scaling, but age_star exactly equal to the
    end age must not apply it.
    """
    config = _dummy_config_for_bolscale(
        bol_scale=3.0, bol_scale_start=0.5, bol_scale_duration=0.5
    )
    hf_row_start = _run_update_instellation_dummy(config, age_star=0.5e9)
    assert hf_row_start['bol_scale'] == pytest.approx(3.0)

    # Discrimination: exactly at the end boundary must already be unscaled,
    # not merely "just past" it
    hf_row_end = _run_update_instellation_dummy(config, age_star=1.0e9)
    assert hf_row_end['bol_scale'] == pytest.approx(1.0)

    hf_row_before_end = _run_update_instellation_dummy(config, age_star=1.0e9 - 1.0)
    assert hf_row_before_end['bol_scale'] == pytest.approx(3.0)


@pytest.mark.physics_invariant
def test_update_instellation_zero_duration_is_no_op():
    """bol_scale_duration=0.0 collapses the window to a single instant, so
    the half-open upper edge makes even age_star exactly equal to
    bol_scale_start fall outside [start, start) and scaling never applies.
    """
    config = _dummy_config_for_bolscale(
        bol_scale=5.0, bol_scale_start=0.5, bol_scale_duration=0.0
    )
    hf_row = _run_update_instellation_dummy(config, age_star=0.5e9)
    assert hf_row['F_ins'] == pytest.approx(1361.0, rel=1e-12)
    assert hf_row['bol_scale'] == pytest.approx(1.0)


@pytest.mark.physics_invariant
def test_update_instellation_disabled_when_bol_scale_start_is_none():
    """bol_scale_start=None disables scaling regardless of bol_scale.

    Enabling condition (not None) requires bol_scale_start to be set.
    """
    config = _dummy_config_for_bolscale(
        bol_scale=2.0, bol_scale_start=None, bol_scale_duration=0.0
    )
    hf_row = _run_update_instellation_dummy(config, age_star=0.7e9)
    assert hf_row['F_ins'] == pytest.approx(1361.0, rel=1e-12)
    assert hf_row['bol_scale'] == pytest.approx(1.0)


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


# ---------------------------------------------------------------------------
# scale_spectrum_to_stellar_surface: inverse of the TOA scaling
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_scale_spectrum_to_stellar_surface_recovers_surface_flux_regardless_of_separation():
    """Scaling a 1 AU spectrum out to the planet and back to the stellar surface
    returns the surface flux, and the recovered value does not depend on the
    orbital separation: the separation that scale_spectrum_to_toa introduces is
    exactly undone by scale_spectrum_to_stellar_surface.

    Discrimination: the round trip is run at two separations two decades apart
    (0.1 AU and 10 AU). The rescale uses (sep / r_star) ** 2; an inverted ratio
    (r_star / sep) ** 2 or a dropped square would leave a residual separation
    dependence, so the two recovered spectra would differ by many decades instead
    of matching. Asserting they both equal the surface flux, and each other, rules
    those out.
    """
    from proteus.star.wrapper import (
        scale_spectrum_to_stellar_surface,
        scale_spectrum_to_toa,
    )
    from proteus.utils.constants import R_sun

    f_surface = np.array([10.0, 20.0, 40.0])
    r_star = 0.9 * R_sun
    # Flux at 1 AU is the surface flux diluted by (r_star / AU) ** 2.
    f_1au = f_surface * (r_star / AU) ** 2

    recovered = []
    for sep in (0.1 * AU, 10.0 * AU):
        f_planet = scale_spectrum_to_toa(f_1au, sep)
        recovered.append(scale_spectrum_to_stellar_surface(f_planet, sep, r_star))

    np.testing.assert_allclose(recovered[0], f_surface, rtol=1e-12)
    np.testing.assert_allclose(recovered[1], f_surface, rtol=1e-12)
    # Separation cancels: an inverted rescale would put these 1e8 apart.
    np.testing.assert_allclose(recovered[0], recovered[1], rtol=1e-12)


# ---------------------------------------------------------------------------
# write_spectrum: the .sflux file stays single-header-line for skiprows=1 readers
# ---------------------------------------------------------------------------


def test_write_spectrum_emits_a_single_header_line(tmp_path):
    """write_spectrum writes exactly one header line, so every consumer that skips
    one row (AGNI, VULCAN, the eclipse loader, the cpl_sflux plotters) reads the
    data unshifted. The written values round-trip through np.loadtxt(skiprows=1).

    Discrimination: a regression that emitted a second header line, or embedded a
    newline in the header string, would make skiprows=1 consume a header line as
    the first data row. That would either raise or return the wrong first
    wavelength, so pinning an exact one-line header and a value-preserving round
    trip (with the true first wavelength recovered) catches it.
    """
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    wl = np.array([100.0, 200.0, 300.0])
    fl = np.array([1.0e3, 2.0e3, 4.0e3])
    hf_row = {'age_star': 4.567e9, 'Time': 42}

    star_wrapper.write_spectrum(wl, fl, hf_row, str(tmp_path))

    path = data_dir / '42.sflux'
    lines = path.read_text().splitlines()
    header_lines = [ln for ln in lines if ln.lstrip().startswith('#')]
    # Exactly one header line, and it is the first line.
    assert len(header_lines) == 1
    assert lines[0].startswith('#')
    # skiprows=1 recovers the data exactly, including the true first wavelength
    # (a second header line would shift it to 200.0 or raise).
    recovered = np.loadtxt(path, skiprows=1).T
    np.testing.assert_allclose(recovered[0], wl, rtol=1e-12)
    np.testing.assert_allclose(recovered[1], fl, rtol=1e-12)


# ---------------------------------------------------------------------------
# flux_weighted_distance: the orbital average behind the instellation
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_flux_weighted_distance_is_the_semimajor_axis_on_a_circular_orbit():
    """Edge case e = 0, the fixed point of the (1 - e^2)^(1/4) factor:
    a circular orbit has nothing to average, so the flux-weighted
    distance is the semi-major axis itself. A row that omits the
    eccentricity is treated as circular rather than producing nan.
    """
    circular = star_wrapper.flux_weighted_distance(
        {'semimajorax': 1.496e11, 'eccentricity': 0.0}
    )
    assert circular == pytest.approx(1.496e11, rel=1e-12)
    assert circular > 0.0

    absent = star_wrapper.flux_weighted_distance({'semimajorax': 1.496e11})
    assert absent == pytest.approx(circular, rel=1e-12)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_flux_weighted_distance_pins_the_orbit_averaged_inverse_square_law():
    """Analytical limit: averaging 1/r^2 over one orbit gives
    1 / (a^2 sqrt(1 - e^2)), so the distance carrying the same flux is
    a (1 - e^2)^(1/4). At e = 0.8 that is a * 0.36^(1/4) = 0.774597 a.

    The competing average, the time-averaged separation a (1 + e^2 / 2),
    gives 1.32 a instead. The two differ in the resulting flux by a
    factor of 2.904 here, which is what makes this eccentricity
    discriminating.
    """
    sma = 1.0e11
    dist = star_wrapper.flux_weighted_distance({'semimajorax': sma, 'eccentricity': 0.8})

    assert dist == pytest.approx(sma * 0.36**0.25, rel=1e-12)
    # Exponent guard: the time-averaged separation lands at 1.32 a.
    assert abs(dist - sma * 1.32) > 0.3 * sma
    # The flux ratio between the two averages is what matters downstream.
    assert (sma * 1.32 / dist) ** 2 == pytest.approx(2.9040, rel=1e-3)
    # Sign and scale guards: an eccentric orbit is effectively closer in,
    # so the distance is positive and shorter than the semi-major axis.
    assert 0.0 < dist < sma


@pytest.mark.physics_invariant
def test_spectrum_normalisation_tracks_the_instellation_it_carries():
    """The broadband spectrum handed to the climate module and the scalar
    instellation derived from the same star must share one orbital
    average, or the energy the atmosphere receives disagrees with the
    number the rest of the run reports.

    Composing scale_spectrum_to_toa with flux_weighted_distance has to
    reproduce the eccentricity dependence of F_ins exactly: both scale as
    1 / sqrt(1 - e^2), which at e = 0.8 is 1 / 0.6. Scaling the spectrum
    by the time-averaged separation instead would give 1 / 1.7424, a
    factor of 2.904 low.
    """
    from unittest.mock import MagicMock

    luminosities = {'Lbol': 3.828e33, 'Lx': 1.0e29, 'Leuv': 2.0e29}
    config = _make_mors_config('spada')
    track = MagicMock()
    track.Value.side_effect = lambda age_myr, key: luminosities[key]

    sma = 1.496e11
    flux_at_1au = np.array([1.0, 2.0, 3.0])
    scaled = {}
    instellation = {}
    for ecc in (0.0, 0.8):
        hf_row = {
            'age_star': 4.6e9,
            'semimajorax': sma,
            'eccentricity': ecc,
            'separation': sma * (1.0 + 0.5 * ecc**2),
        }
        star_wrapper.update_instellation(hf_row, config, stellar_track=track)
        instellation[ecc] = hf_row['F_ins']
        scaled[ecc] = star_wrapper.scale_spectrum_to_toa(
            flux_at_1au, star_wrapper.flux_weighted_distance(hf_row)
        )

    spectrum_ratio = scaled[0.8] / scaled[0.0]
    flux_ratio = instellation[0.8] / instellation[0.0]
    np.testing.assert_allclose(spectrum_ratio, flux_ratio, rtol=1e-12)
    assert flux_ratio == pytest.approx(1.0 / 0.6, rel=1e-12)
    # The competing average would have landed 2.904 times lower, so the
    # equality above is not satisfied by both conventions at once.
    assert abs(flux_ratio - 1.0 / 1.7424) > 1.0
    assert np.all(scaled[0.8] > scaled[0.0])


@pytest.mark.parametrize('ecc', [1.0, 1.5, -0.1], ids=['parabolic', 'hyperbolic', 'negative'])
def test_flux_weighted_distance_rejects_an_unbound_orbit(ecc):
    """An eccentricity at or beyond 1 has no bound-orbit average and
    would put a negative number under the fourth root. The helper reports
    it instead of returning nan. This is an error-contract test, so it
    asserts the contract rather than a physical invariant."""
    with pytest.raises(ValueError) as excinfo:
        star_wrapper.flux_weighted_distance({'semimajorax': 1.0e11, 'eccentricity': ecc})

    message = str(excinfo.value)
    assert 'Eccentricity' in message
    assert str(float(ecc)) in message


@pytest.mark.physics_invariant
def test_update_instellation_spada_shares_one_average_for_bolometric_and_xuv():
    """The Spada branch derives both fluxes from the same orbital
    average, so their ratio is fixed by the luminosity ratio alone and
    cannot depend on the eccentricity. The bolometric flux is pinned
    against the closed-form inverse-square value on a circular orbit and
    against 1 / sqrt(1 - e^2) times it at e = 0.8; a pair of mismatched
    averages would move the ratio by a factor of 2.9 at that
    eccentricity.
    """
    from unittest.mock import MagicMock

    luminosities = {'Lbol': 3.828e33, 'Lx': 1.0e29, 'Leuv': 2.0e29}

    config = _make_mors_config('spada')
    track = MagicMock()
    track.Value.side_effect = lambda age_myr, key: luminosities[key]

    sma = 1.496e11
    fluxes = {}
    for ecc in (0.0, 0.8):
        # The row also carries the time-averaged separation, so a
        # regression reverting either flux to it produces a wrong number
        # rather than a KeyError.
        hf_row = {
            'age_star': 4.6e9,
            'semimajorax': sma,
            'eccentricity': ecc,
            'separation': sma * (1.0 + 0.5 * ecc**2),
        }
        star_wrapper.update_instellation(hf_row, config, stellar_track=track)
        fluxes[ecc] = (hf_row['F_ins'], hf_row['F_xuv'])

    s0_circular = luminosities['Lbol'] * 1e-7 / (4.0 * np.pi * sma**2)
    assert fluxes[0.0][0] == pytest.approx(s0_circular, rel=1e-12)
    # Eccentric orbit: brighter by 1 / sqrt(1 - 0.64) = 1 / 0.6.
    assert fluxes[0.8][0] == pytest.approx(s0_circular / 0.6, rel=1e-12)
    assert fluxes[0.8][1] / fluxes[0.8][0] == pytest.approx(
        fluxes[0.0][1] / fluxes[0.0][0], rel=1e-12
    )
    # Absolute XUV pin, because the ratio above survives any uniform
    # prefactor error in the cgs-to-SI chain. Lx + Leuv = 3.0e29 erg/s
    # over 4 pi (1.496e13 cm)^2 = 2.81236e27 cm^2 gives 106.6714
    # erg/s/cm^2, and 1 erg/s/cm^2 is 1e-3 W/m^2.
    assert fluxes[0.0][1] == pytest.approx(0.1066714, rel=1e-6)
    assert fluxes[0.8][1] == pytest.approx(0.1066714 / 0.6, rel=1e-6)
    # Scale guard: dropping the metre-to-centimetre factor lands at
    # 1066.7 and dropping the unit conversion lands at 106.67.
    assert 0.01 < fluxes[0.0][1] < 1.0
    assert fluxes[0.8][0] > fluxes[0.0][0] > 0.0
    assert fluxes[0.8][1] > fluxes[0.0][1] > 0.0
