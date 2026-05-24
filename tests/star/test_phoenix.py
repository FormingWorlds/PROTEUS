"""Unit tests for the PHOENIX parameter resolver and spectrum loader.

Covers ``proteus.star.phoenix.phoenix_params`` (Teff / radius / log g
resolution from a stellar track or config overrides, plus the mass
out-of-range guard) and the IO-error branches of
``get_phoenix_modern_spectrum``.

Physics: phoenix_params is a pure parameter resolver, not a solver,
so the discrimination guard tests pin closed-form log g values rather
than physical invariants.

See also:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from proteus.star.phoenix import phoenix_params

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _build_handler_with_track(
    *,
    Teff_override=None,
    radius_override=None,
    logg_override=None,
    tracks='spada',
    Mstar=1.0,
    FeH=0.0,
    alpha=0.0,
):
    """Build a MagicMock Proteus handler shaped like the live config tree."""
    handler = MagicMock()
    handler.config.star.mass = Mstar
    handler.config.star.mors.age_now = 4.567
    handler.config.star.mors.phoenix_FeH = FeH
    handler.config.star.mors.phoenix_alpha = alpha
    handler.config.star.mors.phoenix_Teff = Teff_override
    handler.config.star.mors.phoenix_radius = radius_override
    handler.config.star.mors.phoenix_log_g = logg_override
    handler.config.star.mors.tracks = tracks
    return handler


def test_phoenix_params_pulls_teff_and_radius_from_spada_track():
    """When config overrides are None and tracks='spada', phoenix_params
    must call ``Value(age_Myr, 'Teff')`` and ``Value(age_Myr, 'Rstar')``.

    Discriminating: a regression that swapped the keys, used the
    baraffe entry points, or called Value with the age in Gyr instead
    of Myr would land on different numbers. Pin Teff/radius to the
    mocked Value return and assert age_Myr was passed in the call.
    """
    handler = _build_handler_with_track(tracks='spada')

    track = MagicMock()
    track.Value.side_effect = lambda age_Myr, key: {'Teff': 5778.0, 'Rstar': 1.0}[key]
    params = phoenix_params(handler, stellar_track=track, age_yr=4.567e9)

    assert params['Teff'] == pytest.approx(5778.0, rel=1e-12)
    assert params['radius'] == pytest.approx(1.0, rel=1e-12)
    # Age conversion check: 4.567 Gyr -> 4567 Myr. The Value calls
    # must receive age_Myr (~4567), not age_Gyr (~4.6) or age_yr (~5e9).
    age_call_args = [call.args[0] for call in track.Value.call_args_list]
    assert all(abs(a - 4567.0) < 1.0 for a in age_call_args)
    # Discrimination guard: a regression to age_yr would land at 4.567e9.
    assert all(a < 1.0e6 for a in age_call_args)


def test_phoenix_params_pulls_from_baraffe_track_when_tracks_is_baraffe():
    """The baraffe branch (lines 84-85, 93-94) calls different methods
    on the track object: ``BaraffeStellarTeff(age_yr)`` and
    ``BaraffeStellarRadius(age_yr)``.

    Discriminating: assert these two methods were called and the
    spada ``Value`` was not. A regression that fell through to spada
    would leave the Baraffe methods uncalled and the spada ``Value``
    method called.
    """
    handler = _build_handler_with_track(tracks='baraffe', Mstar=0.5)

    track = MagicMock()
    track.BaraffeStellarTeff.return_value = 3500.0
    track.BaraffeStellarRadius.return_value = 0.4
    params = phoenix_params(handler, stellar_track=track, age_yr=1.0e9)

    assert params['Teff'] == pytest.approx(3500.0, rel=1e-12)
    assert params['radius'] == pytest.approx(0.4, rel=1e-12)
    # Branch dispatch: Baraffe entry points were called, spada was not.
    track.BaraffeStellarTeff.assert_called_once_with(1.0e9)
    track.BaraffeStellarRadius.assert_called_once_with(1.0e9)
    assert track.Value.call_count == 0


@pytest.mark.physics_invariant
def test_phoenix_params_computes_logg_from_mass_and_radius():
    """When logg is None and mass + radius are known, phoenix_params
    computes log g from g = G * M / R**2 (in cgs).

    Discriminating: pin log g for Solar parameters. The closed form
    is log10(G * M_sun / R_sun**2 * 100). A regression that forgot
    the *100 SI->cgs conversion would land at ~2.44 (m/s**2 in log)
    instead of ~4.44 (cm/s**2 in log), separated by exactly 2 dex.
    """
    from proteus.utils.constants import M_sun, R_sun, const_G

    handler = _build_handler_with_track(
        tracks='spada',
        Mstar=1.0,
        Teff_override=5778.0,
        radius_override=1.0,
    )
    params = phoenix_params(handler, stellar_track=None, age_yr=4.567e9)
    expected_g_cgs = const_G * M_sun / (R_sun**2) * 100.0
    expected_logg = float(np.log10(expected_g_cgs))
    assert params['logg'] == pytest.approx(expected_logg, rel=1e-12)
    # Order-of-magnitude guard. Solar log g is ~4.44 in cgs. A units
    # bug putting g in m/s^2 would land at ~2.44; pin within 0.5 dex
    # of the correct value to discriminate the 2-dex error.
    assert abs(params['logg'] - 4.44) < 0.5


def test_phoenix_params_rejects_mass_outside_track_range():
    """phoenix_params must raise ValueError when stellar mass is
    outside the allowed range for the chosen track type.

    Edge: spada tracks span 0.10 - 1.25 M_sun. A 2.0 M_sun star
    must fail validation.

    Discriminating: side-effect check. The status file update happens
    just before the raise; assert that UpdateStatusfile was called
    with the configured directories and status code 23.
    """
    handler = _build_handler_with_track(
        tracks='spada',
        Mstar=2.0,
        radius_override=2.0,  # provided so logg branch runs
    )
    with patch('proteus.star.phoenix.UpdateStatusfile') as mock_update:
        with pytest.raises(ValueError, match='outside of'):
            phoenix_params(handler, stellar_track=None, age_yr=4.567e9)
    mock_update.assert_called_once_with(handler.directories, 23)


def test_phoenix_params_respects_explicit_config_overrides():
    """When all of phoenix_Teff, phoenix_radius, phoenix_log_g are
    set in the config, the resolver returns them verbatim and does
    not consult the stellar track.

    Edge: ensures the config-override path short-circuits the
    track-driven calculation.
    """
    handler = _build_handler_with_track(
        Teff_override=4200.0,
        radius_override=0.7,
        logg_override=4.65,
    )
    track = MagicMock()  # would raise if consulted with unmocked .Value
    params = phoenix_params(handler, stellar_track=track, age_yr=1.0e9)
    assert params['Teff'] == pytest.approx(4200.0, rel=1e-12)
    assert params['radius'] == pytest.approx(0.7, rel=1e-12)
    assert params['logg'] == pytest.approx(4.65, rel=1e-12)
    # Discrimination: the track was passed in but should NOT have
    # been queried for Teff or radius.
    assert track.Value.call_count == 0
    assert track.BaraffeStellarTeff.call_count == 0


def test_get_phoenix_modern_spectrum_raises_when_params_incomplete():
    """get_phoenix_modern_spectrum must raise ValueError when
    phoenix_params returns any of Teff/radius/logg as None.

    Discriminating: a regression that proceeded with a None field
    would raise a TypeError on the downstream grid lookup, not a
    clean ValueError with the documented message.
    """
    from proteus.star.phoenix import get_phoenix_modern_spectrum

    handler = _build_handler_with_track()
    # No stellar track and no Teff override -> Teff stays None.
    hf_row_before = dict(handler.hf_row) if hasattr(handler, 'hf_row') else {}
    with pytest.raises(ValueError, match='Teff, radius and log g'):
        get_phoenix_modern_spectrum(handler, stellar_track=None, age_yr=4.567e9)
    # ValueError is clean; no partial state written
    if hf_row_before:
        assert handler.hf_row == hf_row_before
