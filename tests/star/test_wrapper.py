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

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30), pytest.mark.physics_invariant]


# ---------------------------------------------------------------------------
# scale_spectrum_to_toa: inverse-square distance law
# ---------------------------------------------------------------------------


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
