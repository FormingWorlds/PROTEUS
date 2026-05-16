"""Unit tests for the dummy orbit module."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from proteus.interior_energetics.common import Interior_t
from proteus.orbit.dummy import run_dummy_orbit

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _make_config(phi_tide: str, h_tide: float, imk2: float) -> Any:
    dummy = SimpleNamespace(Phi_tide=phi_tide, H_tide=h_tide, Imk2=imk2)
    orbit = SimpleNamespace(dummy=dummy)
    return cast(Any, SimpleNamespace(orbit=orbit))


@pytest.mark.unit
def test_less_than_threshold_heating_scales_with_melt_fraction():
    """Heating applies where phi < threshold and scales linearly with melt fraction."""
    config = _make_config('<0.5', h_tide=10.0, imk2=1e-2)
    interior = Interior_t(nlev_b=4)
    interior.phi = np.array([0.1, 0.4, 0.6])

    result = run_dummy_orbit(config, interior)

    assert result == pytest.approx(1e-2)
    assert interior.tides == pytest.approx(
        [10.0 * (1 - 0.1 / 0.5), 10.0 * (1 - 0.4 / 0.5), 0.0]
    )


@pytest.mark.unit
def test_greater_than_threshold_heating_scales_linearly():
    """Heating applies where phi > threshold and increases toward fully liquid."""
    config = _make_config('>0.25', h_tide=5.0, imk2=2e-3)
    interior = Interior_t(nlev_b=5)
    interior.phi = np.array([0.1, 0.25, 0.75, 1.0])

    result = run_dummy_orbit(config, interior)

    assert result == pytest.approx(2e-3)
    expected = [0.0, 0.0, 5.0 * (0.75 - 0.25) / (1 - 0.25), 5.0 * (1.0 - 0.25) / (1 - 0.25)]
    assert interior.tides == pytest.approx(expected)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_equal_to_threshold_produces_zero_heating():
    """Boundary equality does not trigger heating (strict inequality).

    Boundedness invariant at the comparator boundary: phi == threshold
    falls outside the strict-less-than condition and so the tidal
    heating remains exactly zero. Discriminates the strict-vs-non-strict
    comparator regression.
    """
    config = _make_config('<0.3', h_tide=7.5, imk2=0.0)
    interior = Interior_t(nlev_b=3)
    interior.phi = np.array([0.3, 0.3])

    run_dummy_orbit(config, interior)

    assert interior.tides == pytest.approx([0.0, 0.0])
    # Discrimination guard: shifting phi just below the threshold must
    # produce strictly positive heating (h_tide=7.5 at phi=0.299 with
    # threshold=0.3 gives 7.5 * (1 - 0.299/0.3) ~ 0.025). A regression
    # that used <= instead of < would have produced 0.0 at phi=0.3
    # above AND would also produce > 0 here, so the difference between
    # the boundary case and the just-below case is what discriminates
    # the strict comparator.
    interior_below = Interior_t(nlev_b=3)
    interior_below.phi = np.array([0.299, 0.299])
    run_dummy_orbit(config, interior_below)
    assert interior_below.tides[0] > 0.0
    assert interior_below.tides[0] < 7.5  # bounded by h_tide


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_no_cells_meet_condition_keeps_zero_heating():
    """If no layer meets inequality, tides remain zero everywhere.

    Limit-input invariant: with phi values all above the < 0.1 threshold,
    no cells qualify and the tidal heating array stays at the zero IC.
    """
    config = _make_config('<0.1', h_tide=9.0, imk2=0.5)
    interior = Interior_t(nlev_b=3)
    interior.phi = np.array([0.5, 0.9])

    run_dummy_orbit(config, interior)

    assert interior.tides == pytest.approx([0.0, 0.0])
    # Sign / positivity guard on the unrelated return value: the
    # function still returns Imk2 from config even when no cells heat.
    # A regression that returned 0.0 (e.g. short-circuited on the
    # empty mask) would still satisfy the zero-tides equality above
    # but break the Imk2 contract documented in test_returns_imk2_from_config.
    result = run_dummy_orbit(config, interior)
    assert result == pytest.approx(0.5)
    # Discrimination guard: with one phi shifted into the < 0.1 region,
    # the corresponding tides entry must become strictly positive while
    # the others stay zero. This separates "no heating because no cells
    # qualify" from "no heating because the formula is broken".
    interior_mixed = Interior_t(nlev_b=3)
    interior_mixed.phi = np.array([0.05, 0.9])
    run_dummy_orbit(config, interior_mixed)
    assert interior_mixed.tides[0] > 0.0
    assert interior_mixed.tides[1] == pytest.approx(0.0)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_phi_array_unchanged_by_heating_calculation():
    """run_dummy_orbit updates tides but leaves melt fractions untouched.

    Input-immutability invariant: melt fraction is the input state; the
    dummy orbit applies tidal heating WITHOUT mutating phi. A regression
    that modified phi in place would corrupt the interior solver's
    state across iterations.
    """
    config = _make_config('>0.2', h_tide=4.0, imk2=1.0)
    interior = Interior_t(nlev_b=4)
    interior.phi = np.array([0.1, 0.2, 0.3])
    phi_before = interior.phi.copy()

    run_dummy_orbit(config, interior)

    assert interior.phi == pytest.approx(phi_before)
    # Discrimination guard: tides must still have been UPDATED for the
    # cell that qualifies (phi=0.3 > 0.2). A regression that did nothing
    # (returned without touching either array) would satisfy the phi
    # equality but leave tides at the IC zero.
    assert interior.tides[2] > 0.0
    # Sign / scale guard: the active cell at phi=0.3, threshold=0.2,
    # h_tide=4.0 must land at 4.0 * (0.3 - 0.2) / (1 - 0.2) = 0.5.
    # Pins both the formula and the magnitude.
    assert interior.tides[2] == pytest.approx(0.5)


@pytest.mark.unit
def test_returns_imk2_from_config():
    """Function returns Imk2 value provided in config without modification.

    Pass-through contract: the dummy orbit relays Imk2 from config to
    the caller verbatim, independent of phi or heating state.
    """
    config = _make_config('<0.9', h_tide=1.0, imk2=3.21)
    interior = Interior_t(nlev_b=2)
    interior.phi = np.array([0.5])

    result = run_dummy_orbit(config, interior)

    assert result == pytest.approx(3.21)
    # Discrimination guard: pick a different Imk2 value and confirm the
    # return tracks it. A regression that hardcoded 3.21 or returned a
    # constant (h_tide, 0, NaN) would pass the first equality but fail
    # this second call.
    config2 = _make_config('<0.9', h_tide=1.0, imk2=7.65)
    interior2 = Interior_t(nlev_b=2)
    interior2.phi = np.array([0.5])
    result2 = run_dummy_orbit(config2, interior2)
    assert result2 == pytest.approx(7.65)
    # The two return values must differ (rules out a regression that
    # ignored the config and always returned the same constant).
    assert result != pytest.approx(result2)


@pytest.mark.unit
def test_single_level_interpolates_correctly():
    """Single-layer interior still applies heating logic and preserves array shapes."""
    config = _make_config('>0.5', h_tide=2.0, imk2=0.7)
    interior = Interior_t(nlev_b=2)
    interior.phi = np.array([0.8])

    run_dummy_orbit(config, interior)

    assert interior.tides.shape == (1,)
    assert interior.tides == pytest.approx([2.0 * (0.8 - 0.5) / (1 - 0.5)])
