"""Unit tests for ``proteus.interior_struct.common``.

Covers ``solvus_radius``, the one gate every consumer of the global-miscibility
solvus frame uses (the atmosphere override in the main loop, the SPIDER domain,
the SPIDER entropy remap and the Zalmoxis mesh truncation). The invariant under
test is boundedness: a returned solvus radius lies strictly between the core
and the surface, R_core < R_solvus < R_int, and is finite; anything else keeps
the magma-ocean frame.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from proteus.interior_struct.common import solvus_radius

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

R_INT = 6.371e6  # Earth radius [m]
R_CORE = 3.48e6  # Earth core radius [m]


def _config(miscibility: bool):
    """Stub config exposing the miscibility switch."""
    config = MagicMock()
    config.interior_struct.zalmoxis.global_miscibility = miscibility
    return config


@pytest.mark.physics_invariant
def test_valid_solvus_inside_the_planet_is_returned():
    """A solvus written by a structure solve, strictly inside the planet, is
    the radius the solvus frame uses, returned unchanged."""
    r = solvus_radius(_config(True), 0.9 * R_INT, R_INT, R_inner=R_CORE)
    assert r == pytest.approx(0.9 * R_INT, rel=1e-12)
    assert R_CORE < r < R_INT
    # Edge case just inside the surface still counts as a solvus.
    r_edge = solvus_radius(_config(True), R_INT * (1 - 1e-9), R_INT, R_inner=R_CORE)
    assert r_edge == pytest.approx(R_INT * (1 - 1e-9), rel=1e-12)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'r_solvus',
    [0.0, -1.0e5, 0.3 * R_INT, R_CORE, R_INT, 1.1 * R_INT, np.nan, np.inf, None],
    ids=[
        'zero-initialised-row',
        'negative-radius',
        'inside-the-core',
        'at-the-core',
        'at-the-surface',
        'outside-the-planet',
        'nan',
        'infinite',
        'missing',
    ],
)
def test_unphysical_solvus_keeps_the_magma_ocean_frame(r_solvus):
    """A solvus that is zero (the helpfile initial value), negative, at or
    below the core, at or outside the surface, non-finite or absent is not a
    solvus: the helper returns None so no consumer uses it as a boundary
    radius (a solvus below the core would give SPIDER coresize > 1)."""
    assert solvus_radius(_config(True), r_solvus, R_INT, R_inner=R_CORE) is None
    # Discrimination: a valid solvus between the core and surface does switch frames.
    r = solvus_radius(_config(True), 0.7 * R_INT, R_INT, R_inner=R_CORE)
    assert r == pytest.approx(0.7 * R_INT, rel=1e-12)


@pytest.mark.physics_invariant
def test_miscibility_off_ignores_a_valid_solvus():
    """With global_miscibility off the helpfile solvus is never used, even
    when it holds a physical value."""
    assert solvus_radius(_config(False), 0.9 * R_INT, R_INT, R_inner=R_CORE) is None
    r = solvus_radius(_config(True), 0.9 * R_INT, R_INT, R_inner=R_CORE)
    assert r == pytest.approx(0.9 * R_INT, rel=1e-12)


@pytest.mark.physics_invariant
def test_missing_or_non_finite_bounds_keep_the_magma_ocean_frame():
    """Without a finite outer radius there is no frame to switch to, and a
    missing inner radius falls back to positivity alone."""
    assert solvus_radius(_config(True), 0.5 * R_INT, None) is None
    assert solvus_radius(_config(True), 0.5 * R_INT, np.nan) is None
    assert solvus_radius(_config(True), 0.5 * R_INT, R_INT, R_inner=np.nan) is None
    # Edge case: no inner bound given, so only 0 < R_solvus < R_outer applies.
    r = solvus_radius(_config(True), 0.1 * R_INT, R_INT, R_inner=None)
    assert r == pytest.approx(0.1 * R_INT, rel=1e-12)


@pytest.mark.physics_invariant
def test_negative_inner_radius_keeps_positivity():
    """A negative inner radius does not lower the floor below zero, so a
    negative or zero solvus is still rejected; a positive one is kept."""
    assert solvus_radius(_config(True), -5e4, R_INT, R_inner=-1e5) is None
    assert solvus_radius(_config(True), 0.0, R_INT, R_inner=-1e5) is None
    r = solvus_radius(_config(True), 5e4, R_INT, R_inner=-1e5)
    assert r == pytest.approx(5e4, rel=1e-12)
