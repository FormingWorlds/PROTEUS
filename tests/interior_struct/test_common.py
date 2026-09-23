"""Unit tests for ``proteus.interior_struct.common``.

Covers ``solvus_radius``, the one gate every consumer of the global-miscibility
solvus frame uses (the atmosphere override in the main loop, the SPIDER domain,
the SPIDER entropy remap and the Zalmoxis mesh truncation). The invariant under
test is boundedness: a returned solvus radius lies strictly inside the planet,
0 < R_solvus < R_int, and is finite; anything else keeps the magma-ocean frame.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from proteus.interior_struct.common import solvus_radius

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

R_INT = 6.371e6  # Earth radius [m]


def _config(miscibility: bool):
    """Stub config exposing the miscibility switch."""
    config = MagicMock()
    config.interior_struct.zalmoxis.global_miscibility = miscibility
    return config


@pytest.mark.physics_invariant
def test_valid_solvus_inside_the_planet_is_returned():
    """A solvus written by a structure solve, strictly inside the planet, is
    the radius the solvus frame uses, returned unchanged."""
    r = solvus_radius(_config(True), {'R_solvus': 0.9 * R_INT, 'R_int': R_INT})
    assert r == pytest.approx(0.9 * R_INT, rel=1e-12)
    assert 0.0 < r < R_INT
    # Edge case just inside the surface still counts as a solvus.
    r_edge = solvus_radius(_config(True), {'R_solvus': R_INT * (1 - 1e-9), 'R_int': R_INT})
    assert r_edge == pytest.approx(R_INT * (1 - 1e-9), rel=1e-12)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'r_solvus',
    [0.0, -1.0e5, R_INT, 1.1 * R_INT, np.nan, np.inf, None],
    ids=[
        'zero-initialised-row',
        'negative-radius',
        'at-the-surface',
        'outside-the-planet',
        'nan',
        'infinite',
        'missing',
    ],
)
def test_unphysical_solvus_keeps_the_magma_ocean_frame(r_solvus):
    """A solvus that is zero (the helpfile initial value), negative, at or
    outside the surface, non-finite or absent is not a solvus: the helper
    returns None so no consumer uses it as a boundary radius."""
    row = {'R_int': R_INT}
    if r_solvus is not None:
        row['R_solvus'] = r_solvus
    assert solvus_radius(_config(True), row) is None
    # Discrimination: the same row with a valid solvus does switch frames.
    row['R_solvus'] = 0.5 * R_INT
    assert solvus_radius(_config(True), row) == pytest.approx(0.5 * R_INT, rel=1e-12)


@pytest.mark.physics_invariant
def test_miscibility_off_ignores_a_valid_solvus():
    """With global_miscibility off the helpfile solvus is never used, even
    when it holds a physical value."""
    row = {'R_solvus': 0.9 * R_INT, 'R_int': R_INT}
    assert solvus_radius(_config(False), row) is None
    assert solvus_radius(_config(True), row) == pytest.approx(0.9 * R_INT, rel=1e-12)


@pytest.mark.physics_invariant
def test_explicit_outer_radius_overrides_the_row():
    """``R_outer`` bounds the solvus instead of ``hf_row['R_int']``, as the
    Zalmoxis mesh truncation uses the freshly solved planet radius before the
    row is updated; the row value may also be absent there."""
    row = {'R_solvus': 0.95 * R_INT, 'R_int': 0.9 * R_INT}
    assert solvus_radius(_config(True), row) is None  # outside the stale R_int
    r = solvus_radius(_config(True), row, R_outer=R_INT)
    assert r == pytest.approx(0.95 * R_INT, rel=1e-12)
    # Edge case: no outer radius anywhere means no frame.
    assert solvus_radius(_config(True), {'R_solvus': 0.5 * R_INT}) is None
