"""Property tests of ``proteus.outgas.lid_seal`` over the input domain of the leak step.

Invariants checked: at a constant floor the step kernel gives the closed form
for any held mass, floor and step fraction; a sealed row resumes only with a
finite closing temperature above 0 K.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('hypothesis')

from hypothesis import given
from hypothesis import strategies as st

from proteus.outgas.lid_seal import floor_leak, restore_seal_state

from .test_lid_seal import _config, _row, _sealed_row

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.physics_invariant
@given(
    mass=st.floats(1.0, 1.0e24),
    share=st.floats(0.0, 2.0),
    frac=st.floats(1.0e-12, 1.0),
)
def test_floor_leak_at_a_constant_floor_is_the_closed_form(mass, share, frac):
    """At a constant floor F >= 0 the leak is frac max(M - F, 0), and in [0, frac M]."""
    floor = share * mass
    leak = floor_leak(mass, frac, lambda _: floor)
    assert 0.0 <= leak <= frac * mass
    assert leak == pytest.approx(frac * max(mass - floor, 0.0), rel=1e-12, abs=1e-12 * mass)


@given(temp=st.one_of(st.floats(allow_nan=True, allow_infinity=True), st.just(0.0)))
def test_restore_refuses_a_sealed_row_without_a_closing_temperature(temp):
    """A sealed last row resumes only with a finite seal_Tmagma_close above 0 K;
    an open row resumes whatever the column holds."""
    row = {**_sealed_row(), 'seal_active': 1.0, 'seal_Tmagma_close': temp}
    if np.isfinite(temp) and temp > 0.0:
        assert restore_seal_state(_config(), row) is True
    else:
        with pytest.raises(ValueError, match='seal_Tmagma_close'):
            restore_seal_state(_config(), row)
    assert restore_seal_state(_config(), {**_row(), 'seal_Tmagma_close': temp}) is False
    del row['seal_Tmagma_close']
    with pytest.raises(ValueError, match='seal_Tmagma_close'):
        restore_seal_state(_config(), row)
