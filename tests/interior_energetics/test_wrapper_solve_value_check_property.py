"""Property test of the value check after a Zalmoxis structure solve."""

from __future__ import annotations

import math

import pytest

pytest.importorskip('hypothesis')

import numpy as np  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(60)]

BAD_SOLVE_MSG = 'non-finite or non-positive R_int or M_int'

_VALUES = st.one_of(
    st.floats(allow_nan=True, allow_infinity=True, allow_subnormal=True),
    st.sampled_from([0.0, -0.0, math.nan, math.inf, -math.inf, 5e-324, np.finfo(float).max]),
    st.none(),
)


@pytest.mark.physics_invariant
@settings(max_examples=300, deadline=None)
@given(R_int=_VALUES, M_int=_VALUES)
def test_structure_check_raises_exactly_for_values_not_finite_and_positive(R_int, M_int):
    """The check raises if and only if R_int or M_int is not in (0, inf), and names both."""
    from proteus.interior_energetics.wrapper import _require_finite_structure

    row = {'R_int': R_int, 'M_int': M_int}
    R, M = (float(v or 0.0) for v in (R_int, M_int))  # the row read of the wrapper: -0.0 is 0.0
    if 0.0 < R < math.inf and 0.0 < M < math.inf:
        _require_finite_structure(row, 'test solve')
        return
    with pytest.raises(RuntimeError, match=BAD_SOLVE_MSG) as err:
        _require_finite_structure(row, 'test solve')
    msg = str(err.value)
    assert msg.startswith('test solve: ')
    assert f'R_int={R:.6e} m' in msg and f'M_int={M:.6e} kg' in msg
