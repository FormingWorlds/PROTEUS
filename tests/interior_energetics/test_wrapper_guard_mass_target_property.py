"""Property test of the mass escape of the monotonic-radius guard of the structure update."""

from __future__ import annotations

import pytest

pytest.importorskip('hypothesis')

import numpy as np  # noqa: E402
from hypothesis import HealthCheck, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from proteus.interior_energetics.wrapper import (  # noqa: E402
    _MASS_ESCAPE_RADIUS_OFFSET,
    _MASS_ESCAPE_RADIUS_SLOPE,
    _ZALMOXIS_MASS_ANCHOR_TOL,
)
from tests.interior_energetics.test_wrapper_guard_mass_bound import BOUND_MSG  # noqa: E402
from tests.interior_energetics.test_wrapper_guard_mass_target import (  # noqa: E402
    ACCEPT_MSG,
    M_RETAINED,
    R_RETAINED,
    REJECT_MSG,
    _cooling_case,
    _solve_to,
    _target_ratio_at_anchor,
    run_guard,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.timeout(60),
    pytest.mark.usefixtures('_no_core_table'),
]


@pytest.mark.physics_invariant
@settings(
    max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@example(target_ratio=float(_target_ratio_at_anchor()), radius_step=1.0e-3)
@example(target_ratio=float(np.nextafter(_target_ratio_at_anchor(), 2.0)), radius_step=1.0e-3)
@example(target_ratio=1.05, radius_step=0.1009)
@example(target_ratio=1.05, radius_step=0.1011)
@given(target_ratio=st.floats(0.9, 1.2), radius_step=st.floats(1.0e-6, 0.2))
def test_up_step_kept_if_and_only_if_the_retained_mass_misses_a_larger_target(
    caplog, target_ratio, radius_step
):
    """Under cooling a scalar up-step is kept exactly when the retained mass misses a larger
    target by more than the anchor and the radius step stays within the bound of that mass rise;
    the bound warning comes exactly when the mass miss holds and the bound refuses the step."""
    config, hf_row, base = _cooling_case()
    msgs, _, _ = run_guard(
        caplog, hf_row, config, base, _solve_to(target_ratio, 1.0 + radius_step)
    )

    target = M_RETAINED * target_ratio
    outgrown = (
        bool(target > M_RETAINED) and abs(M_RETAINED / target - 1.0) > _ZALMOXIS_MASS_ANCHOR_TOL
    )
    d_r = R_RETAINED * (1.0 + radius_step) / R_RETAINED - 1.0
    bound = _MASS_ESCAPE_RADIUS_SLOPE * (target / M_RETAINED - 1.0) + _MASS_ESCAPE_RADIUS_OFFSET
    kept = outgrown and bool(d_r <= bound)
    assert any(ACCEPT_MSG in m for m in msgs) is kept
    assert any(BOUND_MSG in m for m in msgs) is (outgrown and not kept)
    assert any(REJECT_MSG in m for m in msgs) is not kept
    expected_R = R_RETAINED * (1.0 + radius_step) if kept else R_RETAINED
    assert hf_row['R_int'] == pytest.approx(expected_R, rel=1e-12)
