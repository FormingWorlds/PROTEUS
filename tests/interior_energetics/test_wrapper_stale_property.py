"""Property test of the re-solve decision of update_structure_from_interior: the routine
ceiling and the recovery of a stale structure after a failed re-solve."""

from __future__ import annotations

import pytest

pytest.importorskip('hypothesis')

from hypothesis import HealthCheck, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from tests.interior_energetics.test_wrapper import (  # noqa: E402
    _call_with_success_solver,
    _stale_ceiling_case,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.timeout(30),
    pytest.mark.usefixtures('_no_core_table'),
]


@pytest.mark.unit
@settings(
    max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@example(stale=True, elapsed=3e4, recovery_time=0.0, interval=5e4)
@example(stale=True, elapsed=2.5e4, recovery_time=2.5e4, interval=5e4)
@example(stale=False, elapsed=3e4, recovery_time=2.5e4, interval=5e4)
@example(stale=True, elapsed=6e4, recovery_time=6e4, interval=5e4)
@given(
    stale=st.booleans(),
    elapsed=st.floats(min_value=0.0, max_value=1e5),
    recovery_time=st.sampled_from([0.0, 1.0, 2.5e4, 5e4, 6e4]) | st.floats(0.0, 1e5),
    interval=st.floats(min_value=1.0, max_value=1e5),
)
def test_the_solver_runs_exactly_when_the_ceiling_or_the_recovery_is_due(
    tmp_path, stale, elapsed, recovery_time, interval
):
    """With no Phi or T_magma change and no floor, a re-solve runs exactly when the
    routine ceiling is due or a stale structure has reached a recovery time above 0, and a
    re-solve clears the stale flag."""
    config, dirs, interior_o, hf_row = _stale_ceiling_case(tmp_path, stale=stale)
    config.interior_struct.zalmoxis.update_interval = interval
    config.interior_struct.zalmoxis.update_stale_ceiling = recovery_time
    last = hf_row['Time'] - elapsed
    _, solver = _call_with_success_solver(config, dirs, interior_o, hf_row, last)
    elapsed_seen = hf_row['Time'] - last
    due = elapsed_seen >= interval or (stale and 0 < recovery_time <= elapsed_seen)
    assert solver.call_count == int(due)
    # a re-solve clears the stale flag; without one the flag is as before
    assert interior_o.structure_stale is (stale and not due)
