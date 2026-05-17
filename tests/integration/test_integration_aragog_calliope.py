"""
Integration test: interior_energetics module schema validator.

The two-timestep aragog + calliope coupling test lives in
``test_slow_aragog_calliope.py`` at the slow tier because Linux GHA
needs > 1200 s for a single aragog setup + solver step. This file
keeps the sub-second error-contract test that exercises the
interior_energetics module schema validator at the integration tier
so it runs on every nightly without burning the integration step
budget.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

import pytest

# Integration tier: the validator round-trip is cheap (< 1 s) but
# exercises the production schema path; integration tier matches the
# rest of the validator-style tests in tests/integration/.
pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


@pytest.mark.integration
def test_interior_energetics_module_validator_rejects_unknown_backend():
    """Interior_energetics ``module`` schema validator rejects backends
    outside the documented {spider, aragog, dummy, boundary} enum.

    Contract from ``src/proteus/config/_interior.py``: the Interior
    dataclass's ``module`` field is validated with
    ``in_(('spider', 'aragog', 'dummy', 'boundary'))``.

    Verifies:
    - ``module='unknown'`` raises ValueError at attrs validator time, BEFORE
      any module dispatch or hf_row write. This prevents a typo'd config
      from silently dispatching to a no-op interior.
    - Each of the four known-good values round-trips without raising, so
      a regression that broke the validator into raising on every input
      is not masked.
    - The default is inside the enum (catches a stale-default regression).
    """
    from proteus.config._interior import Interior

    with pytest.raises(ValueError, match=r'(?i)module'):
        Interior(module='unknown')

    # Discrimination: confirm each documented backend round-trips.
    for known in ('spider', 'aragog', 'dummy', 'boundary'):
        i = Interior(module=known)
        assert i.module == known

    # Discrimination: default is inside the enum.
    default = Interior()
    assert default.module in ('spider', 'aragog', 'dummy', 'boundary'), (
        f'default interior_energetics module unexpectedly outside enum: {default.module!r}'
    )
