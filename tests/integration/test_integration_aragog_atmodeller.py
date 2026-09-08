"""
Integration test: atmodeller solver_multistart schema validator.

The two-timestep aragog + atmodeller coupling test lives in
``test_slow_aragog_atmodeller.py`` at the slow tier because Linux GHA
needs > 1200 s for a single aragog setup + solver step + atmodeller
JAX compile. This file keeps the sub-second error-contract test that
exercises the atmodeller solver_multistart schema validator at the
integration tier so it runs on every nightly without burning the
integration step budget.

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

import pytest

pytest.importorskip('atmodeller')

# Integration tier: the validator round-trip is cheap (< 1 s) but
# exercises the production schema path; integration tier matches the
# rest of the validator-style tests in tests/integration/.
pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


@pytest.mark.integration
def test_atmodeller_solver_multistart_validator_rejects_non_positive():
    """Atmodeller ``solver_multistart`` schema validator rejects zero and
    negative integers.

    Contract from ``src/proteus/config/_outgas.py:113``:
        ``solver_multistart`` must be > 0.

    Verifies:
    - ``solver_multistart=0`` raises ValueError at attrs validator time.
    - ``solver_multistart=-1`` raises ValueError too.
    - Documented positive values (1, 10) round-trip without raising.
    - The default is positive (catches a stale-default regression that
      would otherwise only surface when atmodeller's wrapper crashed
      trying to do ``multistart - 1`` indexing).
    """
    from proteus.config._outgas import Atmodeller

    with pytest.raises(ValueError, match=r'(?i)solver_multistart'):
        Atmodeller(solver_multistart=0)
    with pytest.raises(ValueError, match=r'(?i)solver_multistart'):
        Atmodeller(solver_multistart=-1)

    # Discrimination: confirm known-good positive values round-trip.
    for n in (1, 10):
        a = Atmodeller(solver_multistart=n)
        assert a.solver_multistart == n

    # Discrimination: default must be positive (the attrs validator
    # would not protect a stale default in the factory function).
    default = Atmodeller()
    assert default.solver_multistart > 0, (
        f'default solver_multistart not positive: {default.solver_multistart}'
    )
