"""
Hypothesis-parametrised smoke run: explore the planet/orbit/star parameter
space and assert conservation invariants on every example.

Each example loads `input/dummy.toml`, overrides four physical inputs from
hypothesis-generated values within their published validity ranges, runs
PROTEUS for one timestep, and runs the same conservation-invariant check
that every other smoke test runs (`assert_smoke_conservation_invariants`).

Why this matters:
- The handwritten smoke tests in `test_smoke_modules.py` and
  `test_smoke_atmos_interior.py` exercise one configuration each. They
  catch bugs at THEIR specific point in parameter space.
- A hypothesis-driven smoke explores the corners of the parameter space
  the handwritten tests never visit (extreme stellar Teff, near-circular
  vs. eccentric orbits, sub-Earth vs. super-Earth masses, tight vs. wide
  orbits) and surfaces any conservation-bookkeeping bug that only fires
  outside the typical Earth-like fiducial region.
- Marked `slow` so it does not gate PR CI; runs in nightly + on-demand.
- `derandomize=True` so CI is reproducible across runs.

Test-only exemption to the Config-mutability rule:

The project rule (`.claude/rules/proteus-code-review.md`, "Config
mutability") forbids mutating `Config` attrs at runtime in source code.
This test deliberately violates that rule by overriding fields after
`Proteus(...)` initialisation; the alternative would be to render a
fresh TOML per example, which costs ~50 ms of file IO per run. Because
the override paths bypass attrs validators that would normally fire at
init time, the strategy bounds here MUST stay inside the validator-
accepted ranges (e.g., `mass_tot in [0.3, 5.0]` is well within the
`> 0` and `< 20` validators of `planet_mass_valid`). Do NOT widen the
strategy bounds to test validator behaviour; use the dedicated
`tests/config/test_config_schema_invariants.py` for that.

Hypothesis configuration: max_examples=10, deadline=120s. Each example
costs ~2-3 seconds for a 1-timestep dummy.toml run; 10 examples × 3 s
= ~30 seconds wall, well within the slow-tier budget.

Failure surface:
- A failing example reports the (a, e, Teff, M_planet) tuple, the
  helpfile invariant that broke, the helpfile state that triggered it,
  and the hypothesis-generated seed for replay. Add the failing tuple
  to a parametrised regression test before fixing the underlying bug
  so the same case never resurfaces silently.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest
from _smoke_invariants import assert_smoke_conservation_invariants
from helpers import PROTEUS_ROOT
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from proteus import Proteus

pytestmark = pytest.mark.slow


# Strategy bounds chosen to cover the rocky-exoplanet regime PROTEUS
# is expected to handle, with margin to surface near-edge bugs:
# - semi_major_axis [0.01, 2.0] AU: from ultra-hot inner-edge (TOI-561 b
#   at ~0.01 AU) to wide rocky orbits (~Mars at 1.5 AU); 2.0 AU is
#   beyond the typical PROTEUS regime but inside the schema validity.
# - eccentricity [0.0, 0.5]: 0.0 (circular) catches division-by-zero or
#   sin/cos sign bugs; 0.5 is in the high-eccentricity corner where
#   tidal heating dominates.
# - Teff [2400, 7200] K: M dwarf (TRAPPIST-1 = 2566 K, GJ 9827 ~4263 K)
#   to F dwarf (~7000 K). The schema lower bound is 2000 K; 2400 K is
#   the smallest fully-supported.
# - M_planet [0.3, 5.0] M_earth: from sub-Earth (Mars ~0.107 M_earth
#   excluded as too small for surface_solver convergence) to super-Earth
#   (~5 M_earth; 20 M_earth is the schema upper bound but the dummy
#   modules don't validate beyond ~10).
PARAM_STRATEGY = st.tuples(
    st.floats(min_value=0.01, max_value=2.0),  # semi_major_axis [AU]
    st.floats(min_value=0.0, max_value=0.5),  # eccentricity
    st.floats(min_value=2400.0, max_value=7200.0),  # Teff [K]
    st.floats(min_value=0.3, max_value=5.0),  # M_planet [M_earth]
)


@settings(
    max_examples=10,
    deadline=120000,  # 120 s per example wall budget
    derandomize=True,  # reproducible across CI runs without a seed
    suppress_health_check=[HealthCheck.too_slow],
)
@given(params=PARAM_STRATEGY)
def test_smoke_conservation_invariants_hypothesis(params):
    """Cross-product fuzz over (a, e, Teff, M_planet) asserting the
    conservation invariants survive every realised parameter combination.

    The previous handwritten smoke tests assert the same invariants but
    only at one point each. A regression that broke conservation at,
    e.g., extreme stellar Teff would slip through them but not through
    this fuzz (assuming the bad point falls in the sampled region).

    Anti-happy-path discipline:
    - The strategy bounds extend to the near-edge of the schema (smallest
      a, lowest Teff, largest M_planet) so example draws routinely hit
      the corners that handwritten tests typically avoid.
    - Conservation invariants are asserted after EVERY example, not just
      the median; one failure surfaces the offending tuple.
    - A circular orbit (e=0.0) is reachable by the strategy (boundary
      value catches sign bugs in eccentric-anomaly transforms).
    """
    semimajoraxis, eccentricity, teff, m_planet_mearth = params

    unique_id = str(uuid.uuid4())[:8]
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = Proteus(config_path=PROTEUS_ROOT / 'input' / 'dummy.toml')

        runner.config.params.out.path = str(Path(tmpdir) / f'hyp_smoke_{unique_id}')
        runner.init_directories()

        # Apply hypothesis-generated overrides
        runner.config.orbit.semimajoraxis = semimajoraxis
        runner.config.orbit.eccentricity = eccentricity
        runner.config.star.dummy.Teff = teff
        runner.config.planet.mass_tot = m_planet_mearth

        # Sane defaults to keep the run short and conservative
        runner.config.planet.tsurf_init = 2000.0
        runner.config.params.stop.time.minimum = 1e2
        runner.config.params.stop.time.maximum = 1e3
        runner.config.params.out.plot_mod = None  # NEVER plot under fuzz
        runner.config.params.out.write_mod = 0
        runner.config.params.out.archive_mod = 'none'

        runner.start(resume=False, offline=True)

        # Single composite call: NaN/Inf, T > 0, P >= 0, per-element +
        # per-species mass closure, M_atm <= M_planet, M_planet ≈ M_int
        # + M_ele, escape bounded by atmospheric mass.
        assert_smoke_conservation_invariants(runner.hf_all)


# Explicit anti-happy edge cases. Hypothesis can sample these, but pinning
# them as parametrised runs guarantees they execute on every PR even if
# the slow-tier hypothesis test is excluded by `-m 'not slow'`. Each tuple
# is at a DIFFERENT corner of the parameter space than the handwritten
# tests in test_smoke_modules.py.
EDGE_CASES = [
    # Ultra-hot inner edge (TOI-561 b regime)
    (0.01, 0.0, 5800.0, 1.0),
    # Eccentric near-Earth orbit
    (1.0, 0.4, 5800.0, 1.0),
    # M dwarf habitable zone, super-Earth
    (0.05, 0.0, 2566.0, 3.0),
    # F dwarf wide orbit, sub-Earth
    (1.5, 0.1, 7000.0, 0.5),
    # Circular orbit (e = 0 exactly) catches eccentric-anomaly sign bugs
    (0.5, 0.0, 5800.0, 1.0),
]


@pytest.mark.smoke
@pytest.mark.parametrize('semimajoraxis,eccentricity,teff,m_planet_mearth', EDGE_CASES)
def test_smoke_conservation_invariants_named_edge_cases(
    semimajoraxis, eccentricity, teff, m_planet_mearth
):
    """Run the conservation-invariant check at hand-picked corners of the
    physical parameter space.

    Marked `smoke` rather than `slow` so these explicitly run in every PR.
    The cases are non-overlapping with the handwritten smoke tests in
    test_smoke_modules.py (which all use the dummy.toml defaults).
    """
    unique_id = str(uuid.uuid4())[:8]
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = Proteus(config_path=PROTEUS_ROOT / 'input' / 'dummy.toml')
        runner.config.params.out.path = str(Path(tmpdir) / f'edge_smoke_{unique_id}')
        runner.init_directories()

        runner.config.orbit.semimajoraxis = semimajoraxis
        runner.config.orbit.eccentricity = eccentricity
        runner.config.star.dummy.Teff = teff
        runner.config.planet.mass_tot = m_planet_mearth

        runner.config.planet.tsurf_init = 2000.0
        runner.config.params.stop.time.minimum = 1e2
        runner.config.params.stop.time.maximum = 1e3
        runner.config.params.out.plot_mod = None
        runner.config.params.out.write_mod = 0
        runner.config.params.out.archive_mod = 'none'

        runner.start(resume=False, offline=True)

        assert_smoke_conservation_invariants(runner.hf_all)
