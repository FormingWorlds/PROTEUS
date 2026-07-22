# Test framework

[![codecov](https://img.shields.io/codecov/c/github/FormingWorlds/PROTEUS?label=coverage&logo=codecov)](https://app.codecov.io/gh/FormingWorlds/PROTEUS){target="_blank" rel="noopener"}
[![Unit Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/PROTEUS/coverage-baseline.yml?branch=main&label=Unit%20Tests)](https://github.com/FormingWorlds/PROTEUS/actions/workflows/coverage-baseline.yml){target="_blank" rel="noopener"}
[![Integration Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/PROTEUS/ci-nightly.yml?branch=main&label=Integration%20Tests)](https://github.com/FormingWorlds/PROTEUS/actions/workflows/ci-nightly.yml){target="_blank" rel="noopener"}
[![tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/FormingWorlds/PROTEUS/badges/tests-total.json)](https://proteus-framework.org/testing){target="_blank" rel="noopener"}
[![unit tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/FormingWorlds/PROTEUS/badges/tests-unit.json)](https://proteus-framework.org/testing){target="_blank" rel="noopener"}
[![integration tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/FormingWorlds/PROTEUS/badges/tests-integration.json)](https://proteus-framework.org/testing){target="_blank" rel="noopener"}

These counts refresh automatically: the `Refresh test count badges` workflow regenerates the JSON on the `badges` branch on every push to `main`, and shields.io reads it live.

PROTEUS is scientific simulation software where incorrect results can
propagate silently through coupled modules. The testing framework is
designed to catch real bugs, not just verify that code runs without
crashing. This page explains the design principles behind the test suite.

For practical instructions (running tests, writing tests, CI commands),
see [Testing](../How-to/testing.md).

## Test tier hierarchy

Tests are organized into four tiers of increasing scope and cost:

```
Unit (< 100 ms)  →  Smoke (< 30 s)  →  Integration (minutes)  →  Slow (hours)
     ↑                    ↑                     ↑                      ↑
  Every PR             Nightly              Nightly               Nightly
```

**Unit tests** verify individual Python functions with all external
dependencies mocked. They test logic, error handling, and mathematical
correctness in isolation. Most tests are unit tests.

**Smoke tests** run real binary solvers (SOCRATES, AGNI, SPIDER) for a
single timestep at low resolution. They verify that the binary interface
works and that the solver produces physically valid output.

**Integration tests** couple multiple modules together for several timesteps.
They verify that the data flow between modules is consistent and that the
coupled system converges.

**Slow tests** run full physics simulations at production resolution. They
validate against published benchmarks and cross-implementation checks (e.g.
SPIDER vs Aragog for the same initial conditions). These are the most
expensive tests and run only in the nightly CI.

The badge row at the top of this page carries three kinds of badge. The coverage badge reports the line coverage that codecov records for the main branch. The two status badges report the outcome of the latest unit-coverage run on the main branch and the latest nightly run. The three count badges report how many tests the suite collects in each category: the total, the unit tier, and the combined smoke, integration, and slow tiers. The counts come from `pytest --collect-only`, so they track the real suite as it grows.

## Functional tests vs physics tests

The test suite distinguishes two categories of test intent:

### Functional tests

Functional tests verify the software contract: does the function return the
right type, handle edge cases, raise on invalid input, and dispatch to the
correct backend? These are the bread-and-butter of software testing and apply
to all code regardless of physics content.

Examples:

- Config validator rejects negative planet mass
- Wrapper dispatches to the correct backend based on config
- Helpfile CSV round-trips correctly through write and read
- CLI parses flags and invokes the right subcommand

### Physics tests

Physics tests verify that the code produces physically correct results.
They go beyond "does it run" to "does it compute the right answer." In
scientific simulation code, a test that passes for the wrong reason
generates false confidence.

Physics tests must assert at least one of these invariants:

- **Conservation**: mass closure (sum of reservoirs = total), energy
  balance (input = output within tolerance), angular momentum
- **Positivity or boundedness**: $T > 0$ K, $P > 0$ Pa, mass fractions
  in $[0, 1]$, escape rate $\leq$ atmospheric mass
- **Monotonicity or symmetry**: pressure increasing with depth, reversing
  time integration recovers the initial condition
- **Pinned numeric value**: a closed-form result verified via
  `pytest.approx`, with a discrimination guard showing the most plausible
  wrong formula would give a different answer

Physics tests carry the `@pytest.mark.physics_invariant` marker so their
coverage can be tracked independently of line coverage.

## Discrimination guards

A discrimination guard is an assertion that proves the test is not
trivially true. For example, testing that a function returns 1.0 is
weak if every wrong implementation also returns 1.0. A discrimination
guard would additionally assert that a common wrong formula (e.g. using
$T^3$ instead of $T^4$) gives a different result at the same input.

```python
def test_stefan_boltzmann_flux():
    """Verify F = sigma * T^4 at T = 300 K."""
    T = 300.0
    sigma = 5.670374419e-8
    expected = sigma * T**4  # 459.30 W/m^2

    result = compute_flux(T)
    assert result == pytest.approx(expected, rel=1e-6)

    # Discrimination: T^3 would give 1.531 W/m^2, well outside tolerance
    wrong_result = sigma * T**3
    assert abs(result - wrong_result) > 100.0
```

## Reference pinning

Some tests pin PROTEUS results against published benchmarks. These carry
the `@pytest.mark.reference_pinned` marker and cite the specific paper,
figure, or table they validate against.

Reference-pinned tests are the strongest form of physics validation:
they verify not just that the code conserves the right quantities, but
that it produces the right numbers for a specific physical scenario.

Each physics module directory should contain at least one reference-pinned
test. The module-level inventory is tracked in `docs/Validation/<module>.md`.

## Anti-happy-path rules

Every new test function must include:

1. **At least one edge case**: boundary value ($\phi = 0$ or $1$,
   $e = 0$, $T = T_\mathrm{solidus}$), empty input, or extreme parameter
2. **At least one path exercising the error contract**: a documented
   exception, a guard return, or a graceful clamp. If the function has no
   validation logic, exercise the limit-input behaviour ($e = 0$ is a
   fixed point, Im($k_2$) = 0 leaves state unchanged) and assert the
   mathematical invariant.
3. **Assertion values not trivially derivable from the implementation**:
   discriminating numeric pins or property-based assertions (monotonicity,
   conservation) preferred over point checks

### Forbidden patterns

- Single-assert test functions (except hard-fail invariants like mass closure)
- Standalone weak assertions: `assert result is not None`,
  `assert result > 0`, `assert len(result) > 0` as the only meaningful check
- Tests with no function-level docstring
- Float `==` comparisons (use `pytest.approx`)

## Coverage architecture

PROTEUS uses three coverage gates, all on every pull request:

- **Fast gate**: unit tests only, fixed at 80%. Unit tests alone are not
  expected to reach the 90% ecosystem target, because wrapper code that
  requires real binaries is exercised only by the nightly tiers. That is why
  the gate sits at 80 rather than chasing 90.
- **Estimated total**: the pull request's unit coverage unioned with the
  latest nightly artifact, measured against 90%. That union is how the 90%
  target is reached on a pull request even though smoke, integration, and slow
  tests do not run there. This is the primary KPI; the fast gate is a lower
  bound.
- **Diff-cover**: the lines the pull request changed, held to 80%, again
  unioning the fast coverage with the latest nightly so that wrapper code the
  nightly alone exercises is not counted against the change.

The nightly runs every tier and publishes the coverage artifact the estimated
total and diff-cover union against; it does not itself fail on a coverage
percentage.

The fast gate's 80% and the 90% the estimated total is measured against are
fixed rather than ratcheting, and neither may be lowered:
`tools/update_coverage_threshold.py` holds both and the pull-request threshold
guard fails if either is edited away from its value. The diff-cover threshold is
fixed in the workflow instead.

## Float comparison discipline

All float comparisons use `pytest.approx(val, rel=...)` or
`np.testing.assert_allclose(actual, expected, rtol=..., atol=...)`.
State the tolerance rationale when non-obvious:

```python
# rtol=1e-3 because Cp lookup table truncates to 4 significant figures
assert result == pytest.approx(expected, rel=1e-3)
```

## Determinism

- Set seeds for any randomness: `np.random.seed(42)`, `random.seed(42)`,
  `torch.manual_seed(42)`
- Use `tmp_path` for temporary files (no large outputs)
- Tests must produce the same result on every run
