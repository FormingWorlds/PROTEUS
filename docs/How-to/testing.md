# Testing

This page covers the practical aspects of testing PROTEUS: running tests,
writing tests, checking coverage, and working with CI.

For the conceptual framework behind the testing strategy (tier hierarchy,
physics invariants, validation certification), see
[Test framework](../Explanations/test_framework.md).

## Quick start

Install with `pip install -e ".[develop]"`, then:

```bash
pytest -m "unit and not skip and not slow and not integration"   # Fast unit tests (~2 min)
pytest -m "smoke and not skip"          # Smoke tests with real binaries
pytest --cov=src --cov-report=html      # Generate coverage report
open htmlcov/index.html                 # View coverage in browser
```

Tests covering an optional backend skip when that backend is absent, so
`[develop]` alone runs fewer tests than CI does. To match the CI selection
locally, install the optional extras as well:

```bash
pip install -e ".[develop,vulcan,atmodeller,inference]"
```

Before committing:

1. `pytest -m "unit and not skip and not slow and not integration"` must pass
2. `ruff check src/ tests/ && ruff format src/ tests/` must pass
3. `bash tools/validate_test_structure.sh` must pass

## Test markers

Every test function carries a tier marker that controls when and where it runs:

| Marker | What it tests | Speed budget | CI surface |
|--------|---------------|-------------|------------|
| `@pytest.mark.unit` | Python logic, mocked physics | < 100 ms | Every PR |
| `@pytest.mark.smoke` | Real binaries, 1 timestep, low res | < 30 s | Nightly |
| `@pytest.mark.integration` | Multi-module coupling | Minutes | Nightly |
| `@pytest.mark.slow` | Full physics validation | Hours | Nightly |
| `@pytest.mark.skip` | Deliberately disabled | n/a | Never |

Every test file must have a module-level marker:

```python
pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]
```

Timeout ceilings: 30 s for unit, 60 s for smoke, 300 s for integration,
3600 s for slow.

### Which marker to use

- **Most tests**: `unit`. Mock external dependencies, test one function.
- **Testing real binaries**: `smoke`. SOCRATES, AGNI, or SPIDER actually
  running, 1 timestep.
- **Testing module coupling**: `integration`. Aragog + AGNI working
  together, multiple timesteps.
- **Full science validation**: `slow`. Multi-hour simulations comparing
  against published results.

## Writing tests

### File layout

Tests mirror the source tree:

```
src/proteus/utils/helper.py  →  tests/utils/test_helper.py
src/proteus/escape/wrapper.py  →  tests/escape/test_wrapper.py
```

Validate with `bash tools/validate_test_structure.sh`.

### Basic test structure

```python
import pytest
from proteus.utils.helper import my_function

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

def test_my_function_returns_expected_value():
    """Verify my_function computes the correct result for standard input."""
    result = my_function(input_value=10)
    assert result == pytest.approx(42.0, rel=1e-5)
    # Discrimination: a common off-by-one bug would give 41.0,
    # which is outside the tolerance
    assert result > 41.5
```

### Requirements for every test

1. **Docstring**: state the physical scenario or contract being verified
2. **At least two assertions**: the second discriminates against the most
   plausible wrong answer
3. **At least one edge case**: boundary value, empty input, or extreme
   parameter
4. **No bare float `==`**: use `pytest.approx(val, rel=...)` or
   `np.testing.assert_allclose`

### Mocking

Unit tests mock external calls (SOCRATES, AGNI, SPIDER, file I/O):

```python
from unittest.mock import patch, MagicMock

@pytest.mark.unit
def test_run_atmosphere_dispatches_to_agni():
    """Verify the atmosphere wrapper calls AGNI when module='agni'."""
    with patch('proteus.atmos_clim.wrapper.run_agni') as mock_agni:
        mock_agni.return_value = None
        run_atmosphere(config_with_agni, hf_row)
        mock_agni.assert_called_once()
```

Mock at the narrowest scope (a specific function, not a whole module).
Mocked physics functions must return physically plausible values.

### Fixtures

Shared fixtures live in `tests/conftest.py`:

- `EarthLikeParams`, `UltraHotSuperEarthParams`, `IntermediateSuperEarthParams`:
  pre-configured parameter sets
- `config_earth`, `config_minimal`, `config_dummy`: paths to test configs
- `tmp_path`: pytest built-in for temporary directories

### Optional dependencies

Tests importing optional packages must call `pytest.importorskip`:

```python
pytest.importorskip('atmodeller')
pytest.importorskip('vulcan')
```

This prevents collection failures on CI runners without the optional package.

## Coverage

### Thresholds

| Gate | Tests | Target | Enforced |
|------|-------|--------|----------|
| Fast | unit only | 80% (fixed) | PR checks |
| Estimated total | PR unit coverage unioned with the latest nightly | 90% (fixed) | PR checks |
| Diff-cover | Changed lines, fast coverage unioned with the latest nightly | 80% | PR checks |

All three gates run on pull requests. The nightly runs every tier and
publishes the coverage artifact that the estimated total and diff-cover union
against; it does not itself fail on a coverage percentage.

The fast gate's 80% and the 90% the estimated total is measured against are
fixed rather than ratcheting, and neither may be lowered:
`tools/update_coverage_threshold.py` holds both and a pull-request guard fails
if either is edited away from its value. The diff-cover threshold is fixed in
the workflow instead. Unit tests alone are not expected to reach 90%, because
wrapper code that requires real binaries runs only in the nightly tiers; the
90% target is met through the estimated total.

The gates only warn on draft pull requests and block once the pull request is
marked ready for review. Two of them also fall back to warning when no nightly
artifact is available, and the estimated total allows a small grace margin
below its target before it fails.

### Checking coverage locally

```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

Module-level analysis:

```bash
bash tools/coverage_analysis.sh
```

### Test quality validation

```bash
python tools/check_test_quality.py --check
```

This AST-based linter flags:

- Single-assert test functions
- Weak standalone assertions (`assert result is not None`)
- Missing function-level docstrings
- Float `==` comparisons
- Missing module-level tier markers

## Checking that a refactor changed nothing

A change meant to leave behaviour alone is shown to have left it alone by
running the same configuration before and after and finding the same numbers.
`tests/integration/golden_run.toml` is a fixed all-dummy configuration for
that, and `tests/integration/golden_run.tsv` holds the trajectory it produces:
one line per helpfile column, so a quantity that moves shows up as a single
changed line in a diff.

While working on a refactor, compare the working tree against the recorded
trajectory:

```bash
python tools/record_golden_run.py --dry-run
```

It exits non-zero and names the columns that moved, the row where each moved
most, and by how much relative to the tolerance. The same comparison runs
nightly as `tests/integration/test_integration_golden_run.py`.

How small a change it resolves, measured by scaling the cooling rate in
`src/proteus/interior_energetics/dummy.py` by `1 + eps` and running the
comparison:

| `eps` | outcome |
|---|---|
| `1e-6` | reported, at 49 times the tolerance |
| `1e-7` | reported, at 4.9 times |
| `2e-8` | inside the tolerance |

So the threshold sits just above `2e-8`. The most sensitive column is not the
one perturbed: a change to the interior is amplified about fiftyfold on its
way into the volatile partitioning, because dissolved volatiles depend steeply
on melt fraction near solidification. That threshold is far below any change
of behaviour and well above the last-bit differences between one platform's
maths library and another's.

What it covers is the coupling framework: the main loop, the timestep
controller, the module handshakes, the helpfile round-trip, and every
configuration field the run reads. It does not cover the physics modules
themselves, which run on their dummy backends here.

### Recording the trajectory again

A change to the physics, to the coupling loop or to the configuration moves
the trajectory, and the reference then has to be recorded again:

```bash
python tools/record_golden_run.py
```

Do that in the same commit as the change that moved it. The command prints
what moved against the reference it replaces, and the diff of `golden_run.tsv`
is the record of which quantities changed and by how much, so a reviewer sees
the behavioural consequence of the change rather than only its code.

Editing `golden_run.toml` without recording the trajectory again fails the
check with a message saying so, rather than as an unexplained numerical
difference: the reference stores a digest of the configuration it came from.

## CI/CD pipeline

### Pull request checks

When you open a PR, CI runs:

1. **Structure validation**: `tests/` mirrors `src/proteus/`
2. **Unit tests** (Linux + macOS): `pytest -m "unit and not skip and not slow and not integration"`
3. **Estimated-total coverage**: PR unit coverage unioned with the latest
   nightly, measured against 90%
4. **Diff-cover**: 80% coverage on changed lines
5. **Lint**: `ruff check` and `ruff format`
6. **Editable install**: verifies the package installs correctly
7. **Test quality**: `python tools/check_test_quality.py --check` (reported,
   does not block)

The pull-request cycle runs the unit tier only. The smoke, integration, and
slow tiers run in the nightly workflow.

Runtime: ~5-10 minutes.

### Nightly validation

The nightly workflow runs all tiers:

1. **Unit + smoke** on Linux and macOS
2. **Integration** on Linux and macOS
3. **Slow** across multiple shards (aragog, zalmoxis-coupled, agni,
   janus-inference, etc.)
4. **Coverage aggregate**: combines every tier and publishes the coverage
   artifact that the pull-request gates union against

Runtime: ~2-3 hours.

## Pre-commit checklist

Before every commit:

```bash
pytest -m "unit and not skip and not slow and not integration"   # Tests pass
ruff check --fix src/ tests/            # Lint
ruff format src/ tests/                 # Format
bash tools/validate_test_structure.sh   # Structure
```

The pre-commit hook runs `ruff check` and `ruff format` automatically.
