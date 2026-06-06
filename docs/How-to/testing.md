# Testing

This page covers the practical aspects of testing PROTEUS: running tests,
writing tests, checking coverage, and working with CI.

For the conceptual framework behind the testing strategy (tier hierarchy,
physics invariants, validation certification), see
[Test framework](../Explanations/test_framework.md).

## Quick start

Install with `pip install -e ".[develop]"`, then:

```bash
pytest -m "unit and not skip"           # Fast unit tests (~2 min)
pytest -m "smoke and not skip"          # Smoke tests with real binaries
pytest --cov=src --cov-report=html      # Generate coverage report
open htmlcov/index.html                 # View coverage in browser
```

Before committing:

1. `pytest -m "unit and not skip"` must pass
2. `ruff check src/ tests/ && ruff format src/ tests/` must pass
3. `bash tools/validate_test_structure.sh` must pass

## Test markers

Every test function carries a tier marker that controls when and where it runs:

| Marker | What it tests | Speed budget | CI surface |
|--------|---------------|-------------|------------|
| `@pytest.mark.unit` | Python logic, mocked physics | < 100 ms | Every PR |
| `@pytest.mark.smoke` | Real binaries, 1 timestep, low res | < 30 s | Every PR |
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
pytest.importorskip('zephyrus')
pytest.importorskip('lovepy')
```

This prevents collection failures on CI runners without the optional package.

## Coverage

### Thresholds

| Gate | Tests | Target | Enforced |
|------|-------|--------|----------|
| Fast (every PR) | unit + smoke | Ratcheting toward 90% | PR checks |
| Full (nightly) | unit + smoke + integration + slow | 90% | Nightly CI |
| Diff-cover (every PR) | Changed lines only | 80% | PR checks |

Thresholds auto-ratchet upward (never decrease) and are capped at 90%.

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

## CI/CD pipeline

### Pull request checks

When you open a PR, CI runs:

1. **Structure validation**: `tests/` mirrors `src/proteus/`
2. **Unit tests** (Linux + macOS): `pytest -m "unit and not skip"`
3. **Smoke tests**: `pytest -m "smoke and not skip"`
4. **Diff-cover**: 80% coverage on changed lines
5. **Lint**: `ruff check` and `ruff format`
6. **Editable install**: verifies the package installs correctly

Runtime: ~5-10 minutes.

### Nightly validation

The nightly workflow runs all tiers:

1. **Unit + smoke** on Linux and macOS
2. **Integration** on Linux and macOS
3. **Slow** across multiple shards (aragog, zalmoxis-coupled, agni,
   janus-inference, etc.)
4. **Coverage aggregate**: combines all tiers and checks the 90% gate

Runtime: ~2-3 hours.

## Pre-commit checklist

Before every commit:

```bash
pytest -m "unit and not skip"           # Tests pass
ruff check --fix src/ tests/            # Lint
ruff format src/ tests/                 # Format
bash tools/validate_test_structure.sh   # Structure
```

The pre-commit hook runs `ruff check` and `ruff format` automatically.
