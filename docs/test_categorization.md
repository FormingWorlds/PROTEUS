# Test Categorization for CI/CD

This document explains how PROTEUS tests are categorized and how they flow through the CI/CD pipeline.

> **Related documentation:** For complete testing infrastructure details including test structure, configuration, and ecosystem rollout, see [Testing Infrastructure](test_infrastructure.md).

## Test Categories

All tests in PROTEUS are marked with pytest markers to enable targeted test selection:

### @pytest.mark.unit
**Purpose**: Fast validation of Python logic with mocked physics
**Runtime**: <100ms per test (target)
**Count**: 23 tests
**Runs In**: `ci-pr-checks.yml` (PR validation, ~2-5 minutes total)
**Coverage**: Python interfaces, configuration parsing, utilities

> See [Testing Infrastructure - Best Practices](test_infrastructure.md#best-practices) for guidance on writing effective unit tests.

**Example Tests**:
- `tests/config/test_config.py` - Configuration system
- `tests/test_cli.py` - Command-line interface
- `tests/grid/test_grid.py` - Grid generation utilities
- `tests/plot/test_cpl_*.py` - Plotting functions

### @pytest.mark.smoke
**Purpose**: Quick validation that binaries work with new Python code
**Runtime**: <30s per test (target)
**Count**: 0 currently (none implemented yet)
**Runs In**: `ci-pr-checks.yml` (PR validation, optional)
**Coverage**: Binary execution, real atmospheric models

**When to use**: Tests that run a physics module for 1 timestep at low resolution

**Example tests** (not yet implemented):
- Single SPIDER timestep at 10 radial points
- Single JANUS iteration at 20 atmospheric layers
- SOCRATES single spectral calculation

### @pytest.mark.integration
**Purpose**: Multi-module coupling and interaction tests
**Runtime**: Minutes to hours
**Count**: 23 tests (real integration tests)
**Runs In**: `ci-nightly-science.yml` (nightly validation, ~2 hours)
**Coverage**: PROTEUS workflow, module coupling, data exchange

**Example Tests**:
- `tests/integration/test_integration_dummy.py` - PROTEUS with dummy modules (4 tests)
- `tests/integration/test_integration_dummy_agni.py` - PROTEUS + AGNI atmosphere (4 tests)
- `tests/integration/test_integration_aragog_janus.py` - ARAGOG interior + JANUS atmosphere (5 tests)
- `tests/integration/test_albedo_lookup.py` - Albedo interpolation (3 tests)

### @pytest.mark.slow
**Purpose**: Full scientific validation with comprehensive simulations
**Runtime**: Hours (up to 4 hours total for all slow tests)
**Count**: 0 currently (none implemented - would extend integration tests)
**Runs In**: `ci-nightly-science.yml` (nightly validation, optional job)
**Coverage**: Full physics accuracy, long-term evolution, benchmark comparisons

**When to use**: Tests that simulate Earth magma ocean, Venus greenhouse, Super-Earth evolution, etc.

**Example tests** (not yet implemented):
- Earth magma ocean solidification (1-4 hours)
- Venus runaway greenhouse transition (30 min - 2 hours)
- Super-Earth interior evolution (2-6 hours)
- Full PROTEUS workflow end-to-end (4-8 hours)

### @pytest.mark.skip (Placeholder Tests)
**Purpose**: Structural placeholders for modules that need test implementation
**Runtime**: Skipped (not executed)
**Count**: 9 tests
**Runs In**: Not executed (marked with @pytest.mark.skip)
**Status**: TODO - Need implementation

**Files**:
- `tests/escape/test_escape.py`
- `tests/orbit/test_orbit.py`
- `tests/interior/test_interior.py`
- `tests/atmos_clim/test_atmos_clim.py`
- `tests/outgas/test_outgas.py`
- `tests/utils/test_utils.py`
- `tests/observe/test_observe.py`
- `tests/star/test_star.py`
- `tests/atmos_chem/test_atmos_chem.py`

These placeholder tests exist to maintain test directory structure. Replace their `pass` statements with actual test implementations.

## CI/CD Pipeline

### Fast PR Checks (`ci-pr-checks.yml`)
**Trigger**: Pull requests to `main` or `dev`, pushes to feature branches
**Duration**: ~5-10 minutes
**Strategy**:
1. Use pre-built Docker image with compiled physics
2. Overlay PR code changes
3. Run `@pytest.mark.unit` tests only
4. Optionally run `@pytest.mark.smoke` tests
5. Enforce ruff code quality checks
6. Coverage requirement: 69% (auto-ratcheting)

**Command**:
```bash
pytest -m "unit and not skip" --ignore=tests/examples \
  --cov=src --cov-fail-under=69
```

**Test Output**: Coverage report, ruff errors
**Artifacts**: HTML coverage report

### Nightly Science Validation (`ci-nightly-science.yml`)
**Trigger**: Scheduled at 03:00 UTC daily (1 hour after Docker build)
**Duration**: ~4-6 hours total
**Strategy**:
1. Use latest pre-built Docker image
2. Run comprehensive integration tests
3. Run slow scientific validation tests
4. Generate detailed coverage reports

**Commands**:
```bash
# Job 1: Slow + Integration tests (comprehensive)
pytest -m "slow or integration" --ignore=tests/examples \
  --cov=src

# Job 2: Integration tests only (separate tracking)
pytest -m "integration and not slow" --ignore=tests/examples \
  --cov=src
```

**Test Output**: Coverage reports, physics validation results
**Artifacts**: HTML coverage, simulation outputs, test logs

## Test Discovery & Organization

### Directory Structure
```
tests/
├── examples/          # Example/demonstration tests (excluded from CI)
├── integration/       # Multi-module coupling tests (@pytest.mark.integration)
│   ├── test_integration_dummy.py
│   ├── test_integration_dummy_agni.py
│   ├── test_integration_aragog_janus.py
│   └── test_albedo_lookup.py
├── config/           # Configuration tests (@pytest.mark.unit)
├── grid/             # Grid tests (@pytest.mark.unit)
├── plot/             # Plotting tests (@pytest.mark.unit)
├── escape/           # Placeholder test (@pytest.mark.skip)
├── orbit/            # Placeholder test (@pytest.mark.skip)
├── interior/         # Placeholder test (@pytest.mark.skip)
├── atmos_clim/       # Placeholder test (@pytest.mark.skip)
├── atmos_chem/       # Placeholder test (@pytest.mark.skip)
├── outgas/           # Placeholder test (@pytest.mark.skip)
├── observe/          # Placeholder test (@pytest.mark.skip)
├── star/             # Placeholder test (@pytest.mark.skip)
├── utils/            # Placeholder test (@pytest.mark.skip)
├── test_cli.py       # CLI tests (@pytest.mark.unit)
├── test_init.py      # Init tests (@pytest.mark.unit)
└── inference/        # Inference tests (@pytest.mark.integration)
```

### Marker Counts
| Marker | Count | Location |
|--------|-------|----------|
| `@pytest.mark.unit` | 23 | PR checks (~5 min) |
| `@pytest.mark.smoke` | 0 | PR checks (optional) |
| `@pytest.mark.integration` | 23 | Nightly (~2 hr) |
| `@pytest.mark.slow` | 0 | Nightly (optional) |
| `@pytest.mark.skip` | 9 | Not executed |
| **Total Countable** | **46** | — |
| **Placeholder/Skipped** | **9** | — |
| **Grand Total** | **55** | — |

## Running Tests Locally

### All unit tests (fast local development)
```bash
pytest -m unit
```

### All smoke tests (optional binary validation)
```bash
pytest -m smoke
```

### Unit + Smoke (what PR checks run)
```bash
pytest -m "unit or smoke"
```

### All integration tests (slower, ~2+ hours)
```bash
pytest -m integration
```

### All slow tests (comprehensive, ~4+ hours)
```bash
pytest -m slow
```

### Everything except slow (fast local)
```bash
pytest -m "not slow"
```

### Everything with coverage
```bash
pytest --cov=src --cov-report=html
```

### Skip placeholder tests explicitly
```bash
pytest -m "not skip"
```

## Implementation Checklist for New Tests

When implementing tests for a module, follow this checklist:

1. **Identify the test type**:
   - [ ] Unit test? (tests Python logic, mocks physics) → `@pytest.mark.unit`
   - [ ] Smoke test? (validates binary works, 1 timestep) → `@pytest.mark.smoke`
   - [ ] Integration test? (multi-module, full workflow) → `@pytest.mark.integration`
   - [ ] Slow test? (comprehensive, hours-long) → `@pytest.mark.slow`

2. **Create test file** (if not exists):
   - Location: `tests/<module>/test_<filename>.py`
   - Mirror `src/<package>/` structure

3. **Add test function with marker**:
   ```python
   import pytest

   @pytest.mark.unit
   def test_my_feature():
       """Brief description of what this tests."""
       # Test code here
       assert result == expected
   ```

4. **Remove @pytest.mark.skip** from placeholder:
   ```python
   # Before:
   @pytest.mark.skip(reason="Placeholder test")
   def test_placeholder():
       pass

   # After:
   @pytest.mark.unit
   def test_my_feature():
       # Real test implementation
   ```

5. **Verify coverage**:
   - Run `pytest --cov=src`
   - Check that coverage increased for your module

6. **Run appropriate test marker group**:
   - For unit tests: `pytest -m unit`
   - For integration: `pytest -m integration`
   - Verify tests pass

## Coverage Requirements

- **Threshold**: 69% (auto-ratcheting on main branch)
- **Calculation**: Only real tests count (examples and skipped tests excluded)
- **Tool**: `pytest-cov` (coverage.py)
- **Report**: HTML report at `htmlcov/index.html`

Coverage is enforced in:
- `ci-pr-checks.yml` for unit tests
- `ci-nightly-science.yml` for integration tests

## References

- [Test Infrastructure Documentation](./test_infrastructure.md)
- [Docker CI Architecture](./docker_ci_architecture.md)
- [PROTEUS Copilot Instructions](./.github/copilot-instructions.md)
