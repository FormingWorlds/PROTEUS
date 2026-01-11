# Test Categorization for CI/CD

This document explains how PROTEUS tests are categorized and how they flow through the CI/CD pipeline.

> **Related documentation:** For complete testing infrastructure details including test structure, configuration, and ecosystem rollout, see [Testing Infrastructure](test_infrastructure.md).

Also see the consolidated CI status and planning in [CI/CD Status and Roadmap](test_infrastructure.md#cicd-status-and-roadmap-as-of-2026-01-11).

## Test Categories

All tests in PROTEUS are marked with pytest markers to enable targeted test selection:

### @pytest.mark.unit

**Purpose**: Fast validation of Python logic with mocked physics
**Runtime**: <100ms per test (target)

**Count**: 487 tests (as of 2026-01-11)
**Runs In**: `ci-pr-checks.yml` (PR validation, ~2-5 minutes total)
**Coverage**: Python interfaces, configuration, utilities, wrapper modules

**Implemented Tests**:


- `tests/config/test_config.py` - Configuration system (3 tests)
- `tests/config/test_defaults.py` - Configuration defaults (7 tests)
- `tests/atmos_clim/test_common.py` - Atmosphere common utils (6 tests)
- `tests/utils/test_data.py` - Data management utils (7 tests)
- `tests/test_cli.py` - Command-line interface (3 tests)
- `tests/test_init.py` - Package initialization (1 test)
- `tests/plot/test_cpl_colours.py` - Color mapping (2 tests)
- `tests/plot/test_cpl_helpers.py` - Helper functions (1 test)
- `tests/star/test_star.py` - Stellar physics and luminosity/instellation (14 tests)
- `tests/utils/test_plot.py` - Plotting utilities (mocked matplotlib)
- All module wrappers (observe, outgas, escape, interior, etc.)


### @pytest.mark.smoke

**Purpose**: Quick validation that binaries work with new Python code
**Runtime**: <30s per test (target)
**Count**: 1 test (as of 2026-01-06)

**Runs In**: `ci-pr-checks.yml` (PR validation, ~3-5 min with unit tests)


**Implemented Tests**:

- `tests/integration/test_smoke_minimal.py::test_proteus_dummy_init` - PROTEUS initialization with dummy modules (0.3s)
- `tests/integration/test_smoke_janus.py` - JANUS-Interior coupling (skipped due to binary instability)


### @pytest.mark.integration

**Purpose**: Multi-module coupling and interaction tests

**Runtime**: Minutes to hours

**Example Tests**:

- `tests/integration/test_integration_dummy.py` - PROTEUS with dummy modules (4 tests)
- `tests/integration/test_integration_dummy_agni.py` - PROTEUS + AGNI atmosphere (4 tests)
- `tests/integration/test_integration_aragog_janus.py` - ARAGOG interior + JANUS atmosphere (5 tests)
- `tests/integration/test_albedo_lookup.py` - Albedo interpolation (3 tests)


### @pytest.mark.slow


**Purpose**: Full scientific validation with comprehensive simulations


**When to use**: Tests that simulate Earth magma ocean, Venus greenhouse, Super-Earth evolution, etc.


**Example tests** (not yet implemented):
- Earth magma ocean solidification (1-4 hours)
- Venus runaway greenhouse transition (30 min - 2 hours)

- Super-Earth interior evolution (2-6 hours)

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

**Coverage gates**:

- **Fast gate**: 18% overall (low threshold for quick feedback)
- **Diff-cover**: 80% on changed lines (enforced; uses `--diff-file` to avoid remote fetch inside container)
- **Full gate**: 69% (auto-ratcheting; enforced on nightly runs and merge to main)


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

```text
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

Implemented (as of 2026-01-06)

| Marker | Count | Runs In |
| --- | --- | --- |
| `@pytest.mark.unit` | 10 | PR checks (~2–5 min) |
| `@pytest.mark.smoke` | 1 | PR checks (~3–5 min with unit) |
| `@pytest.mark.integration` | 0 | Nightly |
| `@pytest.mark.slow` | 0 | Nightly |
| `@pytest.mark.skip` | 9 | Excluded from CI |

Planned Targets

| Marker | Target Count | Notes |
| --- | --- | --- |
| `@pytest.mark.unit` | 23 | Coverage expansion priority |
| `@pytest.mark.smoke` | 5–7 | One per major module |
| `@pytest.mark.integration` | 23 | Multi-module coupling |
| `@pytest.mark.slow` | 3–5 | Full scenario validations |

| **Grand Total** | **55** | — |

## Test Fixtures and conftest.py

All PROTEUS tests share common configuration and parameter sets through `tests/conftest.py`. This file provides:

### Parameter Classes for Test Scenarios

Three representative exoplanet scenarios are defined as parameter classes (all values in SI units):

1. **EarthLikeParams** — Modern Earth reference
   - Habitable zone, outgassed interior, thin CO₂-N₂ atmosphere
   - Use for habitability baseline tests

2. **UltraHotSuperEarthParams** — TOI-561 b
   - Ultra-short period (0.45 day), extreme irradiation (~100× Earth)
   - Ultra-low density (4.3 g/cm³) suggests thick volatile envelope
   - Use for atmospheric escape physics and magma ocean tests
   - Reference: Teske et al. (arXiv:2509.17231)

3. **IntermediateSuperEarthParams** — L 98-59 d
   - 3.7-day orbit, H₂-rich atmosphere (MMW ~9 u), permanent magma ocean
   - Bridges habitability and volatile loss regimes
   - Use for volatile retention and tidal heating tests
   - Reference: Nicholls et al. (arXiv:2507.02656)

### Session-Scoped Fixtures

All fixtures use `scope='session'` (cached once per test run for efficiency):

```python
def test_with_fixtures(earth_params, ultra_hot_params, config_minimal):
    """Example: use multiple fixtures in one test."""
    # earth_params: EarthLikeParams instance
    # ultra_hot_params: UltraHotSuperEarthParams instance
    # config_minimal: Path to input/minimal.toml
    pass
```

**Available Fixtures**:

- `earth_params` → EarthLikeParams instance
- `ultra_hot_params` → UltraHotSuperEarthParams instance
- `intermediate_params` → IntermediateSuperEarthParams instance
- `config_earth` → Path to `input/planets/earth.toml`
- `config_minimal` → Path to `input/minimal.toml`
- `config_dummy` → Path to `input/demos/dummy.toml`
- `proteus_root` → Absolute path to repository root

### Physical Constants in Parameters

All parameter classes use PROTEUS constants from `src/proteus/utils/constants.py`:

- Gravitational constant `const_G`
- Stefan-Boltzmann constant `const_sigma`
- Solar mass `M_sun`, Solar radius `R_sun`, Solar luminosity `L_sun`
- Earth mass `M_earth`, Earth radius `R_earth`
- Seconds per year `secs_per_year`

This ensures consistency across the entire test suite and with simulations.

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

1. Identify the test type and marker:

- Unit test (Python logic, mocks physics) → `@pytest.mark.unit`
- Smoke test (binary init, 1 timestep) → `@pytest.mark.smoke`
- Integration test (multi-module workflow) → `@pytest.mark.integration`
- Slow test (comprehensive, hours-long) → `@pytest.mark.slow`

1. Create the test file (if not exists):

- Location: `tests/<module>/test_<filename>.py`
- Mirror `src/<package>/` structure

1. Add a test function with the correct marker:

```python
import pytest

@pytest.mark.unit
def test_my_feature():
   """Brief description of what this tests."""
   # Test code here
   assert result == expected
```

1. Replace @pytest.mark.skip placeholders with real tests:

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

1. Verify coverage improvements:

- Run `pytest --cov=src`
- Confirm coverage increased for your module

1. Run the appropriate test marker group:

- Unit: `pytest -m unit`
- Integration: `pytest -m integration`
- Ensure tests pass

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
- [PROTEUS Copilot Instructions](../.github/copilot-instructions.md)
