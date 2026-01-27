# Testing Infrastructure

This document describes the standardized testing infrastructure for PROTEUS and the wider ecosystem.

> **Related documentation:**
>
> - [Test Categorization](test_categorization.md) — Test markers and CI/CD pipeline flow
> - [Docker CI Architecture](docker_ci_architecture.md) — Detailed Dockerfile, image build strategy, and implementation reference
> - [Test building](test_building.md) — Follow best practices for building robust, physics-compliant tests
> - [conftest.py reference](tests/conftest.py) — Fixtures and parameter classes for common test scenarios

## Table of Contents

1. [CI/CD Status and Roadmap](#cicd-status-and-roadmap-as-of-2026-01-27)
2. [Quick Start](#quick-start)
3. [Developer Workflow](#developer-workflow)
4. [Coverage Analysis](#coverage-analysis-workflow)
5. [Pre-commit Checklist](#pre-commit-checklist)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

---

## CI/CD Status and Roadmap (as of 2026-01-27)

### CI/CD Current Status

**Implementation Status**: Fast PR workflow complete and passing ✓ — **Fast gate >32%**

- ✓ Unit Tests (mocked physics): **480+ tests**, ~2–5 min runtime
- ✓ Smoke Tests (real binaries): multiple smoke tests (some skipped for env/instability), run as `pytest -m "smoke and not skip"`
- ✓ Code Quality (ruff): Pass (`ruff check src/ tests/`, `ruff format --check src/ tests/`)
- ✓ Coverage tracking: **32.03%** fast gate (`[tool.proteus.coverage_fast] fail_under` in `pyproject.toml`)
- ✓ Diff-cover: 80% on changed lines (`diff-cover` with `--diff-file` to avoid remote fetch in container)
- ✓ Coverage ratcheting: **Automatic threshold increase** via `tools/update_coverage_threshold.py`; `github-actions[bot]` commits on push to `main` or `tl/test_ecosystem_v5`

**Key Achievements**:

1. **Large unit test suite**: 480+ unit tests across config, wrappers, utils, plot, star, escape, etc.
2. **Fast gate** above 32% (ratcheted in `pyproject.toml`); **full gate** 69% for nightly runs
3. Auto-ratcheting for both fast (PR) and full (nightly) thresholds
4. CI jobs: unit-tests → smoke-tests (needs unit-tests), lint in parallel
5. Test structure validated by `tools/validate_test_structure.sh` (tests mirror `src/proteus/`)

**Known Issues**:

- Codecov upload may fail with "Token required because branch is protected" (non-blocking)
- GPG verification warnings from codecov action (non-critical)

### CI/CD Architecture (Docker-based)

- **Prebuilt image**: `ghcr.io/formingworlds/proteus:latest` built nightly at 02:00 UTC via `.github/workflows/docker-build.yml`, and on push to `main` / `tl/test_ecosystem_v4` / `tl/test_ecosystem_v5` when relevant paths change (Dockerfile, pyproject.toml, etc.).
- **PR image choice**: On `main`, CI uses `latest`; on other branches (e.g. `tl/test_ecosystem_v5`), CI uses `ghcr.io/formingworlds/proteus:tl-test_ecosystem_v5`.
- **Environment**: Compiled physics (SOCRATES, AGNI, PETSc) + Python deps from `pyproject.toml`; data paths via `FWL_DATA`, `RAD_DIR`, `AGNI_DIR`, `PETSC_DIR`.
- **PR workflow** (`.github/workflows/ci-pr-checks.yml`): Runs in container, overlays PR code onto `/opt/proteus/`, runs `pytest -m "unit and not skip"` with coverage, validates test structure, runs diff-cover, then `pytest -m "smoke and not skip"`. Optionally downloads last **Nightly Science Validation (v5)** coverage artifact to show "estimated total" (unit + integration) in the run summary.
- **Nightly workflows**:
  - `ci-nightly-science.yml`: Triggered by schedule (03:00 UTC) and `workflow_dispatch`. Uses `main` + `latest` image. Jobs: quick-integration-test (dummy + multi_timestep), science-validation (`slow or integration`), integration-tests (`integration and not slow`). Full coverage ratcheting on main.
  - `ci-nightly-science-v5.yml`: Triggered on push to `tl/test_ecosystem_v5` and `workflow_dispatch`. Uses branch image. Runs integration + unit + slow (std_config) with coverage, writes `coverage-integration-only.json` and uploads artifact `v5-branch-nightly-coverage` for use by Fast PR checks.
- **Reusable workflow**: `proteus_test_quality_gate.yml` — `workflow_call` for ecosystem modules (Python version, coverage threshold, working directory, pytest-args).
- **Artifacts**: Coverage XML/HTML and optional Codecov upload (non-blocking when token missing).

### Current Metrics (as of 2026-01-27)

| Metric | Value | Target | Status |
| --- | --- | --- | --- |
| Unit tests | **480+** | — | ✓ |
| Smoke tests | Multiple (some skipped) | 5–7 active | In progress |
| Integration tests | Yes (dummy, aragog_agni, aragog_janus, std_config, etc.) | — | ✓ |
| Fast gate coverage | **32.03%** | ratcheted in pyproject.toml | ✓ |
| Full gate coverage | **69%** | ratcheted in pyproject.toml | ✓ |
| CI runtime (fast) | ~5–10 min | <10 min | ✓ |
| Diff-cover threshold | 80% | 80% | ✓ Met |

### Immediate Next Steps

**Current priorities**:

1. **Fix Codecov integration** (optional): Add `CODECOV_TOKEN` to GitHub repository secrets for full upload instead of non-blocking failures.
2. **Expand active smoke tests**: Several smoke tests are implemented but skipped (JANUS/SOCRATES instability, AGNI/JANUS binary requirements, CALLIOPE reserved for nightly). Goal: 5–7 active smoke tests running in PRs.
3. **Integration and slow tests**: `test_integration_std_config.py` exists; nightly v5 runs integration + slow (std_config). Further slow scenarios (Earth magma ocean, Venus runaway, Super-Earth) can be added with a 3–4 hour nightly budget.
4. **Nightly notifications** (optional): Email or Slack on nightly failures.

### Planned Improvements

#### Phase 1: Fast PR Workflow (✓ Substantially Complete)

- **Unit tests**: 480+ tests; fast gate ratcheted at 32.03% in `pyproject.toml`. Mock-based strategy for Python logic.
- **Smoke tests**: Multiple smoke tests implemented; some skipped (JANUS/SOCRATES, AGNI, CALLIOPE nightly). Target 5–7 active in PRs.
- **Lint**: Ruff check and format run in parallel with unit/smoke.
- **Diff-cover**: 80% on changed lines; threshold-decrease check vs `main` in CI.

#### Phase 2: Nightly Science Validation

- **`ci-nightly-science.yml`**: Schedule 03:00 UTC + `workflow_dispatch`. Quick integration (dummy, multi_timestep), science-validation (`slow or integration`), integration-tests (`integration and not slow`). Full coverage ratcheting on main (69% threshold).
- **`ci-nightly-science-v5.yml`**: On push to `tl/test_ecosystem_v5`. Integration (dummy + `integration and not slow`), unit append, slow (std_config). Produces `coverage-integration-only.json` and artifact `v5-branch-nightly-coverage` for Fast PR “estimated total” summary.
- **Integration suite**: `test_integration_dummy.py`, `test_integration_aragog_agni.py`, `test_integration_aragog_janus.py`, `test_integration_std_config.py`, and others. Standard-config coupling (ARAGOG+AGNI+CALLIOPE+ZEPHYRUS+MORS) exercised in nightly.
- **Full coverage ratcheting**: Implemented via `tools/update_coverage_threshold.py`; full gate 69%.

#### Phase 3: Long-Term (Future)

- Nightly notifications (email/Slack on failure)
- Regression baselines, multi-OS CI, ecosystem test harmonization

### Success Metrics

**Fast PR Workflow** (✓ Achieved):

- ✅ 480+ unit tests, `pytest -m "unit and not skip"`, fast gate 32.03%
- ✅ Smoke tests run as `pytest -m "smoke and not skip"` (some skipped for env)
- ✅ Ruff check and format; test structure validation
- ✅ Auto-ratcheting for fast gate on push to main / `tl/test_ecosystem_v5`
- ✅ CI runtime ~5–10 minutes (under 10 min target)
- ✅ Diff-cover 80% on changed lines

**Nightly Workflow** (✓ In place):

- Integration tests in `ci-nightly-science.yml` and `ci-nightly-science-v5.yml`
- Full coverage ratcheting (69% threshold) on main
- v5 branch nightly artifact feeds Fast PR “estimated total” (unit + integration)

**Ongoing goals**:

- 5–7 active smoke tests in PRs (reduce skips where possible)
- 50%+ unit coverage over time
- More slow/science scenarios in nightly budget

### Decision Points

#### Diff-cover threshold

- Current: `--fail-under=80` (strict)
- Recommendation: Keep strict to encourage good test coverage for changes

#### Unit test dependencies

- Current: `needs: unit-tests` in workflow
- Recommendation: Keep to fail fast on logic errors

#### Codecov integration

- Current: Non-blocking failures due to protected branch
- Recommendation: Add `CODECOV_TOKEN` to GitHub repo secrets if available

---

## Quick Start

### For PR Authors

When you open a PR, the CI system will:

1. ✅ Pull pre-built Docker image (`latest` on main, branch-tagged on feature branches)
2. ✅ Overlay your code onto `/opt/proteus/`
3. ✅ Validate test structure (`tools/validate_test_structure.sh`)
4. ✅ Run unit tests: `pytest -m "unit and not skip"` with coverage (fast gate 32.03%)
5. ✅ Run diff-cover (80% on changed lines), then smoke tests: `pytest -m "smoke and not skip"`
6. ✅ Lint runs in parallel: `ruff check src/ tests/`, `ruff format --check src/ tests/`

**Total**: ~5–10 minutes for typical Python-only PRs. Fortran/AGNI changes trigger smart rebuilds (SOCRATES/AGNI) inside the container.

### For Test Writers

Use pytest markers to categorize your tests:

```python
@pytest.mark.unit
def test_fast_logic():
    """Runs in PR checks. Mock heavy physics."""
    pass

@pytest.mark.smoke
def test_binary_works():
    """Runs in PR checks. 1 timestep, low res."""
    pass

@pytest.mark.integration
def test_module_coupling():
    """Runs nightly. Multi-module tests."""
    pass

@pytest.mark.slow
def test_full_physics():
    """Runs nightly. Hours-long validation."""
    pass
```

### Running Locally

```bash
# Install development dependencies
pip install -e ".[develop]"

# Run unit tests (fast; matches PR unit job)
pytest -m "unit and not skip"

# Run unit + smoke (what PR runs: unit then smoke, each excluding skip)
pytest -m "(unit or smoke) and not skip"

# Run everything except slow
pytest -m "not slow"

# Full test suite
pytest
```

### Performance Improvements

| Workflow | Before | After | Savings |
| --- | --- | --- | --- |
| PR Check (Python changes) | ~60 min | ~10 min | 50 min |
| PR Check (Fortran changes) | ~60 min | ~20 min | 40 min |
| Nightly (Full suite) | ~120 min | ~90 min | 30 min |

### How It Works

**Smart Rebuild** (in `ci-pr-checks.yml` smoke-tests job):

- **SOCRATES**: `socrates/build_code` runs; "Nothing to be done" or full rebuild if Fortran changed.
- **AGNI**: If any `.jl` files changed under `AGNI/`, Julia re-instantiation runs; otherwise skipped.
- **SPIDER**: Not rebuilt in CI (commented out); container relies on pre-built image.
- Python-only PR: No recompilation; smoke tests run with existing binaries.
- Fortran/Julia PR: Only changed components rebuilt (~minutes).

**Test Stratification**: Tests organized by execution time and purpose:

1. Unit Tests (seconds): Python logic, mocked physics
2. Smoke Tests (minutes): Binary validation, minimal resolution
3. Integration Tests (minutes): Multi-module coupling
4. Slow Tests (hours): Full scientific validation

**Container Strategy**:

- Build: Nightly at 02:00 UTC
- Cache: Docker layers + BuildKit cache
- Usage: All CI workflows pull the same image
- Overlay: PR code replaces container code at runtime

---

## Developer Workflow

1. Write or modify code
2. Write or update tests
3. Run tests locally: pytest
4. Check coverage: pytest --cov
5. Fix failing tests
6. Commit changes
7. Push → CI runs automatically
8. Monitor CI results

### Adding New Code

When adding a new module or feature:

1. **Create source file:** `src/proteus/<module>/<file>.py` (or new subdir under `src/proteus/`)
2. **Create test file:** `tests/<module>/test_<filename>.py` so tests mirror `src/proteus/` (see `tools/validate_test_structure.sh`)
3. **Write tests first** (TDD) or alongside code; use `@pytest.mark.unit` / `@pytest.mark.smoke` / `@pytest.mark.integration` / `@pytest.mark.slow` as appropriate
4. **Run tests:** `pytest tests/<module>/` or `pytest -m unit`
5. **Check coverage:** `pytest --cov=src tests/` or `pytest --cov=proteus` (CI uses `--cov=src` for unit, `--cov=proteus` in nightly v5)
6. **Validate structure:** `bash tools/validate_test_structure.sh`

### Test Writing Guidelines

**Basic test structure:**

```python
"""
Tests for <package>.<module>
"""
from __future__ import annotations

import pytest
from <package>.<module> import function_to_test


@pytest.mark.unit
def test_function_basic():
    """Test basic functionality of function"""
    result = function_to_test(input_value)
    assert result == expected_value


@pytest.mark.unit
def test_function_edge_cases():
    """Test edge cases and boundaries"""
    assert function_to_test(0) == expected
    assert function_to_test(-1) == expected

    with pytest.raises(ValueError):
        function_to_test(invalid_input)


@pytest.mark.integration
def test_module_integration():
    """Test interaction between components"""
    # Test multiple functions/classes together
    pass


@pytest.mark.slow
def test_performance():
    """Long-running performance test"""
    # Tests that take significant time
    pass
```

**Using fixtures (conftest.py):**

The `tests/conftest.py` file provides session-scoped fixtures for common test scenarios. This avoids redundant setup code and ensures consistent parameter sets across the test suite.

**Key Fixtures in conftest.py**:

1. **Physical Parameter Classes** — Three representative exoplanet scenarios:
   - `EarthLikeParams`: Modern Earth (habitable reference)
   - `UltraHotSuperEarthParams`: TOI-561 b (ultra-hot, volatile-poor)
   - `IntermediateSuperEarthParams`: L 98-59 d (volatile-rich magma ocean)

2. **Configuration Path Fixtures**:
   - `config_earth` → `input/planets/earth.toml`
   - `config_minimal` → `input/minimal.toml`
   - `config_dummy` → `input/demos/dummy.toml`

3. **Utility Fixtures**:
   - `proteus_root` → Absolute path to repository root
   - `earth_params`, `ultra_hot_params`, `intermediate_params` → Instances of parameter classes

**Example Usage**:

```python
"""Test habitability with Earth-like parameters"""
import pytest


def test_habitability(earth_params):
    """Unit test: validate equilibrium temperature."""
    # All parameters pre-loaded from EarthLikeParams
    assert earth_params.planet_surface_temp > 273  # Above freezing
    assert earth_params.orbital_semimajor == 1.496e11  # 1 AU


@pytest.mark.integration
def test_earth_coupling(config_earth, proteus_root):
    """Integration test: validate PROTEUS initialization with Earth config."""
    assert config_earth.exists()
    # ... load and simulate with real physics modules


@pytest.mark.slow
@pytest.mark.parametrize('params_class', [
    'earth_params',
    'ultra_hot_params',
    'intermediate_params'
])
def test_multi_scenario_evolution(request, params_class):
    """Slow test: run evolution for three scenarios."""
    params = request.getfixturevalue(params_class)
    # ... run full Gyr-scale simulations
```

**Session Scope**: All parameter fixtures use `scope='session'` (cached once per test run) for efficiency. Configuration fixtures depend on `proteus_root`, which must exist for tests to pass.

### Coverage Analysis Workflow

```bash
# 1. Run tests with coverage
pytest --cov

# 2. Generate detailed report
pytest --cov --cov-report=term-missing

# 3. View HTML report
open htmlcov/index.html

# 4. Analyze by module
bash tools/coverage_analysis.sh

# 5. Find uncovered code
coverage report --show-missing --skip-covered

# 6. Focus on priorities
# - Core modules first
# - High-impact code
# - Integration points
```

### Pre-commit Checklist

Before committing:

- [ ] All tests pass locally: `pytest -m "unit and not skip"` (and smoke if applicable)
- [ ] Coverage meets threshold (fast gate 32.03% for unit run; see `pyproject.toml`)
- [ ] No linting errors: `ruff check src/ tests/`
- [ ] Code formatted: `ruff format src/ tests/`
- [ ] New tests added for new code; use `@pytest.mark.unit` / `smoke` / `integration` / `slow` as appropriate
- [ ] Test structure validated: `bash tools/validate_test_structure.sh`

---

## Implementation Phases (Reference)

Docker build and PR workflow are implemented; see [CI/CD Architecture](#cicd-architecture-docker-based) and [Current Metrics](#current-metrics-as-of-2026-01-27).

**Local Docker checks:**

```bash
# Build locally to verify Dockerfile (optional; CI uses pre-built image)
docker build -t proteus-test .

# Run image and run tests
docker run -it ghcr.io/formingworlds/proteus:latest bash
# Inside container:
pytest -m "unit and not skip"
pytest -m "smoke and not skip"
```

**Branch workflows:** Push to `main` or `tl/test_ecosystem_v5` triggers `ci-pr-checks.yml`; image is `latest` on main, `tl-test_ecosystem_v5` on that branch. Use `workflow_dispatch` for manual runs. Nightly v5 (`ci-nightly-science-v5.yml`) runs on push to `tl/test_ecosystem_v5` and uploads coverage artifact for Fast PR “estimated total.”

---

## Troubleshooting

### Common Issues

#### 1. "pytest: error: unrecognized arguments: --cov"

**Cause:** pytest-cov not installed

**Solution:**

```bash
# Install pytest-cov
pip install pytest-cov

# Or reinstall all dev dependencies
pip install -e ".[develop]"

# Verify installation
python -c "import pytest_cov; print('pytest-cov:', pytest_cov.__version__)"
```

#### 2. "Coverage below threshold"

**Cause:** Test coverage dropped below configured threshold

**Solution:**

```bash
# Identify uncovered code
pytest --cov --cov-report=term-missing

# Find specific gaps
coverage report --show-missing --skip-covered

# Add tests for uncovered lines
# or adjust threshold temporarily
```

#### 3. "Tests not found" or "No tests ran"

**Cause:** Test discovery issues

**Solution:**

```bash
# Check what pytest discovers
pytest --collect-only

# Verify test naming (must start with test_)
find tests -name "*.py" | grep -v __pycache__

# Check PYTHONPATH
echo $PYTHONPATH

# Reinstall package
pip install -e ".[develop]"
```

#### 4. "Import errors" in tests

**Cause:** Package not installed or wrong path

**Solution:**

```bash
# Reinstall in editable mode
pip install -e ".[develop]"

# Verify package installed
pip list | grep <package_name>

# Check import
python -c "import <package>; print(<package>.__version__)"

# Verify package structure
ls -la src/<package>/
```

#### 5. "CI passes locally but fails on GitHub"

**Cause:** Environment differences

**Solution:**

- Check Python version matches
- Verify all dependencies in pyproject.toml
- Check for OS-specific code (paths, etc.)
- Review CI logs for specific errors
- Test in clean virtual environment


#### 6. "Ruff linting fails"

**Cause:** Code style violations

**Solution:**

```bash
# Check what fails
ruff check src/ tests/

# Auto-fix many issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/

# Check again
ruff check src/ tests/
```

### Debugging Tests

```bash
# Run with verbose output
pytest -v

# Show local variables on failure
pytest --showlocals

# Stop at first failure
pytest -x

# Run specific test
pytest tests/module/test_file.py::test_function

# Print output (even if test passes)
pytest -s

# Run with debugger on failure
pytest --pdb
```

### Docker CI Troubleshooting

#### 1. Docker Build Fails

**Cause:** Dockerfile syntax error or missing dependencies

**Solution:**

```bash
# Test locally
docker build -t proteus-test .

# Check specific stage
docker build --target <stage> -t proteus-test .

# Inspect layers
docker history proteus-test
```

#### 2. CI Can't Pull Image

**Cause:** Image is private or registry URL incorrect

**Solution:**

- Verify image is public: `ghcr.io/formingworlds/proteus:latest`
- Check registry URL in workflow
- Test pull locally: `docker pull ghcr.io/formingworlds/proteus:latest`

#### 3. Tests Fail in Container

**Cause:** Environment differences or missing dependencies

**Solution:**

```bash
# Run container interactively
docker run -it ghcr.io/formingworlds/proteus:latest bash

# Inside container, run tests
pytest -m unit -v
```

#### 4. Smart Rebuild Not Working

**Cause:** Makefile issues or missing binaries

**Solution:**

- Verify Makefiles are present in container
- Check if binaries have correct timestamps
- Force rebuild: `rm SPIDER/spider && make`

### Getting Help

1. Check this documentation
2. Review [tools/README.md](../tools/README.md)
3. Check [pytest documentation](https://docs.pytest.org/)
4. Check [coverage.py documentation](https://coverage.readthedocs.io/)
5. Open an issue on GitHub

---

## Best Practices

### Working with GitHub Copilot

GitHub Copilot is configured for the PROTEUS ecosystem with specific guidelines (`.github/workflows/copilot-instructions.md`). These instructions ensure consistent code quality and testing practices across all modules.

**Key Copilot Guidelines:**

1. **Test Infrastructure & Organization**
   - Copilot will automatically structure tests to mirror source code exactly
   - For every file in `src/<package>/`, Copilot creates `tests/<package>/test_<filename>.py`
   - Use `pytest --collect-only` to verify test discovery
   - Run `bash tools/validate_test_structure.sh` to validate structure

2. **Testing Standards**
   - Framework: `pytest` exclusively in `tests/` directory
   - Speed: Unit tests must run in <100ms (Copilot will use mocks aggressively)
   - Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`
   - Floats: Never use `==` for floats; use `pytest.approx(val, rel=1e-5)` or `np.testing.assert_allclose`
   - Physics: Ensure physically valid inputs (e.g., T > 0K) unless testing error handling

3. **Coverage Requirements**
   - **Full suite**: `pyproject.toml` `[tool.coverage.report]` `fail_under` (currently 69%)
   - **Fast gate (unit + smoke)**: `pyproject.toml` `[tool.proteus.coverage_fast]` `fail_under` (currently 32.03%)
   - **Auto-ratcheting**: Thresholds increase when coverage improves (never decrease); CI blocks decreases vs main
     - Fast gate: ratcheted on push to main or `tl/test_ecosystem_v5` via `tools/update_coverage_threshold.py --target fast`; `github-actions[bot]` commits `[skip ci]`
     - Full gate: ratcheted on main in nightly via `--target full`
   - All PRs must meet the fast gate and diff-cover (80% on changed lines)

4. **Code Quality & Style**
   - Linting: Follow `ruff` standards (line length < 92 chars, max indentation 3 levels)
   - Type hints: Use standard Python type hints
   - Docstrings: Include brief docstrings describing the physical scenario

5. **Safety & Determinism**
   - Randomness: Explicitly set seeds (e.g., `np.random.seed(42)`) in tests
   - Files: Do not generate tests that produce large output files; use `tempfile` or mocks

**Best Practices for Working with Copilot:**

- **Reference the guidelines:** When asking Copilot to generate tests, mention "following the PROTEUS test infrastructure guidelines"
- **Iterative refinement:** Use Copilot to generate initial test structure, then refine with domain knowledge
- **Validate generated code:** Always run `pytest --collect-only` and validate coverage after Copilot generates tests
- **Provide context:** Give Copilot context about the physical scenario being tested for better docstrings
- **Ecosystem consistency:** Copilot instructions apply to ALL Python modules (PROTEUS, CALLIOPE, JANUS, MORS, etc.)

### Testing Philosophy

1. **Test behavior, not implementation**
   - Focus on what code does, not how
   - Tests should survive refactoring
   - Validate outputs and side effects, not internal state

2. **Write tests first when possible (TDD)**
   - Clarifies requirements
   - Ensures testability
   - Provides instant feedback
   - Prevents over-engineering

3. **Keep tests simple and focused**
   - One concept per test
   - Clear, descriptive test names
   - Easy to understand and maintain
   - Avoid test interdependencies

4. **Use appropriate test types**
   - **Unit tests** (`@pytest.mark.unit`): Single functions/methods, fast (<100ms), isolated
   - **Integration tests** (`@pytest.mark.integration`): Multiple components, moderate speed
   - **Slow tests** (`@pytest.mark.slow`): Full simulation loops, computationally intensive

### Test Organization

1. **Mirror source structure exactly**
   - Tests in `tests/<module>/test_<filename>.py` match `src/proteus/<module>/<filename>.py`
   - `tools/validate_test_structure.sh` checks that each `src/proteus/<module>/` has a corresponding `tests/<module>/` (special dirs `data`, `helpers`, `integration` are skipped in validation)
   - Easy to find related tests; consistent across the ecosystem; enables automated validation

2. **One test file per source file**
   - When practical: `src/proteus/<module>/foo.py` → `tests/<module>/test_foo.py`
   - Keeps tests organized; clear 1:1 mapping
   - Use `bash tools/validate_test_structure.sh` to verify

3. **Group related tests**
   - Use test classes for related tests
   - Share fixtures via `conftest.py`
   - Organize by functionality within test files

4. **Use descriptive names**

```python
# Good: Clear what is being tested
def test_temperature_conversion_celsius_to_kelvin():
    """Test conversion from Celsius to Kelvin returns correct value."""
    result = convert_temperature(100, 'C', 'K')
    assert result == pytest.approx(373.15, rel=1e-5)

# Less good: Vague, unclear what is tested
def test_conversion():
    pass
```

1. **Document test intent**
   - Include docstrings explaining what scenario is tested
   - Add inline comments for non-obvious assertions
   - Reference formulas, physical principles, or domain knowledge
   - See [CALLIOPE test files](https://github.com/FormingWorlds/CALLIOPE/tree/main/tests) for examples

   Note: Placeholder tests exist to maintain directory structure; see the placeholder test list in [Test Categorization](test_categorization.md).

### Coverage Strategy

1. **Focus on critical paths first**
   - Core business logic and calculations
   - Physical models and simulations
   - Error handling and edge cases
   - Public APIs and interfaces

1. **Set realistic thresholds**
   - Start: 20-30% for new modules
   - Q2 target: 35-40%
   - Q4 target: 50-60%
   - Long-term: 80%+ (ecosystem standard)
   - Don't chase 100% - focus on value

1. **Use exclude patterns strategically**
   - Debug code and development utilities
   - Abstract methods that subclasses implement
   - Type checking blocks (`if TYPE_CHECKING:`)
   - Intentionally untestable code (mark with `# pragma: no cover`)

1. **Track trends over time**
   - Coverage going up? ✓ Good progress
   - Coverage dropping? Investigate and address
   - Use automatic ratcheting (CALLIOPE pattern) to prevent regression
   - Review coverage reports in PR reviews

1. **Prioritize based on risk**
   - High-risk code: Aim for 95%+ coverage
   - Medium-risk code: Aim for 80%+ coverage
   - Low-risk code: Aim for 60%+ coverage
   - Use `bash tools/coverage_analysis.sh` to identify gaps

### Test Quality Standards

1. **Write clear, maintainable tests**
   - Use descriptive test names that explain the scenario
   - Include docstrings for complex test cases
   - Add inline comments for non-obvious assertions
   - Document the physical principle being validated

1. **Follow the AAA pattern**

```python
   def test_atmospheric_pressure_at_surface():
       """Test that surface pressure calculation matches expected value."""
       # Arrange: Set up test data
       temperature = 300.0  # K
       gravity = 9.8  # m/s^2

       # Act: Perform the calculation
       pressure = calculate_surface_pressure(temperature, gravity)

       # Assert: Verify the result
       expected = 101325.0  # Pa (standard atmosphere)
       assert pressure == pytest.approx(expected, rel=0.01)
```

1. **Test one concept per test function**
   - Each test should validate a single behavior
   - If a test has multiple asserts, they should all relate to the same concept
   - Split complex scenarios into multiple focused tests

1. **Use appropriate assertions**
   - For floats: `pytest.approx(value, rel=1e-5)` or `np.testing.assert_allclose`
   - For arrays: `np.testing.assert_array_equal` or `assert_allclose`
   - For exceptions: `pytest.raises(ExceptionType)`
   - For warnings: `pytest.warns(WarningType)`

1. **Mock external dependencies**
   - File I/O operations
   - Network calls and APIs
   - Heavy computations (for unit tests)
   - System calls and OS interactions

```python
   from unittest.mock import Mock, patch

   @pytest.mark.unit
   def test_data_loader_calls_file_reader():
       """Test that data loader correctly calls file reader."""
       with patch('module.read_file') as mock_read:
           mock_read.return_value = {'data': [1, 2, 3]}
           result = load_data('dummy_path')
           mock_read.assert_called_once_with('dummy_path')
           assert result['data'] == [1, 2, 3]
   ```

### Test Markers and Organization

Use markers consistently across all ecosystem modules:

```python
@pytest.mark.unit
def test_pure_function():
    """Fast, isolated test of a single function."""
    pass

@pytest.mark.integration
def test_component_interaction():
    """Tests multiple components working together."""
    pass

@pytest.mark.slow
def test_long_computation():
    """Takes >1 second - typically full simulations."""
    pass
```

**Run selectively:**

```bash
# Fast feedback: unit tests only (~seconds)
pytest -m unit

# Before commit: all except slow (~minutes)
pytest -m "not slow"

# Nightly/full CI: everything (~hours for PROTEUS)
pytest
```

**Benefits of markers**:

- **Fast iteration:** Run unit tests while developing
- **Efficient CI:** Skip slow tests on feature branches
- **Clear categorization:** Know what each test validates
- **Selective debugging:** Focus on relevant test category

### Fixture Best Practices

1. **Keep fixtures focused**
   - One purpose per fixture
   - Compose when needed

2. **Use appropriate scope**

   ```python
   @pytest.fixture(scope="function")  # Default, new each test
   def data():
       return {...}

   @pytest.fixture(scope="module")  # Once per test file
   def database():
       return setup_db()
   ```

3. **Clean up resources**

   ```python
   @pytest.fixture
   def temp_file(tmp_path):
       file = tmp_path / "test.txt"
       file.write_text("data")
       yield file
       # Cleanup happens automatically for tmp_path
   ```

### CI/CD Best Practices

1. **Fast feedback**
   - Run unit tests first
   - Parallel test execution
   - Cache dependencies

2. **Informative failures**
   - Clear error messages
   - Upload logs/artifacts
   - Coverage reports

3. **Don't skip CI**
   - Every PR runs tests
   - Every push to main runs tests
   - Enforce branch protection

4. **Monitor and optimize**
   - Track CI duration
   - Identify slow tests
   - Balance thoroughness and speed

### Continuous Improvement

1. **Add tests with every PR**
   - New features: new tests
   - Bug fixes: regression tests

2. **Review test quality**
   - Are tests clear?
   - Do they test the right things?
   - Are they maintainable?

3. **Refactor tests**
   - Like production code
   - Remove duplication
   - Improve clarity

4. **Share knowledge**
   - Document testing patterns
   - Review each other's tests
   - Discuss testing strategy

---

## Skipped Tests (Placeholders and Environment-Limited)

Individual tests (not whole modules) are skipped with `@pytest.mark.skip` where implementation is deferred or the test needs binaries/network that PR CI does not provide. Current skip reasons include:

- **JANUS/SOCRATES**: `test_smoke_janus.py` (runtime instability); `test_smoke_atmos_interior.py` (JANUS/SOCRATES binaries)
- **AGNI**: `test_smoke_atmos_interior.py` (AGNI/Julia binaries)
- **CALLIOPE**: `test_smoke_outgassing.py` (slow, reserved for nightly)
- **Interior**: `test_interior.py` (one test: corefrac=1.0 raises by design)
- **Data**: `test_download_robustness.py` (network/OSF/Zenodo)
- **Utils**: `test_data.py` (network-dependent)

CI runs `pytest -m "unit and not skip"` and `pytest -m "smoke and not skip"`, so skipped tests are excluded from PR coverage and smoke runs. To activate a test: remove or narrow `@pytest.mark.skip`, add the right marker (`unit` / `smoke` / `integration`), and follow the [Test Writing Guidelines](#test-writing-guidelines). See [Test Categorization](test_categorization.md) for marker semantics and CI flow.

---

## Monitoring & Maintenance

**For Maintainers:**

1. **CI/CD Performance:**
   - PR checks should complete in <10 minutes
   - Nightly validation in ~4-6 hours
   - Monitor GitHub Actions for failures

2. **Coverage Tracking:**
   - Fast gate: 32.03% (`[tool.proteus.coverage_fast]`), ratcheted on push to main / `tl/test_ecosystem_v5`
   - Full gate: 69% (`[tool.coverage.report]`), ratcheted on main in nightly
   - Tests marked `skip` are excluded from PR runs (`unit and not skip`, `smoke and not skip`)
   - Review coverage in HTML artifacts and Codecov

3. **Documentation Updates:**
   - Update this guide when adding new test markers
   - Document changes to test requirements
   - Keep CI/CD strategy documentation current

4. **Test Quality:**
   - Review new tests in PRs for clarity and correctness
   - Ensure tests are maintainable and well-documented
   - Refactor tests as needed (like production code)

---

## References

- **pytest:** <https://docs.pytest.org/>
- **coverage.py:** <https://coverage.readthedocs.io/>
- **GitHub Actions:** <https://docs.github.com/en/actions>
- **Reusable Workflows:** <https://docs.github.com/en/actions/using-workflows/reusing-workflows>
- **ruff:** <https://docs.astral.sh/ruff/>

---

## Checklist: Ready to Deploy Quality Gate

### For New Ecosystem Module

- [ ] **Setup** (30 min)
  - [ ] Copy PROTEUS pyproject.toml pytest/coverage sections
  - [ ] Create/update `.github/workflows/ci_tests.yml`
  - [ ] Set initial `fail_under` threshold (20-30%)
  - [ ] Consider implementing automatic ratcheting (tools/update_coverage_threshold.py)
  - [ ] Add pytest-cov to develop dependencies

- [ ] **Test Structure** (30-60 min)
  - [ ] Run `bash tools/validate_test_structure.sh`
  - [ ] Run `bash tools/restructure_tests.sh` if needed
  - [ ] Verify `tests/` mirrors `src/`
  - [ ] Create basic placeholder tests

- [ ] **Local Validation** (15 min)
  - [ ] `pytest` runs without errors
  - [ ] Coverage report generated
  - [ ] Coverage meets threshold
  - [ ] All markers work (@pytest.mark.unit, .integration, .slow)

- [ ] **CI Validation** (10 min)
  - [ ] Push to GitHub
  - [ ] CI workflow runs successfully
  - [ ] Coverage reported correctly
  - [ ] All checks pass

- [ ] **Documentation** (15 min)
  - [ ] Update README with coverage badge
  - [ ] Document testing approach
  - [ ] Link to this guide

**Total Time:** ~2 hours per module

---

**Maintained by:** FormingWorlds team  
**Last updated:** 2026-01-27  
**Questions?** Open an issue on GitHub
