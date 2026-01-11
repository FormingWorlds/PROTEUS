# Testing Infrastructure

This document describes the standardized testing infrastructure for PROTEUS and the wider ecosystem.

> **Related documentation:**
>
> - [Test Categorization](test_categorization.md) — Test markers and CI/CD pipeline flow
> - [Docker CI Architecture](docker_ci_architecture.md) — Detailed Dockerfile, image build strategy, and implementation reference
> - [Test building](test_building.md) — Follow best practices for building robust, physics-compliant tests
> - [conftest.py reference](/test/conftest.py) — Fixtures and parameter classes for common test scenarios

## Table of Contents

1. [CI/CD Status and Roadmap](#cicd-status-and-roadmap-as-of-2026-01-11)
2. [Quick Start](#quick-start)
3. [Developer Workflow](#developer-workflow)
4. [Coverage Analysis](#coverage-analysis-workflow)
5. [Pre-commit Checklist](#pre-commit-checklist)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

---

## CI/CD Status and Roadmap (as of 2026-01-11)

### CI/CD Current Status

**Implementation Status**: Fast PR workflow complete and passing ✓ — **Near 30% coverage target**

- ✓ Unit Tests (mocked physics): **457 tests**, ~1 min runtime
- ✓ Smoke Tests (real binaries): 1 test, ~2 min runtime
- ✓ Code Quality (ruff): Pass
- ✓ Coverage tracking: **29.52%** (fast gate, auto-ratcheting from 18% → 22% → 23% → **29.52%**)
- ✓ Diff-cover: Changed-lines coverage validation (`--diff-file` approach)
- ✓ Coverage ratcheting: **Automatic threshold increase on improvements** (github-actions bot commits)

**Key Achievements**:

1. **Massive test expansion**: 13 → **457 tests** (35x increase in 1 day)
2. **Coverage improvement**: 18.51% → **29.52%** (+11 percentage points)
3. Auto-ratcheting mechanism working perfectly (4 automatic threshold increases)
4. All three CI jobs pass cleanly (unit, smoke, lint)
5. Comprehensive wrapper test coverage: atmos_chem, atmos_clim, escape, interior, observe, outgas
6. Complete config validator coverage: 89 tests across all validation modules

**Known Issues**:

- Codecov upload fails with "Token required because branch is protected" (non-blocking)
- GPG verification warnings from codecov action (non-critical)

### CI/CD Architecture (Docker-based)

- Prebuilt image: `ghcr.io/formingworlds/proteus:latest` built nightly (~02:00 UTC) via `.github/workflows/docker-build.yml` and on main dependency changes
- Feature branch images: Some feature branches use branch-tagged images (e.g., `ghcr.io/formingworlds/proteus:tl-test_ecosystem_v4`) to validate Docker changes before promotion to `latest`.
- Environment: compiled physics (SOCRATES, AGNI, PETSc) + Python deps from `pyproject.toml`; data paths set via `FWL_DATA`, `RAD_DIR`, `AGNI_DIR`, `PETSC_DIR`
- PR workflow: `ci-pr-checks.yml` runs inside the image, overlays PR code, and uses make for smart rebuilds of changed sources
- Nightly workflow: `ci-nightly-science.yml` uses the same image for integration and slow tests, with coverage ratcheting
- Artifacts: coverage XML/HTML uploaded; Codecov upload is non-blocking when token is missing

### Current Metrics (as of 2026-01-11)

| Metric | Value | Target | Status |
| --- | --- | --- | --- |
| Unit tests | **457** | 470+ | **97%** ✓ |
| Smoke tests | 1 | 5–7 | 14–20% |
| Integration tests | 0 | 23 | 0% |
| Coverage (unit) | **29.52%** | 30% | **98%** ✓ |
| CI runtime (fast) | ~3 min | <10 min | ✓ Excellent |
| Diff-cover threshold | 80% | 80% | ✓ Met |

### Immediate Next Steps

**This week (Final push to 30%+)**:

1. **Add 10-15 unit tests** (2-3 hours)
   - Config dataclass defaults (10-15 tests) → +0.3-0.5%
   - Atmos_clim common functions (5-10 tests) → +0.2-0.3%
   - High-impact utils (5-10 tests) → +0.2-0.3%
   - **Target**: Cross 30.0% threshold

2. **Expand smoke tests** (1–2 hours)
   - Add JANUS + dummy interior coupling test
   - Add MORS stellar evolution test
   - Expected: 1 → 3–4 active smoke tests

3. **Fix Codecov integration** (30 minutes)
   - Add `CODECOV_TOKEN` to GitHub repository secrets
   - Enable full upload instead of non-blocking failures

**Next week (Polish & documentation)**:

1. Merge feature branch to main (all checks passing)
2. Update documentation with final metrics
3. Plan integration test strategy for nightly CI

**Following weeks (Integration & slow tests)**:

1. Implement integration tests (JANUS + ARAGOG coupling, multi-step evolution with feedback)
2. Plan slow test strategy (3–4 hour budget per scenario: Earth magma ocean, Venus runaway greenhouse, Super-Earth evolution)
3. Configure nightly notifications (email on test failures; optional Slack integration)

### Planned Improvements

#### Phase 1: Fast PR Workflow Enhancements (✓ Substantially Complete)

> **Status:** 30% coverage target nearly achieved (29.52% → 30.0% estimated within 1-2 days)

**1.1 Expand Smoke Tests** (Estimated: 1-2 hours remaining)

- ✅ Core smoke test implemented (minimal initialization check)
- ⏳ Remaining: Add smoke tests for each major module (SPIDER, JANUS, AGNI, SOCRATES, ARAGOG)
- Use minimal timestep runs with dummy config or fast fixtures
- Early detection of binary incompatibilities with code changes

**1.2 Unit Test Coverage Expansion** (✅ 97% Complete: 457/470+ tests)

- ✅ Increased coverage from 18.51% → 29.52% (11.01 percentage points)
- ✅ Completed modules:
  - Config validators: 89 comprehensive tests
  - Module wrappers: 107 tests (CALLIOPE outgas, ZEPHYRUS escape, JANUS+SPIDER observe)
  - Utils and helpers: Multiple utility function suites
- ⏳ Remaining to 30%: 10-15 tests (config dataclass defaults, atmos_clim commons, high-impact utils)
- Strategy: Mock-based unit tests for Python logic (unittest.mock for external dependencies)

**1.3 Performance Optimization** (✅ Complete)

- ✅ Fast CI runtime maintained at ~3 minutes (well under 10 minute target)
- ✅ Parallel job execution (code-quality + unit-tests + smoke-tests)
- ✅ Optimized Docker image pull/startup

#### Phase 2: Nightly Science Validation (`ci-nightly-science.yml`)

**2.1 Integration Test Suite** (Estimated: 1–2 weeks)

- Implement multi-module coupling tests
- Example: PROTEUS with dummy modules, JANUS + ARAGOG, AGNI + SOCRATES
- Runtime: ~30 min to 2 hours per test suite

**2.2 Slow Science Validation Tests** (Estimated: 2–4 weeks)

- Comprehensive physics accuracy validation
- Examples: Earth magma ocean solidification, Venus runaway greenhouse, Super-Earth evolution
- Marker: `@pytest.mark.slow`
- Budget: 3 hour limit for nightly runs

**2.3 Full Coverage Ratcheting** (Estimated: 1 day)

- Implement automatic threshold increase for full test suite
- Use `tools/update_coverage_threshold.py` on nightly runs
- Target: 69% threshold

**2.4 Nightly Notifications** (Estimated: 4 hours)

- Email notifications for failed tests
- Slack integration
- GitHub annotations with failure details

#### Phase 3: Long-Term Improvements (Future)

- Regression testing (baseline metrics tracking)
- Multi-OS testing (Windows/macOS CI jobs)
- Ecosystem test harmonization (CALLIOPE, JANUS, MORS, VULCAN, ZEPHYRUS)

### Success Metrics

**Fast PR Workflow** (✓ Substantially achieved):

- ✅ 457 unit tests, passing consistently (target: 470+)
- ✅ 1 smoke test, validating binary initialization (target: 5-7)
- ✅ All ruff checks passing (lint + format)
- ✅ Automated coverage threshold enforcement via auto-ratcheting (18.51% → 29.52%)
- ✅ CI runtime ~3 minutes (well under 10 minute target)

**Nightly Workflow** (Next priority, timeline: 2–3 weeks):

- 5+ integration tests (multi-module coupling)
- 30%+ overall coverage (currently at 29.52%)
- Runtime <4 hours total

**End Goal** (by Q2 2026):

- 50%+ unit test coverage
- 20+ integration tests
- 5–7 smoke tests (one per major module)
- 5+ slow/science validation tests
- 3+ other modules using same CI infrastructure

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

1. ✅ Pull pre-built Docker image (instant)
2. ✅ Overlay your code changes (seconds)
3. ✅ Smart rebuild (only changed files, seconds to minutes)
4. ✅ Run unit tests (2–5 minutes)
5. ✅ Run smoke tests (5–10 minutes)
6. ✅ Report back (~10–15 minutes total)

**Impact**: ~60 minutes compilation → ~10–15 minutes for Python changes

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

# Run unit tests (fast)
pytest -m unit

# Run unit + smoke (what PR checks run)
pytest -m "unit or smoke"

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

**Smart Rebuild**: The system only recompiles files that changed:

```bash
# In ci-pr-checks.yml
cd SPIDER
make -q || make -j$(nproc)  # Only rebuild if needed
```

- Python-only PR: No recompilation (~instant)
- Fortran PR: Only changed files (~minutes, not hours)

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

1. **Create source file:** `src/<package>/<module>.py`
2. **Create test file:** `tests/<module>/test_<module>.py`
3. **Write tests first** (TDD) or alongside code
4. **Run tests:** `pytest tests/<module>/`
5. **Check coverage:** `pytest --cov=src/<package>/<module>`
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

- [ ] All tests pass locally
- [ ] Coverage meets threshold
- [ ] No linting errors: `ruff check src/ tests/`
- [ ] Code formatted: `ruff format src/ tests/`
- [ ] New tests added for new code
- [ ] Test structure validated

---

## Implementation Phases

### Phase 2: Expand Testing Coverage (Next 2–3 weeks)

**2.1 Test Docker Image Build** (Estimated: 1–2 hours)

```bash
# Build locally to verify Dockerfile works
docker build -t proteus-test .

# Test the image
docker run -it proteus-test bash
# Inside container:
pytest -m unit
```

**2.2 Push Branch and Monitor** (Estimated: 1 hour)

```bash
git push origin tl/test_ecosystem_v4
```

- Watch GitHub Actions for docker-build.yml
- Verify image pushes to ghcr.io
- Check if it's publicly accessible

**2.3 Test PR Workflow** (Estimated: 2–4 hours)

- Create test PR from this branch
- Verify ci-pr-checks.yml runs
- Check timing improvements
- Validate test results

### Phase 3: Ecosystem Integration (Following week and beyond)

**3.1 Add Test Markers** (Estimated: 1–2 days)

- Mark existing tests with `@pytest.mark.unit`, `@pytest.mark.smoke`, etc.
- Start with critical modules (see placeholder list above)

**3.2 Parallel Run** (Estimated: 3–5 days)

- Keep existing CI active
- Run both systems in parallel
- Compare results and timing

**3.3 Full Transition** (Estimated: 1 week)

- Once validated, deprecate old CI
- Update documentation
- Train team on new system

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
   - **Full integration tests**: Check `pyproject.toml` `[tool.coverage.report]` `fail_under` (currently 69%)
   - **Fast unit tests**: Check `pyproject.toml` `[tool.proteus.coverage_fast]` `fail_under` (currently 22.42%)
   - **Auto-ratcheting**: Both thresholds automatically increase when coverage improves (never decreases)
     - Fast gate ratchets on all branches and auto-commits to branch (encourages unit test coverage)
     - Full gate ratchets only on main branch and auto-commits (production quality standard)
     - Ratcheting uses `tools/update_coverage_threshold.py` with appropriate target (`--target fast` or `--target full`)
     - Commits are made by `github-actions[bot]` with `[skip ci]` to avoid infinite loops
   - All PRs must pass the coverage threshold defined in CI

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
   - Tests in `tests/<package>/test_<module>.py` match `src/<package>/<module>.py`
   - Easy to find related tests
   - Consistent across entire ecosystem
   - Enables automated validation

2. **One test file per source file**
   - When practical
   - Keeps tests organized
   - Clear 1:1 mapping
   - Use `tools/validate_test_structure.sh` to verify

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

## Placeholder Test Modules

Nine modules have placeholder tests requiring implementation: `escape`, `orbit`, `interior`, `atmos_clim`, `outgas`, `utils`, `observe`, `star`, `atmos_chem`.

To implement: Remove `@pytest.mark.skip`, add markers (`@pytest.mark.unit` or `@pytest.mark.integration`), and write tests following the [Test Writing Guidelines](#test-writing-guidelines). See [Test Categorization](test_categorization.md) for examples.

---

## Monitoring & Maintenance

**For Maintainers:**

1. **CI/CD Performance:**
   - PR checks should complete in <10 minutes
   - Nightly validation in ~4-6 hours
   - Monitor GitHub Actions for failures

2. **Coverage Tracking:**
   - Auto-ratcheting enforces quality (69% current threshold)
   - Placeholder tests excluded from coverage
   - Review coverage reports on Codecov

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
**Last updated:** January 2026
**Questions?** Open an issue on GitHub
