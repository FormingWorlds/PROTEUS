# Testing Infrastructure

This document describes the standardized testing infrastructure for PROTEUS and the wider ecosystem.

> **Related documentation:** For details on test categories and CI/CD workflow specifics, see [Test Categorization](test_categorization.md).

## Table of Contents

1. [CI/CD Status and Roadmap](#cicd-status-and-roadmap-as-of-2026-01-06)
2. [Developer Workflow](#developer-workflow)
3. [Coverage Analysis](#coverage-analysis-workflow)
4. [Pre-commit Checklist](#pre-commit-checklist)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)

---

## CI/CD Status and Roadmap (as of 2026-01-06)

### CI/CD Current Status

- Fast PR workflow (`ci-pr-checks.yml`): passing (ruff, 10 unit tests, 1 smoke test)
- Coverage: 18.51% overall; gates — fast: 18%, full: 69%, diff-cover (changed lines): 80%
- Diff-cover uses a workspace-generated diff (`git diff origin/${BASE_REF}...HEAD` + `--diff-file`) to avoid remote fetch issues inside the container
- Smoke coverage: PROTEUS initialization via `tests/integration/test_smoke_minimal.py::test_proteus_dummy_init` using `input/demos/dummy.toml`
- Known issues: Codecov upload needs `CODECOV_TOKEN` on protected branches (currently non-blocking)

### CI/CD Architecture (Docker-based)

- Prebuilt image: `ghcr.io/formingworlds/proteus:latest` built nightly (~02:00 UTC) via `.github/workflows/docker-build.yml` and on main dependency changes
- Environment: compiled physics (SOCRATES, AGNI, PETSc) + Python deps from `pyproject.toml`; data paths set via `FWL_DATA`, `RAD_DIR`, `AGNI_DIR`, `PETSC_DIR`
- PR workflow: `ci-pr-checks.yml` runs inside the image, overlays PR code, and uses make for smart rebuilds of changed sources
- Nightly workflow: `ci-nightly-science.yml` uses the same image for integration and slow tests, with coverage ratcheting
- Artifacts: coverage XML/HTML uploaded; Codecov upload is non-blocking when token is missing

### Current Metrics (as of 2026-01-06)

| Metric | Value | Target | Status |
| --- | --- | --- | --- |
| Unit tests | 10 | 23 | 43% |
| Smoke tests | 1 | 5–7 | 14–20% |
| Integration tests | 0 | 23 | 0% |
| Coverage (unit) | 18.51% | 50% | 37% |
| CI runtime (fast) | ~9 min | <10 min | ✓ On target |
| Diff-cover threshold | 80% | 80% | ✓ Met |

### Immediate Next Steps

**This week (Fast PR enhancements)**:

1. **Merge feature branch** → `main`
   - All checks passing; recommend branch cleanup after merge

2. **Expand smoke tests** (1–2 hours)
   - Add test for PROTEUS + JANUS coupling; keep runtime <10 seconds
   - Expected: 1 → 3–5 smoke tests

3. **Fix Codecov integration** (30 minutes)
   - Add `CODECOV_TOKEN` to GitHub repository secrets
   - Enable full upload instead of non-blocking failures

**Next week (Nightly workflow)**:

1. Implement integration tests (JANUS + ARAGOG coupling, multi-step evolution with feedback)
2. Plan slow test strategy (3–4 hour budget per scenario: Earth magma ocean, Venus runaway greenhouse, Super-Earth evolution)
3. Configure nightly notifications (email on test failures; optional Slack integration)

**Following week (Coverage expansion)**:

1. Unit test coverage targets: grid management 7.6% → 50%, plotting modules 5–23% → 40%
2. Aim for overall 30% fast gate (from 18%) and 70%+ full gate (approaching ecosystem average)

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

```python
"""Test fixtures and configuration"""
from __future__ import annotations

import pytest


@pytest.fixture
def sample_data():
    """Provide sample data for tests"""
    return {"key": "value", "number": 42}


@pytest.fixture
def temp_directory(tmp_path):
    """Provide a temporary directory for tests"""
    return tmp_path / "test_dir"
```

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
   - Check `pyproject.toml` `tool.coverage.report` `fail_under` for current threshold
   - Coverage threshold automatically increases on main branch (never decreases)
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

1. **One test file per source file**
   - When practical
   - Keeps tests organized
   - Clear 1:1 mapping
   - Use `tools/validate_test_structure.sh` to verify

1. **Group related tests**
   - Use test classes for related tests
   - Share fixtures via `conftest.py`
   - Organize by functionality within test files

1. **Use descriptive names**

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

**Benefits of markers:**
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

## Current State & Next Steps

### Current Test Suite Status (January 2026)

PROTEUS has completed a comprehensive test cleanup and categorization effort. For detailed information about each test category and CI/CD pipeline flow, see [Test Categorization](test_categorization.md).

**Test Breakdown:**
- **Unit tests:** 23 tests (marked with `@pytest.mark.unit`)
  - Fast tests with mocked physics for rapid PR feedback
  - Target: <100ms per test
  - Located in: `tests/config/`, `tests/grid/`, `tests/plot/`, `tests/inference/`

- **Integration tests:** 23 tests (marked with `@pytest.mark.integration`)
  - Multi-module coupling and workflow validation
  - Duration: ~2 hours for all integration tests
  - Located in: `tests/integration/`

- **Placeholder tests:** 9 modules marked with `@pytest.mark.skip`
  - Need implementation (currently just structural placeholders)
  - Modules: escape, orbit, interior, atmos_clim, outgas, utils, observe, star, atmos_chem

**CI/CD Integration:**
- **PR Checks** (`ci-pr-checks.yml`): Runs only unit tests (~5-10 minutes)
  - Command: `pytest -m "unit and not skip" --ignore=tests/examples --cov-fail-under=69`
  - Provides rapid feedback for contributors
  - Enforces 69% coverage threshold (auto-ratcheting)

- **Nightly Validation** (`ci-nightly-science.yml`): Runs integration tests (~4-6 hours)
  - Job 1: `pytest -m "slow or integration" --ignore=tests/examples`
  - Job 2: `pytest -m "integration and not slow" --ignore=tests/examples`
  - Comprehensive physics validation

### Implementing Placeholder Tests

The following 9 modules currently have placeholder tests that need implementation:

1. **tests/escape/test_escape.py** - Atmospheric escape module tests
2. **tests/orbit/test_orbit.py** - Orbital mechanics module tests
3. **tests/interior/test_interior.py** - Interior evolution module tests
4. **tests/atmos_clim/test_atmos_clim.py** - Atmospheric climate module tests
5. **tests/outgas/test_outgas.py** - Outgassing module tests
6. **tests/utils/test_utils.py** - Utility functions tests
7. **tests/observe/test_observe.py** - Observation module tests
8. **tests/star/test_star.py** - Stellar evolution module tests
9. **tests/atmos_chem/test_atmos_chem.py** - Atmospheric chemistry module tests

**Implementation Checklist:**

For each placeholder test:
- [ ] Remove `@pytest.mark.skip` decorator
- [ ] Add appropriate marker (`@pytest.mark.unit` or `@pytest.mark.integration`)
- [ ] Write real test functions that:
  - Test actual module functionality
  - Use mocking for external dependencies (unit tests)
  - Validate physics with appropriate tolerances
  - Run in <100ms (unit) or document longer duration (integration)
- [ ] Add comprehensive docstrings and comments
- [ ] Verify coverage contribution
- [ ] Run locally before pushing: `pytest tests/<module>/test_<module>.py -v`

**Example transformation:**

```python
# BEFORE (placeholder):
import pytest

@pytest.mark.skip(reason="Placeholder test - implement real tests for escape module")
def test_escape_placeholder():
    pass

# AFTER (real unit test):
import pytest
from unittest.mock import patch, MagicMock
from proteus.escape import calculate_escape_rate

@pytest.mark.unit
def test_calculate_escape_rate_basic():
    """Test basic escape rate calculation with mocked atmosphere."""
    # Mock atmospheric parameters
    with patch('proteus.escape.get_atmosphere') as mock_atm:
        mock_atm.return_value = MagicMock(
            temperature=300,  # K
            pressure=1e5,     # Pa
            composition={'H2': 0.9, 'He': 0.1}
        )

        rate = calculate_escape_rate(planet_mass=1e24, planet_radius=6e6)

        # Verify escape rate is physically reasonable
        assert rate > 0, "Escape rate must be positive"
        assert rate < 1e10, "Escape rate exceeds physical limits"
```

### Running Tests Locally

**Quick validation:**
```bash
# Run only unit tests (fast: ~30 seconds)
pytest -m "unit and not skip" --ignore=tests/examples

# Run integration tests (slow: ~2 hours)
pytest -m "integration and not slow" --ignore=tests/examples

# Run all tests except placeholders
pytest -m "not skip" --ignore=tests/examples

# Check specific module
pytest tests/escape/test_escape.py -v
```

**With coverage:**
```bash
# PR-equivalent check
pytest -m "unit and not skip" --ignore=tests/examples --cov=src --cov-fail-under=69

# Full coverage report
pytest --cov --cov-report=html
open htmlcov/index.html
```

### Contributing Guidelines

When adding new tests to PROTEUS:

1. **Choose appropriate marker:**
   - `@pytest.mark.unit` for fast, mocked tests (<100ms)
   - `@pytest.mark.integration` for multi-module tests (seconds to minutes)
   - `@pytest.mark.slow` for hours-long simulations
   - `@pytest.mark.smoke` for quick binary validation

2. **Run tests locally before pushing:**
   ```bash
   pytest -m unit                    # Fast: ~30 seconds
   pytest -m "not slow"              # Medium: ~5 minutes
   pytest -m "slow or integration"   # Comprehensive: 2-4 hours
   ```

3. **For placeholder implementation:**
   - See checklist above
   - Reference existing tests in `tests/config/` and `tests/integration/`
   - Ask for review from maintainers

4. **Documentation:**
   - Add docstrings explaining what each test validates
   - Include physical context where relevant
   - Comment on expected values and tolerances

### Monitoring & Maintenance

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
