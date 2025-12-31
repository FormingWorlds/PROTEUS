# Testing Infrastructure

This document describes the standardized testing infrastructure for PROTEUS and the wider ecosystem.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Configuration](#configuration)
4. [Ecosystem Rollout](#ecosystem-rollout)
5. [Developer Workflow](#developer-workflow)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

---

## Quick Start

### Prerequisites

Ensure you have Python 3.11+ and the repository cloned.

### For PROTEUS

```bash
# 1. Install development dependencies (includes pytest-cov)
pip install -e ".[develop]"

# 2. Validate current test structure
bash tools/validate_test_structure.sh

# 3. Restructure tests to mirror source layout (if needed)
bash tools/restructure_tests.sh

# 4. Run tests with coverage
pytest

# 5. View detailed coverage report
open htmlcov/index.html

# 6. Analyze coverage by module
bash tools/coverage_analysis.sh
```

### For Submodules (CALLIOPE, JANUS, MORS, etc.)

```bash
# Navigate to submodule
cd <submodule-directory>

# Install dependencies
pip install -e ".[develop]"

# Run tests
pytest --cov

# View coverage
open htmlcov/index.html
```

### Common Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m "not slow"        # Skip slow tests

# Run specific module
pytest tests/config/

# Check test discovery
pytest --collect-only

# Coverage with missing lines
pytest --cov --cov-report=term-missing
```

---

## Architecture Overview

### System Design

The testing infrastructure consists of three main components:

#### 1. Test Structure
- **Principle:** Tests mirror source code structure exactly
- **Location:** `tests/` directory with subdirectories matching `src/<package>/`
- **Organization:** One test file per source file when possible
- **Benefits:** Predictable, navigable, maintainable

**Example:**
```
src/proteus/
├── config/
│   ├── __init__.py
│   └── _config.py
├── interior/
│   ├── __init__.py
│   └── wrapper.py
└── plot/
    ├── __init__.py
    └── cpl_global.py

tests/
├── config/
│   ├── __init__.py
│   └── test_config.py
├── interior/
│   ├── __init__.py
│   └── test_wrapper.py
└── plot/
    ├── __init__.py
    └── test_cpl_global.py
```

#### 2. Configuration (pyproject.toml)

**pytest Configuration:**
```toml
[tool.pytest.ini_options]
minversion = "8.1"
addopts = [
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
    "--strict-markers",
    "--strict-config",
    "-ra",
    "--showlocals",
]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
```

**Coverage Configuration:**
```toml
[tool.coverage.run]
branch = true
source = ["<package_name>"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__pycache__/*",
    "*/conftest.py",
]

[tool.coverage.report]
fail_under = 5  # Adjust based on current coverage
show_missing = true
precision = 2
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "if typing.TYPE_CHECKING:",
    "@abstractmethod",
    "@abc.abstractmethod",
]

[tool.coverage.html]
directory = "htmlcov"
```

#### 3. CI/CD Pipeline

**GitHub Actions Workflows:**

1. **Main CI Workflow** (`.github/workflows/ci.yml`)
   - Matrix testing: Python 3.11, 3.12, 3.13
   - Runs pytest with coverage
   - Linting with ruff
   - Uploads coverage to Codecov
   - Generates HTML artifacts

2. **Reusable Quality Gate** (`.github/workflows/proteus_test_quality_gate.yml`)
   - Centralized workflow for all PROTEUS modules/repositories
   - Configurable Python version and coverage threshold
   - Can be called by submodule workflows

**CI/CD Flow:**
```
Push/PR → GitHub Actions
    ↓
Matrix Testing (3.11, 3.12, 3.13)
    ↓
Run pytest --cov
    ↓
Check coverage threshold
    ↓
Upload reports (Codecov, HTML)
    ↓
Lint with ruff
    ↓
Pass/Fail → Merge gate
```

### Available Tools

#### 1. `tools/validate_test_structure.sh`
- **Purpose:** Check if tests mirror source structure
- **Output:** Color-coded report of missing directories/files
- **Usage:** `bash tools/validate_test_structure.sh`

#### 2. `tools/restructure_tests.sh`
- **Purpose:** Automatically reorganize tests to mirror source
- **Actions:**
  - Creates missing directories
  - Moves misplaced test files
  - Adds `__init__.py` files
  - Creates placeholder tests
- **Usage:** `bash tools/restructure_tests.sh`

#### 3. `tools/coverage_analysis.sh`
- **Purpose:** Analyze coverage by module and identify priorities
- **Output:** Module-by-module coverage with priority list
- **Usage:** `bash tools/coverage_analysis.sh`

---

## Configuration

### Project Setup

**Required Files:**

1. **pyproject.toml**
   - Add pytest and coverage configurations (see Architecture section)
   - Include `pytest-cov` in `[project.optional-dependencies]`

2. **.github/workflows/ci.yml**
   - Set up matrix testing
   - Configure coverage threshold
   - Add linting step

3. **tests/conftest.py**
   - Define shared fixtures
   - Configure pytest plugins
   - Set up test helpers

4. **.gitignore**
   ```
   .pytest_cache/
   .coverage
   htmlcov/
   coverage.xml
   ```

### Dependencies

**Required packages in `[project.optional-dependencies]`:**
```toml
develop = [
    "pytest >= 8.1",
    "pytest-cov",
    "coverage[toml]",
    # ... other dev dependencies
]
```

### Test Markers

Define markers in `pyproject.toml`:

```toml
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
```

**Usage in tests:**
```python
import pytest

@pytest.mark.unit
def test_basic_functionality():
    assert True

@pytest.mark.integration
def test_module_interaction():
    # Test multiple components together
    pass

@pytest.mark.slow
def test_long_running_process():
    # Tests that take significant time
    pass
```

---

## Ecosystem Rollout

### PROTEUS Ecosystem Components

The testing infrastructure is designed for:
- **PROTEUS** - Main coupling framework
- **CALLIOPE** - Outgassing module
- **JANUS** - Atmosphere-climate module
- **MORS** - Stellar evolution module
- **VULCAN** - Atmospheric chemistry module
- **ZEPHYRUS** - Escape module
- **Zalmoxis** - Interior evolution module
- **aragog** - Interior module (alternative)

To be adapted for future modules as needed:
- **AGNI**
- **OBLIQUA**
- ..

### Rollout Strategy

#### Phase 1: PROTEUS (Main Repository)

1. **Setup Infrastructure**
   - ✅ Create reusable workflow
   - ✅ Create CI workflow
   - ✅ Update pyproject.toml
   - ✅ Create tools (restructure, validate, analyze)
   - ✅ Create documentation

2. **Implement Testing**
   - Run validation script
   - Run restructuring script
   - Add tests to placeholder files
   - Run tests locally
   - Commit and push
   - Verify CI passes

3. **Establish Baseline**
   - Measure current coverage
   - Set realistic threshold
   - Document coverage gaps
   - Create improvement plan

#### Phase 2: Submodules (Parallel Rollout)

For each submodule (CALLIOPE, JANUS, MORS, VULCAN, ZEPHYRUS, Zalmoxis, aragog, etc.):

**1. Configuration Setup**

Copy and adapt from PROTEUS:

**pyproject.toml additions:**
```toml
[tool.coverage.run]
branch = true
source = ["<package_name>"]  # Change to: calliope, janus, mors, etc.
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__pycache__/*",
    "*/conftest.py",
]

[tool.pytest.ini_options]
minversion = "8.1"
addopts = [
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
    "--strict-markers",
    "--strict-config",
    "-ra",
    "--showlocals",
]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]

[tool.coverage.report]
fail_under = 5  # Adjust based on current coverage
show_missing = true
precision = 2
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "@abstractmethod",
]

[tool.coverage.html]
directory = "htmlcov"

# In [project.optional-dependencies]
develop = [
    "pytest >= 8.1",
    "pytest-cov",
    "coverage[toml]",
    # ... existing dependencies
]
```

**2. CI Workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  workflow_dispatch:

jobs:
  test-matrix:
    name: Test Suite
    strategy:
      fail-fast: false
      matrix:
        python-version: ['3.11', '3.12', '3.13']
        os: [ubuntu-latest]

    runs-on: ${{ matrix.os }}

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[develop]"

      - name: Display installed packages
        run: pip list

      - name: Run tests with coverage
        run: |
          pytest \
            --cov=src \
            --cov-report=term-missing \
            --cov-report=xml \
            --cov-report=html \
            --cov-fail-under=5 \
            tests/

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        if: matrix.python-version == '3.11'
        with:
          files: ./coverage.xml
          flags: unittests
          fail_ci_if_error: false
        env:
          CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}

      - name: Upload coverage HTML report
        uses: actions/upload-artifact@v4
        if: matrix.python-version == '3.11'
        with:
          name: coverage-report
          path: htmlcov/
          retention-days: 30

  lint:
    name: Code Quality
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[develop]"

      - name: Run ruff linting
        run: ruff check src/ tests/ || true

      - name: Run ruff formatting check
        run: ruff format --check src/ tests/ || true
```

**3. Test Structure**

Organize tests to mirror source:

```bash
# Analyze current structure
find src/<package> -type d
find tests -type d

# Create missing directories
mkdir -p tests/<matching_structure>

# Add __init__.py files
find tests -type d -exec touch {}/__init__.py \;

# Create basic test files
# tests/<module>/test_<module>.py
```

**4. Validation Checklist**

For each submodule:

- [ ] Configuration added to pyproject.toml
- [ ] CI workflow created
- [ ] Test dependencies installed
- [ ] Test structure mirrors source
- [ ] tests/conftest.py created
- [ ] .gitignore updated
- [ ] Tests run locally: `pytest --cov`
- [ ] Coverage meets threshold
- [ ] CI passes on push
- [ ] Codecov integration (optional)

#### Phase 3: Monitoring & Improvement

**Continuous Tasks:**

1. **Coverage Tracking**
   - Monitor trends weekly
   - Create issues for gaps
   - Gradually increase thresholds

2. **Documentation**
   - Add badges to READMEs
   - Update contributor guides
   - Create testing examples

3. **Maintenance**
   - Review workflows quarterly
   - Update Python versions
   - Optimize CI performance
   - Refactor as code evolves

**Coverage Improvement Strategy:**

```
Current State → Baseline (e.g., 5-50%)
    ↓
Year 1: +10-20% increase
    ↓
Year 2: +10-20% increase
    ↓
Goal: 80%+ coverage
```

Adjust thresholds gradually:
```toml
# Start realistic
fail_under = 5  # or current coverage

# Increase quarterly/semi-annually
fail_under = 20  # Q2
fail_under = 40  # Q4
fail_under = 60  # Year 2
fail_under = 80  # Long-term goal
```

---

## Developer Workflow

### Local Development Cycle

```
1. Write/modify code
    ↓
2. Write/update tests
    ↓
3. Run tests locally: pytest
    ↓
4. Check coverage: pytest --cov
    ↓
5. Fix failing tests
    ↓
6. Commit changes
    ↓
7. Push → CI runs automatically
    ↓
8. Monitor CI results
```

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
3. Check pytest documentation: https://docs.pytest.org/
4. Check coverage.py documentation: https://coverage.readthedocs.io/
5. Open an issue on GitHub

---

## Best Practices

### Testing Philosophy

1. **Test behavior, not implementation**
   - Focus on what code does, not how
   - Tests should survive refactoring

2. **Write tests first (TDD)**
   - Clarifies requirements
   - Ensures testability
   - Provides instant feedback

3. **Keep tests simple and focused**
   - One concept per test
   - Clear test names
   - Easy to understand

4. **Use appropriate test types**
   - Unit tests: Single functions/methods
   - Integration tests: Multiple components
   - System tests: End-to-end workflows

### Test Organization

1. **Mirror source structure**
   - Easy to find related tests
   - Consistent across project

2. **One test file per source file**
   - When practical
   - Keeps tests organized

3. **Group related tests**
   - Use test classes for related tests
   - Share fixtures via conftest.py

4. **Use descriptive names**
   ```python
   # Good
   def test_temperature_conversion_celsius_to_kelvin():
       pass

   # Less good
   def test_conversion():
       pass
   ```

### Coverage Strategy

1. **Focus on critical paths**
   - Core business logic
   - Error handling
   - Edge cases

2. **Don't chase 100%**
   - 80%+ is excellent
   - Diminishing returns above that
   - Some code is hard to test (UI, I/O)

3. **Use exclude patterns**
   - Debug code
   - Abstract methods
   - Type checking blocks

4. **Track trends**
   - Coverage going up? ✓
   - Coverage dropping? Investigate

### Test Markers

Use markers consistently:

```python
@pytest.mark.unit
def test_pure_function():
    """Fast, isolated test"""
    pass

@pytest.mark.integration
def test_component_interaction():
    """Tests multiple components"""
    pass

@pytest.mark.slow
def test_long_computation():
    """Takes >1 second"""
    pass
```

**Run selectively:**
```bash
# Fast feedback: unit tests only
pytest -m unit

# Before commit: all except slow
pytest -m "not slow"

# Nightly: everything
pytest
```

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

## References

- **pytest:** https://docs.pytest.org/
- **coverage.py:** https://coverage.readthedocs.io/
- **GitHub Actions:** https://docs.github.com/en/actions
- **Reusable Workflows:** https://docs.github.com/en/actions/using-workflows/reusing-workflows
- **ruff:** https://docs.astral.sh/ruff/

---

**Maintained by:** FormingWorlds team
**Last updated:** 2025-12-31
**Questions?** Open an issue on GitHub
