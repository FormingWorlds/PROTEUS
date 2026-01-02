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

#### 4. `tools/update_coverage_threshold.py` (Optional - CALLIOPE Pattern)
- **Purpose:** Automatically ratchet coverage threshold upward
- **Trigger:** Runs on main branch when coverage increases
- **Behavior:** Updates `fail_under` in pyproject.toml, prevents regression
- **Usage:** Automated via CI (see CALLIOPE for implementation)

### Ecosystem Integration Standards

#### Codecov Integration

All ecosystem modules should integrate with Codecov for ecosystem-wide coverage tracking:

```yaml
- name: Upload coverage reports to Codecov
  uses: codecov/codecov-action@v4
  if: always()
  with:
    files: ./coverage.xml
    flags: unittests
    name: codecov-${{ matrix.python-version }}-${{ matrix.os }}
    fail_ci_if_error: false  # Non-blocking on feature branches
  env:
    CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}
```

For main branch: Set `CODECOV_TOKEN` as repository secret for full reporting.

#### HTML Artifact Uploads

Archive HTML coverage reports for 30 days:

```yaml
- name: Upload coverage HTML report
  uses: actions/upload-artifact@v4
  if: always()
  with:
    name: coverage-report-${{ matrix.python-version }}-${{ matrix.os }}
    path: htmlcov/
    retention-days: 30
```

#### Test Quality & Documentation

Best practice: Add comprehensive inline comments to test files:
- Module docstring: Explain overall test purpose
- Test comments: Document what each test validates
- Context: Include formulas, principles, or domain knowledge relevant to assertions
- Cross-references: Link to source code when helpful

See [CALLIOPE test files](https://github.com/FormingWorlds/CALLIOPE/tree/main/tests) for exemplary documentation.

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
- Others

### Current Status

**PROTEUS** ✅ Complete
- Coverage: 69.23% (target: 70%+)
- CI duration: ~18 minutes (with dependencies)
- Features: Hash-based caching, dynamic badges, comprehensive reporting

**CALLIOPE** ✅ Phase 2 Pilot Complete
- Coverage: 18.68% (branch coverage, auto-ratcheting at 18%)
- CI duration: ~5 minutes (6-job matrix: 2 OS × 3 Python versions)
- Features: Coverage ratcheting, Codecov integration, HTML artifacts, comprehensive documentation
- Status: Reference implementation for ecosystem integration
- See: [CALLIOPE testing guide](https://proteus-framework.org/CALLIOPE/TESTS) for ratcheting mechanism

**Ecosystem Modules** - Ready for deployment
- CALLIOPE: ✅ Phase 2 Pilot (use as reference implementation)
- JANUS, MORS: Phase 2b/2c (template from CALLIOPE)
- VULCAN, ZEPHYRUS: Need CI setup (can use CALLIOPE pattern)
- aragog: Already integrated in PROTEUS CI

### Rollout Strategy

#### Phase 1: PROTEUS (Main Repository) ✅ COMPLETE

1. **Setup Infrastructure** ✅
   - ✅ Create reusable workflow (`.github/workflows/proteus_test_quality_gate.yml`)
   - ✅ Create CI workflow (`.github/workflows/ci.yml`)
   - ✅ Update pyproject.toml with pytest/coverage configuration
   - ✅ Create tools (restructure, validate, analyze scripts)
   - ✅ Create comprehensive documentation

2. **Implement Testing** ✅
   - ✅ Run validation script
   - ✅ Run restructuring script
   - ✅ Added 68 test files with full coverage
   - ✅ Tests running locally and in CI
   - ✅ CI passes with coverage reporting

3. **Establish Baseline** ✅
   - ✅ Current coverage: 69.23%
   - ✅ Coverage threshold: 69% (enforcement level)
   - ✅ Coverage gaps documented
   - ✅ Improvement plan active

**Key Achievement:** Hash-based caching deployed and validated (saves ~11-15 min on SOCRATES rebuilds)

#### Phase 2: Ecosystem Integration 🚀 STARTING NOW

For each submodule (CALLIOPE, JANUS, MORS, VULCAN, ZEPHYRUS, Zalmoxis, aragog):

### Quick Start: 4-Step Deployment for Ecosystem Modules

**Using CALLIOPE as Reference Implementation**

CALLIOPE (Phase 2 pilot) has completed all ecosystem integration standards and includes innovations beyond the base standard. When implementing for other modules (JANUS, MORS, etc.), use CALLIOPE as a reference:

- Test structure and quality: [CALLIOPE tests](https://github.com/FormingWorlds/CALLIOPE/tree/main/tests)
- Workflow configuration: [CALLIOPE ci_tests.yml](https://github.com/FormingWorlds/CALLIOPE/blob/main/.github/workflows/ci_tests.yml)
- Coverage ratcheting: [CALLIOPE update_coverage_threshold.py](https://github.com/FormingWorlds/CALLIOPE/blob/main/tools/update_coverage_threshold.py)
- Documentation: [CALLIOPE testing guide](https://proteus-framework.org/CALLIOPE/TESTS)

**Step 1: Copy Configuration from CALLIOPE or PROTEUS**

```bash
# Clone PROTEUS repo if you haven't already
git clone https://github.com/FormingWorlds/PROTEUS.git

# Copy relevant sections from PROTEUS pyproject.toml
cp PROTEUS/pyproject.toml <your-module>/pyproject.toml.backup
```

**Step 2: Update pyproject.toml**

Add these sections to your module's `pyproject.toml`:

```toml
# pytest configuration
[tool.pytest.ini_options]
minversion = "8.1"
addopts = [
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

# Coverage configuration
[tool.coverage.run]
branch = true
source = ["<package_name>"]  # Change to: calliope, janus, mors, vulcan, etc.
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__pycache__/*",
    "*/conftest.py",
]

[tool.coverage.report]
fail_under = 30  # Start with realistic threshold; increase by 5-10% quarterly
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

# In [project.optional-dependencies]
develop = [
    "pytest >= 8.1",
    "pytest-cov",
    "coverage[toml]",
    # ... your existing dependencies
]
```

**Coverage Threshold Guidance:**
- **Start:** 20-30% (realistic baseline)
- **Q2:** Increase to 35-40%
- **Q4:** Increase to 50-60%
- **Year 2:** Target 70%+

Example progression for new module:

```toml
fail_under = 30  # January 2026
fail_under = 40  # April 2026
fail_under = 50  # July 2026
fail_under = 60  # October 2026
```

**Advanced: Automatic Coverage Ratcheting (CALLIOPE Innovation)**

For sustainable growth without manual threshold updates, CALLIOPE implements automatic ratcheting:

```toml
[tool.coverage.report]
# Coverage threshold - automatically updated by CI when coverage increases
# See: tools/update_coverage_threshold.py and .github/workflows/ci_tests.yml
# This value can only increase or stay the same (coverage ratcheting mechanism)
fail_under = 18
```

Implementation steps:

1. Create `tools/update_coverage_threshold.py` to read current coverage and update threshold
2. Add CI step that runs on main branch (specific Python/OS combo) to trigger updates
3. Commit updates with `[skip ci]` to prevent cascade builds
4. Document mechanism in pyproject.toml for team visibility

Benefits:
- ✅ Automatic progress tracking
- ✅ Sustainable threshold growth
- ✅ Eliminates manual updates
- ✅ Enforces continuous improvement

See [CALLIOPE implementation](https://github.com/FormingWorlds/CALLIOPE/blob/main/tools/update_coverage_threshold.py) for reference code and [CALLIOPE testing guide](https://proteus-framework.org/CALLIOPE/TESTS) for detailed documentation.

**Step 3: Create/Update CI Workflow**

Create `.github/workflows/ci.yml` in your module. Two options:

**Option A: Use Reusable Workflow (Recommended)**

```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  workflow_dispatch:

jobs:
  test:
    uses: FormingWorlds/PROTEUS/.github/workflows/proteus_test_quality_gate.yml@main
    with:
      python-version: '3.13'
      coverage-threshold: 30
      working-directory: '.'
      pytest-args: ''
```

**Option B: Full Custom Workflow (More Control)**

Copy from PROTEUS `.github/workflows/ci.yml` and customize for your module's dependencies.

**Step 4: Validate and Test**

```bash
# 1. Validate test structure mirrors source
bash tools/validate_test_structure.sh

# 2. Run tests locally
pytest

# 3. Check coverage
pytest --cov

# 4. Push to GitHub
git push

# 5. Monitor CI at: https://github.com/FormingWorlds/<module>/actions
```

#### Advanced Features: Hash-Based Caching Strategy

##### Why Caching Matters

For modules with external dependencies (SOCRATES, AGNI, VULCAN), caching can save **10-15 minutes per run**.

**PROTEUS Implementation:**
- Compiles SOCRATES only when source code changes
- Restores Julia dependencies only when Project.toml/Manifest.toml changes
- Uses hash-based cache keys for deterministic invalidation

##### Hash-Based Caching Pattern

```yaml
# Clone dependencies BEFORE cache restore (critical!)
- name: Clone SOCRATES
  run: git clone https://github.com/nichollsh/SOCRATES.git socrates

# Now cache restore can hash the source files
- name: Restore SOCRATES cache
  uses: actions/cache/restore@v4
  id: cache-socrates
  with:
    path: socrates/
    # Hash changes = cache miss = recompile (correct behavior)
    key: socrates-${{ runner.os }}-${{ hashFiles('socrates/**/*.f90', 'socrates/**/*.c') }}
    restore-keys: |
      socrates-${{ runner.os }}-

# Build if cache missed
- name: Build SOCRATES (if needed)
  if: steps.cache-socrates.outputs.cache-hit != 'true'
  run: cd socrates && ./build_code

# Save for next run
- name: Save SOCRATES cache
  if: steps.cache-socrates.outputs.cache-hit != 'true'
  uses: actions/cache/save@v4
  with:
    path: socrates/
    key: socrates-${{ runner.os }}-${{ hashFiles('socrates/**/*.f90', 'socrates/**/*.c') }}
```

**Key Principles:**
1. **Clone before cache restore** - Lets hashFiles() work
2. **Use source file hashes** - Cache invalidates when code changes
3. **Conditional builds** - Only compile if cache missed
4. **Conditional saves** - Only save if not already cached

##### Performance Expectations

| Scenario | Duration | Notes |
|----------|----------|-------|
| First run (no cache) | 45-60 min | Builds all dependencies |
| Cache hit (no changes) | 12-18 min | Uses pre-built binaries |
| After source change | 25-35 min | Recompiles affected dependencies |

##### Troubleshooting Cache Issues

**Problem:** Cache always misses

```
Key: 'socrates-Linux-' (empty hash?)
```

**Solution:** Verify directories exist before cache restore step

**Problem:** Old cached binaries used after major refactor

```
Key: 'socrates-Linux-old_hash' still matched
```

**Solution:** Update hash patterns when source structure changes

---

#### Phase 3: Monitoring & Improvement (Parallel with Phase 2)

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

**Coverage Threshold Growth Plan:**

Start with realistic baseline, increase gradually:

```toml
# PROTEUS Example (achieved 69.23%, enforcing 69%)
fail_under = 69  # Target: maintain high bar

# New Module Example (starting from 20-30%)
fail_under = 30   # January 2026
fail_under = 40   # April 2026 (+10%)
fail_under = 50   # July 2026 (+10%)
fail_under = 60   # October 2026 (+10%)
fail_under = 70   # January 2027 (+10%)
```

**Why this pace?**
- ✅ Realistic: Allows time to write tests
- ✅ Motivating: Visible progress
- ✅ Sustainable: Doesn't block development
- ✅ Long-term: Reaches 70%+ in 1 year

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

## Checklist: Ready to Deploy Quality Gate

### For New Ecosystem Module

- [ ] **Setup** (30 min)
  - [ ] Copy PROTEUS pyproject.toml pytest/coverage sections
  - [ ] Create/update `.github/workflows/ci.yml`
  - [ ] Set appropriate `fail_under` threshold (20-30%)
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
**Last updated:** 2026-01-02
**Questions?** Open an issue on GitHub
