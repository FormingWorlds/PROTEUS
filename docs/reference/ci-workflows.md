# CI/CD Workflows Reference

## GitHub Actions Infrastructure

The PROTEUS testing infrastructure uses GitHub Actions for automated testing and quality assurance.

### Main CI Workflow (`.github/workflows/ci_tests.yml`)

**Purpose:** Run tests, check coverage, lint code on every push and pull request

**Matrix Testing:**
- Python versions: 3.11, 3.12, 3.13
- Operating systems: Linux, macOS, Windows (as applicable)

**Jobs:**
1. **Test Job**
   - Checkout code
   - Set up Python environment
   - Install dependencies
   - Run pytest with coverage
   - Upload coverage reports (Codecov, HTML artifacts)

2. **Lint Job**
   - Check code style with ruff
   - Enforce linting standards
   - Report violations

**Duration:** ~18 minutes (with dependencies like SOCRATES compilation)

### Reusable Quality Gate (`.github/workflows/proteus_test_quality_gate.yml`)

**Purpose:** Centralized testing workflow for all PROTEUS ecosystem modules

**Features:**
- Configurable Python version
- Customizable coverage threshold
- Reusable across CALLIOPE, JANUS, MORS, VULCAN, ZEPHYRUS, etc.
- Enforces consistent quality standards

**Usage in Submodules:**

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    uses: FormingWorlds/PROTEUS/.github/workflows/proteus_test_quality_gate.yml@main
    with:
      python-version: "3.12"
      coverage-threshold: 40
```

## CI/CD Pipeline Flow

```
Push/PR → GitHub Actions Triggered
    ↓
Checkout Repository
    ↓
Matrix: Python 3.11, 3.12, 3.13
    ↓
Install Dependencies
    ↓
Run pytest with Coverage
    ↓
Check Coverage Threshold (fail_under)
    ↓
Upload Coverage (Codecov)
    ↓
Generate HTML Artifacts
    ↓
Run Linting (ruff)
    ↓
Pass/Fail → Merge Gate
```

## Coverage Integration

### Codecov Reporting

**Configuration:**

```yaml
- name: Upload coverage reports to Codecov
  uses: codecov/codecov-action@v4
  if: always()
  with:
    files: ./coverage.xml
    flags: unittests
    name: codecov-${{ matrix.python-version }}-${{ matrix.os }}
    fail_ci_if_error: false
  env:
    CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}
```

**For Main Branch:**
- Set `CODECOV_TOKEN` as repository secret
- Full ecosystem coverage tracking enabled
- Coverage ratcheting enforced

**For Feature Branches:**
- `fail_ci_if_error: false` allows continuing even if Codecov is unavailable
- Non-blocking coverage validation

### HTML Artifact Upload

**Configuration:**

```yaml
- name: Upload coverage HTML report
  uses: actions/upload-artifact@v4
  if: always()
  with:
    name: coverage-report-${{ matrix.python-version }}-${{ matrix.os }}
    path: htmlcov/
    retention-days: 30
```

**Features:**
- Preserves detailed coverage reports for 30 days
- Downloadable from GitHub Actions interface
- Per-file and per-function coverage details

## Test Quality Benchmarks

### Duration Targets

- **PROTEUS main:** ~18 minutes (with SOCRATES compilation)
- **Submodules (CALLIOPE):** ~5 minutes (6-job matrix: 2 OS × 3 Python)
- **Target:** Keep under 15 minutes for daily developer workflow

### Coverage Targets

- **PROTEUS:** 69%+ (current: 69.23%)
- **CALLIOPE:** 18%+ (auto-ratcheting at 18%)
- **New modules:** Start realistic (30-40%), ratchet upward

### Success Criteria

- All tests pass across Python 3.11, 3.12, 3.13
- Coverage threshold met
- No regressions in existing tests
- Linting passes (ruff)

## Ecosystem Module Workflows

### CALLIOPE Reference Implementation

CALLIOPE includes advanced features (Phase 2 pilot):
- **Coverage ratcheting:** Auto-increases threshold on main branch
- **Advanced CI:** `.github/workflows/ci_tests.yml` with optimizations
- **Reference tests:** Comprehensive test suite with documentation

See [CALLIOPE repository](https://github.com/FormingWorlds/CALLIOPE) for implementation details.

### Other Modules (JANUS, MORS, VULCAN, ZEPHYRUS)

Use this template in `.github/workflows/ci_tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[develop]"
    
    - name: Run tests with coverage
      run: pytest --cov --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v4
      with:
        files: ./coverage.xml
        fail_ci_if_error: false
    
    - name: Lint with ruff
      run: ruff check .
```

## Troubleshooting CI Failures

### Common Issues

**Coverage Threshold Not Met:**
- Check `fail_under` in `pyproject.toml`
- Increase test coverage by adding missing tests
- For new modules, adjust realistic starting threshold

**Test Timeout:**
- Check for infinite loops or missing mocks
- Mark slow tests with `@pytest.mark.slow`
- Skip slow tests in CI: `pytest -m "not slow"`

**Dependency Installation Fails:**
- Verify `setup.py` or `pyproject.toml` is correct
- Check `[project.dependencies]` lists all required packages
- Ensure optional dependencies in `[project.optional-dependencies]`

**Matrix Testing Issues:**
- Verify Python versions in `strategy.matrix.python-version`
- Check if tests require specific Python features
- Test locally with each Python version
