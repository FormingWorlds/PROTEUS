# Test Configuration Reference

## pytest Configuration

### Settings in pyproject.toml

```toml
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
```

### Configuration Meanings

- **minversion:** Minimum pytest version required (8.1+)
- **addopts:** Options automatically applied to all runs
  - `--strict-markers`: Disallow undefined markers
  - `--strict-config`: Enforce strict configuration
  - `-ra`: Show summary of all test outcomes
  - `--showlocals`: Show local variables in tracebacks
- **testpaths:** Directory where pytest discovers tests
- **python_files/classes/functions:** Test naming patterns
- **markers:** Custom markers for categorizing tests

## Coverage Configuration

### Settings in pyproject.toml

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
fail_under = 69
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

### Configuration Meanings

- **branch:** Enable branch coverage (if/else paths)
- **source:** Package name to measure coverage for
- **omit:** Files to exclude from coverage
- **fail_under:** Minimum coverage percentage (enforced in CI)
- **show_missing:** Report which lines are uncovered
- **precision:** Decimal places in coverage reports
- **exclude_lines:** Code patterns to skip in coverage
- **html.directory:** Output folder for HTML reports

## Coverage Thresholds

- **fail_under:** Current threshold (auto-ratcheted on main branch)
- **Ratcheting:** Threshold automatically increases when coverage improves
- **Never decreases:** Prevents coverage regression
- **Check pyproject.toml:** Verify current threshold for your module

## Test Markers

### Available Markers

```python
@pytest.mark.unit
def test_single_function():
    """Fast unit test for isolated functionality."""
    pass

@pytest.mark.integration  
def test_module_interaction():
    """Slower test for multiple components together."""
    pass

@pytest.mark.slow
def test_long_running():
    """Very slow test, deselect with: pytest -m 'not slow'"""
    pass
```

### Running with Markers

```bash
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m "not slow"        # Skip slow tests
pytest -m "unit or integration"  # Unit OR integration
```

## Coverage Reporting

### Local Development

```bash
# Option 1: Using pytest-cov (recommended for local work)
pytest --cov --cov-report=html

# Option 2: Using coverage (matches CI exactly)
coverage run -m pytest
coverage html

# View report
open htmlcov/index.html
```

### Analyze by Module

```bash
bash tools/coverage_analysis.sh
```

Output: Module-by-module coverage with improvement priorities

## Continuous Integration

### GitHub Actions Workflow

- **Trigger:** Push to main or pull request
- **Matrix:** Python 3.11, 3.12, 3.13 × multiple OS
- **Steps:**
  1. Checkout code
  2. Install dependencies
  3. Run pytest with coverage
  4. Check coverage threshold
  5. Upload to Codecov
  6. Generate HTML artifacts
  7. Run linting with ruff

### Required Environment Variables

- `CODECOV_TOKEN`: Set as repository secret for Codecov integration
- CI automatically skips token validation on feature branches
