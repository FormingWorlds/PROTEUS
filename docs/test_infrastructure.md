# Testing Infrastructure

## What This Document Is For

**New to PROTEUS?** This document explains how our testing system works: how to run tests, check code coverage, and troubleshoot common issues.

**Key concepts:**

- **Coverage** measures what percentage of your code is tested. Higher is better.
- **CI (Continuous Integration)** automatically runs tests when you push code.
- **Thresholds** are minimum coverage percentages that must be met.

For test markers and CI pipelines, see [Test Categorization](test_categorization.md). For writing tests, see [Test Building](test_building.md).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [CI/CD Status](#cicd-status)
3. [Developer Workflow](#developer-workflow)
4. [Best Practices](#best-practices)
5. [Coverage Analysis](#coverage-analysis-workflow)
6. [Pre-commit Checklist](#pre-commit-checklist)
7. [Troubleshooting](#troubleshooting)
8. [Reusable Quality Gate](#reusable-quality-gate-for-ecosystem-modules)
9. [References](#references)

---

## Quick Start

**First time?** Install with `pip install -e ".[develop]"`, then run tests:

```bash
pytest -m "unit and not skip"           # Fast unit tests (~2 min)
pytest -m "smoke and not skip"          # Smoke tests with real binaries
pytest --cov=src --cov-report=html      # Generate coverage report
open htmlcov/index.html                 # View coverage in browser
```

**Before committing:**
1. Run `pytest -m "unit and not skip"` — must pass
2. Run `ruff check src/ tests/ && ruff format src/ tests/` — must pass
3. Run `bash tools/validate_test_structure.sh` — must pass

---

## CI/CD Status

### How CI Works

When you open a pull request, CI automatically:

1. Validates test file structure
2. Runs unit tests and checks coverage (must meet threshold)
3. Runs smoke tests
4. Checks code style with ruff

**Current coverage thresholds** (from `pyproject.toml`):

- **Fast gate**: unit + smoke, checked on PRs
- **Full gate**: all tests, checked nightly

### Workflows

| Workflow | Runs When | What It Does |
|----------|-----------|--------------|
| `ci-pr-checks.yml` | Every PR | Unit + smoke tests, lint, ~5-10 min |
| `ci-nightly.yml` | Daily 3am UTC | All tests including slow, updates thresholds |

**Key features:**

- **Grace period**: PRs can merge with ≤0.3% coverage drop (warning posted)
- **Diff-cover**: 80% coverage required on changed lines
- **Auto-ratcheting**: Thresholds only increase, never decrease

---

## Developer Workflow

1. Write or modify code
2. Write or update tests (`tests/<module>/test_<filename>.py` mirrors `src/proteus/<module>/<filename>.py`)
3. Run tests: `pytest tests/<module>/` or `pytest -m unit`
4. Check coverage: `pytest --cov=src` (CI uses `--cov=src` for unit)
5. Validate structure: `bash tools/validate_test_structure.sh`
6. Lint: `ruff check src/ tests/` and `ruff format src/ tests/`
7. Commit and push; CI runs automatically

**Adding new code**: Create `src/proteus/<module>/<file>.py` and `tests/<module>/test_<filename>.py`. Use fixtures from `tests/conftest.py` and `tests/integration/conftest.py`. See [Test Building](test_building.md) for prompts and [Test Categorization](test_categorization.md) for fixtures list.

**Basic test structure**: Use `@pytest.mark.unit` (or other marker), docstrings, `pytest.approx()` for floats, and `unittest.mock` for external/heavy code in unit tests. Example:

```python
@pytest.mark.unit
def test_function_basic():
    """Test basic functionality."""
    result = function_to_test(input_value)
    assert result == pytest.approx(expected, rel=1e-5)
```

---

## Best Practices

| Practice | Guideline |
|----------|----------|
| **Structure** | Mirror `src/proteus/` in `tests/`; one test file per source file |
| **Markers** | Use `unit` / `smoke` / `integration` / `slow` consistently |
| **Floats** | Use `pytest.approx(val, rel=1e-5)` or `np.testing.assert_allclose`; never `==` |
| **Mocking** | Unit tests mock I/O and heavy physics; integration tests use real modules |
| **Docstrings** | Explain physical scenario being tested |
| **Determinism** | Set seeds (`np.random.seed(42)`); use `tmp_path` for temp files |
| **Coverage** | Focus on critical paths; use `bash tools/coverage_analysis.sh` |

See [Test Categorization](test_categorization.md) for marker details and [Test Building](test_building.md) for prompts.

---

## Coverage Analysis Workflow

**Generate reports:**

```bash
pytest --cov=src --cov-report=term-missing   # Terminal report with missing lines
pytest --cov=src --cov-report=html           # HTML report
open htmlcov/index.html                      # View in browser
bash tools/coverage_analysis.sh              # Coverage by module
coverage report --show-missing --skip-covered  # Only uncovered files
```

**Thresholds** (in `pyproject.toml`):
- `[tool.proteus.coverage_fast] fail_under` — Fast gate (PRs)
- `[tool.coverage.report] fail_under` — Full gate (nightly)

Thresholds auto-ratchet upward; never decrease manually.

---

## Pre-commit Checklist

Run these before every commit:

```bash
# 1. Tests pass
pytest -m "unit and not skip"

# 2. Lint passes
ruff check src/ tests/ && ruff format src/ tests/

# 3. Structure valid
bash tools/validate_test_structure.sh
```

**For new code:**

- [ ] Added tests with correct markers (see [Test Categorization](test_categorization.md))
- [ ] Coverage meets fast gate threshold
- [ ] Docstrings explain physical scenarios

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `pytest: unrecognized arguments: --cov` | pytest-cov not installed | `pip install -e ".[develop]"` |
| Coverage below threshold | Coverage dropped | Add tests; see `coverage report --show-missing`; do not lower threshold |
| Tests not found | Discovery / naming | `pytest --collect-only`; ensure `test_*.py` and `test_*` functions |
| Import errors | Package not installed | `pip install -e ".[develop]"`; check `src/` layout |
| CI fails, local passes | Environment / deps | Match Python version and pyproject.toml; check CI logs |
| Ruff fails | Style violations | `ruff check --fix src/ tests/` and `ruff format src/ tests/` |

### Debugging Tests

```bash
pytest -v --showlocals -x tests/module/test_file.py::test_function
pytest --pdb   # Debugger on failure
```

### Stale Nightly Baseline

PR checks compare coverage against the last successful nightly run. If the nightly workflow fails (e.g. data download timeout, CI infrastructure issues, or transient test failures), the baseline becomes stale (>48 hours old) and PRs will fail validation. To fix this, [trigger the nightly workflow manually](https://github.com/FormingWorlds/PROTEUS/actions/workflows/ci-nightly.yml) and wait for it to complete.

### Docker CI

- **Build fails**: `docker build -t proteus-test .` locally; check Dockerfile and deps.
- **Image pull fails**: Verify `ghcr.io/formingworlds/proteus:latest` is public.
- **Tests fail in container**: `docker run -it ghcr.io/formingworlds/proteus:latest bash` and run `pytest -m unit -v` inside.

---

## Reusable Quality Gate for Ecosystem Modules

PROTEUS provides a reusable workflow for ecosystem modules (CALLIOPE, JANUS, MORS, etc.) to adopt consistent testing standards.

**Location:** `.github/workflows/proteus_test_quality_gate.yml`

**Purpose:** Standardized test quality gate that ecosystem modules can call from their own CI workflows.

**Inputs:**

| Input | Default | Description |
|-------|---------|-------------|
| `python-version` | `3.12` | Python version for testing |
| `coverage-threshold` | `30` | Minimum coverage percentage |
| `grace-period` | `0.3` | Allowed coverage drop (percentage points) |
| `working-directory` | `.` | Project subdirectory |
| `pytest-args` | `''` | Additional pytest arguments |

**Why use it:**

- **Consistency**: Same testing standards across all ecosystem modules
- **Maintenance**: Updates to quality gate propagate to all modules
- **Best practices**: Includes coverage reporting, artifact upload, proper caching

### Implementation Guide for Ecosystem Modules

**Step 1: Ensure your module has the required structure**

```text
your-module/
├── src/
│   └── your_module/
│       └── __init__.py
├── tests/
│   └── test_*.py
├── pyproject.toml          # Must have [project.optional-dependencies] develop = [...]
└── .github/
    └── workflows/
        └── tests.yml       # Create this file
```

**Step 2: Add `[develop]` dependencies to `pyproject.toml`**

```toml
[project.optional-dependencies]
develop = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1.0",
]
```

**Step 3: Create `.github/workflows/tests.yml`**

Basic example:
```yaml
name: Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    uses: FormingWorlds/PROTEUS/.github/workflows/proteus_test_quality_gate.yml@main
    with:
      coverage-threshold: 30
```

**Step 4: Configure for your module's needs**

Example configurations for different scenarios:

```yaml
# CALLIOPE - Higher coverage, specific markers
jobs:
  test:
    uses: FormingWorlds/PROTEUS/.github/workflows/proteus_test_quality_gate.yml@main
    with:
      coverage-threshold: 50
      pytest-args: '-m "unit and not skip" -v'

# JANUS - Lower initial threshold, exclude slow tests
jobs:
  test:
    uses: FormingWorlds/PROTEUS/.github/workflows/proteus_test_quality_gate.yml@main
    with:
      coverage-threshold: 25
      pytest-args: '-m "not slow"'

# MORS - Monorepo with subdirectory
jobs:
  test:
    uses: FormingWorlds/PROTEUS/.github/workflows/proteus_test_quality_gate.yml@main
    with:
      working-directory: 'python'
      coverage-threshold: 40
```

### Codecov Integration

To enable Codecov uploads, add the `CODECOV_TOKEN` secret to your repository:

1. Go to [codecov.io](https://codecov.io) and connect your repository
2. Copy the upload token
3. In GitHub: Settings → Secrets → Actions → New repository secret
4. Name: `CODECOV_TOKEN`, Value: your token

The workflow automatically uploads coverage if the secret exists.

### Transitioning from Custom Workflows

If your module has an existing test workflow:

1. **Keep your workflow temporarily**: Rename to `tests-old.yml`
2. **Create new workflow**: Add `tests.yml` using the quality gate
3. **Compare results**: Ensure both pass on a few PRs
4. **Remove old workflow**: Delete `tests-old.yml` once confident

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `pip install` fails | Ensure `pyproject.toml` has valid `[project.optional-dependencies]` |
| Coverage too low | Lower `coverage-threshold` initially, ratchet up over time |
| Tests not found | Check `tests/` directory exists and contains `test_*.py` files |
| Codecov upload fails | Add `CODECOV_TOKEN` secret or ignore (non-fatal) |

---

## References

### PROTEUS Documentation

- [Test Categorization](test_categorization.md) — Markers, CI pipeline, fixtures
- [Test Building](test_building.md) — Prompts for unit/integration tests
- [Docker CI Architecture](docker_ci_architecture.md) — Docker image, CI pipelines
- [AI-Assisted Development](ai_usage.md) — Using AI for tests and code review
- [tests/conftest.py](https://github.com/FormingWorlds/PROTEUS/blob/main/tests/conftest.py) — Shared fixtures
- [AGENTS.md](https://github.com/FormingWorlds/PROTEUS/blob/main/AGENTS.md) — Commands and thresholds

### External Resources

- [pytest Documentation](https://docs.pytest.org/)
- [coverage.py Documentation](https://coverage.readthedocs.io/)
- [ruff Documentation](https://docs.astral.sh/ruff/)
