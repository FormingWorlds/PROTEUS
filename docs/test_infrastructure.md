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
4. [Coverage Analysis](#coverage-analysis-workflow)
5. [Pre-commit Checklist](#pre-commit-checklist)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)
8. [References](#references)

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
- **Fast gate**: 44.45% (unit + smoke, checked on PRs)
- **Full gate**: 59% (all tests, checked nightly)

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

## Coverage Analysis Workflow

```bash
pytest --cov --cov-report=term-missing   # Detailed report
open htmlcov/index.html                 # HTML report
bash tools/coverage_analysis.sh         # By module
coverage report --show-missing --skip-covered  # Uncovered lines
```

Thresholds: see `pyproject.toml` (`[tool.proteus.coverage_fast]` and `[tool.coverage.report]`). Do not lower thresholds; use ratcheting.

---

## Pre-commit Checklist

- [ ] Tests pass: `pytest -m "unit and not skip"` (and smoke if applicable)
- [ ] Coverage meets fast gate (see `pyproject.toml`)
- [ ] Lint: `ruff check src/ tests/` and `ruff format src/ tests/`
- [ ] New tests for new code; correct markers (see [Test Categorization](test_categorization.md))
- [ ] Structure: `bash tools/validate_test_structure.sh`

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

### Docker CI

- **Build fails**: `docker build -t proteus-test .` locally; check Dockerfile and deps.
- **Image pull fails**: Verify `ghcr.io/formingworlds/proteus:latest` is public.
- **Tests fail in container**: `docker run -it ghcr.io/formingworlds/proteus:latest bash` and run `pytest -m unit -v` inside.

---

## Best Practices

- **Structure**: Mirror source; one test file per source file where practical; `tools/validate_test_structure.sh`.
- **Markers**: Use `unit` / `smoke` / `integration` / `slow` consistently. See [Test Categorization](test_categorization.md).
- **Floats**: `pytest.approx(expected, rel=1e-5)` or `np.testing.assert_allclose`; never `==` for floats.
- **Mocking**: Unit tests mock I/O and heavy physics; integration tests may use real modules.
- **Docstrings**: Brief description of the scenario tested; document physical intent where relevant.
- **Determinism**: Set seeds (e.g. `np.random.seed(42)`) in tests; avoid large output files (use `tmp_path` or mocks).

Coverage: focus on critical paths; use `bash tools/coverage_analysis.sh`; thresholds in `pyproject.toml`; ratcheting prevents regression.

---

## References

- [Test Categorization](test_categorization.md) — Markers, CI pipeline, fixtures
- [Test Building](test_building.md) — Prompts for unit/integration tests
- [Docker CI Architecture](docker_ci_architecture.md) — Image build and strategy
- [tests/conftest.py](../tests/conftest.py) — Shared fixtures
- [AGENTS.md](../AGENTS.md) — Commands and thresholds
- pytest: <https://docs.pytest.org/>
- coverage: <https://coverage.readthedocs.io/>
- ruff: <https://docs.astral.sh/ruff/>
