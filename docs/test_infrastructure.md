# Testing Infrastructure

This document describes the testing infrastructure for PROTEUS and the ecosystem. For markers and CI flow see [Test Categorization](test_categorization.md). For prompts and workflow see [Test Building](test_building.md). For status and strategy see [Test Building Strategy](test_building_strategy.md).

---

## Table of Contents

1. [CI/CD Status](#cicd-status)
2. [Quick Start](#quick-start)
3. [Developer Workflow](#developer-workflow)
4. [Coverage Analysis](#coverage-analysis-workflow)
5. [Pre-commit Checklist](#pre-commit-checklist)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)
8. [Skipped Tests](#skipped-tests-placeholders-and-environment-limited)
9. [Monitoring & Maintenance](#monitoring--maintenance)
10. [References](#references)
11. [Checklist: New Ecosystem Module](#checklist-new-ecosystem-module)

---

## CI/CD Status

**Last updated**: 2026-01-28

- **PR workflow** (`ci-pr-checks.yml`): Pre-built Docker image; overlay PR code; validate test structure; `pytest -m "unit and not skip"` with coverage (fast gate); diff-cover 80% on changed lines; `pytest -m "smoke and not skip"`; lint in parallel. Runtime ~5–10 min.
- **Coverage thresholds**: In `pyproject.toml` — fast gate `[tool.proteus.coverage_fast] fail_under`, full gate `[tool.coverage.report] fail_under`. Ratcheted on push to main / `tl/test_ecosystem_v5` via `tools/update_coverage_threshold.py`; do not decrease.
- **Nightly** (`ci-nightly-science-v5.yml` on `tl/test_ecosystem_v5`): Sets `PROTEUS_CI_NIGHTLY=1`; runs unit → smoke (including gated smoke tests) → integration → slow. Combined coverage; full gate enforced.
- **Docker**: Image `ghcr.io/formingworlds/proteus:latest` (and branch tag for `tl/test_ecosystem_v5`). Built via `docker-build.yml`. PR overlays code onto `/opt/proteus/`. Smart rebuild (SOCRATES/AGNI) in smoke job when relevant sources change.
- **Reusable workflow**: `proteus_test_quality_gate.yml` for ecosystem modules.

See [Test Categorization](test_categorization.md) for pipeline details and [Docker CI Architecture](docker_ci_architecture.md) for image build.

---

## Quick Start

**PR authors**: CI will validate structure, run unit tests with coverage (fast gate), diff-cover, smoke tests, and lint. Install: `pip install -e ".[develop]"`. Run locally: `pytest -m "unit and not skip"` and `pytest -m "smoke and not skip"`.

**Test writers**: Use markers `@pytest.mark.unit`, `@pytest.mark.smoke`, `@pytest.mark.integration`, `@pytest.mark.slow` as appropriate. See [Test Categorization](test_categorization.md).

**Running locally**:

```bash
pytest -m "unit and not skip"           # Unit only (matches PR)
pytest -m "(unit or smoke) and not skip"  # What PR runs
pytest -m "not slow"                    # All except slow
pytest --cov=src --cov-report=html      # With coverage
```

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

## Skipped Tests (Placeholders and Environment-Limited)

Some tests use `@pytest.mark.skip` (placeholders or require binaries/network). CI runs `pytest -m "unit and not skip"` and `pytest -m "smoke and not skip"`, so skipped tests are excluded from PR. Examples: JANUS/SOCRATES (instability), AGNI (Julia/binaries), CALLIOPE (slow, nightly), network-dependent data tests. Smoke tests gated by `PROTEUS_CI_NIGHTLY=1` run only in nightly. To enable a test: remove or narrow `skip`, add the right marker, ensure it passes. See [Test Categorization](test_categorization.md).

---

## Monitoring & Maintenance

- **CI**: PR &lt;10 min; nightly as per workflow. Monitor Actions for failures.
- **Coverage**: Fast and full gates in `pyproject.toml`; review HTML artifacts and Codecov.
- **Docs**: Update these docs when adding markers or changing CI; keep thresholds in `pyproject.toml` as source of truth.
- **Quality**: Review new tests in PRs; refactor tests like production code.

---

## References

- [Test Categorization](test_categorization.md) — Markers, CI pipeline, fixtures
- [Test Building](test_building.md) — Prompts for unit/integration tests
- [Test Building Strategy](test_building_strategy.md) — Status and principles
- [Docker CI Architecture](docker_ci_architecture.md) — Image build and strategy
- [tests/conftest.py](../tests/conftest.py) — Shared fixtures
- [AGENTS.md](../AGENTS.md) — Commands and thresholds
- pytest: <https://docs.pytest.org/>
- coverage: <https://coverage.readthedocs.io/>
- ruff: <https://docs.astral.sh/ruff/>

---

## Checklist: Ready to Deploy Quality Gate

**For a new ecosystem module**:

- [ ] Copy PROTEUS `pyproject.toml` pytest/coverage sections; set initial `fail_under` (e.g. 20–30%)
- [ ] Create/update `.github/workflows/ci_tests.yml`; add pytest-cov
- [ ] Run `bash tools/validate_test_structure.sh`; ensure `tests/` mirrors `src/`
- [ ] Local: `pytest` passes; coverage meets threshold; markers work
- [ ] Push; CI runs; coverage reported
- [ ] Document testing approach; link to this guide

**Last updated**: 2026-01-28
