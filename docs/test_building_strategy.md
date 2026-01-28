# Test Building Strategy for PROTEUS

**Last updated**: 2026-01-28

This document summarizes the test implementation status, principles, and roadmap. For markers and CI flow see [Test Categorization](test_categorization.md). For infrastructure and layout see [Test Infrastructure](test_infrastructure.md). For prompts and workflow see [Test Building](test_building.md).

---

## Current Status

- **Unit tests**: 480+ tests; target &lt;100 ms each; run in PR with `pytest -m "unit and not skip"`.
- **Smoke tests**: Multiple tests in `tests/integration/test_smoke_*.py`. Some run only in nightly when `PROTEUS_CI_NIGHTLY=1` (see [Test Categorization](test_categorization.md)).
- **Integration tests**: In `tests/integration/` (dummy, aragog_agni, aragog_janus, std_config, multi_timestep, calliope_multi_timestep, etc.); run in nightly.
- **Coverage**: Fast gate and full gate are in `pyproject.toml` (`[tool.proteus.coverage_fast]` and `[tool.coverage.report]`). Thresholds are ratcheted on push to main; do not decrease them.

---

## Testing Principles

1. **Structure**: Tests mirror source. `src/proteus/<module>/<file>.py` → `tests/<module>/test_<file>.py`. Validate with `bash tools/validate_test_structure.sh`.
2. **Markers**: Use `@pytest.mark.unit` (fast, mocked), `@pytest.mark.smoke` (1 timestep, real binaries), `@pytest.mark.integration`, `@pytest.mark.slow` as appropriate. See [Test Categorization](test_categorization.md).
3. **Mocking**: Unit tests mock external I/O and heavy physics (SOCRATES, AGNI, etc.). Integration tests may use real modules.
4. **Floats**: Use `pytest.approx(expected, rel=1e-5)` or `np.testing.assert_allclose`; never `==` for floats.
5. **Parametrization**: Prefer `@pytest.mark.parametrize` for data-driven cases. Add brief docstrings for the scenario tested.

---

## Quick Reference: Adding New Tests

1. **Create file**: `tests/<module>/test_<filename>.py` (mirror `src/proteus/<module>/<filename>.py`).
2. **Use fixtures**: See `tests/conftest.py` and `tests/integration/conftest.py` for parameter classes and config paths.
3. **Run**: `pytest tests/<module>/ -v`, `pytest -m unit`, or `pytest --cov=src` as needed.
4. **Validate**: `bash tools/validate_test_structure.sh`; ensure coverage meets the fast gate for unit runs.

---

## Roadmap (Summary)

- **PR CI**: Unit + smoke (excluding `skip`). Fast gate enforced; diff-cover 80% on changed lines.
- **Nightly**: Unit, smoke (with `PROTEUS_CI_NIGHTLY=1`), integration, slow. Full coverage gate enforced.
- **Next focus**: Expand integration coverage (e.g. ARAGOG+AGNI, CALLIOPE+ZEPHYRUS); maintain coverage threshold; add slow scenarios as needed. See [Test Infrastructure](test_infrastructure.md) for priorities and [AGENTS.md](../AGENTS.md) for commands.

---

## Resources

- [Test Infrastructure](test_infrastructure.md) — Layout, coverage workflow, troubleshooting
- [Test Categorization](test_categorization.md) — Markers, CI pipeline, fixtures
- [Test Building](test_building.md) — Master Prompt (unit) and Integration Prompt
- [AGENTS.md](../AGENTS.md) — Test commands, coverage thresholds, lint
- [tests/conftest.py](../tests/conftest.py) — Shared fixtures
