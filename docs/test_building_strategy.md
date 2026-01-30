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

---

## 2026-01-28 – Nightly run 21455158655 (assessment and plan)

### Run assessment

- **Run ID**: [21455158655](https://github.com/FormingWorlds/PROTEUS/actions/runs/21455158655)
- **Workflow**: CI - Nightly Science Validation (v5)
- **Conclusion**: `cancelled`
- **Job**: Branch Nightly Coverage (Integration - v5)

**Step outcomes (from API):**

| Step | Conclusion |
|------|------------|
| Set up job, Checkout, Overlay code, Read threshold, Download test data, Free disk / Julia | success |
| Coverage erase + unit tests | success |
| Smoke tests with coverage | success |
| Integration coverage (dummy + integration, not slow) | success |
| Save coverage before slow | success |
| **Run slow integration tests (standard config)** | **skipped** |
| Generate coverage JSON | success |
| **Write workflow summary and test results report** | **failure** |
| Upload coverage XML/HTML | success |

**Root causes inferred:**

1. **Job cancelled**: Run conclusion `cancelled` with job timeout `55` minutes. The "Run slow integration tests" step was **skipped** because the job did not reach it (timeout or manual cancel). Slow integration tests need ~10–15 min; unit + smoke + integration can exceed 55 min, so the job was likely cancelled by timeout before slow tests ran.
2. **Summary step failure**: "Write workflow summary and test results report" failed. The script exits with `sys.exit(1)` when `failed_tests` is non-empty (by design, to fail the run). So either (a) some tests failed and the summary correctly failed the step, or (b) the script raised an exception (e.g. missing file or parse error). Without log access, we treat both: make the summary script robust so it never crashes and always writes a summary; then exit(1) only when there are failed tests.
3. **Compiled binaries / data**: Previous run 21428575411 had `get_Seager_EOS` import fixes and smoke gating via `PROTEUS_CI_NIGHTLY=1`. Data download step and Julia/AGNI setup are already in the workflow; no Docker change required for 21455158655 itself.

### Plan (fix root causes)

1. **Increase nightly job timeout**  
   - In `ci-nightly-science-v5.yml`, set `timeout-minutes` from `55` to `90` so unit + smoke + integration + slow can complete.

2. **Harden summary script**  
   - In the "Write workflow summary and test results report" step, wrap the full summary generation in `try/except`.  
   - On exception: write a minimal summary (e.g. "Summary generation failed: &lt;error&gt;"), then exit(1) so the run still fails.  
   - Ensure all file reads (JUnit XML, coverage JSON, output files) are inside the try block and missing files don’t raise.  
   - Keep existing behaviour: exit(1) at the end if and only if `failed_tests` is non-empty.

3. **No test skips added**  
   - Do not add skips to make the run pass; fix environment/time so tests run.

4. **Validation**  
   - Run `ruff format` on any changed files.  
   - Optionally run `pytest -m "unit and not skip"` locally to confirm nothing regressed.

### Execution log

- **Timeout**: Increased `timeout-minutes` to 90 in `ci-nightly-science-v5.yml`.
- **Summary script**: Wrapped full summary generation in `try/except`; on exception write minimal summary (with traceback) to step summary, then `sys.exit(1)`. Exit(1) only when `failed_tests` non-empty or `summary_error` is set.
- **Ruff**: `ruff format src/ tests/` run (170 files unchanged).
- **Local pytest**: Not run (environment-specific `PermissionError` in pytest faulthandler); no Python test code was changed.

---

## Resources

- [Test Infrastructure](test_infrastructure.md) — Layout, coverage workflow, troubleshooting
- [Test Categorization](test_categorization.md) — Markers, CI pipeline, fixtures
- [Test Building](test_building.md) — Master Prompt (unit) and Integration Prompt
- [AGENTS.md](../AGENTS.md) — Test commands, coverage thresholds, lint
- [tests/conftest.py](../tests/conftest.py) — Shared fixtures
