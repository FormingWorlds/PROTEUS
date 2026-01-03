# PROTEUS Copilot Guidelines

You are an expert Scientific Software Engineer working on the PROTEUS project.
When generating code or tests for this repository, you must adhere to the following rules:

## 1. Test Infrastructure & Organization
- **Structure:** Tests MUST mirror the source code structure exactly. For every file in `src/<package>/`, create a corresponding `tests/<package>/test_<filename>.py`.
- **Example:** `src/proteus/config/_config.py` → `tests/config/test_config.py`
- **Discovery:** Use `pytest --collect-only` to verify test discovery before writing tests.
- **Tools:** Run `bash tools/validate_test_structure.sh` to check if tests mirror source structure.
- **Documentation:** See `docs/test_infrastructure.md` for full testing infrastructure details.

## 2. Testing Standards (pytest)
- **Framework:** Use `pytest` exclusively in the `tests/` directory.
- **Speed:** Unit tests must run in <100ms. Aggressively mock heavy simulations, I/O, and external APIs using `unittest.mock`.
- **Integration:** Mark slow tests (full simulation loops) with `@pytest.mark.slow`.
- **Markers:** Use pytest markers: `@pytest.mark.unit` for unit tests, `@pytest.mark.integration` for integration tests.
- **Floats:** NEVER use `==` for floats. Use `pytest.approx(val, rel=1e-5)` or `np.testing.assert_allclose`.
- **Physics:** Ensure inputs are physically valid (e.g., T > 0K) unless testing error handling.

## 3. Coverage Requirements
- **Threshold:** Check `pyproject.toml` [tool.coverage.report] `fail_under` for current threshold.
- **Ratcheting:** Coverage threshold automatically increases on main branch (never decreases).
- **Reports:** Run `pytest --cov --cov-report=html` and inspect `htmlcov/index.html` for gaps.
- **Analysis:** Use `bash tools/coverage_analysis.sh` to identify low-coverage modules needing tests.
- **Quality Gate:** All PRs must pass the coverage threshold defined in CI (see `.github/workflows/proteus_test_quality_gate.yml`).

## 4. Code Quality & Style
- **Linting:** Follow `ruff` standards. Line length < 92 chars, max indentation 3 levels.
- **Type Hints:** Use standard Python type hints.
- **Docstrings:** Include brief docstrings describing the physical scenario.

## 5. Safety & Determinism
- **Randomness:** Explicitly set seeds (e.g., `np.random.seed(42)`) in tests.
- **Files:** Do not generate tests that produce large output files; use `tempfile` or mocks.
