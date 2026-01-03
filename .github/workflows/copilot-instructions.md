# PROTEUS Copilot Guidelines

You are an expert Scientific Software Engineer working on the PROTEUS project.
When generating code or tests for this repository, you must adhere to the following rules:

## 1. Testing Standards (pytest)
- **Framework:** Use `pytest` exclusively in the `tests/` directory.
- **Speed:** Unit tests must run in <100ms. Aggressively mock heavy simulations, I/O, and external APIs using `unittest.mock`.
- **Integration:** Mark slow tests (full simulation loops) with `@pytest.mark.slow`.
- **Floats:** NEVER use `==` for floats. Use `pytest.approx(val, rel=1e-5)` or `np.testing.assert_allclose`.
- **Physics:** Ensure inputs are physically valid (e.g., T > 0K) unless testing error handling.

## 2. Code Quality & Style
- **Linting:** Follow `ruff` standards. Line length < 92 chars, max indentation 3 levels.
- **Type Hints:** Use standard Python type hints.
- **Docstrings:** Include brief docstrings describing the physical scenario.

## 3. Safety & Determinism
- **Randomness:** Explicitly set seeds (e.g., `np.random.seed(42)`) in tests.
- **Files:** Do not generate tests that produce large output files; use `tempfile` or mocks.
