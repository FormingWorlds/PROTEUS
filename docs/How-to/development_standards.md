# Development standards

This page summarises the code quality standards that apply to all
contributions to PROTEUS, regardless of how the code is written.

For testing specifics, see [Testing](testing.md). For the conceptual
testing framework, see [Test framework](../Explanations/test_framework.md).
For the contribution process, see [Contributing](../Community/CONTRIBUTING.md).

## Code style

PROTEUS enforces style through `ruff` (linting and formatting) and
pre-commit hooks. Run before every commit:

```bash
ruff check --fix src/ tests/
ruff format src/ tests/
```

Key conventions:

- **Line length**: 96 characters maximum (prefer < 92)
- **Indentation**: 4 spaces, maximum 3 levels
- **Naming**: `snake_case` for variables and functions, `UPPER_CASE` for constants
- **Type hints**: standard Python type hints on function signatures
- **Docstrings**: NumPy-style with Parameters, Returns, Raises sections

## Code organization

PROTEUS is developed by many contributors in parallel. Code is organised to
keep changes local, so that two people adding features rarely edit the same
lines. The targets below are advisory, not enforced gates.

**File size**

- Aim to keep a new module under 500 lines.
- When a file grows past roughly 800 lines, split it along concern boundaries
  before adding more to it.

**Function and method size**

- Aim to keep a function or method under 50 lines.
- Past roughly 80 lines, extract named helper functions.
- Express long orchestration as a sequence of named stage functions that the
  top-level routine calls, not as one inline body.

**Module layout**

Each physics module follows a consistent three-part layout:

- `wrapper.py` holds the public entry point and dispatches to the selected backend.
- `<backend>.py` (for example `aragog.py`, `spider.py`) implements one backend each.
- `common.py` holds helpers shared across backends.

Add a new backend as a new `<backend>.py` file plus a dispatch branch in
`wrapper.py`. Do not append a second backend's logic into an existing backend file.

**Append-friendly registries**

Central lists that every module contributes to (the output-schema keys, config
field sets) are written one entry per line with a trailing comma, grouped by
module under a header comment, and ordered alphabetically within each group.
This keeps two independent additions on different lines so they merge cleanly.

**Editing shared files**

- When adding to the main coupling loop, add a stage function and call it rather
  than inlining logic into a shared method.
- When adding an output column, place it in its module's group, not at the end
  of the global list.
- Prefer short-lived branches and frequent small merges, and give collaborators
  a heads-up before a large edit to a known shared file.

## Physical correctness

PROTEUS is scientific simulation software where incorrect results are worse
than crashes. Every code change must satisfy:

- **Conservation**: mass and energy budgets must close. The runtime invariant
  `assert_mass_conservation` checks this on every iteration.
- **Positivity**: temperatures must be positive (Kelvin), pressures must be
  positive, mass fractions must be in \[0, 1].
- **Unit consistency**: config values use "human" units (M$_\oplus$, bar, Gyr);
  internal values use SI (kg, Pa, yr). Verify units at every boundary.
- **Float comparison**: never use `==` for floating-point values. Use
  `pytest.approx` or `np.testing.assert_allclose` with stated tolerances.

## Test requirements

Every code change that modifies `src/proteus/` must include corresponding
tests in `tests/`. The test must:

1. Mirror the source file structure
2. Carry a tier marker (`@pytest.mark.unit`, `smoke`, `integration`, or `slow`)
3. Include at least two assertions per test function
4. Include at least one edge case
5. Use physically plausible mock values for physics functions
6. Have a function-level docstring stating what is being verified

Physics modules additionally require at least one physics invariant assertion
(conservation, positivity, monotonicity, or a pinned numeric value with a
discrimination guard).

## Mocking discipline

- Mock at the narrowest scope: a specific function, not an entire module
- Never mock the function under test
- Mocked physics functions must return physically plausible values
  (not 0.0 or 1.0 for everything)
- Unit tests mock external solvers; smoke and integration tests use real
  binaries

## Configuration immutability

The `Config` attrs object must not be mutated at runtime. Use local variables
instead of setting `config.X.Y = value` inside module code. Config mutations
outside of initialisation are a known source of subtle coupling bugs.

## Commit messages

- First line: under 72 characters, present tense, imperative mood
  ("Add ...", "Fix ...", "Remove ...")
- Body: explain what changed and why, not how
- No abbreviations or internal shorthand without explanation

## Pull requests

- Title: plain language, under 72 characters
- Body: follow the PR template (Description, Validation, Checklist)
- All CI checks must pass before merge
- PRs modifying > 50 lines of test code receive an independent review

## Version control

- Python 3.12 (Linux and macOS only; Windows is not supported)
- Calendar versioning: YY.MM.DD
- Pre-commit hooks are mandatory: `pre-commit install -f`
