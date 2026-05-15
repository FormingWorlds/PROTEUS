---
description: PROTEUS test quality deep-dive. Anti-happy-path patterns, discriminating-value guards, physics-invariant tiering, validation certification markers, adversarial-review trigger. Extends the Testing Standards section in `.github/copilot-instructions.md`.
---

# PROTEUS Test Quality Rules

This file is the canonical deep-dive on test quality. The high-level summary lives in [`.github/copilot-instructions.md`](../../copilot-instructions.md) under "Testing Standards". The two files MUST stay in sync. If you change one, mirror the change in the other.

> **Discovery note.** PROTEUS keeps its Claude-Code rule files under `.github/.claude/rules/` (not the conventional repo-root `.claude/`) so they can be tracked in git and shared across collaborators. Claude does NOT auto-discover them at this path; the repo-root `CLAUDE.md` (symlinked to `.github/copilot-instructions.md`) names this file and `proteus-code-review.md` explicitly so AI tooling and human readers know to load them. **When opening or editing any file under `tests/**` or `src/proteus/**`, read this file first.**

Sister rule files:

- [`.github/copilot-instructions.md`](../../copilot-instructions.md): high-level rules, applied repo-wide.
- [`.github/.claude/rules/proteus-code-review.md`](proteus-code-review.md): review-pass gate, domain-aware code review (physics plausibility, unit boundaries, hf_row override pattern, etc.). Test-marker discipline lives there too.

PROTEUS is scientific simulation code and the test suite is held to physics-grade rigor. Tests exist to catch real bugs. A test that asserts the wrong thing, or that passes for the wrong reason, is worse than no test because it generates false confidence. The rules below codify what "real test" means here.

---

## 1. Anti-happy-path rules (every new test)

Every new test function MUST include:

1. **At least one edge case**: a boundary value (Phi = 0 or 1, e = 0, T = T_solidus), an empty input, or an extreme physical parameter.
2. **At least one path that exercises the error contract**:
   - If the function under test has documented validation (raises on negative T, refuses to dispatch with `module = None`, etc.), test that the error fires AND that no side effect ran.
   - If the function has no validation (closed-form mathematics: orbital mechanics, thermodynamic relations), exercise the **limit-input behavior** (e = 0 is a fixed point, Imk2 = 0 leaves state unchanged) and assert the corresponding mathematical invariant.
   - "No validation in source therefore no error test" is not an exemption; the limit-input substitute is.
3. **Assertion values NOT trivially derivable from the implementation**: discriminating numeric pins (see Section 2 below) or property-based assertions (monotonicity, conservation, symmetry, boundedness).

### Forbidden patterns

These are flagged by `tools/check_test_quality.py` and rejected at PR time.

- **Single-assert test functions**. Two or more assertions per test; the second usually pins the invariant the first hand-waves over. Exception: a single assertion of a hard-fail invariant (mass closure within `1e-12`) is acceptable if the test is the only test of that invariant in the file.
- **Standalone weak assertions** as the only meaningful check:
  - `assert result is not None`
  - `assert result > 0`
  - `assert len(result) > 0`
  - `assert isinstance(result, dict)`
  - `assert result is None` where the function returns `None` implicitly
  These are fine as **secondary** sanity checks alongside a discriminating assertion.
- **Tests with no function-level docstring**. The docstring states which physical scenario or contract clause is being verified.
- **`==` adjacent to a float literal**. Use `pytest.approx(val, rel=...)` or `np.testing.assert_allclose(actual, expected, rtol=..., atol=...)`. Comparing two floats with `==` is a known flake source even for "exact" identities like 0.0 (-0.0 vs +0.0, NaN propagation).
- **Tests asserting on a fixture's implicit default**: e.g. `assert fixture_returning_none() is None`. This is trivially true. Delete the test; do not strengthen it by adding more `is None` assertions.

---

## 2. Discriminating test values

The test contract is: a regression that introduces a plausible bug must fail the test. "Plausible bug" means off-by-one exponent, wrong sign, swapped factor of 2, missing factor of pi, dimensionally-wrong unit. Pick input values where the wrong-formula result is far from the correct one.

### Bad / good examples

| Pattern | Bad (any-exponent-passes) | Good (discriminates) |
|---|---|---|
| `F = sigma * T**4` | Test at `T = 1` (any power of 1 is 1) | Test at `T = 300` and `T = 1500`; pin the ratio |
| Opacity interpolation | Test at grid nodes (interpolation is identity there) | Test at off-grid points where bilinear vs nearest-neighbor differ |
| Mass-fraction normalization | All species equal (1/N each, symmetric so order doesn't matter) | Asymmetric composition (one dominant species + traces) |
| Kepler period | `a = 1` (sma**1.5 = sma) | `a = 2` and `a = 1`; assert ratio = 2**1.5, not 2 or 8 |

### Discrimination guard (REQUIRED for pinned-value tests)

When a test pins a numeric value, include an explicit assertion that the wrong-formula result would differ from the correct one by more than tolerance. This proves the chosen input actually discriminates. Without it, "I picked a good input" is a claim, not a verified one.

Canonical pattern:

```python
def test_de_dt_matches_closed_form_value_for_unit_params():
    val = de_dt(a=2.0, e=0.5, params=_UNIT_PARAMS)
    expected = (21.0 / 2.0) * 0.5 / (2.0**6.5)
    assert val == pytest.approx(expected, rel=1e-12)
    # Discrimination guard: a regression to a**5 lands at 0.164, not 0.058.
    wrong_a5 = (21.0 / 2.0) * 0.5 / (2.0**5)
    assert abs(val - wrong_a5) > 0.05
```

The guard line is mandatory whenever the test's primary assertion is a `pytest.approx` against a hand-calculated value. It is not required for property-based assertions (those are already discriminating by construction).

---

## 3. Physics-invariant assertions (tiered)

### When required

Every unit test on a **physics module** must assert at least one of the four invariants below. Physics modules are:

```
src/proteus/interior_struct/*
src/proteus/interior_energetics/*
src/proteus/atmos_clim/*
src/proteus/atmos_chem/*
src/proteus/escape/*
src/proteus/outgas/*
src/proteus/orbit/*
src/proteus/star/*
src/proteus/observe/*
src/proteus/inference/objective.py
```

Utility modules are exempt from this requirement but still subject to all anti-happy-path rules:

```
src/proteus/utils/*
src/proteus/config/*
src/proteus/plot/*
src/proteus/cli.py
src/proteus/inference/utils.py
src/proteus/inference/{BO,async_BO,gen_D_init,plot}.py
src/proteus/grid/*
src/proteus/tools/* (when present)
```

### The four invariant families

1. **Conservation**
   - Mass closure: `M_atm + M_mantle + M_core <= M_planet`; per-species `species_kg_atm + species_kg_liquid + species_kg_solid ≈ species_kg_total`.
   - Energy balance: LHS = RHS within stated tolerance for the energy ODE right-hand side.
   - Angular-momentum conservation: total system AM constant when no external torque (satellite tests, tidal evolution).
2. **Positivity / boundedness**
   - `T > 0` Kelvin everywhere, `P > 0` Pa everywhere.
   - Mass fractions in `[0, 1]`; volume mixing ratios in `[0, 1]`.
   - Outgassing rates non-negative; escape rate <= atmospheric mass at the current step.
   - Melt fraction in `[0, 1]`; phi `<=` 1 at all depths.
3. **Monotonicity or symmetry**
   - Pressure increases with depth in interior profiles.
   - Density increases with pressure at fixed entropy.
   - Reversing time integration recovers the IC (time-symmetry of conservative ODEs).
   - Doubling stellar mass while fixing semimajor axis shortens the orbital period by `sqrt(2)`.
4. **Pinned numeric value with a discrimination guard**: see Section 2. This is acceptable as the sole invariant when a closed-form result is the contract.

Property-based assertions (monotonicity, conservation, symmetry, boundedness) are preferred over point-value pins when both are possible. They hold for any valid input and so catch bugs across the entire input space, not just at the chosen test point.

### Validation certification markers

Two markers track validation quality independently of line coverage:

- **`@pytest.mark.physics_invariant`** -- this test asserts at least one of the four invariants. Tag every qualifying test in a physics module. `tools/check_test_quality.py` warns when a physics-module test asserts no invariant and is not tagged.
- **`@pytest.mark.reference_pinned`** -- this test pins behavior against a **published benchmark** (paper, figure, table; cite explicitly in the test docstring), an **analytical limit** (Stefan-Boltzmann black-body limit, hydrostatic equilibrium, isentropic Kepler relations), or a **cross-implementation cross-check** (SPIDER vs Aragog at the same IC, CALLIOPE vs atmodeller at the same fO2 shift). Each physics module must contain at least one `reference_pinned` test. The per-module inventory is tracked in `docs/Validation/<module>.md`; the missing-tests punch list is computed by `tools/check_test_quality.py --reference-pinned-audit`.

Both markers are registered in `pyproject.toml` under `[tool.pytest.ini_options] markers`. They do not gate CI on their own; their coverage is a separate KPI surfaced in the PR summary comment.

---

## 4. Mocking discipline

- Default to `unittest.mock` for ALL external calls in unit tests: SOCRATES, AGNI, SPIDER, Aragog solver, Zalmoxis solver, file I/O, HTTP, subprocess.
- Mock at the **narrowest scope**: patch the specific function (`unittest.mock.patch('proteus.foo.calc_X')`), not the whole module.
- A mocked physics function MUST return **physically plausible** values. A mock that returns `0.0` or `1.0` for everything will mask sign / clamp / fallback bugs.
- NEVER mock the function under test. If you're tempted to, the test is asking the wrong question.
- Smoke tests use real binaries; integration tests use real submodules. The rules in this file still apply to those tiers, but the mocking discipline is relaxed because the real call is the contract.

---

## 5. Optional-dependency imports

Any test that imports an optional dependency MUST call `pytest.importorskip` at module top so Docker `--no-deps` runs do not fail collection:

```python
import pytest

hypothesis = pytest.importorskip('hypothesis')
# ... or for a module-level helper that requires the dep:
pytest.importorskip('boreas')
```

Optional deps that have hit this trap on `tl/interior-refactor`: `hypothesis` (three times), `boreas`, `atmodeller`, `lovepy`, `mors`, `vulcan`, `zalmoxis` (when not installed via editable).

The lint script flags `import <optional_dep>` statements at module top not paired with `importorskip`.

---

## 6. Module-level constants and `monkeypatch`

When the source under test reads an env var into a **module-level constant** at import time, `monkeypatch.setenv` alone is not sufficient. The constant is frozen at the original import.

Trap from `tl/interior-refactor`:

```python
# Source: src/proteus/utils/data.py
FWL_DATA_DIR = Path(os.environ.get('FWL_DATA', '...'))  # frozen at import
```

```python
# Test (wrong):
monkeypatch.setenv('FWL_DATA', str(tmp_path))   # too late; constant already cached

# Test (right):
monkeypatch.setenv('FWL_DATA', str(tmp_path))   # for downstream code that re-reads
monkeypatch.setattr('proteus.utils.data.FWL_DATA_DIR', tmp_path, raising=False)
```

When in doubt, do both. The lint script flags `monkeypatch.setenv` calls in test files that touch source modules with module-level env-derived constants.

---

## 7. Marker discipline and timeouts

### Module-level marker is mandatory

Every test file MUST begin with:

```python
import pytest

pytestmark = [pytest.mark.<tier>, pytest.mark.timeout(<budget>)]
```

with budgets:

- `unit` -> `timeout(30)` (target wall-time per test is `< 100 ms`; the 30 s cap is a defensive net).
- `smoke` -> `timeout(60)` (target `< 30 s`).
- `integration` -> `timeout(300)`.
- `slow` -> `timeout(3600)`.

CI runs `pytest -m "unit and not skip"`. Tests without the tier marker are invisible to CI and shipped untested. The lint script blocks any file missing the module-level `pytestmark`.

### Per-function markers

Per-function `@pytest.mark.<tier>` markers are **additive**, not a replacement for the module-level marker. They are useful when a file's tests span multiple tiers (rare; prefer separate files).

### Timeout is a safety net, not a target

The `timeout` ceiling exists so a future regression that introduces a hang (real solver call, infinite loop, network retry) surfaces as a specific-test failure rather than a generic job timeout. Current test wall times are 100x below the ceiling; if you find yourself needing the full 30 s for a unit test, something has gone wrong and you should reduce scope or move the test to a slower tier.

---

## 8. Float and numerical comparison

- NEVER use `==` for floats. Use `pytest.approx(val, rel=1e-5)` or `np.testing.assert_allclose(actual, expected, rtol=..., atol=...)`.
- State the tolerance rationale in a comment when the choice is non-obvious. E.g. "`rtol=1e-3` because the Cp lookup truncates entries to 4 sig fig".
- For pinned numeric values, include a **discrimination guard** (Section 2).
- For property-based assertions, use `pytest.approx` against the exact symbolic relation, with the tightest tolerance the implementation can hit (typically `rel=1e-12` for closed-form algebra; looser for ODE integrations).

---

## 9. Voice rule for test artifacts

The repo-wide voice rule (zero AI-process disclosure in any public artifact) applies to test code with the same strictness as to source. Banned in:

- Test-skip reasons (`@pytest.mark.skip(reason='...')`).
- Test-file docstrings.
- Parametrize ids (`@pytest.mark.parametrize('name', [...], ids=[...])`).
- Log-capture assertions (regex against `caplog.records`).
- Test-function names.
- Test-class names.

Banned phrases: "audit", "review pass", "adversarial review", "Phase X", "T1.x", "Group A/B/C/D", `claude-config/...` paths, "Generated with Claude", em-dashes, en-dashes (except in bibliographic page ranges within citations).

Write the OUTCOME (what the test verifies) never the PROCESS (how the rule was derived). First-person Tim voice. Going-forward only, no history rewrite.

---

## 10. Fixture and parameter conventions

- Use SI units in test parameters unless the function under test explicitly expects config units (M_sun, bar, Gyr, K).
- Use the conftest parameter classes (`EarthLikeParams`, `UltraHotSuperEarthParams`, `IntermediateSuperEarthParams`) for realistic test scenarios. They are session-scoped and cheap.
- Use `@pytest.mark.parametrize` when the same logic spans multiple physical regimes (Earth-like, super-Earth, sub-Neptune). Each parametrize id must read like a physical scenario, not a tuple of numbers.
- Set seeds for any randomness:
  ```python
  np.random.seed(42)
  torch.manual_seed(42)
  random.seed(42)
  ```
  All three must be seeded if all three are used. Today's BO convergence regression confirmed: seeding only `torch.manual_seed` is not enough.
- Use `tmp_path` (pytest fixture) for temporary files. Do not produce large outputs in the test path.

---

## 11. Documentation per test

- **File-level docstring**: name the module under test, list the invariants and contract clauses the file exercises, link to the three test docs. Required.
- **Function-level docstring**: state the physical scenario or contract clause in plain language. Required (lint-enforced).
- **Inline comments**: explain **why** a specific input range was chosen ("T=300 K and T=1500 K so the T**3 vs T**4 difference is resolved well above tolerance").

---

## 12. Naming

- Test names describe behavior, not the called function: `test_opacity_monotonic_with_temperature`, NOT `test_get_opacity`.
- Test names use snake_case and read as full sentences.
- Group related tests in classes (`class TestOpacity:`) when they share setup; use the class to thread a single fixture through several scenarios.

---

## 13. Adversarial review trigger

Any single commit that adds or substantially modifies **> 50 lines** of test code (cumulative across all `tests/**` paths) triggers an adversarial-review pass before merge.

The reviewer's mandate:

- Cite the anti-happy-path rule (Section 1) and the discrimination-guard requirement (Section 2).
- Flag single-assert tests, weak `is not None` patterns, missing module-level marker, missing `physics_invariant` tag on a physics-module test, missing `reference_pinned` tag on a per-module benchmark test, dead tests (passes for the wrong reason), tests that mock the function under test.
- Verify discriminating values: re-derive the expected value from a plausible wrong formula and assert the test fails with that wrong formula.
- Verify physics module coverage of the four invariants: which of the four does this test exercise? If none, why is the test in `tests/<physics_module>/`?

The reviewer is a separate process from the test author. For Claude-Code workflow this means spawning a `proteus-review` skill or a `code-reviewer` agent with the test files in scope; the review must complete and surface findings before the test commit is pushed.

The review's findings are addressed in a follow-up commit (not amended into the test commit). The follow-up subject line is in plain language describing the OUTCOME ("sharpen orbital period assertions to distinguish a**1.5 from a**1.0", NOT "address review findings").

---

## 14. Tooling

The repo provides:

- `bash tools/validate_test_structure.sh` -- structural check (`tests/` mirrors `src/proteus/`).
- `python tools/check_test_quality.py --check` -- CI mode: AST scan for the forbidden patterns in Section 1 and the marker requirement in Section 7. Fails the PR if NEW violations exceed the baseline.
- `python tools/check_test_quality.py --baseline` -- after a deliberate sweep, regenerates `tools/test_quality_baseline.json`. Only run when you have intentionally reduced violations.
- `python tools/check_test_quality.py --reference-pinned-audit` -- prints physics modules missing a `reference_pinned` test.
- `bash tools/coverage_analysis.sh` -- coverage by module, sorted by gap.
- `ruff check src/ tests/` and `ruff format src/ tests/` -- run before commit.

The lint script is wired into PR CI as a non-blocking warning step initially; once the legacy violations are swept and the baseline is reset to zero, the step becomes blocking.

---

## 15. Coverage strategy (operator's view)

PROTEUS uses two coverage gates with explicit sub-targets. The fast gate is for PR cycle time; the estimated total is the real KPI.

| Gate | Tests | Target | When |
|---|---|---|---|
| Fast gate (`tool.proteus.coverage_fast`) | unit + smoke | ratcheting toward **70%** | Every PR |
| Estimated total (PR union with nightly artifact) | unit + smoke + integration | **90%** (PROTEUS-ecosystem ceiling) | Every PR |
| Full gate (`tool.coverage.report`) | unit + smoke + integration + slow | **90%** | Nightly |
| Diff-cover | changed lines | 80% (hard-coded) | Every PR |

Unit tests alone are not expected to reach 90%. Wrapper code that requires real binaries (SOCRATES, AGNI, SPIDER) is covered by smoke / integration / slow tests in nightly; the nightly artifact is downloaded into PR runs and unioned with the PR's own coverage to estimate the total.

What this means for adding tests:

- A new function in a physics module that wraps a real binary: write a unit test with mocks (counts toward fast gate), AND a smoke or integration test with the real binary (counts toward estimated total / full gate).
- A new closed-form helper in a utility module: a unit test is sufficient.
- A new orchestration function in `proteus.py` or `cli.py`: a unit test for argument parsing and dispatch, plus an integration test for the actual call path.

The ratchet is one-way (`tools/update_coverage_threshold.py`), capped at 90%. Never manually decrease the threshold.

---

## 16. Failure modes to recognize on review

These are real patterns that have shipped in the past. The lint script catches some of them mechanically; reviewers catch the rest.

| Pattern | Example | Why it slipped | Fix |
|---|---|---|---|
| Silent skip in helper | `def _enum_for(field): ...; if actual is None: continue` masks broken introspection | Helper hides a real failure as a no-op | Hard assertion: `assert actual is not None, ...` |
| Centered-target convergence | BO test asserts `(initial_best - final_best) / initial_best > 0.5` while the target is `(0.5, 0.5)`; a regression returning constant `(0.5, 0.5)` passes | The "improvement" metric is dominated by where the target sits | Add a proximity check: `||x_best - target|| < 0.5 * initial_min_dist` |
| Log-line-only assertion | Test captures a log line and asserts on its text; a regression that changes the code path but keeps the log still passes | Logs are not the contract | Capture the call kwarg and assert on the value passed in |
| Module-level constant patched only via env var | `monkeypatch.setenv('FWL_DATA', ...)` on a source that read it at import time | Constants are frozen at import; setenv is too late | `monkeypatch.setattr('mod.CONST', ...)` in addition to setenv |
| Optional dep imported unconditionally | `import hypothesis` at module top | Docker `--no-deps` build skips the optional install | `pytest.importorskip('hypothesis')` at module top |
| Stale marker after refactor | File moved from `interior/` to `interior_energetics/`, kept `@pytest.mark.unit` but missing the module-level `pytestmark` | CI marker filter still passed because of per-function markers; coverage tier became invisible | Add module-level `pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]` |
| Trivially-true on implicit None | `def fixture(): pass`; `def test_x(fixture): assert fixture is None` | Fixture returned None implicitly; test passes for the wrong reason | Delete the test |

When you spot a new variant of these, add it here.

---

## 17. Sister rules (cross-link)

- `.github/copilot-instructions.md` "Testing Standards" -- the high-level summary readers without `tests/**` context see first.
- `.claude/rules/proteus-code-review.md` "Test marker discipline" -- the review-pass gate that backs up the rules in this file. Also contains domain-aware physics checks (Stefan-Boltzmann exponent, hf_row override pattern, IC consistency, whole-element aggregation symmetry) that apply when reviewing the **source** code that tests cover.

Any change to the rule set: update both files in the same commit and call out the cross-reference in the commit body.
