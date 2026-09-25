# PROTEUS test instructions

<!-- fwl-tests-core:begin sha256=dc5361ba219c6b60 -->
## Test rules shared by the PROTEUS ecosystem

Each test file starts with a module-level tier marker and a timeout. CI selects tests by marker, so a file without one runs in no CI job; the timeout stops a hang, it is not a target.

```python
pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]
```

| tier | what it tests | timeout |
|---|---|---|
| `unit` | Python logic with the heavy physics mocked; aim for < 100 ms | 30 s |
| `smoke` | the real binaries or solvers, one step, low resolution | 60 s |
| `integration` | several modules coupled | 300 s |
| `slow` | full physics validation | 3600 s |

Per-function markers add to the module marker; they do not replace it. `skip` marks a placeholder that no CI job runs.

Every new test covers an edge case (a boundary value, an empty input, an extreme physical parameter), exercises the error contract (a documented exception, a guard, a clamp, or the limit input of the formula), and asserts values that do not follow trivially from the implementation. A pinned value of 1 that every exponent reproduces checks nothing.

`python tools/check_test_quality.py --check` reports violations that are new against `tools/test_quality_baseline.json`: a file without a tier marker, a test without a docstring, a test with one assertion or none, a weak assertion as the only one (`is None`, `is not None`, `> 0`, `len(...) > 0`, `isinstance`), `==` next to a float literal, and an optional dependency imported without `pytest.importorskip`. `bash tools/validate_test_structure.sh` runs the repository's own structure check; `tests/AGENTS.md` below says what it checks here.

Physics tests carry markers so their coverage is tracked apart from line coverage:
- `@pytest.mark.physics_invariant` on each test function that asserts a conservation law, a bound (T > 0, fractions in [0, 1]), a monotonicity or symmetry, or a pinned value with a discrimination guard. The marker goes on the function, not the module: structural tests in the same file do not carry it.
- `@pytest.mark.reference_pinned` (together with `physics_invariant`) on a test that pins a published benchmark, an analytical limit or a cross-implementation result; cite the paper, table or figure in the docstring.
- A discrimination guard asserts that the most plausible wrong formula (a missing factor, a swapped exponent, the wrong unit) gives a result outside the tolerance of the pinned value.

Floats: compare with `pytest.approx` or `np.testing.assert_allclose` and a tolerance you can justify from the method, not the one that makes the test pass.

Mocks: mock at the narrowest scope (the one external call), and return physically plausible values, so the code under test runs its real branches. Set random seeds and write files only under `tmp_path`.

A module-level constant read from an environment variable at import time does not change with `monkeypatch.setenv`; patch the constant with `monkeypatch.setattr`.
<!-- fwl-tests-core:end -->

## PROTEUS specifics

Structure: `src/proteus/<module>/<file>.py` is tested in `tests/<module>/test_<file>.py`. `bash tools/validate_test_structure.sh` checks only that every source directory has a test directory with at least one test file, so the one-to-one file rule is yours to keep. Shared fixtures are in `tests/conftest.py` (`EarthLikeParams`, `UltraHotSuperEarthParams`, `IntermediateSuperEarthParams`, `config_minimal`, `config_dummy`); read it before you write a test.

CI: pull requests run the unit tier only (`pytest -m "unit and not skip and not slow and not integration"`); smoke, integration and slow run nightly. `tools/check_test_quality.py --check` runs in the PR job with `continue-on-error` and compares against `tools/test_quality_baseline.json`; a new violation shows in the log. Regenerate the baseline (`--baseline`) only after a sweep that removed violations.

### Physics modules

A unit test of these sources asserts at least one invariant and carries `physics_invariant`: `interior_struct`, `interior_energetics`, `atmos_clim`, `atmos_chem`, `escape`, `outgas`, `orbit`, `star`, `observe`, and `inference/objective.py`, `inference/BO.py`, `inference/async_BO.py` (the Bayesian optimisation drives the simulator). A helper in a physics directory counts as physics when its output feeds physics; only pure plumbing (logging, paths, type coercion) is exempt. `utils`, `config`, `plot`, `grid`, `cli.py` and the other `inference` files are exempt from the invariant rule, not from the rest.

Each physics source file whose public API returns a physical quantity has its own `reference_pinned` test (`aragog.py` and `spider.py` each need one), listed in `docs/Validation/<module>/<file>.md` with the reference, the test ids and the date of the last comparison. `python tools/check_test_quality.py --reference-pinned-audit` reports per directory only, so check the per-file pages by hand.

Invariants used here: `M_atm + M_mantle + M_core <= M_planet` and per-species `kg_atm + kg_liquid + kg_solid` equal to the species total; the energy ODE right-hand side balance; angular momentum without external torque; T > 0 K and P > 0 Pa; melt fraction in [0, 1]; P increasing with depth; density increasing with pressure at fixed entropy. Prefer a property over a point value: it holds across the input space.

### Discriminating values

Choose inputs where the wrong formula gives a different answer: `sigma * T**4` at T = 300 K and 1500 K, not T = 1; interpolation off the grid nodes; an asymmetric composition, not equal fractions; a Kepler period at a = 2 AU, where `a**1.5` differs from `a` and `a**2`. A test that pins a hand-calculated value with `pytest.approx` adds three guards:

```python
def test_de_dt_matches_closed_form_value_for_unit_params():
    """de/dt at a = 2, e = 0.5 matches the closed form and not its a**5 variant."""
    val = de_dt(a=2.0, e=0.5, params=_UNIT_PARAMS)
    assert val == pytest.approx((21.0 / 2.0) * 0.5 / 2.0**6.5, rel=1e-12)
    assert abs(val - (21.0 / 2.0) * 0.5 / 2.0**5) > 0.05  # exponent error lands at 0.164
    assert val > 0  # sign error
    assert 1e-3 < val < 1.0  # scale error (kg against g, a missing division)
```

When the primary assertion is a closure (`sum(parts) == pytest.approx(total)`), it already catches a factor error; the sign and scale guards stay. The single-assert exception: one assertion of a hard closure (mass within 1e-12) is enough when it is the only test of that invariant in the file.

### Mocks, fixtures, seeds

- Unit tests mock SOCRATES, AGNI, SPIDER, the Aragog and Zalmoxis solvers, file I/O, HTTP and subprocesses, at the narrowest scope (`patch('proteus.foo.calc_X')`), with physically plausible return values; never mock the function under test. Smoke tests use the real binaries and integration tests the real modules.
- `pytest.importorskip` at module top for `hypothesis`, `boreas`, `atmodeller`, `lovepy`, `mors`, `vulcan`, `zalmoxis`, `torch`, `botorch`, `gpytorch`: the PR image installs PROTEUS without them.
- `proteus.utils.data.FWL_DATA_DIR` is read from `FWL_DATA` at import; patch it with `monkeypatch.setattr(..., raising=False)` as well as `setenv`.
- Test parameters are SI unless the function takes config units (M_sun, bar, Gyr, K). Parametrize ids name the physical scenario.
- Seed every generator in use (`np.random.seed`, `torch.manual_seed`, `random.seed`); seeding only torch leaves the Bayesian optimisation tests non-deterministic.
- A slow test with `interior_struct.module = 'zalmoxis'` and dummy outgassing loops in the initial equilibration (surface pressure stays near 0, so it never converges); use `**minimal_zalmoxis_overrides()` from `tests/integration/conftest.py`.

### Docstrings and names

The file docstring names the source under test and lists the invariants it checks. Each test docstring states the physical scenario or contract clause. A comment explains why an input was chosen ("T = 300 K and 1500 K resolve T**3 against T**4"). Names describe behaviour (`test_opacity_monotonic_with_temperature`, not `test_get_opacity`).

### Coverage gates

| gate | tests | target |
|---|---|---|
| fast (`[tool.proteus.coverage_fast]`) | PR unit tier | 80 % |
| estimated total (PR unioned with the latest nightly) | all tiers | 90 % (`[tool.coverage.report]`) |
| diff-cover | changed lines, same union | 80 % |

The gates warn on draft pull requests and block once a pull request is ready for review. The targets are fixed (`CEILINGS` in `tools/update_coverage_threshold.py`), and a pull request that edits `fail_under` away from them fails. A function that wraps a real binary gets a mocked unit test and a smoke or integration test with the binary; a closed-form helper needs only a unit test. `bash tools/coverage_analysis.sh` lists coverage by module.

### Patterns that passed CI and were wrong

- A helper that turns a failed lookup into `continue` hides the failure; assert instead.
- A convergence test whose target sits where a constant output also scores well; add a distance-to-target check.
- A test that asserts only on a log line; assert on the value passed to the call.
- A test of a fixture's implicit `None`; delete it.
- A file with per-function markers and no module `pytestmark`; the tier filter misses it.

A pull request that adds or changes more than 50 lines under `tests/` (`git diff origin/main...HEAD -- tests/`) gets an independent review of its tests before merge: rule compliance, and each pinned value rechecked against a plausible wrong formula.
