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

Each test carries exactly one tier marker, so that the tier filters select it in the one CI job meant for it; a function marker for a second tier breaks that. Split a file whose tests need different tiers. `skip` marks a placeholder that no CI job runs.

Every new test covers an edge case (a boundary value, an empty input, an extreme physical parameter), exercises the error contract (a documented exception, a guard, a clamp, or the limit input of the formula), and asserts values that do not follow trivially from the implementation. A pinned value of 1 that every exponent reproduces checks nothing.

`python tools/check_test_quality.py --check` fails when the count of any rule rises above `tools/test_quality_baseline.json`; the offenders it prints are the first few of that rule in the tree, not necessarily yours. The rules: a file without a tier marker, a test without a docstring, a test with one assertion or none, a weak assertion as the only one (`is None`, `is not None`, `> 0`, `len(...) > 0`, `isinstance`), `==` next to a float literal, and an optional dependency imported without `pytest.importorskip`. `bash tools/validate_test_structure.sh` runs the repository's own structure check; `tests/AGENTS.md` below says what it checks here.

Physics tests carry markers so their coverage is tracked apart from line coverage:
- `@pytest.mark.physics_invariant` on each test function that asserts a conservation law, a bound (T > 0, fractions in [0, 1]), a monotonicity or symmetry, or a pinned value with a discrimination guard. The marker goes on the function, not the module: structural tests in the same file do not carry it.
- `@pytest.mark.reference_pinned` (together with `physics_invariant`) on a test that pins a published benchmark, an analytical limit or a cross-implementation result; cite the paper, table or figure in the docstring.
- A discrimination guard asserts that the most plausible wrong formula (a missing factor, a swapped exponent, the wrong unit) gives a result outside the tolerance of the pinned value.

Floats: compare with `pytest.approx` or `np.testing.assert_allclose` and a tolerance you can justify from the method, not the one that makes the test pass.

Mocks: mock at the narrowest scope (the one external call), and return physically plausible values, so the code under test runs its real branches. Set random seeds and write files only under `tmp_path`.

A module-level constant read from an environment variable at import time does not change with `monkeypatch.setenv`; patch the constant with `monkeypatch.setattr`.
