# PROTEUS short instructions

The full instructions are in `AGENTS.md` (repository rules and review points) and `tests/AGENTS.md` (test rules). This file repeats what a review or a chat needs when it cannot read them.

## Commands

```bash
pip install -e ".[develop,vulcan,atmodeller,inference]"
pytest -m "unit and not skip and not slow and not integration" --ignore=tests/examples
ruff check src/ tests/ tools/ && ruff format --check src/ tests/ tools/
bash tools/validate_test_structure.sh
python tools/check_test_quality.py --check  # no rule's Current count may exceed the same run on origin/main
python tools/agents/check_agents_md.py
proteus start -c <config.toml> --offline
```

## Review checklist

- Physics: T > 0 K; P > 0 and increasing with depth; mass fractions sum to 1; the mass escaped in one step stays within its reservoir (`limit_escape_step`, `ESCAPE_STEP_MAX_FRAC`); outgassing >= 0; radiative flux uses `sigma * T**4`.
- Units at each boundary: `GetHelpfileKeys` in `src/proteus/utils/coupler.py` states each `hf_row` unit (atmospheric pressures in bar, interior pressures such as `P_cmb` in Pa, model time in years, orbital periods in s); each config key states its unit in `docs/Reference/config/` (stellar mass in M_sun, planet mass in M_earth).
- `Config` is not changed during a run except inside `try`/`finally` that restores it; temporary `hf_row` overrides are restored in `finally`.
- Every site that sums element masses includes oxygen, and `M_ele` and `M_planet` leave rock-vapour elements out; `assert_mass_conservation` is not weakened and `atol_frac` not widened (the `outgas.vapourise` relaxation is the one exception).
- A user value that a solver derives again gets a one-time check at the initial condition.
- Physical constants come from one definition.
- EOS tables: P-S tables are complete rectangles; each Aragog P-T table is a full rectangular grid, never filtered by phase.
- Every attrs validator is tested with a valid and an invalid input.
- Tests: `src/proteus/<module>/<file>.py` is tested in `tests/<module>/test_<file>.py`; each file has `pytestmark = [pytest.mark.<tier>, pytest.mark.timeout(<s>)]`; each test has a docstring, at least 2 assertions, an edge case and the error path; floats are compared with a tolerance; a test that asserts a physical invariant carries `physics_invariant`, a test against a published or analytical value also carries `reference_pinned`, and pinned values have sign, scale and wrong-formula guards.
- Commit messages, pull-request text, comments, test names and CI job names describe the change: no tool attribution, no internal plan labels, no em or en dashes; an inline comment block has at most 4 lines.

This checklist repeats the Review section of `AGENTS.md`; change both together.
