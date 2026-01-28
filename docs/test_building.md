# Building Robust Tests

This guide describes the developer workflow and standard prompts for generating unit and integration tests. For test structure, markers, and CI see [Test Categorization](test_categorization.md) and [Test Infrastructure](test_infrastructure.md).

---

## Developer Workflow ("Context Sandwich")

1. **Open source**: The file under test (e.g. `src/proteus/utils/helper.py`).
2. **Open destination**: The test file (e.g. `tests/utils/test_helper.py`).
3. **Open fixtures**: `tests/conftest.py` so available fixtures are in context.
4. **Prompt**: Paste the **Master Prompt** (unit) or **Integration Prompt** below into the chat.

---

## Master Prompt (Unit Tests)

Copy into the chat when generating unit tests:

> **Act as a Senior Scientific Software Engineer for PROTEUS.**
>
> I need robust unit tests for the open file. Follow these strict guidelines:
>
> 1. **Architecture:** Mirror the source. If testing `class Convection`, create `class TestConvection`. File: `tests/<module>/test_<filename>.py` for `src/proteus/<module>/<filename>.py`.
> 2. **Mocking:** This is a unit test. **Aggressively mock** heavy physics (SOCRATES, AGNI) and I/O with `unittest.mock`. Tests must run in &lt;100 ms.
> 3. **Precision:** Use `pytest.approx(expected, rel=1e-5)` for all float comparisons. Never use `==` for floats.
> 4. **Physics:** Use physically valid inputs (e.g. T &gt; 0 K, P &gt; 0) unless testing error handling.
> 5. **Coverage:** Aim for high coverage; include edge cases (None, empty arrays, invalid values where relevant).
> 6. **Style:** Use `@pytest.mark.parametrize` for data-driven tests. Add a brief docstring per test describing the scenario. Use `@pytest.mark.unit`.
> 7. **Format:** Run `ruff format` on test files before committing.
>
> **Generate the tests now.**

---

## Integration Prompt (Standard Configuration)

Use when adding or extending integration tests (e.g. ARAGOG+AGNI+CALLIOPE+ZEPHYRUS+MORS):

> **Act as a Senior Scientific Software Engineer for PROTEUS.**
>
> I need an integration test for the Standard Configuration (e.g. test_std_config.py or multi-module coupling).
>
> 1. **Scope:** Test coupling of the relevant modules (ARAGOG, AGNI, CALLIOPE, ZEPHYRUS, MORS as needed).
> 2. **Mocking:** **Do not mock** internal physics between these modules. Mock only external I/O (e.g. network downloads) with `unittest.mock`.
> 3. **Config:** Use fixtures from `tests/conftest.py` and `tests/integration/conftest.py` (e.g. `intermediate_params`, config paths).
> 4. **Verification:** Stable evolution (multiple timesteps, no crash/NaN); energy and mass conservation with stated tolerances; feedback checks (e.g. T_surf ↔ outgassing ↔ atmos_mass).
> 5. **Marker:** Use `@pytest.mark.integration` (and `@pytest.mark.slow` if long-running).
>
> **Generate the integration test skeleton.**

---

## See Also

- [Test Categorization](test_categorization.md) — Markers, CI pipeline, fixtures
- [Test Infrastructure](test_infrastructure.md) — Layout, coverage, troubleshooting
- [Test Building Strategy](test_building_strategy.md) — Status and principles
- [AGENTS.md](../AGENTS.md) — Test commands and coverage thresholds
