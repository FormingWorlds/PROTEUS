# Building Robust Tests

## What This Document Is For

**New to testing?** This guide helps you write tests for PROTEUS code. Tests are small programs that check if your code works correctly. When you change code, tests catch bugs before they reach production.

**Key concept:** For most code changes, you only need **unit tests** (fast, isolated tests). Integration tests are for advanced scenarios involving multiple physics modules working together.

For test markers and CI pipelines, see [Test Categorization](test_categorization.md). For coverage analysis and troubleshooting, see [Test Infrastructure](test_infrastructure.md).

---

## Quick Start: Writing Your First Test

1. **Create test file**: For `src/proteus/utils/helper.py`, create `tests/utils/test_helper.py`
2. **Add a test function**: Start with `test_` prefix, add `@pytest.mark.unit` marker
3. **Run it**: `pytest tests/utils/test_helper.py -v`
4. **Check coverage**: `pytest --cov=src tests/utils/`

```python
import pytest
from proteus.utils.helper import my_function

@pytest.mark.unit
def test_my_function_basic():
    """Test that my_function returns expected value."""
    result = my_function(input_value=10)
    assert result == pytest.approx(expected_value, rel=1e-5)
```

---

## Developer Workflow

1. **Open source**: The file under test (e.g., `src/proteus/utils/helper.py`)
2. **Open destination**: The test file (e.g., `tests/utils/test_helper.py`)
3. **Open fixtures**: `tests/conftest.py` for available fixtures
4. **Write tests**: Use the prompts below if using AI assistance
5. **Run and verify**: `pytest -m unit` then `ruff format tests/`

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
- [Test Infrastructure](test_infrastructure.md) — Layout, coverage, reusable quality gate
- [Docker CI Architecture](docker_ci_architecture.md) — Docker image, CI pipelines
- [.github/copilot-instructions.md](https://github.com/FormingWorlds/PROTEUS/blob/main/.github/copilot-instructions.md) — Test commands and coverage thresholds
