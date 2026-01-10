# Guide: Building Robust Tests with GitHub Copilot

This guide outlines how to configure Copilot and use the standard "Master Prompt" to generate high-quality, physics-compliant unit tests for the PROTEUS ecosystem.

## 1. Configuration (`copilot-instructions.md`)

Add the following subsection to the end of **Section 2. Testing Standards** in your `copilot-instructions.md` file. This ensures Copilot understands our specific constraints (mocking, physics, precision) by default.

```markdown
- **Copilot Test Generation:**
  - **Context:** Always read `conftest.py` of the current module before generating tests to utilize existing fixtures.
  - **Mocking Strategy:** Default to `unittest.mock` for ALL external calls (network, disk I/O, heavy computation modules like `SOCRATES`/`AGNI`). Only use real calls if explicitly requested for integration tests.
  - **Floats:** Automatically generate assertions using `pytest.approx()` for any floating-point comparisons.
  - **Parametrization:** Prefer `@pytest.mark.parametrize` over writing multiple similar test functions.
  - **Physics Checks:** Add comments explaining *why* a specific input range was chosen (e.g., "T=300K for habitable zone").

## 2. Developer Workflow ("The Context Sandwich")

To get the best results, set up your IDE context before prompting:

1.  **Open Source:** Open the file you are testing (e.g., `src/janus/convection.py`).
2.  **Open Destination:** Open/Create the test file (e.g., `tests/janus/test_convection.py`).
3.  **Open Fixtures:** Open `tests/conftest.py` so Copilot sees available fixtures.
4.  **Prompt:** Paste the **Master Prompt** (below) into Copilot Chat.

## 3. The Master Prompt

Copy and paste this strictly into the Copilot Chat window:

> **Act as a Senior Scientific Software Engineer for PROTEUS.**
>
> I need robust unit tests for the open file. Follow these strict guidelines:
>
> 1.  **Architecture:** Mirror the source code structure. If testing `class Convection`, create `class TestConvection`.
> 2.  **Mocking:** This is a unit test. **Aggressively mock** any heavy physics modules (e.g., `SOCRATES`, `AGNI`) or I/O operations using `unittest.mock`. The test must run in <100ms.
> 3.  **Precision:** Use `pytest.approx(expected, rel=1e-5)` for all float comparisons. Never use `==`.
> 4.  **Physics:** Ensure test inputs are physically valid (e.g., Kelvin > 0, Pressure > 0) unless testing error handling.
> 5.  **Coverage:** Target >90% coverage. Handle edge cases (None, empty arrays, negative values where physically impossible).
> 6.  **Style:** Use `@pytest.mark.parametrize` for data-driven tests. Add a brief docstring to each test explaining the physical scenario being tested.
> 7.  **Format:** Ruff format all test files before committing.
>
> **Generate the tests now.**
