# Guide: Building Robust Tests with GitHub Copilot

This guide outlines how to configure Copilot and use the standard "Master Prompt" to generate high-quality, physics-compliant unit tests for the PROTEUS ecosystem.

## 1. Developer Workflow ("The Context Sandwich")

To get the best results, set up your IDE context before prompting:

1.  **Open Source:** Open the file you are testing (e.g., `src/janus/convection.py`).
2.  **Open Destination:** Open/Create the test file (e.g., `tests/janus/test_convection.py`).
3.  **Open Fixtures:** Open `tests/conftest.py` so Copilot sees available fixtures.
4.  **Prompt:** Paste the **Master Prompt** (below) into Copilot Chat.

## 2. The Master Prompt (Unit Tests)

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

## 3. The Integration Prompt (Standard Configuration)

For Phase 2 validation (coupling ARAGOG+AGNI+CALLIOPE+ZEPHYRUS+MORS), use this prompt:

> **Act as a Senior Scientific Software Engineer for PROTEUS.**
>
> I need an integration test for the Standard Configuration (test_std_config.py).
>
> 1.  **Scope:** Test the full coupling of ARAGOG, AGNI, CALLIOPE, ZEPHYRUS, and MORS.
> 2.  **Mocking:** **Do NOT mock** the internal physics interactions between these modules. Mock only *external* I/O (like network downloads) using `unittest.mock`.
> 3.  **Config:** Use `tests/conftest.py` fixtures (e.g., `intermediate_params`) to set up a realistic super-Earth scenario.
> 4.  **Verification:**
>     - **Stable Evolution:** Run for 3+ timesteps without crashing or NaN values.
>     - **Conservation:** Assert global energy and mass conservation (tolerances: 1e-4).
>     - **Feedback:** Verify that `T_surf` updates affect `outgassing_rate`, which affects `atmos_mass`, which affects `T_surf`.
> 5.  **Marker:** Use `@pytest.mark.integration`.
>
> **Generate the integration test skeleton.**
