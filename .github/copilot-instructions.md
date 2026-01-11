# PROTEUS Ecosystem Copilot Guidelines

You are an expert Scientific Software Engineer working on the PROTEUS ecosystem.

## Ecosystem Structure

PROTEUS is a coupled atmosphere-interior framework with a modular architecture:

- **[PROTEUS](https://github.com/FormingWorlds/PROTEUS)** (main repository): Core coupling framework and orchestration
- **[AGNI](https://github.com/nichollsh/AGNI)**: Radiative-convective atmospheric energy module (Julia)
- **[SOCRATES](https://github.com/nichollsh/SOCRATES)**: Spectral radiative transfer code (Fortran)
- **[CALLIOPE](https://github.com/FormingWorlds/CALLIOPE)**: Volatile in-/outgassing and thermodynamics module (Python)
- **[JANUS](https://github.com/FormingWorlds/JANUS)**: 1D convective atmosphere module (Python)
- **[MORS](https://github.com/FormingWorlds/MORS)**: Stellar evolution module (Python)
- **[ARAGOG](https://github.com/FormingWorlds/aragog)**: Interior thermal evolution module based on T-P formalism (Python)
- **[SPIDER](https://github.com/djbower/spider)**: Interior thermal evolution module based on T-S formalism (C)
- **[VULCAN](https://github.com/FormingWorlds/VULCAN)**: Atmospheric chemistry module (Python)
- **[ZEPHYRUS](https://github.com/FormingWorlds/ZEPHYRUS)**: Atmospheric escape module (Python)
- **[Love.jl](https://github.com/FormingWorlds/Love.jl)**: Tidal evolution module (Julia)

**Important:** Each module is maintained in its own GitHub repository but is typically cloned/installed within the PROTEUS directory structure for integrated development. When working on any module in the ecosystem, apply these guidelines consistently.

## Scope of These Guidelines

**These guidelines apply to ALL Python modules in the PROTEUS ecosystem.** Whether you are working in:
- The main PROTEUS repository
- A standalone module (CALLIOPE, JANUS, MORS, etc.)
- Tests for any ecosystem component

Follow the same standards for testing, coverage, code quality, and infrastructure.

## Installation & Dependencies

For installation instructions and dependency management across the ecosystem:
- **Main installation guide:** `docs/installation.md` - Standard user and developer installation procedures
- **Local machine setup:** `docs/local_machine_guide.md` - Platform-specific setup (macOS, Linux, Windows)
- **Cluster setup:** `docs/kapteyn_cluster_guide.md` - HPC cluster configuration (see also `habrok_cluster_guide.md`, `snellius_cluster_guide.md`)

When helping with installation or dependency issues, always reference these guides first. The `proteus install-all` command handles most submodule installations automatically. However, whenever possible, prefer the developer installation steps outlined in the installation guide for editable installs.

## 1. Test Infrastructure & Organization
- **Structure:** Tests MUST mirror the source code structure exactly. For every file in `src/<package>/`, create a corresponding `tests/<package>/test_<filename>.py`.
- **Example:** `src/proteus/config/_config.py` → `tests/config/test_config.py`
- **Discovery:** Use `pytest --collect-only` to verify test discovery before writing tests.
- **Tools:** Run `bash tools/validate_test_structure.sh` to check if tests mirror source structure.
- **Documentation:** See `docs/test_infrastructure.md` for full testing infrastructure details.

## 2. Testing Standards (pytest)
- **Framework:** Use `pytest` exclusively in the `tests/` directory.
- **Coverage Tool:** Two equivalent approaches are supported:
  - Local: `pytest --cov` (uses pytest-cov plugin, convenient)
  - CI/Local: `coverage run -m pytest` (matches CI exactly, compatible with ratcheting)
  - Choose based on preference; both work correctly.
- **Speed:** Unit tests must run in <100ms. Aggressively mock heavy simulations, I/O, and external APIs using `unittest.mock`.
- **Integration:** Mark slow tests (full simulation loops) with `@pytest.mark.slow`.
- **Markers:** Use pytest markers: `@pytest.mark.unit` for unit tests, `@pytest.mark.integration` for integration tests.
- **Floats:** NEVER use `==` for floats. Use `pytest.approx(val, rel=1e-5)` or `np.testing.assert_allclose`.
- **Physics:** Ensure inputs are physically valid (e.g., T > 0K) unless testing error handling.
- **Copilot Test Generation:**
  - **Context:** Always read the `conftest.py` of the current module before generating tests to utilize existing fixtures.
  - **Mocking Strategy:** Default to `unittest.mock` for ALL external calls (e.g., network, disk I/O, heavy computation modules like `SOCRATES` or `AGNI`). Only use real calls if explicitly requested for integration tests.
  - **Floats:** Automatically generate assertions using `pytest.approx()` for any floating-point comparisons.
  - **Parametrization:** Prefer `@pytest.mark.parametrize` over writing multiple similar test functions.
  - **Physics Checks:** detailed comments explaining *why* a specific input range was chosen (e.g., "Temperature set to 300K to represent habitable zone conditions").
  - **Instructions:** See `docs/test_building.md` for best practices on building robust tests and `test_building_strategy.md` for current test building strategy.
  -**Documentation:** Add detailed docstrings to each test explaining the physical scenario being tested. In the header of the test file, include a brief overview of what is being tested and any important context, including a link to all docuementation about testing standards: `docs/test_infrastructure.md`, `docs/test_categorization.md` and `docs/test_building.md`.

## 3. Coverage Requirements
- **Threshold:** Check `pyproject.toml` [tool.coverage.report] `fail_under` for current threshold.
- **Automatic Ratcheting:** Coverage threshold automatically increases on main branch via `tools/update_coverage_threshold.py` (never decreases). See CALLIOPE for reference implementation.
- **Reports:** Run `pytest --cov --cov-report=html` and inspect `htmlcov/index.html` for gaps.
- **Analysis:** Use `bash tools/coverage_analysis.sh` to identify low-coverage modules needing tests.
- **Quality Gate:** All PRs must pass the coverage threshold defined in CI (see `.github/workflows/proteus_test_quality_gate.yml`).

## 4. Code Quality & Style
- **Linting:** Follow `ruff` standards. Line length < 92 chars, max indentation 3 levels.
- **Type Hints:** Use standard Python type hints.
- **Docstrings:** Include brief docstrings describing the physical scenario.

## 5. Safety & Determinism
- **Randomness:** Explicitly set seeds (e.g., `np.random.seed(42)`) in tests.
- **Files:** Do not generate tests that produce large output files (unless explicitly instructed); use `tempfile` or mocks.
