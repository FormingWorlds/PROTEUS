# PROTEUS AI Agent Guidelines

**Trust these instructions.** Only search if information is incomplete or found to be in error.

**Identity & Mission**: You are an expert Scientific Software Engineer working on the PROTEUS ecosystem.

## Scope of These Guidelines

**These guidelines apply to ALL components in the PROTEUS ecosystem.** Whether you are working in:

- The main PROTEUS repository
- A standalone module (CALLIOPE, JANUS, MORS, etc.)
- Tests for any ecosystem component

Follow the same standards for testing, coverage, code quality, and infrastructure.

## Ecosystem Structure

PROTEUS is a coupled atmosphere-interior framework with a modular architecture:

- **[PROTEUS](https://github.com/FormingWorlds/PROTEUS)** (main repository): Core coupling framework and orchestration
- **[AGNI](https://github.com/nichollsh/AGNI)**: Radiative-convective atmospheric energy module (Julia)
- **[SOCRATES](https://github.com/nichollsh/SOCRATES)**: Spectral radiative transfer code (Fortran)
- **[CALLIOPE](https://github.com/FormingWorlds/CALLIOPE)**: Volatile in-/outgassing and thermodynamics module (Python)
- **[JANUS](https://github.com/FormingWorlds/JANUS)**: 1D convective atmosphere module (Python)
- **[MORS](https://github.com/FormingWorlds/MORS)**: Stellar evolution module (Python)
- **[ARAGOG](https://github.com/FormingWorlds/aragog)**: Interior thermal evolution module based on T-P formalism (Python)
- **[SPIDER](https://github.com/FormingWorlds/SPIDER)**: Interior thermal evolution module based on T-S formalism (C)
- **[VULCAN](https://github.com/FormingWorlds/VULCAN)**: Atmospheric chemistry module (Python)
- **[ZEPHYRUS](https://github.com/FormingWorlds/ZEPHYRUS)**: Atmospheric escape module (Python)
- **[Obliqua](https://github.com/FormingWorlds/Obliqua)**: Tidal evolution module (Julia)

**Important:** Each module is maintained in its own GitHub repository but is typically cloned/installed within the PROTEUS directory structure for integrated development. When working on any module in the ecosystem, apply these guidelines consistently.


**Project Type**: Scientific simulation framework

**Languages**: Python 3.12 (primary), Julia, Fortran, C

**Size**: ~98 Python files in `src/proteus/`, multiple submodules

**Target Runtime**: Python 3.12 (Linux/macOS only; Windows not supported)

## Build & Validation

For installation instructions and dependency management across the ecosystem:

- **Main installation guide:** `docs/installation.md` - Standard user and developer installation procedures
- **Local machine setup:** `docs/local_machine_guide.md` - Platform-specific setup (macOS, Linux, Windows)
- **Cluster setup:** `docs/kapteyn_cluster_guide.md` - HPC cluster configuration (see also `docs/habrok_cluster_guide.md`, `docs/snellius_cluster_guide.md`)

When helping with installation or dependency issues, always reference these guides first. The `proteus install-all` command handles most submodule installations automatically. However, whenever possible, prefer the developer installation steps outlined in the installation guide for editable installs.

### Environment Setup

**Prerequisites**:

1. Python 3.12 (via conda/miniforge or miniconda)
2. Julia (official installer: `curl -fsSL https://install.julialang.org | sh`)
3. Git (install via conda if needed: `conda install git`)
4. ~20 GB disk space

**Developer Install** (full editable installation - always use for development):

```bash
# 1. Set environment variables (REQUIRED - do this first)
mkdir /your/local/path/FWL_DATA
echo "export FWL_DATA=/your/local/path/FWL_DATA/" >> "$HOME/.bashrc"
source "$HOME/.bashrc"

# 2. Clone PROTEUS base
git clone git@github.com:FormingWorlds/PROTEUS.git
cd PROTEUS

# 3. Create conda environment
conda create -n proteus python=3.12
conda activate proteus

# 4. Install SOCRATES (Fortran radiative transfer)
./tools/get_socrates.sh
echo "export RAD_DIR=$PWD/socrates/" >> "$HOME/.bashrc"
source "$HOME/.bashrc"

# 5. Install AGNI (Julia radiative-convective atmosphere model)
git clone git@github.com:nichollsh/AGNI.git
cd AGNI
bash src/get_agni.sh 0  # Argument 0 skips tests
cd ../

# 6. Install Python submodules as editable (in order)
# MORS (stellar evolution)
git clone git@github.com:FormingWorlds/MORS
pip install -e MORS/.

# JANUS (1D convective atmosphere)
git clone git@github.com:FormingWorlds/JANUS
pip install -e JANUS/.

# CALLIOPE (volatile in-/outgassing)
git clone git@github.com:FormingWorlds/CALLIOPE
pip install -e CALLIOPE/.

# ARAGOG (interior thermal evolution)
git clone git@github.com:FormingWorlds/aragog.git
pip install -e aragog/.

# ZEPHYRUS (atmospheric escape)
git clone git@github.com:FormingWorlds/ZEPHYRUS
pip install -e ZEPHYRUS/.

# 7. Install PETSc (numerical computing library - specific version from OSF)
# NOTE: Must be done in Python <= 3.12 environment
./tools/get_petsc.sh
# Sets PETSC_DIR and PETSC_ARCH automatically (arch-linux-c-opt or arch-darwin-c-opt)

# 8. Install SPIDER (interior evolution - requires PETSc)
./tools/get_spider.sh

# 9. Install PROTEUS framework
pip install -e ".[develop]"

# 10. Enable pre-commit hooks
pre-commit install -f

# 11. Optional modules (if needed)
# LovePy (multi-phase tidal heating - Julia)
./tools/get_lovepy.sh

# VULCAN (chemical kinetics atmosphere model)
./tools/get_vulcan.sh
```

**User Install** (simpler, non-editable):

```bash
git clone https://github.com/FormingWorlds/PROTEUS.git
cd PROTEUS
conda env create -f environment.yml
conda activate proteus
pip install -e .
proteus install-all --export-env
```

**Important Notes**:

- **FWL_DATA** and **RAD_DIR** must be set before running PROTEUS
- **PETSc** is downloaded as a specific pre-compiled version from OSF (not built from source)
- **SPIDER** requires PETSc to be installed first
- All Python submodules should be installed as editable (`-e`) for development
- After installation, reload shell: `source ~/.bashrc` or `conda activate proteus`
- After each file change or edit, ruff format all changed files with `ruff check --fix ` and `ruff format --check`

### Build Commands

**No explicit build step** for Python code (installed via `pip install -e .`). Submodules require compilation:

- **SOCRATES** (Fortran): `cd socrates && ./build_code`
- **SPIDER** (C): `cd SPIDER && make`
- **AGNI** (Julia): `julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'`

**Always run** `pip install -e ".[develop]"` after code changes to update installation.

### Test Commands

**Run all tests**:

```bash
pytest
```

**Run by category** (matches CI):

```bash
pytest -m unit              # Fast unit tests (<100ms each, mocked physics)
pytest -m smoke             # Binary validation (1 timestep, low res)
pytest -m "unit or smoke"   # What PR checks run
pytest -m integration       # Multi-module coupling
pytest -m "not slow"        # Everything except slow tests
```

**With coverage**:

```bash
# Option 1: pytest-cov (convenient)
pytest --cov=src --cov-report=html

# Option 2: coverage run (matches CI exactly)
coverage run -m pytest
coverage report
coverage html
```

**Coverage thresholds** (in `pyproject.toml`):

- Fast gate: `[tool.proteus.coverage_fast] fail_under = 44.45`
- Full suite: `[tool.coverage.report] fail_under = 59`

**Validate test structure**:

```bash
bash tools/validate_test_structure.sh
```

**Coverage analysis**:

```bash
bash tools/coverage_analysis.sh
```

### Lint Commands

**Always run before committing**:

```bash
ruff check src/ tests/        # Check for issues
ruff check --fix src/ tests/ # Auto-fix issues
ruff format src/ tests/      # Format code
```

**Pre-commit hook** (runs automatically on commit):

```bash
pre-commit install -f
```

### Validation Pipeline

**CI runs on PRs** (`.github/workflows/ci-pr-checks.yml`):

1. **Unit tests**: `pytest -m "unit and not skip" --cov=src --cov-fail-under=44.45`
2. **Smoke tests**: `pytest -m "smoke and not skip"`
3. **Lint**: `ruff check src/ tests/` and `ruff format --check src/ tests/`
4. **Diff-cover**: 80% coverage on changed lines (enforced)
5. **Test structure**: `bash tools/validate_test_structure.sh`

**All must pass** before merge. Coverage thresholds auto-ratchet upward (never decrease).

## Project Layout

### Key Directories

- `src/proteus/` - Main Python source code
  - `cli.py` - Command-line interface entry point
  - `proteus.py` - Core `Proteus` class
  - `config/` - Configuration system (TOML parsing, validation)
  - `atmos_clim/`, `atmos_chem/`, `escape/`, `interior/`, `outgas/`, `observe/`, `orbit/`, `star/` - Physics module wrappers
  - `utils/` - Utilities (data, logging, plotting, helpers)
  - `grid/`, `inference/`, `plot/` - Specialized functionality

- `tests/` - Test suite (MUST mirror `src/proteus/` structure)
  - `tests/<module>/test_<filename>.py` for each `src/proteus/<module>/<filename>.py`
  - `tests/conftest.py` - Shared fixtures (parameter classes, config paths)
  - `tests/integration/` - Multi-module integration tests

- `input/` - TOML configuration files
- `output/` - Simulation results (gitignored)
- `tools/` - Build/utility scripts
- `docs/` - Documentation (MkDocs)

### Configuration Files

- `pyproject.toml` - Package metadata, pytest config, coverage thresholds, ruff rules
- `environment.yml` - Conda environment (user install)
- `mkdocs.yml` - Documentation configuration
- `.github/workflows/` - CI/CD pipelines
  - `ci-pr-checks.yml` - Fast PR validation (unit + smoke + lint)
  - `code-style.yaml` - Pre-commit hooks
  - `proteus_test_quality_gate.yml` - Reusable test workflow

### Entry Points

- **CLI**: `proteus start -c <config.toml> -o <output_dir>` (defined in `src/proteus/cli.py`)
- **Python API**: `from proteus import Proteus; p = Proteus(config_path, output_path)`

## Testing Standards

**Structure**: Tests MUST mirror source exactly. `src/proteus/config/_config.py` → `tests/config/test_config.py`

**Framework:** Use `pytest` exclusively in the `tests/` directory.

**Markers** (use consistently):

- `@pytest.mark.unit` - Fast Python logic tests (<100ms, mock heavy physics)
- `@pytest.mark.smoke` - Real binary validation (1 timestep, <30s)
- `@pytest.mark.integration` - Multi-module coupling
- `@pytest.mark.slow` - Full physics validation (hours)

**Rules**:

- **Never** use `==` for floats. Use `pytest.approx(val, rel=1e-5)` or `np.testing.assert_allclose`
- **Always** mock external calls (SOCRATES, AGNI, file I/O, network) in unit tests
- **Always** use physically valid inputs (T > 0K, P > 0) unless testing error handling
- **Always** read `tests/conftest.py` before writing tests to use existing fixtures
- **Always** add docstrings explaining the physical scenario being tested

- **Coverage Tool:** Two equivalent approaches are supported:
  - Local: `pytest --cov` (uses pytest-cov plugin, convenient)
  - CI/Local: `coverage run -m pytest` (matches CI exactly, compatible with ratcheting)
  - Choose based on preference; both work correctly.
- **Speed:** Unit tests must run in <100ms. Aggressively mock heavy simulations, I/O, and external APIs using `unittest.mock`.
- **Integration:** Mark slow tests (full simulation loops) with `@pytest.mark.slow`.
- **Markers:** Use pytest markers: `@pytest.mark.unit` for unit tests, `@pytest.mark.integration` for integration tests.
- **Floats:** NEVER use `==` for floats. Use `pytest.approx(val, rel=1e-5)` or `np.testing.assert_allclose`.
- **Physics:** Ensure inputs are physically valid (e.g., T > 0K) unless testing error handling.
- **Context:** Always read the `conftest.py` of the current module before generating tests to utilize existing fixtures.
- **Mocking Strategy:** Default to `unittest.mock` for ALL external calls (e.g., network, disk I/O, heavy computation modules like `SOCRATES` or `AGNI`). Only use real calls if explicitly requested for integration tests.
- **Floats:** Automatically generate assertions using `pytest.approx()` for any floating-point comparisons.
- **Parametrization:** Prefer `@pytest.mark.parametrize` over writing multiple similar test functions.
- **Physics Checks:** detailed comments explaining *why* a specific input range was chosen (e.g., "Temperature set to 300K to represent habitable zone conditions").
- **Instructions:** See `docs/test_building.md` for best practices on building robust tests.
- **Documentation:** Add detailed docstrings to each test explaining the physical scenario being tested. In the header of the test file, include a brief overview of what is being tested and any important context, including a link to all docuementation about testing standards: `docs/test_infrastructure.md`, `docs/test_categorization.md` and `docs/test_building.md`.
- **Formatting:** Ruff format all test files before committing.

### Coverage Requirements
- **Threshold:** Check `pyproject.toml` [tool.coverage.report] `fail_under` for current threshold.
- **Automatic Ratcheting:** Coverage threshold automatically increases on main branch via `tools/update_coverage_threshold.py` (never decreases).
- **Reports:** Run `pytest --cov --cov-report=html` and inspect `htmlcov/index.html` for gaps.
- **Analysis:** Use `bash tools/coverage_analysis.sh` to identify low-coverage modules needing tests.
- **Quality Gate:** All PRs must pass the coverage threshold defined in CI (see `.github/workflows/proteus_test_quality_gate.yml`).

## Safety & Determinism
- **Randomness:** Explicitly set seeds (e.g., `np.random.seed(42)`) in tests.
- **Files:** Do not generate tests that produce large output files (unless explicitly instructed); use `tempfile` or mocks.

## Code Quality

**Style** (enforced by ruff):

- Line length < 96 chars (config allows 96, but prefer < 92)
- Max indentation 3 levels
- Variables/functions: `snake_case`
- Constants: `UPPER_CASE`
- Type hints: Standard Python type hints
- Docstrings: Brief descriptions of physical scenarios

**Pre-commit**: Runs `ruff check` and `ruff format` automatically. Fix issues before committing.

## Common Workflows

### Making a Code Change

1. **Create branch**: `git checkout -b feature-name`
2. **Make changes** in `src/proteus/`
3. **Write/update tests** in `tests/` (mirror structure)
4. **Run tests locally**: `pytest -m unit` (fast feedback)
5. **Check coverage**: `pytest --cov=src --cov-report=html`
6. **Lint**: `ruff check --fix src/ tests/ && ruff format src/ tests/`
7. **Lint all new files**: `ruff check --fix` and `ruff format` on all newly changed files
7. **Validate structure**: `bash tools/validate_test_structure.sh`
8. **Commit**: `git commit -m "feat: description"`
9. **Push**: CI runs automatically on PR

### Adding a New Module

1. Create `src/proteus/<module>/<file>.py`
2. Create `tests/<module>/test_<file>.py` (mirror structure)
3. Add tests with appropriate markers
4. Run `bash tools/validate_test_structure.sh`
5. Ensure coverage meets threshold

### Debugging Test Failures

```bash
pytest -v --showlocals              # Verbose with local variables
pytest -x                           # Stop at first failure
pytest tests/module/test_file.py::test_function  # Run specific test
pytest --pdb                        # Drop into debugger on failure
```

## Key Dependencies

**Not obvious from layout**:

- **SOCRATES** (Fortran): Radiative transfer (compiled, requires `RAD_DIR`)
- **AGNI** (Julia): Atmospheric energy balance (Julia packages)
- **SPIDER** (C): Interior evolution (compiled, requires PETSc)
- **PETSc**: Numerical library (compiled)
- **Submodules**: CALLIOPE, JANUS, MORS, ARAGOG, ZEPHYRUS (Python packages, see above for installation instructions)

**Data**: Large input files stored on Zenodo/OSF, downloaded automatically on first run (unless `--offline`).

## Important Notes

- **Docker CI**: Uses pre-built image `ghcr.io/formingworlds/proteus:latest`. PR code is overlaid, only changed files recompiled.
- **Coverage ratcheting**: Thresholds auto-increase when coverage improves (committed by `github-actions[bot]`). Never manually decrease.
- **Test placeholders**: Some tests marked `@pytest.mark.skip` are placeholders. Excluded from CI.
- **Windows**: Not supported. Linux/macOS only.
- **Python version**: Must be 3.12 (PETSc/SPIDER require Python <= 3.12).

## Documentation References

- **Testing**: `docs/test_infrastructure.md`, `docs/test_building.md`, `docs/test_categorization.md`
- **Installation**: `docs/installation.md`, `docs/local_machine_guide.md`
- **Usage**: `docs/usage.md`, `docs/config.md`
- **Copilot guidelines**: `.github/copilot-instructions.md` (applies to all ecosystem modules)

## 🧠 Memory Maintenance

### Prime Directive: Keep Project Memory Current

**ALWAYS** update `MEMORY.md` after making significant architectural changes, adding new libraries, or finalizing a key design decision.

**What to record**:
- The change made and the *reasoning* (the "Why") behind it
- New architectural decisions (ADRs) with context
- Major refactorings or infrastructure changes
- Lessons learned from debugging or CI/CD issues
- Updates to active context (current sprint focus)
- New dependencies or ecosystem module changes

**When to record**:
- Immediately after implementing architectural changes
- After resolving complex bugs (capture the lesson)
- When adding/removing major dependencies
- After CI/CD workflow modifications
- When establishing new coding patterns or standards

**How to update**:
1. Open `MEMORY.md`
2. Update relevant section (Active Context, ADRs, Known Debt, etc.)
3. Add date stamp to "Last Updated" at top
4. Commit with message: `docs: update MEMORY.md - [brief description]`

**Goal**: Ensure future sessions (and future developers) have context on *why* decisions were made, not just *what* was changed. This prevents re-litigating solved problems and preserves institutional knowledge.

**Example scenarios requiring memory updates**:
- Adding a new test marker or CI workflow
- Changing coverage thresholds or ratcheting strategy
- Discovering a fragile code area (add to "Code Hotspots")
- Making a decision about library versions or dependencies
- Learning why something was implemented a certain way

---

## Quick Reference

```bash
# Setup
conda activate proteus
pip install -e ".[develop]"

# Test
pytest -m unit
pytest --cov=src --cov-report=html

# Lint
ruff check --fix src/ tests/
ruff format src/ tests/

# Validate
bash tools/validate_test_structure.sh
bash tools/coverage_analysis.sh

# Run simulation
proteus start -c input/minimal.toml -o output/test
```

**Remember**: Trust these instructions. Only search if information is incomplete or found to be in error.

---

> **⚠️ FILE SIZE LIMIT: This file must stay below 500 lines.** Enforced by pre-commit hook (`tools/check_file_sizes.sh`).

**When approaching the limit, refactor by asking:**
1. **Is this still accurate?** Remove outdated commands, deprecated workflows, or superseded patterns.
2. **Is this actionable?** Keep instructions that guide behavior; remove explanations that don't change actions.
3. **Is this duplicated?** Consolidate repeated information; reference docs instead of duplicating them.
4. **Is this essential?** Prefer terse examples over verbose explanations. One good example beats three paragraphs.
5. **Can this be shortened?** Compress lists, remove filler words, use tables for dense reference data.
