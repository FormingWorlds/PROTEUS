# 🧠 Project Memory

**Last Updated**: 2026-03-14

This document captures the living context of PROTEUS—the "why" behind architectural decisions, the current development focus, and critical knowledge for maintaining consistency across sessions.

---

## 1. Project Identity & Stack

### Core Identity
- **Name**: PROTEUS (/ˈproʊtiəs, PROH-tee-əs)
- **Type**: Coupled atmosphere-interior framework for rocky planet evolution
- **Philosophy**: Modular, adaptable scientific simulation inspired by the Greek god of elusive sea change
- **Version**: 25.11.19 (CalVer: YY.MM.DD)
- **License**: Apache 2.0

### Primary Technology Stack
- **Languages**: Python 3.12 (primary), Julia, Fortran, C
- **Python Framework**: setuptools-based package (`fwl-proteus`)
- **Testing**: pytest with markers (unit, smoke, integration, slow)
- **Coverage**: coverage.py with automatic ratcheting (current: 59% full, 44.45% fast)
- **Linting**: ruff (line-length 96, quote-style single)
- **CI/CD**: GitHub Actions with Docker-based workflows
- **Documentation**: Zensical (wraps MkDocs Material; serve with `zensical serve`, NOT `mkdocs serve`)

### Key External Dependencies
- **SOCRATES** (Fortran): Spectral radiative transfer code
- **AGNI** (Julia): Radiative-convective atmospheric energy module
- **SPIDER** (C): Interior thermal evolution (T-S formalism, requires PETSc)
- **PETSc**: Numerical computing library (specific OSF version, not built from source)

### Python Ecosystem Modules (Editable Installs)
- **CALLIOPE**: Volatile in-/outgassing and thermodynamics
- **JANUS**: 1D convective atmosphere module
- **MORS**: Stellar evolution module
- **ARAGOG**: Interior thermal evolution (T-P formalism)
- **ZEPHYRUS**: Atmospheric escape module
- **BOREAS**: Hydrodynamic atmospheric escape module
- **ZALMOXIS**: Interior structure solver (hydrostatic equilibrium, EOS)

### Environment Requirements
- **Python**: 3.12 (strict requirement for PETSc/SPIDER compatibility)
- **Platforms**: Linux/macOS only (Windows not supported)
- **Disk Space**: ~20 GB
- **Critical Env Vars**: `FWL_DATA`, `RAD_DIR`, `PETSC_DIR`, `PETSC_ARCH`

---

## 2. Active Context (The "Now")

### Current Focus (as of 2026-03-14)

**Recently Merged (since 2026-02-13)**:
1. **PR #648: Zalmoxis-SPIDER coupling** (merged 2026-03-10) - External mesh from Zalmoxis structure solver passed to SPIDER thermal evolution. Physics-based structure update triggers. ~1,900 lines coupling code, ~3,300 lines tests.
2. **PR #596: VULCAN online chemistry** (merged) - In-loop chemical kinetics via `src/proteus/atmos_chem/`. VULCAN now runs at every snapshot, not just post-processing.
3. **PR #642: Download bug fixes** (merged 2026-03-08) - New `download_zenodo_file()` for single-file downloads, PHOENIX spectrum unzipping, custom stellar spectrum path expansion.
4. **PR #643: PETSc/SPIDER macOS 26+ fix** (merged 2026-02-23) - CFLAGS quoting fix for RHEL 9 / Rocky Linux.
5. **PR #630: CI improvements** (merged 2026-02-16) - macOS unit tests in PR checks, nightly deduplication, Codecov aggregate coverage.

**Open PRs**:
- **PR #654**: Zensical documentation style update (branch: `ks/docs`) - Diataxis restructuring, new landing page, custom CSS
- **PR #634**: Albedo feature (branch: `lj/albedo`)

### Recent Architectural Changes
- **Zalmoxis-SPIDER coupling**: External mesh mode lets Zalmoxis compute density structure, SPIDER evolves thermal state on that grid. Physics-based update triggers (dT/T, dPhi, floor/ceiling). Config: `struct.module = "zalmoxis"`.
- **VULCAN online chemistry**: `atmos_chem/` module runs VULCAN in-loop at every snapshot. Produces `vulcan_<time>.csv` per atmosphere snapshot.
- **BOREAS escape module**: Alternative hydrodynamic escape model (`escape.module = "boreas"`), added in PR #589.
- **Documentation**: Migrated to Zensical (wraps MkDocs Material). Diataxis structure: `docs/How-to/`, `docs/Explanations/`, `docs/Reference/`, `docs/Community/`. Build with `zensical serve`, NOT `mkdocs serve`.
- **Docker CI Architecture**: Pre-built images (`ghcr.io/formingworlds/proteus:latest`), cross-platform (Linux + macOS), Codecov integration
- **Dual-Tool Agent Instructions**: `CLAUDE.md` is a symlink to `.github/copilot-instructions.md`. Pre-commit 500-line limit on `copilot-instructions.md` constrains both.
- **File Size Limits**: Pre-commit enforced limits on .github/copilot-instructions.md (500) and .github/copilot-memory.md (1000)

---

## 3. Architectural Decisions (ADRs)

### ADR-001: Docker-Based CI/CD (2026-01)
**Decision**: Use pre-built Docker images for all CI/CD workflows instead of compiling on every run.

**Reasoning**:
- Compilation of SOCRATES, PETSc, SPIDER, AGNI takes ~60 minutes per PR
- Pre-built image reduces PR feedback time from 60+ min to 10-15 min
- Smart rebuild only recompiles changed files (make handles this)
- Nightly builds at 02:00 UTC ensure image stays current

**Implementation**: `Dockerfile`, `.github/workflows/docker-build.yml`, image at `ghcr.io/formingworlds/proteus:latest`

**Trade-offs**: Adds complexity (Docker maintenance) but massive developer experience improvement

---

### ADR-002: Four-Tier Test Categorization (2026-01)
**Decision**: Categorize tests into unit, smoke, integration, and slow with pytest markers.

**Reasoning**:
- **Unit** (<100ms, mocked): Fast feedback on Python logic
- **Smoke** (1 timestep, real binaries): Binary validation without full physics
- **Integration**: Multi-module coupling tests
- **Slow** (hours): Full scientific validation

**Why This Matters**:
- PR checks run unit + smoke (~10 min) for fast feedback
- Nightly runs full suite including slow tests
- Prevents regression while maintaining developer velocity

**Implementation**: `@pytest.mark.{unit,smoke,integration,slow}` in `pyproject.toml`

---

### ADR-003: Automatic Coverage Ratcheting (2026-01)
**Decision**: Coverage thresholds automatically increase, never decrease.

**Reasoning**:
- Prevents coverage regression
- Encourages incremental improvement
- Two thresholds: fast gate (unit+smoke) and full gate (all tests)
- Script `tools/update_coverage_threshold.py` runs on main branch pushes

**Current Thresholds**:
- Fast gate: 44.45% (`[tool.proteus.coverage_fast]`)
- Full gate: 59% (`[tool.coverage.report]`)

**Why This Matters**: Ensures test quality never degrades, even as codebase grows

---

### ADR-004: Editable Installs for All Ecosystem Modules (Ongoing)
**Decision**: All Python submodules installed with `pip install -e .` for development.

**Reasoning**:
- Enables live code changes without reinstallation
- Simplifies debugging across module boundaries
- Required for integrated ecosystem development
- Installation order matters (dependencies: MORS → JANUS → CALLIOPE → ARAGOG → ZEPHYRUS)

**Trade-off**: More complex setup, but essential for multi-repo development

---

### ADR-005: Test Structure Mirrors Source Structure (Established)
**Decision**: `tests/<module>/test_<filename>.py` must mirror `src/proteus/<module>/<filename>.py`

**Reasoning**:
- Enforces 1:1 mapping between source and tests
- Validated by `tools/validate_test_structure.sh`
- Makes it obvious where tests belong
- Prevents orphaned tests

**Enforcement**: CI runs validation script; PRs fail if structure violated

---

### ADR-006: No Float Equality Comparisons (Established)
**Decision**: Never use `==` for float comparisons; always use `pytest.approx()` or `np.testing.assert_allclose()`.

**Reasoning**:
- Floating-point arithmetic is inherently imprecise
- Physics simulations accumulate numerical errors
- Prevents flaky tests from rounding differences

**Enforcement**: Documented in `.github/copilot-instructions.md`, enforced in code review

---

### ADR-007: PETSc from OSF, Not Built from Source (Established)
**Decision**: Download pre-compiled PETSc from OSF instead of building from source.

**Reasoning**:
- Building PETSc from source takes 30+ minutes
- Specific version required for SPIDER compatibility
- Pre-compiled version is platform-specific (arch-linux-c-opt, arch-darwin-c-opt)
- Reduces installation complexity

**Implementation**: `tools/get_petsc.sh` downloads from OSF

---

## 4. Known Debt & "Watch Outs"

### Documentation Drift
- **Issue**: Some test documentation references old workflow names
- **Impact**: Low (workflows themselves are correct)
- **Action**: Ongoing cleanup as workflows stabilize

### Code Hotspots (Fragile/Complex Areas)

#### 1. AGNI Integration (`src/proteus/atmos_clim/agni.py`)
- **Why Fragile**: Julia-Python bridge via juliacall
- **Recent Changes**: Atmosphere allocation fixes (commit 980b441b)
- **Watch Out**: Memory management across language boundary
- **Test Coverage**: Integration tests in `tests/integration/test_integration_aragog_agni.py`

#### 2. Data Download System (`src/proteus/utils/data.py`)
- **Why Complex**: Handles Zenodo, OSF, network failures, offline mode
- **Recent Changes**: Security improvements, validation enhancements (commits 9986961d, 1eec4bad)
- **Watch Out**: Network-dependent tests require careful mocking
- **Test Coverage**: `tests/utils/test_data.py` with unit tests

#### 3. Configuration System (`src/proteus/config/`)
- **Why Complex**: TOML parsing, validation, type conversion, defaults
- **Recent Changes**: Enhanced unit tests (commit 6d4394a6)
- **Watch Out**: Nested configuration validation, type coercion edge cases
- **Test Coverage**: `tests/config/test_config.py`, `test_converters.py`, `test_options.py`

#### 4. Zalmoxis-SPIDER Coupling (`src/proteus/interior/wrapper.py`, `zalmoxis.py`)
- **Why Complex**: Phase 2 feedback loop mutates config temporarily (prescribed T-mode override with try/finally restore). T-profile interpolation bridges SPIDER's mantle nodes to Zalmoxis's full-planet grid. WolfBower2018 T-dependent EOS causes ~5% M_int shift on first structure update.
- **Recent Changes**: Hybrid physics-based structure update triggers (dT/T + dPhi + floor/ceiling), removed deprecated `weight_iron_frac`, per-layer EOS config, `update_structure_from_interior()` now returns `(time, Tmagma, Phi)` tuple.
- **Defaults Changed**: SPIDER is default interior module, Zalmoxis is default structure module, `temperature_mode` default is `adiabatic`, `num_levels` default is 100, `mantle_eos` uses colon format (`WolfBower2018:MgSiO3`).
- **Watch Out**: `zalmoxis_solver()` unconditionally writes Aragog files even when using SPIDER. `solve_structure()` permanently mutates `config.orbit.module='dummy'` for Zalmoxis. `validate_mesh_fields.py` in SPIDER is dead validation (needs rewrite).
- **Test Coverage**: `tests/interior/test_zalmoxis.py`, `tests/integration/test_smoke_zalmoxis_spider.py`, `test_regression_aw_zalmoxis.py`, `test_regression_structure_update.py`

#### 5. CI Workflow Summary Generation (`.github/workflows/ci-nightly.yml`)
- **Why Fragile**: Parses JUnit XML, coverage JSON, handles failures
- **Recent Changes**: Hardened with try/except, better error handling (commit 7ed06597)
- **Watch Out**: Missing files, parse errors can crash summary step
- **Mitigation**: Wrapped in exception handling, writes minimal summary on error

### Unfinished Business (TODOs in Codebase)

Found 4 TODOs across codebase:
1. `src/proteus/grid/manage.py` - Grid management optimization
2. `src/proteus/interior/aragog.py` - Interior evolution edge cases
3. `src/proteus/observe/wrapper.py` - Observation module enhancements
4. `tests/utils/test_utils.py` - Additional utility test cases

**Priority**: Low (not blocking current work)

### Test Skips (Placeholders)
- **JANUS/SOCRATES**: Some tests skipped due to instability
- **AGNI**: Some tests require Julia binaries (nightly only)
- **CALLIOPE**: Slow tests run in nightly only
- **Data tests**: Network-dependent tests mocked in unit, real in integration

**Strategy**: Gradually remove skips as stability improves; use `PROTEUS_CI_NIGHTLY=1` gating for expensive tests

### Coverage Gaps (As of 2026-02-01)
- **Current**: 59% overall, 44.45% fast suite
- **Target**: Incremental improvement via ratcheting
- **Focus Areas**: Use `bash tools/coverage_analysis.sh` to identify low-coverage modules

### TODO: Coverage Estimation Math Issue
**Status**: Identified, needs investigation

**Problem**: The file `coverage-integration-only.json` is misnamed - it actually contains COMBINED coverage (unit + smoke + integration) due to `--cov-append`. When the PR workflow estimates total coverage by combining fast (unit+smoke) results with this nightly artifact, stale lines from nightly could mask coverage regressions on PRs.

**Affected files**:
- `.github/workflows/ci-nightly.yml` lines 402-411 (creates the file)
- `.github/workflows/ci-pr-checks.yml` lines 312-321 (uses the file)

**Potential fix options**:
1. Rename file to `coverage-unit-smoke-integration.json` (cosmetic only)
2. Create truly separate coverage files per test type (requires `coverage erase` before integration)
3. Validate current math works correctly before any changes

**Priority**: Low - needs investigation to confirm if this is actually causing issues

---

## 5. Critical Workflows & Commands

### Developer Daily Commands
```bash
# Activate environment
conda activate proteus

# Run fast tests (what PR checks run)
pytest -m "unit and not skip"
pytest -m "smoke and not skip"

# Check coverage
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Lint (always before commit)
ruff check --fix src/ tests/
ruff format src/ tests/

# Validate test structure
bash tools/validate_test_structure.sh

# Coverage analysis
bash tools/coverage_analysis.sh
```

### CI/CD Pipeline Flow
1. **PR Opened**: `ci-pr-checks.yml` runs (Linux: unit + smoke + coverage; macOS: unit; lint; summary, ~10-15 min)
2. **PR Merged to main**: Coverage ratcheting updates thresholds
3. **Nightly 02:00 UTC**: `docker-build.yml` rebuilds image → dispatches `ci-nightly.yml`
4. **Nightly (post-build)**: `ci-nightly.yml` runs full suite (unit → smoke → integration → slow), uploads aggregate coverage to Codecov, ratchets thresholds
5. **Nightly 03:00 UTC**: `ci-nightly.yml` cron fallback (skips if already dispatched by docker-build)

### Installation Sequence (Developer)
**Critical**: Follow exact order due to dependencies
1. Set `FWL_DATA` and `RAD_DIR` environment variables
2. Install SOCRATES (Fortran)
3. Install AGNI (Julia)
4. Install Python submodules in order: MORS → JANUS → CALLIOPE → ARAGOG → ZEPHYRUS
5. Install PETSc (from OSF)
6. Install SPIDER (requires PETSc)
7. Install PROTEUS framework: `pip install -e ".[develop]"`
8. Enable pre-commit: `pre-commit install -f`

---

## 6. Ecosystem Context

### Multi-Repository Structure
PROTEUS is the orchestrator; each module is a separate GitHub repository:
- FormingWorlds/PROTEUS (main)
- FormingWorlds/CALLIOPE
- FormingWorlds/JANUS
- FormingWorlds/MORS
- FormingWorlds/aragog
- FormingWorlds/ZEPHYRUS
- FormingWorlds/BOREAS
- FormingWorlds/VULCAN
- FormingWorlds/Zalmoxis
- nichollsh/AGNI
- nichollsh/SOCRATES
- FormingWorlds/SPIDER

**Implication**: Changes may require coordinated updates across repositories

### Fork Policy
All ecosystem repos live under the **FormingWorlds** GitHub organisation (or contributor forks like `nichollsh/`). When creating PRs, **always target the FormingWorlds fork** (or the contributor fork we cloned from). Never open PRs against upstream/original repositories (e.g. `djbower/spider`). Check `git remote -v` to confirm `origin` points to the correct fork before pushing or creating PRs.

### Testing Standards Apply Ecosystem-Wide
All modules follow same standards:
- Test structure mirrors source
- Coverage ratcheting
- pytest markers (unit, smoke, integration, slow)
- ruff formatting
- Same CI/CD patterns

---

## 7. Recent Lessons Learned

### Lesson 1: Nightly Workflow Timeouts (2026-01-28)
**Problem**: Nightly workflow cancelled at 55 minutes, slow tests never ran.

**Root Cause**: Unit + smoke + integration exceeded 55-minute timeout.

**Solution**: Increased timeout to 90 minutes (commit 7ed06597).

**Takeaway**: Always budget time for full test suite; slow tests need 10-15 min alone.

---

### Lesson 2: Summary Script Robustness (2026-01-28)
**Problem**: Summary generation step failed when tests failed, hiding actual test failures.

**Root Cause**: Script exited before writing summary if files missing or parse errors.

**Solution**: Wrapped all summary generation in try/except; write minimal summary on error.

**Takeaway**: Error handling in CI scripts must be bulletproof; always write *something* to help debugging.

---

### Lesson 3: AGNI Memory Management (2026-01)
**Problem**: AGNI integration tests occasionally failed with memory errors.

**Root Cause**: Atmosphere allocation not properly managed across Julia-Python boundary.

**Solution**: Enhanced allocation logic in `src/proteus/atmos_clim/agni.py` (commit 980b441b).

**Takeaway**: Cross-language boundaries require explicit resource management.

---

### Lesson 4: Docker/CI Infrastructure (2026-01, consolidated) ✅ RESOLVED
Key takeaways from Julia, CI, and test infrastructure debugging:
- **Julia in Docker**: Use official tarballs, not `juliaup`. Set PATH via ENV, not symlinks (symlinks break library resolution).
- **Data downloads**: Sanitize Zenodo IDs with regex (`^[0-9]+$`) before passing to subprocess.
- **CI utilities**: Don't assume system tools like `bc` exist in containers; use Python instead.
- **Smoke test data**: Trace ALL data dependencies through config files. Use `download_sufficient_data()` as reference.
- **CI timeouts**: Nightly workflow uses 240-minute timeout. Slow tests have per-test `@pytest.mark.timeout()` marks (30/60 min). Physics simulation runtime is unpredictable from config alone.

---

## 8. Future Roadmap (Known Priorities)

### Open Items
- 📋 **Plot test reference images**: 12 plot tests are xfail due to outdated references
- ⚠️ **LovePy investigation**: multi_timestep test skipping due to LovePy exception
- Centralise physical bounds (T_surf, P_surf limits) into conftest.py constants
- Expand integration test coverage (ARAGOG+AGNI, CALLIOPE+ZEPHYRUS)

### Medium-Term
- Multi-architecture Docker support (ARM64 for Apple Silicon)
- Matrix testing across Python 3.11, 3.12, 3.13
- Performance profiling and benchmarking

### Long-Term
- Semantic versioning for stable releases
- Artifact caching for FWL_DATA between CI runs

---

## 9. Key People & Roles

### Core Maintainers
- **Tim Lichtenberg** (tim.lichtenberg@rug.nl): Project lead
- **Harrison Nicholls** (harrison.nicholls@physics.ox.ac.uk): AGNI, SOCRATES

### Contact Points
- **Discussions**: https://github.com/orgs/FormingWorlds/discussions
- **Issues**: https://github.com/FormingWorlds/PROTEUS/issues
- **Documentation**: https://proteus-framework.org/proteus/

---

## 10. References & Resources

### Essential Documentation
- **Installation**: `docs/How-to/installation.md`
- **Testing Infrastructure**: `docs/How-to/test_infrastructure.md`
- **Test Categorization**: `docs/How-to/test_categorization.md`
- **Test Building**: `docs/How-to/test_building.md`
- **Docker CI Architecture**: `docs/Explanations/docker_ci_architecture.md`
- **Docs Development**: `docs/How-to/documentation.md` (build with `zensical serve`)
- **Agent Guidelines**: `.github/copilot-instructions.md`

### External Resources
- **Paper**: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024JE008576
- **Website**: https://proteus-framework.org
- **GitHub**: https://github.com/FormingWorlds/PROTEUS

---

**Note**: This document should be updated whenever significant architectural decisions are made, major features are added, or critical lessons are learned. See `.github/copilot-instructions.md` for the Memory Maintenance Prime Directive.

> **⚠️ FILE SIZE LIMIT: This file must stay below 1000 lines.** Enforced by pre-commit hook (`tools/check_file_sizes.sh`). File located at `.github/copilot-memory.md`.

**When approaching the limit, refactor by asking:**
1. **Is this still relevant?** Archive completed decisions, resolved issues, or obsolete context to a separate `docs/archive/` file if historically valuable, otherwise delete.
2. **Is this decision or context?** Keep the *why* behind decisions; remove transient status updates that no longer matter.
3. **Is this duplicated elsewhere?** Reference `.github/copilot-instructions.md`, docs, or code comments instead of duplicating.
4. **Can sections be condensed?** Merge related items, use bullet points over prose, compress verbose explanations.
5. **What would a new contributor need?** Prioritize information that prevents mistakes over historical trivia.
