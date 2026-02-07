# Docker-Based CI/CD Architecture for PROTEUS

## What This Document Is For

**New to PROTEUS CI?** This document explains how our Docker-based testing infrastructure works. Docker containers provide a consistent environment with pre-compiled physics modules, making CI runs fast and reproducible.

**Key concept:** Instead of compiling SOCRATES, AGNI, PETSc, and SPIDER on every CI run (~60 min), we use a pre-built Docker image (~5 min startup).

For test markers and categories, see [Test Categorization](test_categorization.md). For coverage workflows, see [Test Infrastructure](test_infrastructure.md). For writing tests, see [Test Building](test_building.md).

---

## Overview

This architecture solves slow compilation times by using a pre-built Docker image containing the full PROTEUS environment with compiled physics modules. The image is built on demand and used by all CI/CD workflows.

## Architecture Components

### 1. Dockerfile

**Location:** `/Dockerfile`

**Purpose:** Define the pre-built environment with all dependencies and compiled physics modules.

**Key Features:**
- Base: Python 3.12 on Debian Bookworm (slim)
- System dependencies: gfortran, make, cmake, git, NetCDF libraries
- Julia installation via official installer
- Compiles all physics modules:
  - SOCRATES (radiative transfer)
  - PETSc (numerical computing)
  - SPIDER (interior evolution)
  - AGNI (radiative-convective atmosphere)
- Installs Python packages from `pyproject.toml`
- Optimized for size with cache cleanup

**Environment Variables:**
```bash
FWL_DATA=/opt/proteus/fwl_data
RAD_DIR=/opt/proteus/socrates
AGNI_DIR=/opt/proteus/AGNI
PETSC_DIR=/opt/proteus/petsc
PETSC_ARCH=arch-linux-c-opt
PROTEUS_DIR=/opt/proteus
```

### 2. docker-build.yml (The Updater)

**Location:** `.github/workflows/docker-build.yml`

**Purpose:** Build and push the Docker image to GitHub Container Registry.

**Triggers:**
- Schedule: Nightly at 02:00 UTC
- Push to `main` when dependencies change:
  - `pyproject.toml`
  - `environment.yml`
  - `Dockerfile`
  - `tools/get_*.sh` scripts

**Output:** `ghcr.io/formingworlds/proteus:latest`

**Tags:**
- `latest` (on main branch)
- `<branch>-<sha>` (commit-specific)
- `nightly-YYYYMMDD` (daily builds)

**Optimization:**
- BuildKit cache for faster rebuilds
- Layer caching from previous builds
- Multi-stage optimization potential

### 3. ci-pr-checks.yml (Fast Feedback)

**Location:** `.github/workflows/ci-pr-checks.yml`

**Purpose:** Fast PR validation using pre-built Docker image.

**Triggers:**
- Pull requests to `main` or `dev`
- Push to `main`, `dev`, or feature branches
- Manual dispatch

**Strategy:**
1. **Container:** Runs inside `ghcr.io/formingworlds/proteus:latest` (or branch-specific tag)
2. **Threshold check:** Prevents coverage decreases vs main
3. **Code Overlay:** Overlays PR code onto container (excludes compiled modules)
4. **Structure validation:** `tools/validate_test_structure.sh`
5. **Sequential testing:** Unit → Smart rebuild → Smoke
6. **Coverage coordination:** Downloads nightly artifact for estimated total

**Steps (in order):**

1. **Prevent threshold decreases** — Fails if `fail_under` decreased vs main
2. **Overlay PR code** — `rsync` excludes SPIDER, SOCRATES, PETSc, AGNI
3. **Validate test structure** — Ensures `tests/` mirrors `src/proteus/`
4. **Run unit tests** — `pytest -m "unit and not skip"` with coverage
5. **Smart rebuild** — Recompile SOCRATES/AGNI only if sources changed
6. **Run smoke tests** — `pytest -m "smoke and not skip"` (appends coverage)
7. **Download nightly coverage** — For estimated total calculation
8. **Check staleness** — Fails if nightly artifact >48h old
9. **Validate coverage** — Grace period of 0.3% for drops
10. **Diff-cover** — 80% coverage required on changed lines
11. **Lint** — `ruff check` and `ruff format --check`

**Coverage coordination:**
- Fast gate threshold from `[tool.proteus.coverage_fast] fail_under` (currently 44.45%)
- Estimated total = union of PR lines + nightly integration lines
- Grace period allows ≤0.3% drop with warning
- Diff-cover enforces 80% on changed lines

See [Test Categorization](test_categorization.md) for marker details and [Test Infrastructure](test_infrastructure.md) for coverage thresholds.

**Key Innovation - Smart Rebuild:**
```yaml
- name: Smart rebuild of physics modules
  run: |
    # Only rebuild if source files changed
    cd SPIDER
    make -q || make -j$(nproc)  # -q checks if build is up-to-date
```

Since the container already has compiled binaries:
- If PR changes only Python files: No recompilation needed (~instant)
- If PR changes Fortran/C files: Only changed files recompile (~seconds to minutes)
- Full compilation avoided (~30-60 minutes saved)

### 4. ci-nightly.yml (Deep Validation)

**Location:** `.github/workflows/ci-nightly.yml`

**Purpose:** Comprehensive scientific validation and coverage baseline.

**Triggers:**
- Schedule: Nightly at 03:00 UTC
- Manual dispatch
- Push to feature branches (workflow file changes only)

**Environment:**
- Sets `PROTEUS_CI_NIGHTLY=1` — enables additional smoke tests
- Timeout: 240 minutes (4 hours)
- Downloads ~200MB minimal data for smoke tests

**Strategy:**
1. Use branch-specific Docker image
2. Overlay code (excludes compiled modules)
3. Download minimal data (spectral files, stellar spectra, lookup tables)
4. Configure Julia environment for Python integration
5. Run all test tiers sequentially
6. Generate coverage artifacts for PR coordination
7. Ratchet coverage threshold on success

**Test sequence:**
1. **Unit tests** — `pytest -m "unit and not skip"` with coverage
2. **Smoke tests** — `pytest -m "smoke and not skip"` (coverage appended)
3. **Integration tests** — `pytest -m "integration and not slow"` (coverage appended)
4. **Slow tests** — `pytest -m slow` (if time permits)

**Artifacts uploaded:**
- `nightly-coverage/coverage-integration-only.json` — For PR estimated total
- `nightly-coverage/nightly-timestamp.txt` — For staleness detection
- `nightly-coverage/coverage-by-type.json` — Breakdown by test type

**Coverage ratcheting:**
- Full threshold from `[tool.coverage.report] fail_under` (currently 59%)
- Auto-commits threshold increase on successful main runs

See [Test Infrastructure](test_infrastructure.md) for coverage coordination details.

## Test Markers

Tests are categorized using pytest markers defined in `pyproject.toml`:

```python
# Unit test (fast, mocked physics)
@pytest.mark.unit
def test_config_parsing():
    # Test Python logic without heavy dependencies
    pass

# Smoke test (quick real binary check)
@pytest.mark.smoke
def test_spider_single_timestep():
    # Run SPIDER for 1 timestep at low resolution
    # Ensures binary actually works
    pass

# Integration test (multi-module)
@pytest.mark.integration
def test_atmosphere_interior_coupling():
    # Test interaction between JANUS and SPIDER
    pass

# Slow test (full scientific validation)
@pytest.mark.slow
def test_earth_evolution_1gyr():
    # Run full 1 Gyr simulation
    # Validate against known results
    pass
```

## Workflow Sequence

### Nightly (Main Branch)
```
02:00 UTC: docker-build.yml (scheduled)
  ↓
  Build and push Docker image
  ↓
03:00 UTC: ci-nightly.yml (scheduled)
  ↓
  Pull Docker image
  ↓
  Overlay code, download data (~200MB)
  ↓
  Run unit tests with coverage
  ↓
  Run smoke tests (PROTEUS_CI_NIGHTLY=1 enables extras)
  ↓
  Run integration tests
  ↓
  Run slow tests (if time permits)
  ↓
  Upload nightly-coverage artifact
  ↓
  Ratchet threshold if coverage increased
  ↓
  If either nightly workflow FAILED:
  ↓
ci-self-heal.yml (automatic)
  ↓
  Triage failure → Create Issue → AI agent fix → Validate → Open PR
```

### Pull Request
```
PR opened/updated
  ↓
ci-pr-checks.yml
  ↓
Pull Docker image (instant)
  ↓
Check threshold not decreased vs main
  ↓
Overlay PR code onto container
  ↓
Validate test structure
  ↓
Run unit tests with coverage (~2-5 min)
  ↓
Smart rebuild (only if Fortran/Julia changed)
  ↓
Run smoke tests (~5-10 min)
  ↓
Download nightly artifact, check staleness
  ↓
Compute estimated total coverage
  ↓
Diff-cover (80% on changed lines)
  ↓
Lint with ruff
  ↓
Fast feedback (~10-15 min total)
```

## Benefits

### Speed Improvements
- **Before:** Every PR compiles SOCRATES, PETSc, SPIDER, AGNI (~60 minutes)
- **After:** Use pre-built image, smart rebuild only (~5-10 minutes for Python-only changes)
- **Savings:** ~50 minutes per PR iteration

### Resource Efficiency
- Docker layer caching reduces rebuild time
- Smart recompilation only builds changed files
- Parallel job execution where possible

### Scientific Rigor
- Nightly comprehensive validation ensures correctness
- PR checks provide fast feedback without compromising quality
- Separation of fast unit tests from slow integration tests

### Developer Experience
- Fast PR checks (~10-15 min) enable rapid iteration
- Clear test markers guide test writing
- Comprehensive nightly validation catches regressions

## Image Maintenance

### When Docker Image Rebuilds
1. Nightly at 02:00 UTC (scheduled)
2. Changes to `pyproject.toml` (dependency updates)
3. Changes to `environment.yml` (conda dependencies)
4. Changes to `Dockerfile` (build process)
5. Changes to `tools/get_*.sh` (compilation scripts)

### Image Size Management
- Cleanup layers remove apt cache, Python cache
- Multi-stage builds potential for further optimization
- Current estimated size: ~2-3 GB (with compiled modules)

### Cache Strategy
- BuildKit cache stored in registry
- Layer caching from previous builds
- Fast incremental builds

## Coverage Coordination

The two-tier coverage system coordinates between nightly and PR workflows:

| Feature | Value | Description |
|---------|-------|-------------|
| Fast gate | 44.45% | PR threshold (unit + smoke) |
| Full gate | 59% | Nightly threshold (all tests) |
| Grace period | 0.3% | PRs can merge with small drops |
| Staleness | 48h | PR fails if nightly too old |
| Diff-cover | 80% | Required on changed lines |

**How estimated total works:**
1. PR runs unit + smoke → `coverage-unit.json`
2. Download nightly's `coverage-integration-only.json`
3. Compute union of covered lines
4. Compare against full threshold

See [Test Infrastructure](test_infrastructure.md) for threshold details.

## Troubleshooting

### Image Build Fails
- Check GitHub Actions logs in `docker-build.yml`
- Verify compilation scripts work locally
- Test Dockerfile locally: `docker build -t proteus-test .`

### Smart Rebuild Not Working
- Verify make is installed in container
- Check if Makefiles are copied correctly
- Manual rebuild: Remove binaries and rebuild

### Tests Fail in Container
- Test locally with: `docker run -it ghcr.io/formingworlds/proteus:latest bash`
- Verify environment variables are set
- Check file permissions

### Image Too Large
- Review cleanup steps in Dockerfile
- Consider multi-stage builds
- Analyze layers: `docker history ghcr.io/formingworlds/proteus:latest`

## Future Enhancements

1. **Multi-architecture Support:** Build for ARM64 (Apple Silicon)
2. **Version Tagging:** Semantic versioning for stable releases
3. **Matrix Testing:** Multiple Python versions (3.11, 3.12, 3.13)
4. **Performance Profiling:** Benchmark tests across versions
5. **Artifact Caching:** Cache FWL_DATA between runs
6. ~~**Self-Healing CI:**~~ ✅ Implemented — see [Self-Healing CI](self_healing_ci.md)

## References

### PROTEUS Documentation
- [Test Infrastructure](test_infrastructure.md) — Coverage workflows, thresholds, troubleshooting
- [Test Categorization](test_categorization.md) — Test markers, CI pipelines, fixtures
- [Test Building](test_building.md) — Writing tests, prompts, best practices
- [Self-Healing CI](self_healing_ci.md) — AI-powered automatic failure diagnosis and repair
- [AI-Assisted Development](ai_usage.md) — Using AI for tests and code review

### External Resources
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [GitHub Actions: Container Jobs](https://docs.github.com/en/actions/using-jobs/running-jobs-in-a-container)
- [pytest Markers](https://docs.pytest.org/en/stable/example/markers.html)
- [coverage.py Documentation](https://coverage.readthedocs.io/)
