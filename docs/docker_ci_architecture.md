# Docker-Based CI/CD Architecture for PROTEUS

## Overview

This architecture solves slow compilation times by using a pre-built Docker image containing the full PROTEUS environment with compiled physics modules. The image is built nightly and used by all CI/CD workflows.

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

### 3. ci-pr-checks.yml (The Consumer - Fast Feedback)

**Location:** `.github/workflows/ci-pr-checks.yml`

**Purpose:** Fast PR validation using pre-built Docker image.

**Triggers:**
- Pull requests to `main` or `dev`
- Push to `main` or `dev`

**Strategy:**
1. **Container:** Runs inside `ghcr.io/formingworlds/proteus:latest`
2. **Code Overlay:** Checks out PR code and overlays it onto the container
3. **Smart Rebuild:** Only recompiles changed files (make handles this automatically)
4. **Two Job Pipeline:**
   - **Unit Tests:** Fast tests with mocked physics modules
   - **Smoke Tests:** Quick validation with real binaries (1 timestep, low res)

**Jobs:**

#### Job 1: Unit Tests
- Runs: `pytest -m unit`
- Coverage: Reports to Codecov
- Duration: ~2-5 minutes
- Purpose: Validate Python logic without heavy physics

#### Job 2: Smoke Tests
- Runs: `pytest -m smoke`
- Coverage: Not required
- Duration: ~5-10 minutes
- Purpose: Ensure binaries work with new Python code

#### Job 3: Lint
- Runs: `ruff check` and `ruff format --check`
- Purpose: Code quality enforcement

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

### 4. ci-nightly-science.yml (Deep Validation)

**Location:** `.github/workflows/ci-nightly-science.yml`

**Purpose:** Comprehensive scientific validation on main branch.

**Triggers:**
- Schedule: Nightly at 03:00 UTC (1 hour after Docker build)
- Manual dispatch

**Strategy:**
1. Use latest Docker image
2. Run full scientific test suite
3. Generate comprehensive coverage reports
4. Archive simulation outputs

**Jobs:**

#### Job 1: Science Validation
- Runs: `pytest -m slow`
- Duration: Up to 4 hours
- Purpose: Full physics simulations for correctness
- Coverage: Comprehensive validation

#### Job 2: Integration Tests
- Runs: `pytest -m integration`
- Duration: Up to 2 hours
- Purpose: Multi-module interaction testing
- Coverage: Module coupling validation

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
02:00 UTC: docker-build.yml
  ↓
  Build new Docker image with latest main
  ↓
  Push to ghcr.io/formingworlds/proteus:latest
  ↓
03:00 UTC: ci-nightly-science.yml
  ↓
  Pull latest image
  ↓
  Run @pytest.mark.slow (4 hours)
  ↓
  Run @pytest.mark.integration (2 hours)
  ↓
  Upload comprehensive coverage and outputs
```

### Pull Request
```
PR opened/updated
  ↓
ci-pr-checks.yml
  ↓
Pull ghcr.io/formingworlds/proteus:latest (instant)
  ↓
Overlay PR code onto container
  ↓
Smart rebuild (only changed files)
  ↓
Job 1: Unit tests (2-5 min)
Job 2: Smoke tests (5-10 min)
Job 3: Lint (1-2 min)
  ↓
Fast feedback to developer (~10-15 min total)
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

## Migration Strategy

### Phase 1: Parallel Testing
- Keep existing `ci_tests.yml` alongside new workflows
- Run both systems in parallel
- Compare results and performance

### Phase 2: Gradual Transition
- Route PRs to new system
- Keep nightly on old system initially
- Verify coverage equivalence

### Phase 3: Full Migration
- Deprecate `ci_tests.yml`
- All CI/CD uses Docker-based system
- Update documentation

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

## References

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [GitHub Actions: Container Jobs](https://docs.github.com/en/actions/using-jobs/running-jobs-in-a-container)
- [pytest Markers](https://docs.pytest.org/en/stable/example/markers.html)
- PROTEUS Documentation: `docs/test_infrastructure.md`
