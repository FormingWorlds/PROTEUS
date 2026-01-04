# Docker-Based CI/CD Implementation - Quick Start

This branch (`tl/test_ecosystem_v4`) implements a Docker-based CI/CD architecture for PROTEUS to solve slow compilation times.

## What Was Created

### 1. Core Files

- **`Dockerfile`** - Pre-built environment with compiled physics modules
- **`.github/workflows/docker-build.yml`** - Nightly Docker image builder
- **`.github/workflows/ci-pr-checks.yml`** - Fast PR validation
- **`.github/workflows/ci-nightly-science.yml`** - Deep scientific validation

### 2. Documentation

- **`docs/docker_ci_architecture.md`** - Complete architecture documentation
- **`tests/examples/test_marker_usage.py`** - Example tests with markers

### 3. Configuration Updates

- **`pyproject.toml`** - Added `smoke` pytest marker

## Quick Start Guide

### For PR Authors

When you open a PR, the new CI will:

1. ✅ Pull pre-built Docker image (instant)
2. ✅ Overlay your code changes (seconds)
3. ✅ Smart rebuild (only changed files, seconds to minutes)
4. ✅ Run unit tests (2-5 minutes)
5. ✅ Run smoke tests (5-10 minutes)
6. ✅ Report back (~10-15 minutes total)

**Before:** ~60 minutes of compilation per PR
**After:** ~10-15 minutes for Python changes

### For Test Writers

Use pytest markers to categorize your tests:

```python
@pytest.mark.unit
def test_fast_logic():
    """Runs in PR checks. Mock heavy physics."""
    pass

@pytest.mark.smoke
def test_binary_works():
    """Runs in PR checks. 1 timestep, low res."""
    pass

@pytest.mark.integration
def test_module_coupling():
    """Runs nightly. Multi-module tests."""
    pass

@pytest.mark.slow
def test_full_physics():
    """Runs nightly. Hours-long validation."""
    pass
```

### Running Locally

```bash
# Install development dependencies
pip install -e ".[develop]"

# Run unit tests (fast)
pytest -m unit

# Run unit + smoke (what PR checks run)
pytest -m "unit or smoke"

# Run everything except slow
pytest -m "not slow"

# Full test suite
pytest
```

## Implementation Steps

### Phase 1: Initial Setup ✅ (This Branch)

- [x] Create Dockerfile
- [x] Create docker-build.yml workflow
- [x] Create ci-pr-checks.yml workflow
- [x] Create ci-nightly-science.yml workflow
- [x] Add `smoke` pytest marker
- [x] Document architecture
- [x] Create example tests

### Phase 2: Testing (Next Steps)

1. **Test Docker Image Build**
   ```bash
   # Build locally to verify Dockerfile works
   docker build -t proteus-test .

   # Test the image
   docker run -it proteus-test bash
   # Inside container:
   pytest -m unit
   ```

2. **Push Branch and Monitor**
   ```bash
   git push origin tl/test_ecosystem_v4
   ```

   - Watch GitHub Actions for docker-build.yml
   - Verify image pushes to ghcr.io
   - Check if it's publicly accessible

3. **Test PR Workflow**
   - Create test PR from this branch
   - Verify ci-pr-checks.yml runs
   - Check timing improvements
   - Validate test results

### Phase 3: Migration

1. **Add Test Markers**
   - Mark existing tests with `@pytest.mark.unit`, `@pytest.mark.smoke`, etc.
   - Start with critical modules

2. **Parallel Run**
   - Keep existing `ci_tests.yml` active
   - Run both systems in parallel
   - Compare results and timing

3. **Full Transition**
   - Once validated, deprecate old CI
   - Update documentation
   - Train team on new system

## Expected Improvements

### Timing Comparison

| Workflow | Before | After | Savings |
|----------|--------|-------|---------|
| PR Check (Python changes) | ~60 min | ~10 min | 50 min |
| PR Check (Fortran changes) | ~60 min | ~20 min | 40 min |
| Nightly (Full suite) | ~120 min | ~90 min | 30 min |

### Resource Usage

- **Before:** Compile from scratch every PR
- **After:** Reuse pre-built image, incremental compilation
- **Storage:** ~2-3 GB Docker image (acceptable for GitHub Container Registry)

## Architecture Highlights

### Smart Rebuild

The system only recompiles files that changed:

```yaml
# In ci-pr-checks.yml
- name: Smart rebuild of physics modules
  run: |
    cd SPIDER
    make -q || make -j$(nproc)  # Only rebuild if needed
```

**Result:**
- Python-only PR: No recompilation (~instant)
- Fortran PR: Only changed files (~minutes, not hours)

### Test Stratification

Tests are organized by execution time and purpose:

1. **Unit Tests** (seconds): Python logic, mocked physics
2. **Smoke Tests** (minutes): Binary validation, minimal resolution
3. **Integration Tests** (minutes): Multi-module coupling
4. **Slow Tests** (hours): Full scientific validation

### Container Strategy

- **Build:** Nightly at 02:00 UTC
- **Cache:** Docker layers + BuildKit cache
- **Usage:** All CI workflows pull the same image
- **Overlay:** PR code replaces container code at runtime

## Troubleshooting

### Docker Build Fails

```bash
# Test locally
docker build -t proteus-test .

# Check specific stage
docker build --target <stage> -t proteus-test .

# Inspect layers
docker history proteus-test
```

### CI Can't Pull Image

- Verify image is public or token has permissions
- Check registry URL: `ghcr.io/formingworlds/proteus:latest`
- Test pull locally: `docker pull ghcr.io/formingworlds/proteus:latest`

### Tests Fail in Container

```bash
# Run container interactively
docker run -it ghcr.io/formingworlds/proteus:latest bash

# Inside container, run tests
pytest -m unit -v
```

### Smart Rebuild Not Working

- Verify Makefiles are present in container
- Check if binaries have correct timestamps
- Force rebuild: `rm SPIDER/spider && make`

## Files Changed Summary

```
├── Dockerfile (NEW)
├── .github/workflows/
│   ├── docker-build.yml (NEW)
│   ├── ci-pr-checks.yml (NEW)
│   └── ci-nightly-science.yml (NEW)
├── docs/
│   └── docker_ci_architecture.md (NEW)
├── tests/examples/
│   ├── __init__.py (NEW)
│   └── test_marker_usage.py (NEW)
├── pyproject.toml (MODIFIED - added smoke marker)
└── README.md (THIS FILE)
```

## Next Actions

### Immediate (For Reviewer)

1. Review Dockerfile for security and best practices
2. Check workflow configurations
3. Verify pytest marker integration
4. Test build locally if possible

### Short-term (After Merge)

1. Monitor first nightly build (02:00 UTC)
2. Test PR workflow with real PR
3. Gather timing metrics
4. Adjust resource limits if needed

### Long-term (Future PRs)

1. Add markers to existing tests
2. Optimize Docker image size
3. Add matrix testing (multiple Python versions)
4. Implement artifact caching for FWL_DATA

## Contact

For questions or issues with this implementation:

- **Author:** Tim Lichtenberg (tim.lichtenberg@rug.nl)
- **Documentation:** `docs/docker_ci_architecture.md`
- **Reference Tests:** `tests/examples/test_marker_usage.py`
- **Architecture Guide:** This README

## References

- PROTEUS Test Infrastructure: `docs/test_infrastructure.md`
- Installation Guide: `docs/installation.md`
- Docker Best Practices: https://docs.docker.com/develop/dev-best-practices/
- GitHub Actions Container Jobs: https://docs.github.com/en/actions/using-jobs/running-jobs-in-a-container
