# Test Building Strategy for PROTEUS

**Last Updated**: 2026-01-27
**Status**: Unit-test coverage raised above **32.03%** fast-gate threshold (run 21413731346). Added unit tests for `utils/data.py` (`check_needs_update`, `GetFWLData`) and `config` (`read_config`, `read_config_object`). ~492+ tests, 7 active smoke tests (exceeds 5-7 target). **No test skips**: See [Plan: Fix Nightly Integration Failures (No Test Skips)](#plan-fix-nightly-integration-failures-no-test-skips) for root-cause fixes (AGNI data, fixture, Docker, workflow). See [Plan: What Comes Next](#plan-what-comes-next) for prioritized next tasks.

---

## Current Status

### Coverage Metrics

- **Total tests**: 492+ (442+ unit + 7 active smoke + 4 skipped smoke + 39 other)
- **Current coverage**: Above 32.03% fast-gate threshold (run 21413731346)
- **Target**: 32.03% fast gate coverage (`[tool.proteus.coverage_fast] fail_under`) + 5-7 smoke tests
- **Status**: 🎯 **Unit Test Goal Exceeded** — threshold (32.03%) met via added `check_needs_update`/`GetFWLData`/config tests, all tests passing

### Test Status Summary

| Category | Current | Target | Status | Gap |
|----------|---------|--------|--------|-----|
| Unit tests | 442 | 470+ | ✅ **94%** | On track |
| Smoke tests (active) | 7 | 5-7 | ✅ **100%** | **Target exceeded** |
| Integration tests | 6 | 23 | ⏸️ **26%** | **-17 tests** |
| Coverage (fast gate) | 31.24% | 30% | ✅ **104%** | Exceeded |

### Active Smoke Tests

| Test | Status | Notes |
|------|--------|-------|
| `test_proteus_dummy_init` | ✅ **ACTIVE** | Minimal initialization |
| `test_smoke_dummy_atmos_dummy_interior_flux_exchange` | ✅ **ACTIVE** | Fixed (ini_tmagma=2000K) |
| `test_smoke_escape_dummy_atmos` | ✅ **ACTIVE** | Escape module + dummy atmosphere coupling |
| `test_smoke_star_instellation` | ✅ **ACTIVE** | Star module + dummy atmosphere coupling |
| `test_smoke_orbit_tidal_heating` | ✅ **ACTIVE** | Orbit module + dummy interior coupling |
| `test_smoke_outgas_atmos_volatiles` | ✅ **ACTIVE** | CALLIOPE outgassing + dummy atmosphere coupling |
| `test_smoke_dummy_full_chain` | ✅ **ACTIVE** | All dummy modules in sequence |
| `test_smoke_janus_dummy_coupling` | ⏸️ **SKIPPED** | JANUS/SOCRATES runtime instability (hangs) |
| `test_integration_calliope_dummy_atmos_outgassing` | ✅ **INTEGRATION** | Moved to nightly CI |

---

## Recent Achievements

### Data Download Robustness (2026-01-XX) ✅
- **Multi-tier fallback system**: `zenodo_client` → `zenodo_get` → web download → OSF fallback
- **Retry logic**: 3 attempts with exponential backoff (5s, 10s, 20s)
- **Rate limiting**: 2s cooldown between Zenodo API requests
- **Token detection**: Automatic fallback when API token unavailable
- **Impact**: Integration tests should be more resilient to data download failures
- **Files**: `src/proteus/utils/data.py` (~1883 lines changed)

### All CI Tests Passing (2026-01-11) ✅
- **Fixed 4 VULCAN unit tests**: Resolved complex import/mocking issues with `proteus.atmos_chem.vulcan` module
- **All 441 unit tests passing** in latest CI run
- **All 7 smoke tests passing** consistently
- **Coverage maintained** above 30% threshold (31.24%)

### VULCAN Test Fixes
The VULCAN chemistry module tests were failing due to:
1. Module import order issues when mocking `vulcan` package
2. Complex dependency chain between `wrapper.py`, `vulcan.py`, and `common.py`
3. Solution: Mock `vulcan` and `proteus.atmos_chem.vulcan` in `sys.modules` at module level, create actual CSV files for `read_result` to process

### Integration Test Infrastructure ✅
- **Created integration test fixtures** (`tests/integration/conftest.py`):
  - `proteus_multi_timestep_run`: Factory fixture for multi-timestep PROTEUS runs
  - `validate_energy_conservation`: Energy balance validation helper
  - `validate_mass_conservation`: Mass conservation validation helper
  - `validate_stability`: Stability checks (no runaway temperatures/pressures)
- **Created multi-timestep integration tests**:
  - `test_integration_dummy_multi_timestep`: 5-timestep validation with dummy modules
  - `test_integration_dummy_extended_run`: 10-timestep extended run validation
  - `test_integration_calliope_multi_timestep`: 5-timestep CALLIOPE outgassing coupling
  - `test_integration_calliope_extended_run`: 10-timestep CALLIOPE extended run
  - `test_integration_std_config_multi_timestep`: 5-timestep standard config (all real modules)
  - `test_integration_std_config_extended_run`: 10-timestep extended standard config (@pytest.mark.slow)
- **Updated CI workflow** to include new integration tests in nightly runs
- **Standard configuration test** implemented using `input/all_options.toml` (Priority 2.1)
- **Foundation established** for Phase 2: Integration Test Foundation

---

## Completed Work

Tests are prioritized by **ease of implementation** × **impact on coverage** × **foundational value**.

### Priority 1: Quick Wins (High impact, low effort) ✓ **COMPLETED**

These modules have simple, pure functions with few dependencies. Tests run in <10ms each.

#### 1.1 `utils/helper.py` ✓ **COMPLETED**
- **Status**: 53 unit tests created and passing
- **Coverage**: ~80% of functions tested
- **Impact**: High — utilities used throughout codebase

#### 1.2 `utils/logs.py` ✓ **COMPLETED**
- **Status**: 41 unit tests created and passing
- **Coverage**: 91.49% of logs.py
- **Impact**: High — logging used throughout all simulation runs

#### 1.3 `config/_converters.py` ✓ **COMPLETED**
- **Status**: 27 unit tests created and passing
- **Coverage**: 100% of _converters.py
- **Impact**: High — converters used for all TOML config parsing

#### 1.4 `orbit/dummy.py` ✓ **COMPLETED**
- **Status**: 7 unit tests created and passing
- **Coverage**: 100% of dummy tidal heating logic
- **Impact**: Medium — tidal heating calculations

### Priority 2: Moderate Effort, High Value ✓ **COMPLETED**

These modules have more complex logic but are still testable without heavy computations.

#### 2.1 `utils/terminate.py` ✓ **COMPLETED**
- **Status**: 19 unit tests created and passing
- **Coverage**: All termination criteria (solidification, energy balance, escape, disintegration, time/iteration limits, keepalive)

#### 2.2 `star/dummy.py` ✓ **COMPLETED**
- **Status**: 14 unit tests created and passing
- **Coverage**: Radius scaling, spectrum generation, luminosity (Stefan-Boltzmann), instellation (inverse-square)

#### 2.3 `interior/dummy.py` ✓ **COMPLETED**
- **Status**: 13 unit tests created and passing
- **Coverage**: calculate_simple_mantle_mass() + run_dummy_int() + melt fraction physics

#### 2.4 `utils/coupler.py` ✓ **COMPLETED**
- **Status**: 55 unit tests created and passing
- **Coverage**: Helpfile operations, lock file management, version getters, print functions

### Priority 3: Foundation/Core Modules ✓ **COMPLETED**

These are heavily used but require careful test design to isolate logic from physics.

#### 3.1 `config/` modules ✓ **SUBSTANTIALLY COMPLETED**
- **Status**: 89 validator tests created and passing
- **Coverage**: Core config, interior validators, params validators, observe validators, outgas validators
- **Remaining**: 10-20 additional tests for dataclass defaults and edge-case TOML parsing

#### 3.2 `atmos_clim/wrapper.py` ✓ **COMPLETED**
- **Status**: 20 unit tests created and passing
- **Coverage**: Complete wrapper interface for dummy atmosphere module

#### 3.3 `atmos_chem/wrapper.py` ✓ **COMPLETED**
- **Status**: 9 unit tests created and passing (all fixed 2026-01-11)
- **Coverage**: VULCAN chemistry module interface
- **Recent fixes**: Resolved VULCAN import/mocking issues in CI, all 4 VULCAN tests now passing

### Priority 3.5: Module Wrappers ✓ **COMPLETED**

These modules interface with external physics solvers and require careful mocking.

#### 3.5.1 `escape/wrapper.py` ✓ **COMPLETED**
- **Status**: 25 unit tests created and passing
- **Coverage**: ZEPHYRUS escape module interface

#### 3.5.2 `interior/wrapper.py` ✓ **COMPLETED**
- **Status**: 12 unit tests created and passing
- **Coverage**: Interior module interface (run_interior dispatcher, dummy/spider/aragog modules)

#### 3.5.3 `observe/wrapper.py` ✓ **COMPLETED**
- **Status**: 10 unit tests created and passing
- **Coverage**: Observation module interface

#### 3.5.4 `outgas/wrapper.py` ✓ **COMPLETED**
- **Status**: 60 unit tests created and passing
- **Coverage**: CALLIOPE outgassing interface (comprehensive validation of all major functions)

### Priority 4: Lower Priority ✓ **PARTIALLY COMPLETED**

#### 4.1 `utils/plot.py` ✓ **COMPLETED**
- **Status**: 9 unit tests created and passing
- **Coverage**: Plotting utilities (mocked matplotlib)

#### 4.2 `utils/data.py` ✅ **SUBSTANTIALLY COMPLETED**
- **Status**: 48+ unit tests in `tests/utils/test_data.py` (download wrappers, OSF/Zenodo paths, validation, **check_needs_update**, **GetFWLData**, get_Seager_EOS; some branches skipped for integration)
- **Added 2026-01-27**: Tests for `check_needs_update` (missing dir, no zenodo, valid/invalid folder) and `GetFWLData` (absolute path) to raise coverage above 32.03% threshold
- **Remaining**: Optional—hash-mismatch and OSF exception paths better covered by integration tests
- **Effort**: LOW for further unit gains; integration coverage is the main lever

### Priority 2.5: Smoke Tests (Fast Integration Validation) ⏸️ **IN PROGRESS**

**Purpose**: End-to-end coupling tests with real binaries but minimal computation (1 timestep, low resolution, <30s each).

**Current status**: 3 smoke tests active (target: 5-7)

#### 2.5.1 `Atmosphere-Interior Coupling` ✓ **COMPLETED**
- **Implemented**: `test_smoke_dummy_atmos_dummy_interior_flux_exchange` (✓ PASSING, ~10s)
- **Status**: Fixed and re-enabled (2026-01-11). Lowered `ini_tmagma` from 3500K to 2000K to prevent runaway heating.
- **Validation**: Flux exchange (F_atm ↔ F_int), T_surf updates, T_magma bounds (200-1e6 K)

#### 2.5.2 `JANUS-Interior Coupling` ✓ **FIXED**
- **Implemented**: `test_smoke_janus_dummy_coupling` (with timeout protection)
- **Status**: Fixed (2026-01-11). Added 60s timeout, reduced resolution (num_levels=15, spectral_bands=8).
- **Note**: May skip if JANUS/SOCRATES binaries unavailable, but won't hang CI.

#### 2.5.3 `CALLIOPE Outgassing` ✓ **MOVED TO INTEGRATION**
- **Implemented**: `test_integration_calliope_dummy_atmos_outgassing`
- **Status**: Moved to integration tests (2026-01-11). Changed marker from `@pytest.mark.smoke` to `@pytest.mark.integration`.
- **Rationale**: CALLIOPE initialization overhead better suited for nightly CI.

**Implementation strategy**:
1. Start with dummy module combinations (fastest, easiest to debug)
2. Add real module tests one at a time
3. Use low-resolution grids (JANUS: 15 levels minimum, SOCRATES: minimal bands)
4. Mock external data files (stellar spectra, opacities) where possible
5. Each smoke test should complete in <30s

---

## Roadmap & Next Steps

### Phase 1: Smoke Test Expansion (Immediate — Next 1-2 Weeks) ✅ **COMPLETED**

**Goal**: Reach 5-7 active smoke tests for fast PR validation
**Current**: 7 active tests ✅ **Target exceeded**
**Status**: All priority smoke tests implemented and passing
**Recent Achievement**: All CI tests now passing (441 tests), including fixed VULCAN unit tests

#### Priority 1.1: Add New Smoke Tests

**1.1.1 Escape Module Smoke Test** (HIGH)
- **Target**: `test_smoke_escape_dummy_atmos`
- **Scope**: ZEPHYRUS escape + dummy atmosphere coupling
- **Validates**: Mass loss calculations, elemental inventory updates
- **Configuration**: 1 timestep, minimal resolution
- **Effort**: 2-3 hours
- **Impact**: +1 active smoke test

**1.1.2 Star Module Smoke Test** (MEDIUM)
- **Target**: `test_smoke_star_instellation`
- **Scope**: MORS stellar evolution + dummy atmosphere
- **Validates**: Stellar luminosity, instellation calculations
- **Configuration**: Single timestep, fixed orbit
- **Effort**: 2-3 hours
- **Impact**: +1 active smoke test

**1.1.3 Orbit Module Smoke Test** (MEDIUM)
- **Target**: `test_smoke_orbit_tidal_heating`
- **Scope**: LovePy tidal heating + dummy interior
- **Validates**: Tidal heating calculations, orbital evolution
- **Configuration**: 1 timestep, simple orbit
- **Effort**: 3-4 hours
- **Impact**: +1 active smoke test

**1.1.4 Outgassing-Atmosphere Coupling** (MEDIUM)
- **Target**: `test_smoke_outgas_atmos_volatiles`
- **Scope**: CALLIOPE outgassing + dummy atmosphere (simpler than full CALLIOPE)
- **Validates**: Volatile mass exchange, fO2 updates
- **Configuration**: 1 timestep, minimal volatiles
- **Effort**: 2-3 hours
- **Impact**: +1 active smoke test

**1.1.5 Multi-Module Dummy Chain** (LOW)
- **Target**: `test_smoke_dummy_full_chain`
- **Scope**: All dummy modules in sequence (star → orbit → interior → atmos → escape)
- **Validates**: Full coupling loop with minimal physics
- **Configuration**: 1 timestep, all dummy modules
- **Effort**: 1-2 hours
- **Impact**: +1 active smoke test, validates coupling infrastructure

**Smoke Test Implementation Template**:
```python
# Template for new smoke tests
@pytest.mark.smoke
def test_smoke_MODULE1_MODULE2_coupling(tmp_path):
    """Test MODULE1 + MODULE2 coupling (1 timestep).

    Physical scenario: [Brief description]

    Validates:
    - [Key validation point 1]
    - [Key validation point 2]
    - No NaN or Inf values

    Runtime: <30s (1 timestep, minimal resolution)
    """
    # 1. Create minimal TOML config
    # 2. Initialize Proteus
    # 3. Run 1 timestep
    # 4. Validate outputs (no NaN, physical ranges)
    # 5. Check coupling variables updated
```

**Phase 1 Success Criteria**:
- ✅ 5-7 active smoke tests (achieved 7, exceeds target)
- ✅ All smoke tests run in <30s each
- ✅ Smoke tests pass in CI consistently

### Phase 2: Integration Test Foundation (Next 2-4 Weeks) ⏸️ **IN PROGRESS**

**Goal**: Establish integration test infrastructure and first comprehensive test

**Current Status**: Infrastructure created (2026-01-11)
- ✅ Integration test fixtures (`tests/integration/conftest.py`)
- ✅ Validation helpers (energy/mass conservation, stability checks)
- ✅ Simple multi-timestep integration test (dummy modules)
- ⏸️ Standard configuration test (requires real modules)

#### Priority 2.1: Standard Configuration Integration Test (HIGH)

**Target**: `tests/integration/test_std_config.py`

**Configuration**: `input/all_options.toml` (comprehensive PROTEUS configuration)

**Scope**: Full PROTEUS "standard candle" configuration
- **Star**: MORS (stellar evolution)
- **Orbit**: None (tidal heating disabled in `all_options.toml`)
- **Interior**: ARAGOG (thermal evolution)
- **Atmosphere**: AGNI (radiative-convective equilibrium)
- **Outgassing**: CALLIOPE (volatile supply)
- **Escape**: ZEPHYRUS (mass loss)
- **Chemistry**: VULCAN (atmospheric chemistry, optional)

**Note**: The `all_options.toml` configuration uses `orbit.module = "none"` (tidal heating disabled).
This is the actual standard configuration and may differ from ideal "all modules enabled" scenario.
The test validates the configuration as-is.

**Requirements**:
- Run for multiple timesteps (5-10 timesteps)
- Validate energy conservation
- Validate mass conservation across reservoirs
- Check stable feedback loops
- Runtime: <5 minutes (low resolution)

**Implementation Plan**:
1. **Week 1**: Create test structure and minimal config ✅ **COMPLETED** (2026-01-11)
   - ✅ Set up test fixtures for multi-timestep runs (`proteus_multi_timestep_run`)
   - ✅ Add validation helpers (energy/mass conservation, stability)
   - ✅ **Standard configuration TOML identified**: `input/all_options.toml`
     - This is the comprehensive PROTEUS configuration with all real modules
     - Must be tested during nightly Science validation
     - Uses: MORS (star), LovePy (orbit), ARAGOG (interior), AGNI (atmosphere), CALLIOPE (outgas), ZEPHYRUS (escape)

2. **Week 2**: Implement core test logic ✅ **COMPLETED** (2026-01-13)
   - ✅ Initialize all modules (with graceful skip if modules unavailable)
   - ✅ Run coupling loop for N timesteps
   - ✅ Collect state variables at each timestep
   - ✅ Created `test_integration_std_config.py` with 2 tests:
     - `test_integration_std_config_multi_timestep`: 5-timestep validation
     - `test_integration_std_config_extended_run`: 10-timestep extended run (@pytest.mark.slow)

3. **Week 3**: Add validation checks ✅ **COMPLETED** (2026-01-13)
   - ✅ Energy balance validation (flux convergence checks for magma oceans)
   - ✅ Mass conservation checks (20% tolerance for escape/outgassing)
   - ✅ Stability checks (no runaway temperatures/pressures, bounded growth validation)
   - ✅ Comprehensive physical parameter validation (stellar, orbital, interior, atmospheric, volatile, escape)

4. **Week 4**: Debug and stabilize ⏸️ **IN PROGRESS**
   - ✅ Test structure complete with comprehensive validation
   - ⏸️ Verify tests pass in nightly CI with improved data download system
   - ⏸️ Monitor for any runtime issues with real modules
   - ⏸️ Optimize resolution if needed for <5 min runtime
   - ✅ Already integrated into nightly CI (`ci-nightly-science-v5.yml`)

**Effort**: 16-24 hours (4 weeks, 4-6 hours/week)
**Impact**: Foundation for all future integration tests

#### Priority 2.2: Integration Test Infrastructure ✅ **COMPLETED** (2026-01-11)

**2.2.1 Test Fixtures** ✅ **COMPLETED**
- ✅ Created reusable fixtures for multi-timestep runs (`proteus_multi_timestep_run`)
- ✅ Added helpers for conservation law validation (`validate_energy_conservation`, `validate_mass_conservation`, `validate_stability`)
- ✅ Created integration test templates:
  - `test_integration_multi_timestep.py`: Dummy module multi-timestep tests (2 tests)
  - `test_integration_calliope_multi_timestep.py`: CALLIOPE outgassing multi-timestep tests (2 tests)

**2.2.2 CI Integration** ⏸️ **PENDING**
- ⏸️ Add integration test job to `ci-nightly-science.yml`
- ⏸️ Set up separate job for slow integration tests
- ⏸️ Configure notifications for failures

**Effort**: 4-6 hours (2.2.1 completed, ~2 hours; 2.2.2 pending, ~2-4 hours)
**Impact**: Enables systematic integration test development

**Phase 2 Success Criteria**:
- ✅ Standard configuration integration test implemented (`test_integration_std_config.py`)
- ✅ Comprehensive validation checks implemented (energy, mass, stability, physical parameters)
- ⏸️ Test runs stably for 5-10 timesteps (requires all real modules + data - should be more resilient with improved download system)
- ⏸️ Conservation law validations pass (implemented, needs verification in CI with real modules)
- ✅ Integrated into nightly CI (`ci-nightly-science-v5.yml`)
- ✅ **Nightly run 21376308695**: Integration and slow tests passed; §1 (Verify Phase 2 in CI) complete.
- ⏸️ **Data Download Robustness Enhanced (2026-01-XX)**:
  - ✅ **Multi-tier fallback system implemented**: `zenodo_client` → `zenodo_get` → web download → OSF fallback
  - ✅ **Retry logic**: 3 attempts with exponential backoff (5s, 10s, 20s)
  - ✅ **Rate limiting**: 2s cooldown between Zenodo API requests
  - ✅ **Token detection**: Automatic fallback when API token unavailable
  - ⏸️ **Integration test status**: Tests implemented with graceful skip logic; should now be more resilient to data download failures
  - **Next step**: Verify integration tests pass in nightly CI with improved download system
  - **Tests affected**: `test_integration_std_config.py` (both slow tests), `test_integration_aragog_janus.py` (5 tests)

### Phase 3: Coverage Expansion (Ongoing)

**Goal**: Maintain and improve unit test coverage above 30%

#### Priority 3.1: Remaining High-Value Modules

**3.1.1 `utils/data.py` Expansion** ✅ **SUBSTANTIALLY COMPLETED**
- **Current**: 43 unit tests in `tests/utils/test_data.py` (download wrappers, OSF paths, validation, GetFWLData, get_Seager_EOS)
- **Remaining**: Optional edge cases; some branches better covered by integration tests
- **Effort**: LOW for further unit gains

**3.1.2 Core Module Edge Cases** (LOW)
- Add tests for error paths in existing modules
- Test boundary conditions
- Validate error messages
- **Effort**: Ongoing, 1-2 hours per module
- **Impact**: Improves robustness

**Phase 3 Success Criteria**:
- ✅ Unit test coverage maintained >30%
- ✅ New code changes include tests
- ✅ Coverage ratcheting continues to work

### Timeline Summary

| Phase | Duration | Milestones | Deliverables |
|-------|----------|------------|--------------|
| **Phase 1** | 1-2 weeks | 5-7 active smoke tests | 2-4 new smoke tests |
| **Phase 2** | 2-4 weeks | First integration test | `test_std_config.py` + infrastructure |
| **Phase 3** | Ongoing | Maintain >30% coverage | Incremental improvements |

### Risk Assessment

#### High-Risk Items

1. **JANUS/SOCRATES Runtime Instability**
   - **Risk**: Smoke test may never stabilize
   - **Mitigation**: Timeout added, consider alternative test or move to integration tests only

2. **Standard Configuration Test Complexity**
   - **Risk**: May require significant debugging time
   - **Mitigation**: Start with 2-3 modules, add incrementally

3. **CI Runtime Growth**
   - **Risk**: Adding smoke tests may slow CI
   - **Mitigation**: Run smoke tests in parallel, optimize test configurations

#### Medium-Risk Items

1. **Dummy Module Physics Issues**
   - **Risk**: Dummy modules may not be stable enough for smoke tests
   - **Mitigation**: Fixed T_magma issue, use real modules with minimal resolution if needed

2. **Integration Test Maintenance**
   - **Risk**: Integration tests may be fragile
   - **Mitigation**: Focus on stability checks, not exact values

### Recommendations

#### Immediate Actions (This Week)

1. **Verify Integration Tests in Nightly CI** — **PRIORITY** (see [Plan: What Comes Next](#plan-what-comes-next), §1)
   - Monitor `ci-nightly-science-v5.yml`
   - Confirm `test_integration_std_config.py` and `test_integration_aragog_janus.py` pass or have documented issues

2. **utils/data.py** — ✅ Substantially done (43 tests). Optional: add edge-case unit tests when touching that code.

#### Short-Term Actions (Next 2 Weeks)

1. **Complete Phase 2: Integration Test Foundation**
   - ✅ Standard configuration test implemented
   - ✅ Validation checks implemented
   - ✅ Verify tests pass in nightly CI (run 21376308695 passed)
   - ⏸️ Document any runtime optimizations needed
   - ⏸️ Add more module combination tests (e.g., ARAGOG+JANUS, AGNI+CALLIOPE)

2. **Expand Integration Test Suite**
   - Add tests for specific module combinations:
     - ARAGOG + AGNI (interior-atmosphere coupling)
     - CALLIOPE + ZEPHYRUS (outgassing-escape coupling)
     - MORS + AGNI (stellar-atmosphere coupling)
   - Create test templates for new integration tests
   - Establish best practices documentation

#### Long-Term Actions (Next Month)

1. **Phase 3: Coverage Expansion**
   - Maintain >30% unit test coverage
   - Target: 35-40% coverage by end of Q1
   - Focus on high-impact modules with low coverage
   - Continue coverage ratcheting mechanism

2. **Slow Test Strategy**
   - Plan comprehensive physics validation tests
   - Examples: Earth magma ocean, Venus runaway greenhouse, Super-Earth evolution
   - Budget: 3-4 hours per scenario for nightly CI
   - Document test scenarios and expected outcomes

---

## Coverage Analysis

### Current Modules by Coverage

| Module | Lines | Est. Functions | Status | Tests | Priority |
| ------ | ----- | -------------- | ------ | ----- | -------- |
| `utils/helper.py` | 328 | 15 | ✓ DONE | 53 | 1 |
| `utils/logs.py` | 150 | 8 | ✓ DONE | 41 | 1 |
| `config/_converters.py` | 100 | 6 | ✓ DONE | 27 | 1 |
| `orbit/dummy.py` | 54 | 1 | ✓ DONE | 7 | 1 |
| `utils/terminate.py` | 291 | 5 | ✓ DONE | 19 | 2 |
| `star/dummy.py` | 147 | 4 | ✓ DONE | 14 | 2 |
| `interior/dummy.py` | 131 | 4 | ✓ DONE | 13 | 2 |
| `utils/coupler.py` | 979 | 15+ | ✓ DONE | 55 | 2 |
| `config/*` (all validators) | 800+ | 20+ | ✓ DONE | 89 | 3 |
| `atmos_clim/wrapper.py` | 200+ | 8+ | ✓ DONE | 20 | 3 |
| `atmos_chem/wrapper.py` | 150+ | 6+ | ✓ DONE | 9 | 3 |
| `escape/wrapper.py` | 250+ | 10+ | ✓ DONE | 25 | 3.5 |
| `interior/wrapper.py` | 200+ | 8+ | ✓ DONE | 12 | 3.5 |
| `observe/wrapper.py` | 150+ | 6+ | ✓ DONE | 10 | 3.5 |
| `outgas/wrapper.py` | 400+ | 15+ | ✓ DONE | 60 | 3.5 |
| `utils/plot.py` | 308 | 12+ | ✓ DONE | 9 | 4 |
| `utils/data.py` | 707 | 15+ | ✅ SUBSTANTIAL | 43 | 4 |

### Coverage Milestones

| Milestone | Unit Tests | Coverage | Timeline |
| --------- | ---------- | -------- | -------- |
| Baseline | 13 | 18.51% | 2026-01-10 |
| After Priority 1 | 134 | 22.42% | 2026-01-11 (AM) |
| After Priority 2 | 242 | 23.03% | 2026-01-11 (Mid) |
| After Priority 3 (partial) | 457 | 29.52% | 2026-01-11 (PM) |
| **Final State (30% Exceeded)** | **487** | **31.45%** | **2026-01-11 (Final)** |
| **Current State (All Tests Passing)** | **492** | **31.24%** | **2026-01-11 (CI Fixed)** |

---

## Testing Principles (Reference)

When implementing tests, follow these guidelines from PROTEUS Copilot Instructions:

### 1. Structure Mirrors Source

```text
src/proteus/utils/helper.py        →  tests/utils/test_helper.py
src/proteus/config/_converters.py  →  tests/config/test_converters.py
src/proteus/orbit/dummy.py         →  tests/orbit/test_dummy.py
```

### 2. Test Markers

```python
@pytest.mark.unit          # <100ms, no heavy physics
@pytest.mark.smoke         # Real binaries, 1 timestep
@pytest.mark.integration   # Multi-module coupling
@pytest.mark.slow          # Full physics validation (hours)
```

### 3. Mocking Strategy

- **Mock everything external**: Files, network, heavy computations
- **Default tool**: `unittest.mock`
- **Integration tests only**: Use real calls

### 4. Float Comparisons

```python
# WRONG:
assert value == 3.14

# RIGHT:
assert value == pytest.approx(3.14, rel=1e-5)
```

### 5. Parametrization

```python
@pytest.mark.parametrize('status,expected', [
    (0, 'Started'),
    (1, 'Running'),
    (10, 'Completed (solidified)'),
])
def test_comment_from_status(status, expected):
    assert CommentFromStatus(status) == expected
```

---

## Quick Reference: Adding New Tests

### Step 1: Create test file

```bash
touch tests/MODULE/test_SUBMODULE.py
```

### Step 2: Copy template

```python
"""
Unit tests for proteus.MODULE.SUBMODULE module.

Tests DESCRIPTION with TESTABLE_ASPECT.
"""
from __future__ import annotations

import pytest
from proteus.MODULE.SUBMODULE import function_to_test


@pytest.mark.unit
def test_function_name():
    """Test description."""
    result = function_to_test(input_value)
    assert result == expected_value
```

### Step 3: Run and verify

```bash
pytest tests/MODULE/test_SUBMODULE.py -v
pytest tests/MODULE/test_SUBMODULE.py --cov=src/proteus/MODULE
```

### Step 4: Commit and push

```bash
git add tests/MODULE/test_SUBMODULE.py
git commit -m "test(MODULE): add unit tests for SUBMODULE (N tests)"
git push
```

---

## Next Steps Summary

See **[Plan: What Comes Next](#plan-what-comes-next)** for the canonical, prioritized list of tasks. Summary:

- **Phase 2**: Verify integration tests in nightly CI; expand integration coverage (module-combination tests).
- **Phase 3**: Maintain >30% coverage; target 35–40%; add core-module edge-case tests.

---

## Plan: Fix Nightly Integration Failures (No Test Skips)

**Objective**: All integration tests run and pass in nightly CI. Do **not** skip tests; fix root causes (data, environment, Docker).

### Principles

1. **No skips**: Never add `--ignore` or similar to avoid failing tests. Fix the underlying issue instead.
2. **Root causes**: Ensure required data is downloaded, the Docker image has the correct build (AGNI, Julia, env vars), and the workflow prepares data for every test config.
3. **Speed rule**: If any single test exceeds **1 hour**, reduce timesteps or resolution so it stays under 1 h. This is the only acceptable "reduction" of scope.

### Root Causes Addressed

| Cause | Where it shows up | Fix |
|-------|-------------------|-----|
| **AGNI data not pre-downloaded** | `test_integration_aragog_agni_multi_timestep` fails when `runner.start(..., atmos_clim=agni)` runs; AGNI needs spectral files, surface albedos, stellar data | Workflow "Download test data" step calls `download_sufficient_data` for a config with `atmos_clim.module='agni'` (e.g. `tests/integration/aragog_janus.toml` with override) so AGNI-specific data is present before tests run |
| **Fixture only triggered download for ARAGOG** | Fixture called `download_sufficient_data` only when `interior.module == 'aragog'`; tests that use AGNI without ARAGOG would not get AGNI data | Fixture now calls `download_sufficient_data` when `interior.module == 'aragog'` **or** `atmos_clim.module == 'agni'` |
| **Julia depot not set in image** | In Docker, Julia may use `~/.julia` and fill the home volume; CI step sets `JULIA_DEPOT_PATH` but image build did not | Dockerfile sets `JULIA_DEPOT_PATH=/opt/julia_depot` and creates `/opt/julia_depot` so AGNI/Julia use a dedicated path from first use |
| **Test was excluded** | Workflow had `--ignore=tests/integration/test_integration_aragog_agni.py` | Removed; ARAGOG+AGNI test runs in the "integration and not slow" step |

### Actions Taken (Implementation Checklist)

- [x] **Workflow "Download test data"**: After `download_sufficient_data(all_options.toml)`, add a second call using `tests/integration/aragog_janus.toml` with `atmos_clim.module = 'agni'` so spectral files, surface albedos, and any other AGNI-required data are downloaded.
- [x] **Fixture** (`tests/integration/conftest.py`): Change condition from `if runner.config.interior.module == 'aragog'` to `if runner.config.interior.module == 'aragog' or runner.config.atmos_clim.module == 'agni'` so AGNI-using tests always trigger data download when offline is temporarily disabled.
- [x] **Dockerfile**: Set `JULIA_DEPOT_PATH=/opt/julia_depot` in `ENV` and ensure `mkdir -p /opt/julia_depot` so the image uses a dedicated Julia depot from build time.
- [x] **Workflow "Run integration coverage"**: Remove `--ignore=tests/integration/test_integration_aragog_agni.py`; only `test_integration_dummy_agni.py` and `test_albedo_lookup.py` remain excluded (external-data heavy / special setup).
- [ ] **Verify**: Re-run nightly after rebuilding the Docker image (so `JULIA_DEPOT_PATH` and `/opt/julia_depot` are in the image). If `test_integration_aragog_agni_multi_timestep` still fails, capture the exact traceback and fix the next root cause (e.g. missing file path, AGNI wrapper error).

### Speed Rule (Tests >1 h)

If any single test exceeds **1 hour**:

1. Reduce `num_timesteps` or shorten `max_time` / `min_time` in the test or fixture.
2. Reduce resolution (e.g. AGNI `num_levels`, spectral bands) in the config overlay used by that test.
3. Document the runtime and configuration in the test docstring and in this section.

Do **not** skip the test; make it faster so it stays within the 1 h per-test cap.

---

## Plan: What Comes Next

This section is the **canonical plan** for continuing the test roadmap. It aligns with [AGENTS.md](../AGENTS.md), [test_infrastructure.md](test_infrastructure.md), [test_categorization.md](test_categorization.md), and [test_building.md](test_building.md).

### Current Snapshot (as of doc update)

- **Unit tests**: ~490+ (target 470+ ✅)
- **Smoke tests**: 7 active (target 5–7 ✅)
- **Integration tests**: ~6–10 (target 23; gap ~13–17)
- **Coverage (fast gate)**: >30% ✅
- **utils/data.py**: 43 unit tests in `tests/utils/test_data.py` ✅

### Priority Order for Next Work

#### 1. Verify Phase 2 in CI (Do First) ✅ **DONE**

| Task | Owner | Effort | Success criterion |
|------|--------|--------|--------------------|
| Monitor nightly CI `ci-nightly-science-v5.yml` for integration/slow runs | Dev | 0.5 h | Green run or documented failure |
| If failures: fix `test_integration_std_config.py` / data-download paths | Dev | 1–4 h | Std-config runs 5–10 steps in nightly |
| If failures: fix or document `test_integration_aragog_janus.py` | Dev | 1–2 h | Clear skip reason or passing tests |

**Deliverable**: Nightly integration/science run is green or has documented, tracked issues.

**Status**: Nightly run **21376308695** (workflow *CI - Nightly Science Validation (v5)*, branch `tl/test_ecosystem_v5`) completed successfully. Branch Nightly Coverage (Integration - v5) job passed; integration (dummy + integration and not slow) and slow integration (standard config) steps both succeeded. **§1 complete** for current branch.

#### 2. Expand Integration Test Coverage (Short-Term)

| Task | Owner | Effort | Success criterion |
|------|--------|--------|--------------------|
| Add **ARAGOG + AGNI** (interior–atmosphere) integration test | Dev | 3–6 h | 1 test, multi-step, stability checks |
| Add **CALLIOPE + ZEPHYRUS** (outgassing–escape) integration test | Dev | 2–4 h | 1 test, mass/volatile consistency |
| Add **MORS + AGNI** (stellar–atmosphere) integration test | Dev | 2–4 h | 1 test, instellation/luminosity coupling |
| Use `tests/integration/conftest.py` fixtures and [Test Building](test_building.md) integration prompt | — | — | All new tests use shared helpers |

**Deliverable**: +3 integration tests; total integration tests move toward 10+.

#### 3. Phase 3: Coverage and Quality (Ongoing)

| Task | Owner | Effort | Success criterion |
|------|--------|--------|--------------------|
| Keep fast-gate coverage ≥ current threshold | All | — | No decrease in `[tool.proteus.coverage_fast] fail_under` |
| Add **core-module edge-case** unit tests (error paths, boundaries) | Dev | 1–2 h/module | Fewer “missing” lines in `coverage report --show-missing` |
| Run `bash tools/coverage_analysis.sh` and tackle next low-coverage file | Dev | 1–2 h/session | Steady increase in covered lines |

**Deliverable**: Coverage trend flat or up; no unexplained drops.

#### 4. Slow Test Strategy (Backlog)

| Task | Owner | Effort | Success criterion |
|------|--------|--------|--------------------|
| Define 2–3 **slow** scenarios (e.g. Earth magma ocean, Venus greenhouse) | Dev/ Science | 2–4 h | Doc section in this file + `@pytest.mark.slow` stubs |
| Implement one slow scenario end-to-end | Dev | 4–8 h | 1 test in nightly, runtime documented |

**Deliverable**: Written slow-test plan and one implemented scenario.

### Suggested Order of Execution

1. **This week**: Execute **§1 (Verify Phase 2 in CI)**. If nightly is green, move to §2.
2. **Next 1–2 weeks**: Execute **§2 (Expand Integration Coverage)**. Add the three module-combination tests.
3. **Ongoing**: Do **§3 (Coverage and Quality)** in the background (e.g. when touching a module).
4. **When capacity allows**: Start **§4 (Slow Test Strategy)** and implement one scenario.

### Where to Look When Implementing

- **Markers and CI**: [test_categorization.md](test_categorization.md) — `@pytest.mark.unit`, `smoke`, `integration`, `slow` and which runs they use.
- **How to write tests**: [test_building.md](test_building.md) — Master Prompt for unit tests, Integration Prompt for standard-config-style tests.
- **Infrastructure and layout**: [test_infrastructure.md](test_infrastructure.md) — layout, fixtures, coverage, CLI.
- **Agent constraints**: [AGENTS.md](../AGENTS.md) — test commands, coverage thresholds, lint, structure.

### Phase 2 Completion Checklist (from earlier summary)

**Status**: Infrastructure and validation in place; data-download robustness improved.

- [x] **Verify integration tests in CI** (1–2 days): Nightly run 21376308695 green; §1 complete.
- [ ] **Expand integration coverage** (1–2 weeks): +3 module-combination tests as in §2.
- [ ] **Optimize runtime** (if needed): Tune configs so nightly stays within time limits.

### Phase 3 Completion Checklist (from earlier summary)

**Current**: Fast-gate coverage >30%.

- [x] **utils/data.py**: 43 unit tests in place.
- [ ] **Core-module edge cases**: Add tests for error paths and boundaries (see §3).
- [ ] **Maintain threshold**: Rely on ratcheting; do not lower `fail_under`.

## Resources

- [Test Infrastructure Documentation](./test_infrastructure.md)
- [Test Categorization Guide](./test_categorization.md)
- [Test Building Guide](./test_building.md)
- [PROTEUS AGENT Instructions](../AGENTS.md)
- [Test conftest.py fixtures](../tests/conftest.py)
