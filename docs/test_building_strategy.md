# Test Building Strategy for PROTEUS

## Current Status (2026-01-10)

### Coverage Metrics

- **Baseline unit/smoke tests**: 13 tests marked with `@pytest.mark.unit`
- **New tests added**:
  - 53 tests for `utils/helper.py` (9 component classes, 40 distinct functions)
  - 41 tests for `utils/logs.py` (logging infrastructure)
  - 27 tests for `config/_converters.py` (type conversion utilities)
  - 7 tests for `orbit/dummy.py` (tidal heating dummy module)
  - 19 tests for `utils/terminate.py` (termination criteria logic)
  - 14 tests for `star/dummy.py` (blackbody stellar physics)
- **Total**: ~174 unit tests (as of 2026-01-11)
- **Current coverage**: 22.42% (2164/8433 lines) — auto-ratcheted fast gate
- **Target**: 30% fast gate coverage milestone

### Test File Created
- `tests/utils/test_helper.py` — 53 comprehensive unit tests covering:
  - `multiple()` — Robust modulo checking (9 tests)
  - `mol_to_ele()` — Molecular formula parsing (9 tests)
  - `natural_sort()` — Natural alphanumeric sorting (7 tests)
  - `CommentFromStatus()` — Status code interpretation (9 tests)
  - `UpdateStatusfile()` — Status file management (3 tests)
  - `CleanDir()` — Directory cleaning with safety (4 tests)
  - `find_nearest()` — Nearest array value finding (4 tests)
  - `recursive_get()` — Nested dictionary access (5 tests)
  - `create_tmp_folder()` — Temporary folder creation (3 tests)

---

## Prioritized Test Targets

Tests are prioritized by **ease of implementation** × **impact on coverage** × **foundational value**.

### PRIORITY 1: Quick Wins (High impact, low effort)

These modules have simple, pure functions with few dependencies. Tests run in <10ms each.

#### 1.1 `utils/helper.py` ✓ **COMPLETED**
- **Status**: 53 unit tests created and passing
- **Coverage**: ~80% of functions tested
- **Impact**: High — utilities used throughout codebase
- **Effort**: Low — pure functions, no external dependencies

#### 1.2 `utils/logs.py` ✓ **COMPLETED**
- **Status**: 41 unit tests created and passing
- **Coverage**: 91.49% of logs.py
- **Impact**: High — logging used throughout all simulation runs
- **Effort**: Low — simple logging utilities

#### 1.3 `config/_converters.py` ✓ **COMPLETED**
- **Status**: 27 unit tests created and passing
- **Coverage**: 100% of _converters.py (4 functions: none_if_none, zero_if_none, dict_replace_none, lowercase)
- **Impact**: High — converters used for all TOML config parsing
- **Effort**: Low — pure conversion functions, no dependencies

#### 1.4 `orbit/dummy.py` ✓ **COMPLETED**

- **Status**: 7 unit tests created and passing
- **Coverage**: 100% of dummy tidal heating logic
- **Scope**: < / > threshold heating, boundary equals zero, no-heat path, Imk2 return value, single-layer handling, phi immutability
- **Effort**: Low — pure arithmetic with simple config mock

---

### PRIORITY 2: Moderate Effort, High Value (2-4 hours each)

These modules have more complex logic but are still testable without heavy computations.

#### 2.1 `utils/terminate.py` ✓ **COMPLETED**
- **Status**: 19 unit tests created and passing
- **Coverage**: All termination criteria (solidification, energy balance, escape, disintegration, time/iteration limits, keepalive)
- **Scope**: Non-strict and strict termination logic with full criterion matrix
- **Effort**: Moderate — config/handler mocks, comprehensive scenario coverage

#### 2.2 `star/dummy.py` ✓ **COMPLETED**
- **Status**: 14 unit tests created and passing
- **Coverage**: Radius scaling (direct + empirical), spectrum generation, luminosity (Stefan-Boltzmann), instellation (inverse-square)
- **Physics**: Solar normalization, temperature dependencies (T⁴), boundary conditions (zero/minimum T)
- **Effort**: Moderate — real constants and numpy/scipy physics

#### 2.3 `interior/dummy.py` ✓ **COMPLETED**
- **Status**: 13 unit tests created and passing
- **Coverage**: calculate_simple_mantle_mass() + run_dummy_int() + melt fraction physics
- **Physics**: Phase boundaries (solid/partial/molten), radiogenic/tidal heating, Interior_t arrays
- **Effort**: Moderate — config mocks, Interior_t(nlev_b=2) instantiation

---

### PRIORITY 2.5: Smoke Tests (Fast Integration Validation)

**Purpose**: End-to-end coupling tests with real binaries but minimal computation (1 timestep, low resolution, <30s each).

**Current status**: 1 smoke test implemented (as of 2026-01-06)
**Target**: 5-7 smoke tests for fast PR checks

These tests validate that modules actually couple together correctly with real compiled binaries (not mocks), catching integration issues that unit tests miss. They run quickly enough for PR CI but use real physics solvers.

#### 2.5.1 `Atmosphere-Interior Coupling` (TARGET: 2-3 tests)
- **Scope**: JANUS/AGNI → SPIDER/dummy interior coupling
- **Tests needed**:
  - Dummy atmos + dummy interior (1 step, verify flux exchange)
  - JANUS + dummy interior (1 step, radiation balance check)
  - AGNI + dummy interior (1 step, verify convergence)
- **Effort**: MODERATE (2-3 hours total)
- **Validation**: F_atm ↔ F_int exchange, T_surf updates, no NaN/Inf

#### 2.5.2 `Volatile Outgassing Coupling` (TARGET: 1-2 tests)
- **Scope**: CALLIOPE ↔ atmosphere module interaction
- **Tests needed**:
  - Outgassing → JANUS (1 step, verify fO2 and volatiles update)
  - Melt-atmosphere equilibration (1 step, check H2O/CO2 partitioning)
- **Effort**: LOW-MODERATE (1-2 hours)
- **Validation**: Volatile masses conserved, pressures physically reasonable

#### 2.5.3 `Stellar Evolution Coupling` (TARGET: 1 test)
- **Scope**: MORS → instellation update
- **Tests needed**:
  - MORS evolution (1 step, verify L_star and spectrum updates)
- **Effort**: LOW (30 min)
- **Validation**: Luminosity changes with time, instellation scales correctly

#### 2.5.4 `Atmospheric Escape Coupling` (TARGET: 1 test)
- **Scope**: ZEPHYRUS → atmosphere mass loss
- **Tests needed**:
  - Escape rate calculation (1 step, verify mass loss from upper atmosphere)
- **Effort**: LOW-MODERATE (1 hour)
- **Validation**: Escape rate > 0 for hot atmospheres, conserves mass

**Implementation strategy**:
1. Start with dummy module combinations (fastest, easiest to debug)
2. Add real module tests one at a time
3. Use low-resolution grids (JANUS: 10 levels, SOCRATES: minimal bands)
4. Mock external data files (stellar spectra, opacities) where possible
5. Each smoke test should complete in <30s

**Coverage impact**: Minimal direct coverage (integration tests don't count toward unit coverage), but critical for validating that unit-tested code actually works together.

---

### PRIORITY 3: Foundation/Core Modules (4-8 hours each)

These are heavily used but require careful test design to isolate logic from physics.

#### 3.1 `utils/coupler.py`
- **Lines**: 979 (LARGE)
- **Key functions**: 15+ functions
- **Target tests**: 30-40 unit tests
- **Effort**: HIGH
  - Coupling orchestration
  - File I/O (helpfile handling)
  - Version checking
  - Module configuration printing

**Strategy**:
- Start with pure utility functions (find_git_revision, version parsing)
- Mock file I/O for helpfile tests
- Test CSV reading/writing with temp files

#### 3.2 `config/` modules (multiple files)
- **`_config.py`**: 400+ lines (core config)
- **`_params.py`**: 298 lines (parameter definitions)
- **`_interior.py`**: 271 lines (interior config)
- **Target**: 50-80 unit tests across all files
- **Effort**: HIGH (but many already exist)
  - Expand existing `test_config.py` with more scenarios
  - Test validators and defaults
  - Parametrize across different config combinations

---

### PRIORITY 4: Lower Priority (Nice to have)

These have fewer users or are harder to test without integration.

#### 4.1 `utils/plot.py`
- **Lines**: 308
- **Functions**: 12+ plotting utilities
- **Target**: 20-25 unit tests
- **Effort**: MODERATE
  - Plot configuration and styling
- **Strategy**: Mock matplotlib, test configuration assembly

#### 4.2 `utils/data.py`
- **Lines**: 707 (LARGE)
- **Functions**: 15+ data management functions
- **Effort**: HIGH
  - External data downloads
  - File I/O
  - Mocking: Network calls, file downloads

---

## Implementation Progress (Jan 11, 2026)

### Completed (187 tests, 22.42% coverage — awaiting CI update)
- ✓ Priority 1.1-1.4: 134 fast unit tests (helper, logs, converters, orbit/dummy)
- ✓ Priority 2.1-2.3: 53 moderate-effort tests (terminate, star/dummy, interior/dummy)
- ✓ Auto-ratcheting working: 18.00% → 22.42% threshold increase confirmed
- ✓ Auto-commit mechanism: github-actions bot commits threshold updates automatically

### Upcoming (Next priorities to reach 30% target)

**Short-term (Unit Tests)**:
1. **Priority 3.1**: `utils/coupler.py` (15-20 tests, 3-4 hrs) → target 25-26% coverage
2. **Priority 2.4**: `utils/plot.py` (15-20 tests, 2-3 hrs) → target 27-28% coverage
3. **Config expansion**: Parametrized tests for edge cases → target 30%+ coverage

**Parallel track (Smoke Tests)**:
4. **Priority 2.5.1**: Atmosphere-interior coupling (2-3 smoke tests, 2-3 hrs)
5. **Priority 2.5.2**: Volatile outgassing tests (1-2 smoke tests, 1-2 hrs)
6. **Priority 2.5.3-4**: Stellar/escape coupling (2 smoke tests, 1.5 hrs)

**Target milestones**:
- Unit test coverage: 30% fast gate (200+ tests)
- Smoke test suite: 5-7 tests covering all major coupling pathways
- Timeline: End of Jan 2026

---

## Testing Principles (Reference)

When implementing tests, follow these guidelines from PROTEUS Copilot Instructions:

### 1. Structure Mirrors Source
```
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

## Coverage Analysis

### Current Modules by Coverage Need

| Module | Lines | Est. Functions | Status | Priority |
|--------|-------|-----------------|--------|----------|
| `utils/helper.py` | 328 | 15 | ✓ DONE (53 tests) | 1 |
| `utils/logs.py` | 150 | 8 | PLANNED | 1 |
| `config/_converters.py` | 100 | 6 | PLANNED | 1 |
| `orbit/dummy.py` | 54 | 1 | PLANNED | 1 |
| `utils/terminate.py` | 291 | 5 | PLANNED | 2 |
| `star/dummy.py` | 147 | 4 | PLANNED | 2 |
| `interior/dummy.py` | 131 | 4 | PLANNED | 2 |
| `utils/coupler.py` | 979 | 15+ | LATER | 3 |
| `config/_params.py` | 298 | 10+ | LATER | 3 |
| `utils/plot.py` | 308 | 12+ | LATER | 4 |
| `utils/data.py` | 707 | 15+ | LATER | 4 |

### Expected Coverage by Milestone

| Milestone | Unit Tests | Est. Coverage | Timeline |
|-----------|-----------|---------------|----------|
| Current | 13 | 18.51% | 2026-01-10 |
| After Priority 1 | 66 | 22-25% | 2026-01-17 |
| After 50% of Priority 2 | 100-120 | 25-28% | 2026-01-20 |
| **Target (30% fast gate)** | **130-150** | **~30%** | **2026-01-24** |
| After all Priority 2 | 150-170 | 30-32% | 2026-01-31 |
| With Priority 3 start | 200+ | 35%+ | 2026-02-07 |

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

## Resources

- [Test Infrastructure Documentation](./test_infrastructure.md)
- [Test Categorization Guide](./test_categorization.md)
- [PROTEUS Copilot Instructions](../.github/copilot-instructions.md)
- [Test conftest.py fixtures](../tests/conftest.py)
