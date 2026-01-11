# Test Building Strategy for PROTEUS

## Current Status (2026-01-11 - Updated Evening)

### Coverage Metrics

- **Baseline unit/smoke tests**: 13 tests marked with `@pytest.mark.unit`
- **New tests added**:
  - 53 tests for `utils/helper.py` (9 component classes, 40 distinct functions)
  - 41 tests for `utils/logs.py` (logging infrastructure)
  - 27 tests for `config/_converters.py` (type conversion utilities)
  - 89 tests for `config` validators across `_config.py`, `_interior.py`, `_params.py`, `_observe_validators.py`, `_outgas_validators.py` (comprehensive validation coverage)
  - 7 tests for `orbit/dummy.py` (tidal heating dummy module)
  - 19 tests for `utils/terminate.py` (termination criteria logic)
  - 14 tests for `star/dummy.py` (blackbody stellar physics)
  - 13 tests for `interior/dummy.py` (interior thermal evolution)
  - 55 tests for `utils/coupler.py` (helpfile I/O, version getters, print functions)
  - 9 tests for `atmos_chem/wrapper.py` (VULCAN chemistry module interface)
  - 20 tests for `atmos_clim/wrapper.py` (dummy atmosphere wrapper)
  - 25 tests for `escape/wrapper.py` (ZEPHYRUS escape module interface)
  - 12 tests for `interior/wrapper.py` (interior module interface)
  - 10 tests for `observe/wrapper.py` (observation module interface)
  - 60 tests for `outgas/wrapper.py` (CALLIOPE outgassing interface)
  - 1 smoke test for `atmosphere-interior coupling` (dummy modules, ~2s)
- **Total**: **457 tests** (456 unit + 1 smoke)
- **Current coverage**: **~29.5%** (passing CI as of run 20898076292)
- **Target**: 30% fast gate coverage + 5-7 smoke tests
- **Status**: 🎯 **On track** — need ~10-15 more unit tests to reach 30%

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

#### 2.4 `utils/coupler.py` ✓ **COMPLETED (NEW)**

- **Status**: 55 unit tests created and passing (36 initial + 19 additional)
- **Coverage**: Helpfile operations (creation, row extension, CSV I/O), lock file management, state printing, version getters (`_get_spider_version`, `_get_petsc_version`, `_get_git_revision`, `_get_socrates_version`, `_get_agni_version`, `_get_julia_version`), print functions (`print_header`, `print_stoptime`, `print_system_configuration`)
- **Scope**: Complete helpfile lifecycle, version validation infrastructure, runtime diagnostics, edge case handling (missing keys, negative fluxes, large time values)
- **Physics validation**: Realistic Earth values, extreme exoplanet parameters, float precision preservation, scientific notation consistency, negative flux handling
- **Effort**: Low-Moderate — pure I/O + mocked subprocess calls for version checking

---

### PRIORITY 2.5: Smoke Tests (Fast Integration Validation)

**Purpose**: End-to-end coupling tests with real binaries but minimal computation (1 timestep, low resolution, <30s each).

**Current status**: 2 smoke tests implemented (1 baseline + 1 new)
**Target**: 5-7 smoke tests for fast PR checks

These tests validate that modules actually couple together correctly with real compiled binaries (not mocks), catching integration issues that unit tests miss. They run quickly enough for PR CI but use real physics solvers.

#### 2.5.1 `Atmosphere-Interior Coupling` (TARGET: 2-3 tests) **IN PROGRESS**

- **Implemented**: `test_smoke_dummy_atmos_dummy_interior_flux_exchange` (✓ PASSING, ~2s)
  - Validates flux exchange (F_atm ↔ F_int) with dummy modules
  - Tests T_surf updates and physical value ranges
  - Runtime: 2.07s (excellent for PR CI)
- **Still needed**:
  - JANUS + dummy interior (1 step, radiation balance check) **NEXT**
  - AGNI + dummy interior (1 step, verify convergence)
- **Effort**: MODERATE (1.5-2 hrs remaining)
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

#### 3.1 `config/` modules ✓ **SUBSTANTIALLY COMPLETED**

- **`_config.py`**: 400+ lines (core config)
- **`_params.py`**: 298 lines (parameter definitions)
- **`_interior.py`**: 271 lines (interior config)
- **`_observe_validators.py`**: NEW — observation config validation
- **`_outgas_validators.py`**: NEW — outgassing config validation
- **Completed**: 89 validator tests covering:
  - Core config: instellation/escape combinations, directory paths, module selection (27 tests)
  - Interior validators: energy/temperature guards, module compatibility (8 tests)
  - Params validators: path existence, modulus/max-min checks, physical ranges (10 tests)
  - **NEW**: Observe validators: comprehensive coverage of observation config edge cases (24 tests in `test_observe_validators.py`)
  - **NEW**: Outgas validators: full matrix coverage of outgassing scenarios including volcanism, solvevol, prevent_warming interactions (20 tests in `test_outgas_validators.py`)
- **Remaining**: 10-20 additional tests for dataclass defaults and edge-case TOML parsing
- **Effort**: LOW (mostly complete)

#### 3.2 `atmos_clim/wrapper.py` ✓ **COMPLETED**

- **Status**: 20 unit tests created and passing
- **Coverage**: Complete wrapper interface for dummy atmosphere module
- **Scope**: Fixed surface mode, transparent/opaque atmosphere, skin temperature convergence, prevent warming logic, scale height, zenith angle effects, albedo calculation, output key validation
- **Physics validation**: Realistic Earth scenarios, extreme exoplanet cases, temperature equilibration, radiative balance
- **Effort**: COMPLETED

#### 3.3 `atmos_chem/wrapper.py` ✓ **COMPLETED**

- **Status**: 9 unit tests created and passing
- **Coverage**: VULCAN chemistry module interface
- **Scope**: Module disabled checks, file I/O operations, result parsing, config preservation, realistic atmospheric chemistry scenarios
- **Validation**: DataFrame structure, whitespace preservation, error handling
- **Effort**: COMPLETED

---

### PRIORITY 3.5: Module Wrappers ✓ **COMPLETED**

These modules interface with external physics solvers and require careful mocking.

#### 3.5.1 `escape/wrapper.py` ✓ **COMPLETED**

- **Status**: 25 unit tests created and passing
- **Coverage**: ZEPHYRUS escape module interface (run_escape, run_zephyrus, calc_new_elements)
- **Scope**: Tidal heating contributions, elemental inventory updates (bulk/outgas reservoirs), mass conservation, escape rate calculations, invalid module handling
- **Physics validation**: Hot Jupiter scenarios, Earth-like planets, escape efficiency, elemental fractionation
- **Effort**: COMPLETED

#### 3.5.2 `interior/wrapper.py` ✓ **COMPLETED**

- **Status**: 12 unit tests created and passing
- **Coverage**: Interior module interface (run_interior dispatcher, dummy/spider/aragog modules)
- **Scope**: Module selection validation, solver integration, error handling for invalid modules
- **Validation**: Dispatcher logic, module-specific parameter passing
- **Effort**: COMPLETED

#### 3.5.3 `observe/wrapper.py` ✓ **COMPLETED**

- **Status**: 10 unit tests created and passing
- **Coverage**: Observation module interface (run_observe, file I/O, result parsing)
- **Scope**: Module disabled checks, header/data parsing, spectral output handling, error cases
- **Validation**: File structure, numeric precision, missing file handling
- **Effort**: COMPLETED

#### 3.5.4 `outgas/wrapper.py` ✓ **COMPLETED**

- **Status**: 60 unit tests created and passing
- **Coverage**: CALLIOPE outgassing interface (comprehensive validation of all major functions)
- **Scope**: 13 validator functions covering module compatibility, pressure/temperature ranges, mass inventory management, fO2/H2O relationships, melt fraction physics, prevent_warming logic
- **Physics validation**: Realistic magma ocean scenarios, extreme degassing conditions, thermodynamic consistency, mass conservation
- **Effort**: COMPLETED — most comprehensive wrapper test suite

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

## Implementation Progress (Jan 11, 2026 - Evening Update)

### Completed (457 tests, **29.5% coverage** ✓)

- ✓ Priority 1.1-1.4: 134 fast unit tests (helper, logs, converters, orbit/dummy)
- ✓ Priority 2.1-2.4: 108 moderate-effort tests (terminate, star/dummy, interior/dummy, coupler)
- ✓ Priority 3.1: **89 config validator tests** (comprehensive coverage of all validator modules)
  - Core config: 27 tests
  - Interior validators: 8 tests  
  - Params validators: 10 tests
  - **NEW**: Observe validators: 24 tests (`test_observe_validators.py`)
  - **NEW**: Outgas validators: 20 tests (`test_outgas_validators.py`)
- ✓ Priority 3.2-3.3: **29 wrapper tests** (atmos_clim, atmos_chem)
- ✓ Priority 3.5: **107 module wrapper tests** (escape, interior, observe, outgas)
  - escape/wrapper: 25 tests
  - interior/wrapper: 12 tests
  - observe/wrapper: 10 tests
  - outgas/wrapper: 60 tests (most comprehensive)
- ✓ Priority 2.5.1 (partial): 1 smoke test (dummy atmos + dummy interior)
- ✓ **Auto-ratcheting working**: 18.00% → 22.42% → 23.03% → **29.52%** threshold increases
- ✓ Auto-commit mechanism: github-actions bot commits threshold updates automatically
- ✓ All CI checks passing (Code Quality + Unit Tests + Smoke Tests)
- ✓ Ruff formatting issues resolved

### Upcoming (Final push to 30%+ target)

**Current Status**: 29.5% coverage, 457 tests — **need ~10-15 more unit tests to reach 30%**

**Immediate priorities (to cross 30% threshold)**:

1. **Config dataclass defaults** (Priority 3.1 remaining): 10-15 tests for default value initialization, TOML edge cases, cross-field validation paths → **+0.3-0.5% coverage**
2. **Atmos_clim common functions** (Priority 3.2 extension): 5-10 tests for read_atmosphere_data, physical validators, specification parsing → **+0.2-0.3% coverage**
3. **Utils expansion** (Priority 4 selected): 5-10 tests for high-impact utility functions in plot.py or data.py → **+0.2-0.3% coverage**

**Estimated effort**: 2-3 hours to reach **30.0%+** coverage

**Smoke test track (parallel, lower priority)**:

1. **Priority 2.5.1**: Fix dummy atmos + interior physics inputs and unskip test
2. **Priority 2.5.2-4**: Add JANUS+dummy, MORS evolution, ZEPHYRUS escape smoke tests (1-2 hours each)

**Target milestones**:

- ✓ Unit test coverage: 29.5% achieved
- ⏳ **30.0%+ fast gate** (470+ tests total) — **2-3 hours remaining**
- ⏳ Smoke test suite: 3-5 active tests for fast PR checks
- Timeline: **Week of Jan 13, 2026** for 30%+ completion

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

## Coverage Analysis

### Current Modules by Coverage Need

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
| `utils/plot.py` | 308 | 12+ | REMAINING | - | 4 |
| `utils/data.py` | 707 | 15+ | REMAINING | - | 4 |

### Expected Coverage by Milestone

| Milestone | Unit Tests | Est. Coverage | Timeline |
| --------- | ---------- | ------------- | -------- |
| Baseline | 13 | 18.51% | 2026-01-10 |
| After Priority 1 | 134 | 22.42% | 2026-01-11 (AM) |
| After Priority 2 | 242 | 23.03% | 2026-01-11 (Mid) |
| After Priority 3 (partial) | 457 | **29.52%** | **2026-01-11 (PM)** |
| **Target (30% fast gate)** | **470+** | **30.0%+** | **2026-01-13** |
| After all Priority 3 | 500+ | 32-35% | 2026-01-20 |
| With Priority 4 start | 600+ | 38%+ | 2026-01-31 |

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
- [Test Building Guide](./test_building.md)
- [PROTEUS Copilot Instructions](../.github/copilot-instructions.md)
- [Test conftest.py fixtures](../tests/conftest.py)
