# Test Building Strategy for PROTEUS

## Current Status (2026-01-10)

### Coverage Metrics
- **Current unit/smoke tests**: 13 tests marked with `@pytest.mark.unit`
- **New tests added**: 
  - 53 tests for `utils/helper.py` (9 component classes, 40 distinct functions tested)
  - 41 tests for `utils/logs.py` (logging infrastructure)
  - 27 tests for `config/_converters.py` (type conversion utilities)
- **Total**: ~134 unit tests (as of 2026-01-10)
- **Current coverage**: 20.22% (fast gate), 69% target (full gate)
- **Target**: 30% fast gate coverage (by expanding unit tests)

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

#### 1.4 `orbit/dummy.py` (NEW)
- **Lines**: 54
- **Functions**: 1 main function + helper initialization
- **Target tests**: 8-10 unit tests
- **Effort**: LOW
  - Dummy module (simplified physics)
  - Input: config, interior object
  - Output: orbital parameters
  - Easy to mock inputs

**Implementation plan:**
```python
# tests/orbit/test_dummy.py
@pytest.mark.unit
def test_dummy_orbit_returns_zero_eccentricity():
    """Dummy orbit returns e=0 (circular)."""

@pytest.mark.unit
def test_dummy_orbit_preserves_period():
    """Dummy orbit preserves input orbital period."""
```

---

### PRIORITY 2: Moderate Effort, High Value (2-4 hours each)

These modules have more complex logic but are still testable without heavy computations.

#### 2.1 `utils/terminate.py`
- **Lines**: 291
- **Functions**: 5 main functions
- **Target tests**: 20-30 unit tests
- **Effort**: MODERATE
  - Termination criteria checking (solidification, time, volatiles, etc.)
  - Logic-based comparisons
  - Mocking: Mock interior object, config object

**Test categories:**
- Solidification checks
- Time-based termination
- Volatile loss checks
- Maximum iteration limits

#### 2.2 `star/dummy.py`
- **Lines**: 147
- **Functions**: 3-4 key functions
- **Target tests**: 12-15 unit tests
- **Effort**: MODERATE
  - Dummy stellar evolution
  - Input: config, time step
  - Output: stellar radius, luminosity updates
  - Easy mocking

#### 2.3 `interior/dummy.py`
- **Lines**: 131
- **Functions**: 3-4 key functions
- **Target tests**: 10-15 unit tests
- **Effort**: MODERATE
  - Dummy interior model
  - Simple evolution equations
  - Mocking: thermodynamic properties

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

## Implementation Roadmap (Next 2 weeks)

### Week 1 (Jan 13-17)
**Goal**: Add 40-60 more unit tests, reach 25-30% fast gate coverage

1. **Mon (2 hours)**: Implement `test_logs.py` (20-25 tests)
2. **Tue (2 hours)**: Implement `test_converters.py` (15-20 tests)
3. **Wed (1.5 hours)**: Implement `test_orbit_dummy.py` (8-10 tests)
4. **Thu (2 hours)**: Start `test_terminate.py` (first 10-15 tests)
5. **Fri (1 hour)**: Testing, cleanup, coverage analysis

**Expected result**: 100-130 unit tests, ~25-28% coverage

### Week 2 (Jan 20-24)
**Goal**: Build remaining dummy modules, start config expansion

1. **Mon-Tue (4 hours)**: Complete `test_terminate.py` + `test_star_dummy.py`
2. **Wed-Thu (4 hours)**: Expand `test_config.py` with parametrized scenarios
3. **Fri (1 hour)**: Coverage analysis, prepare for nightly integration tests

**Expected result**: 150-170 unit tests, ~30% coverage (target reached)

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
