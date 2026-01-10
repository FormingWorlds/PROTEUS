# Unit Test Building Strategy for PROTEUS

## Achievement Summary

**53 NEW UNIT TESTS CREATED** for `utils/helper.py` — all passing ✓

This document outlines the complete strategy for expanding PROTEUS unit test coverage from 18.51% → 30%+ through systematic, prioritized test development.

---

## Current Status (2026-01-10)

### Metrics
- Starting point: 13 unit tests (18.51% coverage)
- **New tests added: 53** for `utils/helper.py`
- Total unit tests: ~66
- Target: 130-150 tests for 30% coverage

### What Was Built
File: `tests/utils/test_helper.py`

9 test classes, 53 comprehensive unit tests covering:
- `multiple()` — 9 tests (robust divisibility checking)
- `mol_to_ele()` — 9 tests (molecular formula parsing)
- `natural_sort()` — 7 tests (natural alphanumeric sorting)
- `CommentFromStatus()` — 9 tests (status code interpretation)
- `UpdateStatusfile()` — 3 tests (status file I/O)
- `CleanDir()` — 4 tests (directory management with git safety)
- `find_nearest()` — 4 tests (nearest array value)
- `recursive_get()` — 5 tests (nested dict access)
- `create_tmp_folder()` — 3 tests (temp directory creation)

**Key Achievement**: Established test pattern that mirrors source code structure exactly, following PROTEUS ecosystem standards.

---

## Prioritized Test Building Roadmap

Tests are prioritized by: **Implementation ease × Coverage impact × Foundation value**

### PRIORITY 1: Quick Wins (1-3 hours each, high value)

These have simple, isolated logic with minimal dependencies. Target: 20-30 tests each.

#### 1.1 ✓ COMPLETED: `utils/helper.py`
- Status: 53 unit tests created, all passing
- Coverage: 80%+ of testable functions
- Value: HIGH — foundational utilities used throughout

#### 1.2 NEXT: `utils/logs.py`
- Estimated tests: 20-25
- Functions: setup_logger, GetCurrentLogfileIndex, GetLogfilePath, StreamToLogger, CustomFormatter
- Effort: LOW — pure logging utilities, minimal dependencies
- **Implementation**: Write to temp log files, verify log structure and content

#### 1.3 NEXT: `config/_converters.py`
- Estimated tests: 15-20
- Functions: type conversion, enum parsing, boolean conversion
- Effort: LOW — pure conversion functions
- **Implementation**: Parametrized tests for all conversion cases

#### 1.4 NEXT: `orbit/dummy.py`
- Estimated tests: 8-10
- Function: run_dummy_orbit() → returns dummy orbital parameters
- Effort: LOW — simple dummy module with predictable behavior
- **Implementation**: Mock config and interior objects, verify output format

#### 1.5 NEXT: `star/dummy.py`
- Estimated tests: 12-15
- Functions: Dummy stellar evolution (radius, luminosity, age updates)
- Effort: LOW — simple physical model
- **Implementation**: Mock config, verify parameter updates

### PRIORITY 2: Moderate Effort (2-4 hours each)

#### 2.1 `utils/terminate.py`
- Estimated tests: 20-30
- Functions: check_termination() and related criteria checks
- Effort: MODERATE — logic-heavy, multiple conditions
- **Implementation**: Mock interior/atmosphere objects, test each termination path

#### 2.2 `interior/dummy.py`
- Estimated tests: 10-15
- Function: run_dummy_interior() → dummy cooling/evolution
- Effort: MODERATE — simple thermo model
- **Implementation**: Mock config, verify thermal evolution logic

#### 2.3 `utils/coupler.py` (partial)
- Estimated tests: 20-30 (of 15+ functions)
- Functions: ReadHelpfileFromCSV, WriteHelpfileToCSV, version checking
- Effort: MODERATE — file I/O, CSV parsing
- **Implementation**: Use tempfiles, test CSV read/write roundtrips

### PRIORITY 3: Higher Effort (4-8 hours)

#### 3.1 Expand `tests/config/test_config.py`
- Target: 20-30 additional tests (currently ~5)
- Focus: Config validation, type conversions, default values
- Effort: HIGH but necessary — core module
- **Implementation**: Parametrize across config scenarios

#### 3.2 `utils/plot.py` (partial)
- Estimated tests: 15-20 (of 12+ functions)
- Focus: Plot configuration, color schemes, axis labels
- Effort: MODERATE — mock matplotlib, test config logic

#### 3.3 `utils/data.py` (partial)
- Estimated tests: 10-15 (of 15+ functions)
- Focus: Data validation, file path construction
- Effort: HIGH — network/file I/O heavy, extensive mocking needed

---

## Test Structure Pattern (Established)

**All new tests follow this structure:**

```python
"""
Unit tests for proteus.MODULE.SUBMODULE module.

Tests [DESCRIPTION] with [TESTABLE ASPECT].
"""
from __future__ import annotations

import pytest
from proteus.MODULE.SUBMODULE import function_to_test


class TestFunctionGroup:
    """Group related tests in classes."""

    @pytest.mark.unit
    def test_basic_functionality(self):
        """Test core behavior."""
        result = function_to_test(input_value)
        assert result == expected_value

    @pytest.mark.unit
    def test_edge_case(self):
        """Test boundary conditions."""
        result = function_to_test(edge_input)
        assert result == expected_edge_value
```

**Key Rules**:
- Mark all tests with `@pytest.mark.unit`
- Group related tests in classes
- Use descriptive docstrings explaining WHAT and WHY
- Target <100ms per test (no heavy computations)
- Mock external dependencies (file I/O, network, heavy physics)

---

## Coverage Expansion Timeline

### Week 1: Consolidate Quick Wins (Jan 13-17)
**Goal**: Reach 25-28% coverage with ~120 tests

| Day | Task | Est. Tests | Duration |
|-----|------|-----------|----------|
| Mon | `utils/logs.py` | +20 | 2 hrs |
| Tue | `config/_converters.py` | +18 | 2 hrs |
| Wed | `orbit/dummy.py` | +10 | 1.5 hrs |
| Thu | `star/dummy.py` (partial) | +12 | 2 hrs |
| Fri | Testing + Analysis | — | 1 hr |

**Result**: 120 unit tests, 25-28% coverage

### Week 2: Complete Priority 1 & Start Priority 2 (Jan 20-24)
**Goal**: Reach 30% coverage target with ~140-150 tests

| Day | Task | Est. Tests | Duration |
|-----|------|-----------|----------|
| Mon | `star/dummy.py` (complete) | +5 | 1 hr |
| Tue | `interior/dummy.py` | +12 | 2 hrs |
| Wed | `utils/terminate.py` (partial) | +15 | 3 hrs |
| Thu | `utils/coupler.py` (partial) | +10 | 2 hrs |
| Fri | Coverage analysis, cleanup | — | 1 hr |

**Result**: 150+ unit tests, **30% coverage achieved** ✓

### Weeks 3-4: Continued Growth (Jan 27 - Feb 7)
**Goal**: Push toward 35% with 170-200+ tests

- Expand config tests (20+ additional)
- Complete `utils/coupler.py` (remaining functions)
- Start `utils/data.py` (selected functions)
- Begin plotting utilities tests

---

## Quick Start: Adding Tests

### Create a new test file
```bash
# Follow the structure: src/proteus/X/Y.py → tests/X/test_Y.py
touch tests/utils/test_logs.py
```

### Start writing tests
```python
"""Unit tests for proteus.utils.logs module."""
from __future__ import annotations

import tempfile
import pytest
from proteus.utils.logs import setup_logger


class TestSetupLogger:
    @pytest.mark.unit
    def test_creates_log_file(self):
        """Logger setup creates log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logger(logpath=f"{tmpdir}/test.log")
            assert os.path.exists(f"{tmpdir}/test.log")
```

### Run and verify
```bash
# Run the tests
pytest tests/utils/test_logs.py -v

# Check coverage for that module
pytest tests/utils/test_logs.py --cov=src/proteus/utils/logs
```

### Commit
```bash
git add tests/utils/test_logs.py
git commit -m "test(utils): add unit tests for logs module (25 tests)"
git push
```

---

## Best Practices Applied

### 1. No External Dependencies in Unit Tests
```python
# ✗ WRONG: Makes real files
def test_reading_data():
    data = read_real_file('input/data.toml')  # Fragile!

# ✓ RIGHT: Mocks file I/O
def test_reading_data():
    from unittest.mock import mock_open, patch
    with patch('builtins.open', mock_open(read_data='[config]')):
        data = parse_config('[config]')
```

### 2. Parametrized Tests Reduce Duplication
```python
# ✗ WRONG: Nine nearly identical test functions
def test_status_0(): assert CommentFromStatus(0) == 'Started'
def test_status_1(): assert CommentFromStatus(1) == 'Running'
# ... 7 more ...

# ✓ RIGHT: One parametrized test
@pytest.mark.parametrize('status,expected', [
    (0, 'Started'), (1, 'Running'), ...
])
def test_comment_from_status(status, expected):
    assert CommentFromStatus(status) == expected
```

### 3. Floating-Point Comparisons
```python
# ✗ WRONG: Fails due to floating-point precision
assert find_nearest([1.0, 2.0, 3.0], 2.05)[0] == 2.0

# ✓ RIGHT: Robust comparison
assert find_nearest([1.0, 2.0, 3.0], 2.05)[0] == pytest.approx(2.0)
```

### 4. Physical Reasonability in Edge Cases
```python
# Test status codes that represent real scenarios
# Status 10 = solidified (core cool enough to crystallize)
# Status 15 = volatiles escaped (entire atmosphere lost)
# Status 20 = generic error

# Include explanatory comments for "why this test"
def test_volatile_escape_status():
    """Status 15: planet lost all volatiles.

    This can occur in:
    - Ultra-hot planets with extreme stellar irradiation
    - Super-Earths after long-term XUV erosion
    """
```

---

## Module Coverage Status

| Module | Est. LOC | Functions | Tests Needed | Status |
|--------|----------|-----------|-------------|--------|
| `utils/helper.py` | 328 | 15 | 53 | ✓ DONE |
| `utils/logs.py` | 150 | 8 | 20-25 | NEXT |
| `config/_converters.py` | 100 | 6 | 15-20 | NEXT |
| `orbit/dummy.py` | 54 | 1 | 8-10 | NEXT |
| `star/dummy.py` | 147 | 4 | 12-15 | PLANNED |
| `interior/dummy.py` | 131 | 4 | 10-15 | PLANNED |
| `utils/terminate.py` | 291 | 5 | 20-30 | PLANNED |
| `utils/coupler.py` | 979 | 15+ | 20-30 | LATER |
| `config/_params.py` | 298 | 10+ | 15-25 | LATER |
| `config/_interior.py` | 271 | 10+ | 15-20 | LATER |

---

## Resources & References

- **Test Infrastructure**: `docs/test_infrastructure.md`
- **Test Categorization**: `docs/test_categorization.md`
- **Copilot Instructions**: `.github/copilot-instructions.md`
- **Conftest Fixtures**: `tests/conftest.py`
- **Implementation Example**: `tests/utils/test_helper.py` (53 tests)

---

## Success Metrics

| Milestone | Timeline | Unit Tests | Est. Coverage |
|-----------|----------|-----------|---------------|
| Current | 2026-01-10 | 13 | 18.51% |
| After Priority 1a | 2026-01-15 | 66 | 22-25% |
| After Priority 1b | 2026-01-17 | 120 | 25-28% |
| **30% Target** | **2026-01-24** | **150+** | **30%** |
| After Priority 2 | 2026-01-31 | 180-200 | 32-35% |
| Full Priority 1+2 | 2026-02-07 | 200+ | 35%+ |

---

## Next Immediate Step

**Build `tests/utils/test_logs.py`** (20-25 tests)

This will be followed by systematic expansion through the prioritized modules, targeting the 30% coverage goal by January 24.
