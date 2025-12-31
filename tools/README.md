# PROTEUS Tools

This directory contains utility scripts and tools for PROTEUS development.

## Available Tools

### `validate_test_structure.sh`

**Purpose:** Validate that the `tests/` directory properly mirrors the `src/proteus/` structure.

**What it does:**
1. Checks for missing test directories
2. Verifies test files exist in each directory
3. Ensures `__init__.py` files are present
4. Provides a summary report with colored output

**Usage:**
```bash
# From repository root
bash tools/validate_test_structure.sh
```

**Example output:**
```
🔍 Validating test structure...

Checking for missing test directories...
✓ Found: tests/config
✗ Missing: tests/escape (for src/proteus/escape)
✓ Found: tests/grid

Summary:
  Test directories found: 10
  Test directories missing: 3
  __init__.py files missing: 2

⚠ Run 'bash tools/restructure_tests.sh' to fix issues
```

**Exit codes:**
- `0`: All checks passed
- `1`: Issues found (missing directories or __init__.py files)

### `restructure_tests.sh`

**Purpose:** Restructure the `tests/` directory to mirror the `src/proteus/` structure.

**What it does:**
1. Creates missing test directories for all source modules
2. Moves misplaced test files to appropriate subdirectories
3. Creates placeholder test files for untested modules
4. Adds `__init__.py` files for proper Python package structure

**Usage:**
```bash
# From repository root
bash tools/restructure_tests.sh
```

**Before:**
```
tests/
├── conftest.py
├── grid/
├── inference/
├── integration/
├── test_cli.py
├── test_config.py
├── test_cpl_colours.py
└── test_cpl_helpers.py
```

**After:**
```
tests/
├── conftest.py
├── atmos_chem/
│   └── test_atmos_chem.py
├── atmos_clim/
│   └── test_atmos_clim.py
├── config/
│   └── test_config.py
├── escape/
│   └── test_escape.py
├── grid/
│   └── test_grid.py
├── inference/
│   └── test_inference.py
├── interior/
│   └── test_interior.py
├── observe/
│   └── test_observe.py
├── orbit/
│   └── test_orbit.py
├── outgas/
│   └── test_outgas.py
├── plot/
│   ├── test_cpl_colours.py
│   └── test_cpl_helpers.py
├── star/
│   └── test_star.py
├── utils/
│   └── test_utils.py
├── integration/
│   └── ... (unchanged)
├── test_cli.py (stays at root)
└── test_init.py (stays at root)
```

**Safe to run multiple times:** The script checks for existing files before moving them.

### `coverage_analysis.sh`

**Purpose:** Analyze test coverage by module and identify testing priorities.

**What it does:**
1. Runs pytest with coverage
2. Shows coverage percentage for each module
3. Color-codes results (green ≥80%, yellow ≥50%, red <50%)
4. Lists priority modules needing tests
5. Shows overall coverage summary

**Usage:**
```bash
# From repository root
bash tools/coverage_analysis.sh
```

**Example output:**
```
🔍 Analyzing test coverage by module...

Running tests with coverage...

==========================================
Coverage by Module:
==========================================
✓ src/proteus/config/__init__.py: 85%
⚠ src/proteus/interior/common.py: 65%
✗ src/proteus/observe/observe.py: 25%

==========================================
Priority Modules (Coverage < 50%):
==========================================
- src/proteus/observe/observe.py (25%)
- src/proteus/escape/wrapper.py (30%)

==========================================
Overall Coverage:
==========================================
TOTAL: 58%

💡 Tips:
  - View detailed report: open htmlcov/index.html
  - Test specific module: pytest tests/[module]/
  - Check missing lines: coverage report --show-missing
```

**Prerequisites:**
- `coverage[toml]` must be installed
- Tests should be runnable with pytest

## Contributing

When adding new tools:
1. Make scripts executable: `chmod +x tools/your_script.sh`
2. Add documentation to this README
3. Include help text in the script: `your_script.sh --help`
