# Test Structure Reference

## Directory Organization

Tests MUST mirror the source code structure exactly.

### Structure Principle

- For every file in `src/<package>/`, create a corresponding `tests/<package>/test_<filename>.py`
- Directory hierarchy must match
- One test file per source file (when possible)

### Example Structure

```
src/proteus/
├── config/
│   ├── __init__.py
│   └── _config.py
├── interior/
│   ├── __init__.py
│   └── wrapper.py
└── plot/
    ├── __init__.py
    └── cpl_global.py

tests/
├── config/
│   ├── __init__.py
│   └── test_config.py
├── interior/
│   ├── __init__.py
│   └── test_wrapper.py
└── plot/
    ├── __init__.py
    └── test_cpl_global.py
```

## Naming Conventions

- **Test files:** `test_<source_filename>.py`
- **Test classes:** `Test<SourceClassName>` (when appropriate)
- **Test functions:** `test_<functionality>` (lowercase with underscores)
- **Test markers:** `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`

## Validation & Restructuring

### Validate Current Structure

```bash
bash tools/validate_test_structure.sh
```

Output: Color-coded report showing:
- Missing test directories
- Missing test files
- Misplaced test files

### Automatic Restructuring

```bash
bash tools/restructure_tests.sh
```

Actions:
- Creates missing directories
- Moves misplaced test files
- Adds `__init__.py` files where needed
- Creates placeholder tests for missing coverage

## Test Discovery

Verify pytest can find all tests:

```bash
pytest --collect-only
```

This lists all discovered tests organized by module.
