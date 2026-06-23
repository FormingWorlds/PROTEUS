# Integration Tests for petitRADTRANS and Observe Wrapper

## Overview

Two comprehensive integration test files have been created for testing the observation and spectral synthesis features of PROTEUS:

1. **test_integration_observe_wrapper.py** - Wrapper function and full pipeline tests
2. **test_integration_petitradtrans_helpers.py** - Helper function unit/integration tests

## Test File Locations

Both test files are located in: `/data/rdc49-2/PROTEUS/tests/integration/`

## Test Summary

### test_integration_observe_wrapper.py (8 tests)

Tests for wrapper functions in `src/proteus/observe/wrapper.py` and public functions in `src/proteus/observe/petitRADTRANS.py`:

| Test Name | Purpose | Key Validations |
|-----------|---------|-----------------|
| `test_observe_wrapper_calc_synthetic_spectra` | Tests `calc_synthetic_spectra()` wrapper | Output file creation, spectrum format, physical ranges |
| `test_observe_wrapper_run_observe` | Tests `run_observe()` wrapper | Full observation pipeline, output directory |
| `test_transit_depth_spectrum_format` | Validates transit depth spectra | Wavelength units (μm), depth units (ppm), monotonicity |
| `test_eclipse_depth_spectrum_format` | Validates eclipse depth spectra | Wavelength units (μm), depth units (ppm), monotonicity |
| `test_observe_files_persist` | Tests file I/O consistency | Data persistence across reads, file integrity |
| `test_wrapper_invalid_module` | Tests error handling | ValueError on invalid observe module config |
| `test_observe_directory_structure` | Tests output organization | Directory creation, file naming conventions |
| `test_transit_eclipse_depth_ratio` | Tests physical relationships | Consistency between transit and eclipse spectra |

### test_integration_petitradtrans_helpers.py (13 tests)

Tests for internal helper functions in `src/proteus/observe/petitRADTRANS.py`:

| Test Name | Function Tested | Key Validations |
|-----------|-----------------|-----------------|
| `test_vmrs_to_mass_fractions_basic` | `_vmrs_to_mass_fractions()` | Shape preservation, mass fraction normalization, non-negative values |
| `test_vmrs_to_mass_fractions_normalization` | `_vmrs_to_mass_fractions()` | Handles unnormalized VMR input correctly |
| `test_vmrs_to_mass_fractions_empty` | `_vmrs_to_mass_fractions()` | Edge case: empty input handling |
| `test_vmrs_to_mass_fractions_single_gas` | `_vmrs_to_mass_fractions()` | Single species mass fraction = 1.0 |
| `test_prioritize_broadest_coverage_species` | `_prioritize_broadest_coverage_species()` | Broadest-coverage species moved to front, order preservation |
| `test_get_supported_rayleigh_species` | `_get_supported_rayleigh_species()` | Rayleigh species filtering, flag respect |
| `test_get_supported_cia_species` | `_get_supported_cia_species()` | CIA pair validation, component checking |
| `test_interpolate_prt_profiles_basic` | `_interpolate_prt_profiles()` | Output shape, temperature bounds, monotonicity |
| `test_get_ptr_basic` | `_get_ptr()` | Array extraction, data order |
| `test_get_ptr_with_vmrs` | `_get_ptr()` | VMR handling and ordering |
| `test_get_ptr_pressure_ordering` | `_get_ptr()` | Inverted pressure correction, temperature clipping |
| `test_get_input_data_path` | `_get_input_data_path()` | Path discovery, validation, error handling |
| `test_build_prt_composition_missing_gases` | `_build_prt_composition()` | Graceful handling of sparse gas data |

## Features Tested

### Observation Wrapper (wrapper.py)
- ✅ `calc_synthetic_spectra()` - Compute transit and eclipse spectra
- ✅ `run_observe()` - Full observation orchestration pipeline
- ✅ Output file I/O and format validation
- ✅ Error handling for missing atmosphere data

### Spectral Synthesis (petitRADTRANS.py)
- ✅ `transit_depth()` - Transit spectrum computation
- ✅ `eclipse_depth()` - Eclipse spectrum computation  
- ✅ VMR to mass fraction conversion
- ✅ Gas mixture extraction and filtering
- ✅ Profile interpolation onto log-spaced pressure grids
- ✅ Pressure/temperature/radius extraction
- ✅ Line species and CIA species discovery
- ✅ Broadest-coverage line-species prioritization
- ✅ Input data path resolution

### Validation Coverage
- ✅ Physical range validation (wavelength > 0, depth > 0)
- ✅ Unit validation (wavelengths in μm, depths in ppm)
- ✅ Data consistency and persistence
- ✅ Error handling and edge cases
- ✅ File naming conventions
- ✅ Directory structure creation
- ✅ Composition assembly and gas filtering

## Test Configuration

**Base Configuration**: `tests/integration/dummy.toml`
- All dummy modules for speed
- ~2-3 timesteps to generate atmosphere data
- Disabled plotting and archiving

**Marker**: `@pytest.mark.integration`
- Timeout: 300 seconds (5 minutes)
- Can be run with: `pytest -m integration`

**Physics Validation**: `@pytest.mark.physics_invariant`
- Tests that validate physical laws and constants
- Ensures output is within reasonable ranges

## Running the Tests

### Run all observation/synthesis tests:
```bash
conda activate proteus
cd /data/rdc49-2/PROTEUS
python -m pytest tests/integration/test_integration_observe_wrapper.py \
                  tests/integration/test_integration_petitradtrans_helpers.py -v
```

### Run only helper function tests:
```bash
python -m pytest tests/integration/test_integration_petitradtrans_helpers.py -v
```

### Run only wrapper tests:
```bash
python -m pytest tests/integration/test_integration_observe_wrapper.py -v
```

### Run with physics validation only:
```bash
python -m pytest tests/integration/ -m physics_invariant -v
```

### Run specific test:
```bash
python -m pytest tests/integration/test_integration_observe_wrapper.py::test_transit_depth_spectrum_format -v
```

## Test Dependencies

- **pytest** - Testing framework
- **numpy** - Array operations and validation
- **proteus** - Main PROTEUS package
- **petitRADTRANS** - Spectral synthesis (optional for helper tests)

## Expected Runtime

- **Helper tests** (13 tests): ~20-30 seconds (mostly unit tests with no PROTEUS run required)
- **Wrapper tests** (8 tests): ~30-60 seconds per test (requires PROTEUS simulation)
- **Total**: ~3-6 minutes for full test suite

## Files Created

1. `/data/rdc49-2/PROTEUS/tests/integration/test_integration_observe_wrapper.py`
   - Lines: 452
   - Tests: 8
   - Fixtures: 1

2. `/data/rdc49-2/PROTEUS/tests/integration/test_integration_petitradtrans_helpers.py`
   - Lines: 541
   - Tests: 13
   - Fixtures: 1

**Total**: 21 integration tests, ~1000 lines of test code

## Test Coverage

- **Public API**: 2 functions tested (`calc_synthetic_spectra`, `run_observe`)
- **Spectral computation**: 2 functions tested (`transit_depth`, `eclipse_depth`)
- **Helper functions**: 9 functions tested (VMR conversion, mixing, interpolation, etc.)
- **Output validation**: File I/O, format, physical ranges, error handling
- **Configuration variations**: Multiple OBS_SOURCES, module combinations
- **Error conditions**: Invalid configs, missing data, edge cases

## Notes

- Tests use the integration marker for CI/CD categorization
- Fixture creates temporary output directories to avoid polluting the workspace
- Tests skip gracefully when required modules/data are unavailable
- All tests validate physical correctness in addition to software functionality
- Error messages include source/context for easier debugging
