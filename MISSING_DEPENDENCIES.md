# Dependencies for Full Functionality Testing

## Summary

**Status:** ✅ **ALL DEPENDENCIES NOW AVAILABLE**

All required dependencies have been installed in the miniforge environment.
Full download functionality testing is now possible.

## Required Dependencies

### 1. **osfclient** ✅ INSTALLED
- **Purpose**: OSF (Open Science Framework) downloads and fallback functionality
- **Used in**: `download_OSF_folder()`, OSF fallback in `download()` function
- **Import**: `from osfclient.api import OSF`
- **Install**: `pip install osfclient`
- **Status**: ✅ Available in miniforge environment (version 0.0.5)
- **Note**: Should be added to `pyproject.toml` dependencies

### 2. **zenodo-client** ✅ INSTALLED
- **Purpose**: Primary Zenodo download method using official client library
- **Used in**: `download_zenodo_folder_client()`, `download_zenodo_file_client()`
- **Import**: `from zenodo_client import Zenodo`
- **Install**: `pip install zenodo-client`
- **Status**: ✅ Installed in miniforge environment (version 0.4.1)
- **Note**: Already in `pyproject.toml` (line 71)

### 3. **pystow** ✅ INSTALLED
- **Purpose**: Configuration management for zenodo_client (API token storage)
- **Used in**: Token checking, zenodo_client configuration
- **Import**: `import pystow` (optional, used when available)
- **Install**: `pip install pystow`
- **Status**: ✅ Installed in miniforge environment (version 0.7.15)
- **Note**: Optional dependency but recommended for token management

### 4. **juliacall** ✅ INSTALLED
- **Purpose**: Julia integration for PROTEUS core functionality
- **Used in**: Main `proteus` package import
- **Import**: `from juliacall import Main as jl`
- **Install**: `pip install juliacall`
- **Status**: ✅ Available in miniforge environment (version 0.9.24)
- **Note**: Already in `pyproject.toml` (line 55)

## Installation Commands

**All dependencies are now installed in the miniforge environment.**

To install in a different environment:

```bash
# Core dependencies for download functionality
pip install osfclient zenodo-client pystow

# For full PROTEUS import (if needed)
pip install juliacall
```

Or install from the project:

```bash
# Install all project dependencies
pip install -e .

# Or install specific packages
pip install osfclient zenodo-client pystow
```

## Test Results

**Full download tests completed successfully:**
- ✅ Module imports: All 5 functions available
- ✅ osfclient: Imported successfully
- ✅ zenodo_client: Imported successfully
- ✅ pystow: Imported successfully
- ✅ Actual download with mapping: Downloaded 27 files (4.7 MB)
- ✅ Single file download: Downloaded _readme.md (983 bytes)
- ✅ Force parameter: Working correctly

**All 7 tests passed!**

## Impact on Testing

### Without These Dependencies:

1. **osfclient**:
   - ❌ Cannot test OSF fallback functionality
   - ❌ OSF downloads will fail
   - ✅ Zenodo downloads still work

2. **zenodo-client**:
   - ❌ Primary download method unavailable
   - ✅ Falls back to `zenodo_get` command-line tool
   - ✅ Falls back to web-based downloads
   - ⚠️ Missing the most robust download method

3. **pystow**:
   - ⚠️ zenodo_client cannot use API token from config
   - ✅ Falls back to environment variable or public access
   - ⚠️ May have rate limiting issues without token

4. **juliacall**:
   - ❌ Cannot import full `proteus` package
   - ✅ Can still test `proteus.utils.data` module directly
   - ✅ Current tests work around this limitation

## Current Test Status

The test suite (`test_download_functionality.py`) currently:
- ✅ Tests Zenodo API access (works without dependencies)
- ✅ Tests curl downloads with --fail flag (works)
- ✅ Tests code structure and patterns (works)
- ⚠️ Cannot test full OSF integration (needs osfclient)
- ⚠️ Cannot test zenodo_client integration (needs zenodo-client)
- ⚠️ Cannot test with API tokens (needs pystow)

## Recommendations

1. **For Basic Testing**: Current setup is sufficient
   - Tests code structure ✅
   - Tests Zenodo web downloads ✅
   - Tests curl --fail flag ✅

2. **For Full Integration Testing**: Install missing dependencies
   ```bash
   pip install osfclient zenodo-client pystow
   ```

3. **For Complete PROTEUS Testing**: Install all dependencies
   ```bash
   pip install -e .
   ```

## Adding Missing Dependencies to pyproject.toml

Consider adding:
- `osfclient` - Required for OSF downloads
- `pystow` - Optional but recommended for zenodo_client token management

These are already in pyproject.toml:
- ✅ `zenodo-client` (line 71)
- ✅ `juliacall` (line 55)
