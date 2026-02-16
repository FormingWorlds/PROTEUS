# PROTEUS Tools

This directory contains utility scripts and tools for PROTEUS development, configuration management, data retrieval, and testing.

## Testing & Quality Assurance

### validate_test_structure.sh

**Purpose:** Validate that the `tests/` directory properly mirrors the `src/proteus/` structure.

**What it does:**
- Checks for missing test directories
- Verifies test files exist in each directory
- Ensures `__init__.py` files are present
- Provides a summary report with colored output

**Usage:**

```bash
# From repository root
bash tools/validate_test_structure.sh
```

**Exit codes:**
- `0`: All checks passed
- `1`: Issues found (missing directories or __init__.py files)

### restructure_tests.sh

**Purpose:** Restructure the `tests/` directory to mirror the `src/proteus/` structure.

**What it does:**
- Creates missing test directories for all source modules
- Moves misplaced test files to appropriate subdirectories
- Creates placeholder test files for untested modules
- Adds `__init__.py` files for proper Python package structure

**Usage:**

```bash
# From repository root
bash tools/restructure_tests.sh
```

**Safe to run multiple times:** The script checks for existing files before moving them.

### coverage_analysis.sh

**Purpose:** Analyze test coverage by module and identify testing priorities.

**What it does:**
- Runs pytest with coverage
- Shows coverage percentage for each module
- Color-codes results (green ≥80%, yellow ≥50%, red <50%)
- Lists priority modules needing tests
- Shows overall coverage summary

**Usage:**

```bash
# From repository root
bash tools/coverage_analysis.sh
```

**Prerequisites:**
- `coverage[toml]` must be installed
- Tests should be runnable with pytest

### update_coverage_threshold.py

**Purpose:** Automatically ratchet (increase) the coverage threshold when test coverage improves on the main branch.

**What it does:**
- Reads current test coverage from `coverage.json`
- Compares against the threshold in `pyproject.toml`
- If coverage is higher, automatically updates `pyproject.toml`
- Creates a git commit with the new threshold
- Enforces the ratcheting mechanism (never decreases)

**Usage:**

```bash
# Usually called by CI/CD pipeline
python tools/update_coverage_threshold.py
```

**Requirements:**
- `tomllib` (Python 3.11+) or `tomli` package
- `tomlkit` (≥0.11.0) for preserving TOML formatting
- `coverage.json` in the current directory

**Exit codes:**
- `0`: Success (threshold updated or no update needed)
- `1`: Error (missing file, invalid configuration, etc.)

## External Repository Management

These scripts download and build external dependencies required by PROTEUS.

### get_socrates.sh

**Purpose:** Download and compile the SOCRATES radiative transfer code.

**What it does:**
- Clones SOCRATES repository from GitHub
- Configures the build environment
- Compiles the Fortran code
- Sets up spectral data files

**Usage:**

```bash
# Default: downloads to ./socrates/
bash tools/get_socrates.sh

# Custom path:
bash tools/get_socrates.sh /path/to/socrates
```

**Requirements:**
- Fortran compiler (gfortran)
- SSH or HTTPS access to GitHub
- ~2 GB disk space

### get_petsc.sh

**Purpose:** Download, configure, and build PETSc (Portable Extensible Toolkit for Scientific Computing).

**Usage:**

```bash
bash tools/get_petsc.sh
```

### get_spider.sh

**Purpose:** Download and configure SPIDER interior thermal evolution model.

**Usage:**

```bash
bash tools/get_spider.sh
```

### get_vulcan.sh

**Purpose:** Download and prepare VULCAN atmospheric chemistry module.

**Usage:**

```bash
bash tools/get_vulcan.sh
```

### get_lovepy.sh

**Purpose:** Download and install the Love.jl tidal evolution module.

**Usage:**

```bash
bash tools/get_lovepy.sh
```

### get_platon.sh

**Purpose:** Download and configure PLATON atmosphere model.

**Usage:**

```bash
bash tools/get_platon.sh
```

## Data & Configuration Tools

### get_stellar_spectrum.py

**Purpose:** Download and convert stellar spectra from online databases for use in PROTEUS simulations.

**What it does:**
- Queries online spectral databases (MUSCLES, VPL, NREL)
- Downloads spectral data for specified star
- Converts to PROTEUS-compatible format
- Scales spectra to appropriate distance

**Usage:**

```bash
python tools/get_stellar_spectrum.py <star_name> [distance_au]
```

**Available stars include:** Sun, Trappist-1, GJ 1132, GJ 667C, HD 40307, and many others

**Requirements:**
- `numpy`
- Internet connection
- ~100 MB disk space for all available spectra

### chili_generate.py

**Purpose:** Generate PROTEUS configuration files for the CHILI exoplanet intercomparison project.

**What it does:**
- Loads base configuration templates
- Generates multiple model configurations
- Creates organized output structure
- Prepares files for ensemble runs

**Usage:**

```bash
python tools/chili_generate.py
```

**Input:** Configuration files in `input/chili/intercomp/`

**Output:** Generated configs in `input/chili/` and/or scratch folder

**See also:** `input/chili/readme.txt` for full intercomparison documentation

### chili_postproc.py

**Purpose:** Post-process output from CHILI intercomparison project simulations.

**What it does:**
- Reads simulation output files
- Processes and aggregates results
- Generates comparison statistics
- Creates visualization-ready data

**Usage:**

```bash
python tools/chili_postproc.py
```

## Workflow & Results Management

### make_example.sh

**Purpose:** Convert a completed simulation from the `output/` directory into a public example in `examples/`.

**What it does:**
- Validates output directory exists
- Copies result files to examples folder
- Cleans up unnecessary intermediate files
- Prepares documentation

**Usage:**

```bash
# Create example from output/my_simulation/
bash tools/make_example.sh my_simulation
```

**Creates:** `examples/my_simulation/` with clean, publishable results

## Post-Processing & Analysis

### postprocess.jl

**Purpose:** Julia script for general post-processing of PROTEUS simulation outputs.

**What it does:**
- Reads HDF5 output files
- Performs data transformations
- Generates analysis plots
- Exports processed data

**Usage:**

```bash
julia tools/postprocess.jl <input_file> [options]
```

**Requirements:**
- Julia language environment
- HDF5 and associated Julia packages

### postprocess_grid.jl

**Purpose:** Julia script specifically for analyzing grid-related outputs from simulations.

**What it does:**
- Processes grid structure data
- Analyzes spatial resolution
- Generates grid visualization data

**Usage:**

```bash
julia tools/postprocess_grid.jl <input_file> [grid_options]
```

### multiprofile_postprocess.jl

**Purpose:** Post-process multiple simulation profiles simultaneously for comparative analysis.

**What it does:**
- Aggregates data from multiple runs
- Performs ensemble statistics
- Generates comparative plots
- Exports aggregated results

**Usage:**

```bash
julia tools/multiprofile_postprocess.jl <profile1.h5> <profile2.h5> ...
```

### rheological.ipynb

**Purpose:** Jupyter notebook for analyzing and visualizing rheological properties computed during simulations.

**What it does:**
- Interactive exploration of viscosity data
- Rheology model comparisons
- Temperature-pressure rheology diagrams
- Custom visualization and analysis

**Usage:**

```bash
jupyter notebook tools/rheological.ipynb
```

**Requirements:**
- Jupyter notebook environment
- Output HDF5 files with rheological data

## Contributing

When adding new tools:

1. Ensure scripts include proper shebang: `#!/bin/bash` (bash) or `#!/usr/bin/env python3` (Python)
2. Add documentation to this README
3. Invoke scripts as: `bash tools/your_script.sh` (or make executable with `chmod +x` and call directly)
4. Include help text or documentation in the script
