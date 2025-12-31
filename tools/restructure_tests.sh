#!/bin/bash
# Script to restructure tests/ to mirror src/proteus/ directory structure
# Run from repository root: bash tools/restructure_tests.sh

set -e

echo "Restructuring tests to mirror src/proteus structure..."

# Create missing test directories to mirror src/proteus
mkdir -p tests/atmos_chem
mkdir -p tests/atmos_clim
mkdir -p tests/config
mkdir -p tests/escape
mkdir -p tests/interior
mkdir -p tests/observe
mkdir -p tests/orbit
mkdir -p tests/outgas
mkdir -p tests/plot
mkdir -p tests/star
mkdir -p tests/utils

# Move top-level test files to appropriate subdirectories
# test_config.py -> tests/config/
if [ -f tests/test_config.py ]; then
    mv tests/test_config.py tests/config/test_config.py
    echo "Moved test_config.py -> config/"
fi

# test_cpl_*.py files are plot-related -> tests/plot/
if [ -f tests/test_cpl_colours.py ]; then
    mv tests/test_cpl_colours.py tests/plot/test_cpl_colours.py
    echo "Moved test_cpl_colours.py -> plot/"
fi

if [ -f tests/test_cpl_helpers.py ]; then
    mv tests/test_cpl_helpers.py tests/plot/test_cpl_helpers.py
    echo "Moved test_cpl_helpers.py -> plot/"
fi

# test_cli.py and test_init.py stay at top level as they test root-level functionality

# Create __init__.py files in test directories for proper Python package structure
find tests -type d -name "[!_]*" -exec touch {}/__init__.py \;

# Create placeholder test files for modules without tests yet
for module in atmos_chem atmos_clim escape interior observe orbit outgas star utils; do
    if [ ! -f "tests/${module}/test_${module}.py" ]; then
        cat > "tests/${module}/test_${module}.py" << EOF
"""
Tests for proteus.${module} module
"""
from __future__ import annotations

import pytest


def test_placeholder():
    """Placeholder test - replace with actual tests"""
    pass
EOF
        echo "Created placeholder: tests/${module}/test_${module}.py"
    fi
done

echo "Test restructuring complete!"
echo ""
echo "Summary:"
echo "  - Created missing test directories to mirror src/proteus/"
echo "  - Moved test files to appropriate subdirectories"
echo "  - Created placeholder test files for untested modules"
echo ""
echo "Next steps:"
echo "  1. Review the changes: git status"
echo "  2. Add actual tests to placeholder files"
echo "  3. Run tests: pytest tests/"
