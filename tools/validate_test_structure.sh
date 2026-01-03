#!/bin/bash
# Script to validate test structure mirrors src/proteus structure
# Run from repository root: bash tools/validate_test_structure.sh

set -e

echo "[*] Validating test structure..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
missing_count=0
found_count=0

# Get all directories in src/proteus (excluding __pycache__)
echo "Checking for missing test directories..."
for src_dir in $(find src/proteus -type d -not -path "*/__pycache__" -not -path "src/proteus" | sort); do
    # Extract module name
    module=$(basename "$src_dir")
    test_dir="tests/$module"

    if [ ! -d "$test_dir" ]; then
        echo "[X] Missing: $test_dir (for src/proteus/$module)"
        ((missing_count++))
    else
        echo "[+] Found: $test_dir"
        ((found_count++))
    fi
done


echo ""
echo "Checking for test files in each directory..."
for test_dir in tests/*/; do
    module=$(basename "$test_dir")

    # Skip special directories
    if [[ "$module" == "data" || "$module" == "helpers" || "$module" == "integration" || "$module" == *__pycache__* ]]; then
        continue
    fi

    # Count test files
    test_files=$(find "$test_dir" -name "test_*.py" 2>/dev/null | wc -l)

    if [ "$test_files" -eq 0 ]; then
        echo "[!]  No test files in $test_dir"
    else
        echo "[+] $test_files test file(s) in $test_dir"
    fi
done

echo ""
echo "Checking for __init__.py files..."
init_missing=0
for test_dir in tests/*/; do
    module=$(basename "$test_dir")

    # Skip special directories
    if [[ "$module" == "data" || "$module" == "helpers" || "$module" == *__pycache__* ]]; then
        continue
    fi

    if [ ! -f "${test_dir}__init__.py" ]; then
        echo -e "${YELLOW}!${NC}  Missing: ${test_dir}__init__.py"
        ((init_missing++))
    fi
done

if [ "$init_missing" -eq 0 ]; then
    echo "[+] All test directories have __init__.py"
fi

echo ""
echo "=========================================="
echo "Summary:"
echo "  Test directories found: $found_count"
echo "  Test directories missing: $missing_count"
echo "  __init__.py files missing: $init_missing"
echo ""

if [ "$missing_count" -eq 0 ] && [ "$init_missing" -eq 0 ]; then
    echo "[+] Test structure is complete!"
    exit 0
else
    echo "[!] Run 'bash tools/restructure_tests.sh' to fix issues"
    exit 1
fi
