#!/bin/bash
# Script to analyze coverage by module and identify priorities
# Run from repository root: bash tools/coverage_analysis.sh

set -e

echo "🔍 Analyzing test coverage by module..."
echo ""

# Check if coverage is installed
if ! command -v coverage &> /dev/null; then
    echo "Error: coverage not installed. Run: pip install coverage[toml]"
    exit 1
fi

# Run tests with coverage
echo "Running tests with coverage..."
pytest --cov=src --cov-report= --quiet tests/ 2>/dev/null || true

echo ""
echo "=========================================="
echo "Coverage by Module:"
echo "=========================================="

# Generate coverage report by module
coverage report --include="src/proteus/*" --omit="*/tests/*,*/__pycache__/*" | tail -n +3 | head -n -2 | while read -r line; do
    # Extract filename and coverage percentage
    file=$(echo "$line" | awk '{print $1}')
    coverage=$(echo "$line" | awk '{print $NF}' | tr -d '%')

    # Color code based on coverage
    if [ ! -z "$coverage" ] && [ "$coverage" -eq "$coverage" ] 2>/dev/null; then
        if [ "$coverage" -ge 80 ]; then
            color="\033[0;32m"  # Green
            status="✓"
        elif [ "$coverage" -ge 50 ]; then
            color="\033[1;33m"  # Yellow
            status="⚠"
        else
            color="\033[0;31m"  # Red
            status="✗"
        fi

        echo -e "${color}${status} ${file}: ${coverage}%\033[0m"
    fi
done

echo ""
echo "=========================================="
echo "Priority Modules (Coverage < 50%):"
echo "=========================================="

# List modules needing attention
coverage report --include="src/proteus/*" --omit="*/tests/*,*/__pycache__/*" | tail -n +3 | head -n -2 | while read -r line; do
    file=$(echo "$line" | awk '{print $1}')
    coverage=$(echo "$line" | awk '{print $NF}' | tr -d '%')

    if [ ! -z "$coverage" ] && [ "$coverage" -eq "$coverage" ] 2>/dev/null; then
        if [ "$coverage" -lt 50 ]; then
            echo "- $file (${coverage}%)"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Overall Coverage:"
echo "=========================================="
coverage report --include="src/proteus/*" --omit="*/tests/*,*/__pycache__/*" | tail -n 1

echo ""
echo "💡 Tips:"
echo "  - View detailed report: open htmlcov/index.html"
echo "  - Test specific module: pytest tests/[module]/"
echo "  - Check missing lines: coverage report --show-missing"
