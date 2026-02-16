#!/bin/bash
# Validate line limits for .github/copilot-instructions.md and MEMORY.md
# .github/copilot-instructions.md: max 500 lines
# MEMORY.md: max 1000 lines

set -e

AGENTS_MAX=500
MEMORY_MAX=1000

EXIT_CODE=0

if [ -f ".github/copilot-instructions.md" ]; then
    AGENTS_LINES=$(wc -l < .github/copilot-instructions.md | tr -d ' ')
    if [ "$AGENTS_LINES" -gt "$AGENTS_MAX" ]; then
        echo "ERROR: .github/copilot-instructions.md exceeds $AGENTS_MAX lines (current: $AGENTS_LINES)"
        EXIT_CODE=1
    else
        echo "OK: .github/copilot-instructions.md has $AGENTS_LINES lines (max: $AGENTS_MAX)"
    fi
else
    echo "WARNING: .github/copilot-instructions.md not found"
fi

if [ -f "MEMORY.md" ]; then
    MEMORY_LINES=$(wc -l < MEMORY.md | tr -d ' ')
    if [ "$MEMORY_LINES" -gt "$MEMORY_MAX" ]; then
        echo "ERROR: MEMORY.md exceeds $MEMORY_MAX lines (current: $MEMORY_LINES)"
        EXIT_CODE=1
    else
        echo "OK: MEMORY.md has $MEMORY_LINES lines (max: $MEMORY_MAX)"
    fi
else
    echo "WARNING: MEMORY.md not found"
fi

exit $EXIT_CODE
