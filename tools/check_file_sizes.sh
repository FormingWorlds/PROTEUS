#!/bin/bash
# Validate line limits for AGENTS.md and MEMORY.md
# AGENTS.md: max 500 lines
# MEMORY.md: max 1000 lines

set -e

AGENTS_MAX=500
MEMORY_MAX=1000

EXIT_CODE=0

if [ -f "AGENTS.md" ]; then
    AGENTS_LINES=$(wc -l < AGENTS.md | tr -d ' ')
    if [ "$AGENTS_LINES" -gt "$AGENTS_MAX" ]; then
        echo "ERROR: AGENTS.md exceeds $AGENTS_MAX lines (current: $AGENTS_LINES)"
        EXIT_CODE=1
    else
        echo "OK: AGENTS.md has $AGENTS_LINES lines (max: $AGENTS_MAX)"
    fi
else
    echo "WARNING: AGENTS.md not found"
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
