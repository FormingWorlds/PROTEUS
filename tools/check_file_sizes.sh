#!/bin/bash
# Validate the line limit on .github/copilot-instructions.md.
# The cap exists so the file stays readable as an entry point; the
# Claude-Code rule deep-dives live under .github/.claude/rules/ and
# are not subject to this cap.

set -e

AGENTS_MAX=750

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

exit $EXIT_CODE
