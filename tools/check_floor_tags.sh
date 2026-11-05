#!/bin/bash
# Usage: bash tools/check_floor_tags.sh. For each module a setup script clones at its floor tag, lists
# the tags of its remote and prints the one tag of the floor; exits 1 when a module has no single tag
# or its remote cannot be read. PROTEUS_MODULE_GIT_BASE sets the URL prefix.

set -euo pipefail

# Shared helpers: see tools/_get_common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_get_common.sh" || exit 1

# <repository>:<distribution> of each setup script; the repository is its folder in the PROTEUS root.
modules="MORS:fwl-mors JANUS:fwl-janus CALLIOPE:fwl-calliope ZEPHYRUS:fwl-zephyrus
aragog:fwl-aragog Zalmoxis:fwl-zalmoxis VULCAN:fwl-vulcan Morrigan:fwl-morrigan"
base="${PROTEUS_MODULE_GIT_BASE:-https://github.com/}"

status=0
for entry in $modules; do
    repo="${entry%%:*}"
    dist="${entry#*:}"
    url="${base}FormingWorlds/$repo.git"
    if ! floor=$(python "$proteus_tools_dir/_module_pins.py" "$dist" floor 2>&1); then
        echo "$dist floor ?: NO SINGLE TAG ($floor)"
        status=1
        continue
    fi
    if ! refs=$(git ls-remote --tags "$url" 2>&1); then
        echo "$dist floor $floor: NO SINGLE TAG (cannot read the tags of $url: ${refs%%$'\n'*})"
        status=1
        continue
    fi
    # An annotated tag is listed twice, the second time as <name>^{} with its commit.
    if tag=$(printf '%s\n' "$refs" |
        awk '$2 !~ /\^\{\}$/ { sub(/^refs\/tags\//, "", $2); print $2 }' |
        python "$proteus_tools_dir/_module_pins.py" "$dist" floor-tag 2>&1); then
        echo "$dist floor $floor: tag $tag"
    else
        echo "$dist floor $floor: NO SINGLE TAG ($tag)"
        status=1
    fi
done
exit "$status"
