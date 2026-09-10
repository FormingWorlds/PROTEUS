#!/bin/bash
# Download and setup BOREAS (optional hydrodynamic escape module) as an
# editable sibling checkout.
#
# Clones ExoInteriors/BOREAS into ./BOREAS/ inside the PROTEUS root,
# checks out the commit pinned in pyproject.toml
# ([tool.proteus.modules.boreas]), and installs it editable into the
# active Python environment. BOREAS is not on PyPI under a usable name,
# so the pin is resolved from pyproject.toml rather than a version floor.

set -euo pipefail

echo "Set up BOREAS..."

# Shared helpers: see tools/_get_common.sh.
_get_common="$(dirname "${BASH_SOURCE[0]}")/_get_common.sh"
if [ ! -f "$_get_common" ]; then
    echo "ERROR: $_get_common is missing; use a complete PROTEUS checkout." >&2
    exit 1
fi
source "$_get_common"

# Path to PROTEUS folder
root="$proteus_root"

get_parse_args "$@"
workpath="$root/BOREAS/"
guard_dirty_checkout "$workpath" get_boreas.sh

# Make room
rm -rf "$workpath"

# Detect SSH access to GitHub.
use_ssh=$(github_use_ssh)

# Resolve the pinned URL + ref from pyproject.toml.
resolve_module_pin boreas
b_url="$module_url"
b_ref="$module_ref"

echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    uri=$(github_ssh_url "$b_url")
else
    uri="$b_url"
fi
echo "    $uri @ $b_ref -> $workpath"
git clone "$uri" "$workpath" || { echo "ERROR: git clone failed" >&2; exit 1; }
git -C "$workpath" checkout --quiet "$b_ref" \
    || { echo "ERROR: cannot checkout $b_ref" >&2; exit 1; }

# Install boreas package as editable
pip install -U -e "$workpath" || { echo "ERROR: editable install failed" >&2; exit 1; }

# Done
echo "Done!"
