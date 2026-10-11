#!/bin/bash
# Clone VULCAN (optional atmospheric chemistry) into ./VULCAN/ at the release tag of the fwl-vulcan
# floor in pyproject.toml, build fastchem and install it editable. To develop VULCAN, switch the
# clone to a branch and install it again. --force discards local work.

set -euo pipefail

echo "Set up VULCAN..."

# Shared helpers: see tools/_get_common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_get_common.sh" || exit 1

# Path to PROTEUS folder
root="$proteus_root"

get_parse_args "$@"
workpath="$root/VULCAN/"
guard_dirty_checkout "$workpath" get_vulcan.sh

# Make room
rm -rf "$workpath"

# Detect SSH access to GitHub.
use_ssh=$(github_use_ssh)

# Download
echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    uri="git@github.com:FormingWorlds/VULCAN.git"
else
    uri="https://github.com/FormingWorlds/VULCAN.git"
fi
echo "    $uri -> $workpath"
git clone "$uri" "$workpath" || { echo "ERROR: git clone failed" >&2; exit 1; }

checkout_floor_tag fwl-vulcan "$workpath"

# Compile fastchem
cd "$workpath/fastchem_vulcan/" || { echo "ERROR: fastchem dir missing" >&2; exit 1; }
make || { echo "ERROR: fastchem build failed" >&2; exit 1; }
cd "$workpath"

# Install vulcan package as editable
pip install -U -e . || { echo "ERROR: editable install failed" >&2; exit 1; }

# Back to old folder
cd "$root"

# Done
echo "Done!"
