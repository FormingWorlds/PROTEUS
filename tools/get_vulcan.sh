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
unchanged=""
if [ -e "$workpath" ]; then
    unchanged="; $workpath is unchanged"
fi

# The new clone replaces the checkout only once it is at the release tag and fastchem is built.
tmp_clone="${workpath%/}.tmp.$$"
rm -rf "$tmp_clone"
trap 'rm -rf "$tmp_clone"' EXIT

# Detect SSH access to GitHub.
use_ssh=$(github_use_ssh)

# Download
echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    uri="git@github.com:FormingWorlds/VULCAN.git"
else
    uri="https://github.com/FormingWorlds/VULCAN.git"
fi
echo "    $uri -> $tmp_clone"
git clone "$uri" "$tmp_clone" || { echo "ERROR: git clone failed$unchanged" >&2; exit 1; }

checkout_floor_tag fwl-vulcan "$tmp_clone"

# Compile fastchem before the swap: its makefiles use relative paths only, so the build moves along.
cd "$tmp_clone/fastchem_vulcan/" || { echo "ERROR: fastchem dir missing" >&2; exit 1; }
make || { echo "ERROR: fastchem build failed$unchanged" >&2; exit 1; }

cd "$root"
swap_in_checkout "$tmp_clone" "$workpath" || exit 1
trap - EXIT
cd "$workpath"

# Install vulcan package as editable
pip install -U -e . || { echo "ERROR: editable install failed" >&2; exit 1; }

# Back to old folder
cd "$root"

# Done
echo "Done!"
