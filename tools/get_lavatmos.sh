#!/bin/bash
# Download and setup LavAtmos.

echo "Set up LavAtmos..."

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
workpath="$root/LavAtmos/"

# Already setup?
if [ -n "$LAVA_DIR" ]; then
    echo "WARNING: You already have LavAtmos installed"
    echo "         LAVA_DIR=$LAVA_DIR"
    echo ""
    echo "Installing LavAtmos into '$workpath'..."
    echo ""
    sleep 5
fi

guard_dirty_checkout "$workpath" get_lavatmos.sh

# Make room
rm -rf "$workpath"

# Detect SSH access to GitHub.
use_ssh=$(github_use_ssh)

# Resolve the pinned URL + ref from pyproject.toml.
resolve_module_pin lavatmos
l_url="$module_url"
l_ref="$module_ref"

echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    uri=$(github_ssh_url "$l_url")
else
    uri="$l_url"
fi
echo "    $uri @ $l_ref -> $workpath"
git clone "$uri" "$workpath" || { echo "ERROR: git clone failed" >&2; exit 1; }
git -C "$workpath" checkout --quiet "$l_ref" \
    || { echo "ERROR: cannot checkout $l_ref" >&2; exit 1; }

# Inform user
echo " "
echo "You must now run the following command:"
echo "    export LAVA_DIR='$workpath'"
echo "You should also add this command to your shell rc file (e.g. ~/.bashrc)"
