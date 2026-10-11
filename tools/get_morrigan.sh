#!/bin/bash
# Clone Morrigan into ./Morrigan/ at the tag of the fwl-morrigan floor and install it editable; an
# existing folder is installed as it is, and --force replaces it. For standard installations, use
# `pip install "fwl-proteus[morrigan]"`.

set -euo pipefail

echo "Set up Morrigan..."

# Shared helpers: see tools/_get_common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_get_common.sh" || exit 1

# Path to PROTEUS folder
root="$proteus_root"

get_parse_args "$@"
workpath="$root/Morrigan/"
install_existing_clone fwl-morrigan "$workpath" get_morrigan.sh

# The new clone replaces the checkout only once it is at the release tag.
tmp_clone="${workpath%/}.tmp.$$"
rm -rf "$tmp_clone"
trap 'rm -rf "$tmp_clone"' EXIT

# Detect SSH access to GitHub.
use_ssh=$(github_use_ssh)

echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    uri="git@github.com:FormingWorlds/Morrigan.git"
else
    uri="https://github.com/FormingWorlds/Morrigan.git"
fi
echo "    $uri -> $tmp_clone"
git clone "$uri" "$tmp_clone" ||
    { echo "ERROR: git clone failed$(unchanged_note "$workpath")" >&2; exit 1; }

checkout_floor_tag fwl-morrigan "$tmp_clone"

swap_in_checkout "$tmp_clone" "$workpath" || exit 1
trap - EXIT

# Install morrigan package as editable
pip install -U -e "$workpath" || { echo "ERROR: editable install failed" >&2; exit 1; }

# Done
echo "Done!"
