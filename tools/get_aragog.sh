#!/bin/bash
# Clone Aragog into ./aragog/ and install it editable. A new clone, and the one --force puts in
# place of an old one, takes the Aragog branch named like the PROTEUS feature branch, if any, else
# the fwl-aragog floor tag. An existing folder is installed as it is; only --force replaces it.

echo "Set up Aragog..."

# Shared helpers: see tools/_get_common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_get_common.sh" || exit 1

# Path to PROTEUS folder
root="$proteus_root"

get_parse_args "$@"
get_pairs_branch=true
workpath="$root/aragog/"
install_existing_clone fwl-aragog "$workpath" get_aragog.sh

# The new clone replaces the checkout only once it is on its branch or release tag.
tmp_clone="${workpath%/}.tmp.$$"
rm -rf "$tmp_clone"
trap 'rm -rf "$tmp_clone"' EXIT

# Check SSH access to GitHub
use_ssh=$(github_use_ssh)

# Download
echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    uri="git@github.com:FormingWorlds/aragog.git"
else
    uri="https://github.com/FormingWorlds/aragog.git"
fi
echo "    $uri -> $tmp_clone"
git clone "$uri" "$tmp_clone" ||
    { echo "ERROR: git clone failed$(unchanged_note "$workpath")" >&2; exit 1; }

checkout_paired_or_floor fwl-aragog "$tmp_clone" aragog

swap_in_checkout "$tmp_clone" "$workpath" || exit 1
trap - EXIT

# Install aragog package as editable
pip install -U -e "${workpath%/}" || { echo "ERROR: editable install failed" >&2; exit 1; }

# Done
echo "Done!"
