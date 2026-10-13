#!/bin/bash
# Clone atmodeller into ./Atmodeller/ at the pinned commit and install it editable; an
# existing folder is installed as it is, and --force replaces it.

set -euo pipefail

echo "Set up atmodeller..."

# Shared helpers: see tools/_get_common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_get_common.sh" || exit 1

# Path to PROTEUS folder
root="$proteus_root"

get_parse_args "$@"
workpath="$root/Atmodeller/"

resolve_module_pin atmodeller
install_existing_clone atmodeller "$workpath" get_atmodeller.sh "" "$module_ref"

# The new clone replaces the checkout only once it is at the pinned commit.
tmp_clone="${workpath%/}.tmp.$$"
rm -rf "$tmp_clone"
trap 'rm -rf "$tmp_clone"' EXIT

# Detect SSH access to GitHub.
use_ssh=$(github_use_ssh)

echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    uri=$(github_ssh_url "$module_url")
else
    uri="$module_url"
fi
echo "    $uri -> $tmp_clone"
git clone "$uri" "$tmp_clone" ||
    { echo "ERROR: git clone failed$(unchanged_note "$workpath")" >&2; exit 1; }

git -C "$tmp_clone" checkout --quiet "$module_ref" ||
    { echo "ERROR: git checkout failed$(unchanged_note "$workpath")" >&2; exit 1; }

swap_in_checkout "$tmp_clone" "$workpath" || exit 1
trap - EXIT

# Install atmodeller package as editable
pip install -U -e "$workpath" || { echo "ERROR: editable install failed" >&2; exit 1; }

# Done
echo "Done!"
