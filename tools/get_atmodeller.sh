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

# Detect SSH access to GitHub.
uri="$module_url"
[ "$(github_use_ssh)" != true ] || uri=$(github_ssh_url "$module_url")

echo "Cloning from GitHub"
echo "    $uri -> ${workpath%/}.tmp.$$"
clone_at_ref "$uri" "$module_ref" "$workpath"

# Install atmodeller package as editable
pip install -U -e "$workpath" || { echo "ERROR: editable install failed" >&2; exit 1; }

# Done
echo "Done!"
