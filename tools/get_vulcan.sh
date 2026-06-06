#!/bin/bash
# Download and setup VULCAN (optional atmospheric chemistry module) as an
# editable sibling checkout.
#
# Clones FormingWorlds/VULCAN into ./VULCAN/ inside the PROTEUS root,
# checks out the git tag matching the fwl-vulcan version floor pinned in
# pyproject.toml ([project.optional-dependencies].vulcan), builds fastchem,
# and installs it editable. Pinning to the floor tag keeps the editable
# checkout and the PyPI fwl-vulcan release in lock-step instead of tracking
# whatever the default branch points at. To develop against the latest
# VULCAN, run `git checkout main` inside ./VULCAN and reinstall.

set -euo pipefail

echo "Set up VULCAN..."

portable_realpath() {
    if command -v realpath >/dev/null 2>&1; then
        realpath "$1"
    else
        python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$1"
    fi
}

# Path to PROTEUS folder
root=$(dirname "$(portable_realpath "$0")")
root=$(portable_realpath "$root/..")

# Make room
workpath="$root/VULCAN/"
rm -rf "$workpath"

# Detect SSH access to GitHub. `ssh -T git@github.com` exits 1 when
# authentication succeeds (GitHub refuses the shell), so a plain call
# would trip `set -e`; keeping it as the `if` condition keeps it in
# scope where a non-zero exit is expected rather than fatal.
if ssh -T git@github.com; then
    use_ssh=false
else
    if [ $? -eq 1 ]; then
        use_ssh=true
    else
        use_ssh=false
    fi
fi

# Download
echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    uri="git@github.com:FormingWorlds/VULCAN.git"
else
    uri="https://github.com/FormingWorlds/VULCAN.git"
fi
echo "    $uri -> $workpath"
git clone "$uri" "$workpath" || { echo "ERROR: git clone failed" >&2; exit 1; }

# Pin the checkout to the fwl-vulcan version floor declared in PROTEUS's
# pyproject.toml, so the editable install matches the PyPI release across
# machines and CI instead of tracking whatever the default branch points at.
floor=$(grep -oE 'fwl-vulcan>=[0-9][0-9.]*' "$root/pyproject.toml" | head -1 | sed 's/.*>=//')
cd "$workpath" || { echo "ERROR: cannot enter $workpath" >&2; exit 1; }
if [ -n "$floor" ]; then
    echo "Pinning to fwl-vulcan floor: $floor"
    git checkout "tags/$floor" || { echo "ERROR: cannot checkout tag $floor" >&2; exit 1; }
else
    echo "WARNING: could not read fwl-vulcan floor from pyproject.toml; using HEAD" >&2
fi

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
