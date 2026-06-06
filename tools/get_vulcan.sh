#!/bin/bash
# Download and setup VULCAN (optional atmospheric chemistry module) as an
# editable sibling checkout. Clones the pinned commit
# ([tool.proteus.modules.vulcan]), builds fastchem, and installs editable.

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

# Resolve the pinned URL + ref from pyproject.toml.
vc_url=$(python "$root/tools/_module_pins.py" vulcan url)
vc_ref=$(python "$root/tools/_module_pins.py" vulcan ref)
if [ -z "$vc_url" ] || [ -z "$vc_ref" ]; then
    echo "ERROR: could not resolve vulcan url/ref from pyproject.toml" >&2
    exit 1
fi

echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    # Rewrite https://github.com/ -> git@github.com: for SSH transport.
    uri=${vc_url/https:\/\/github.com\//git@github.com:}
else
    uri="$vc_url"
fi
echo "    $uri @ $vc_ref -> $workpath"
git clone "$uri" "$workpath" || { echo "ERROR: git clone failed" >&2; exit 1; }
git -C "$workpath" checkout --quiet "$vc_ref" \
    || { echo "ERROR: cannot checkout $vc_ref" >&2; exit 1; }

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
