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

# Refuse to delete a checkout holding local work unless --force is given.
# Keep this guard in sync across the get_* scripts that refresh checkouts.
# Guarded states: modified tracked files, and commits not on any remote.
# Untracked files (build artifacts, egg-info) do not block the refresh.
force=false
for arg in "$@"; do
    [ "$arg" = "--force" ] && force=true
done
workpath="$root/BOREAS/"
if [ -d "$workpath/.git" ] && [ "$force" != true ]; then
    dirty=$(git -C "$workpath" status --porcelain --untracked-files=no 2>/dev/null | head -1)
    unpushed=$(git -C "$workpath" log HEAD --not --remotes --oneline 2>/dev/null | head -1)
    if [ -n "$dirty" ] || [ -n "$unpushed" ]; then
        echo "ERROR: $workpath has uncommitted changes or commits not on a remote." >&2
        echo "       Refusing to delete it. Commit and push your work, or run" >&2
        echo "       bash tools/get_boreas.sh --force  to discard the checkout." >&2
        exit 1
    fi
fi

# Make room
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
b_url=$(python "$root/tools/_module_pins.py" boreas url)
b_ref=$(python "$root/tools/_module_pins.py" boreas ref)
if [ -z "$b_url" ] || [ -z "$b_ref" ]; then
    echo "ERROR: could not resolve boreas url/ref from pyproject.toml" >&2
    exit 1
fi

echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    # Rewrite https://github.com/ -> git@github.com: for SSH transport.
    uri=${b_url/https:\/\/github.com\//git@github.com:}
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
