#!/bin/bash
# Download and setup Zalmoxis as an editable sibling checkout.
#
# Clones FormingWorlds/Zalmoxis into ./Zalmoxis/ inside the PROTEUS
# root, checks out the fwl-zalmoxis version floor pinned in
# pyproject.toml, and installs it editable into the active Python
# environment. The editable install takes precedence over the PyPI
# fwl-zalmoxis pin on sys.path, so any local edits to Zalmoxis/src/ are
# picked up by `import zalmoxis` without reinstalling. To develop
# against the latest Zalmoxis, run `git checkout main` inside
# ./Zalmoxis and reinstall.

echo "Set up Zalmoxis..."

portable_realpath() {
    if command -v realpath >/dev/null 2>&1; then
        realpath "$1"
    else
        python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$1"
    fi
}

# Path to PROTEUS folder
root=$(dirname $(portable_realpath $0))
root=$(portable_realpath "$root/..")

# Refuse to delete a checkout holding local work unless --force is given.
# Keep this guard in sync across the get_* scripts that refresh checkouts.
# Guarded states: modified tracked files, and commits not on any remote.
# Untracked files (build artifacts, egg-info) do not block the refresh.
force=false
for arg in "$@"; do
    [ "$arg" = "--force" ] && force=true
done
workpath=$root/Zalmoxis/
if [ -d "$workpath/.git" ] && [ "$force" != true ]; then
    dirty=$(git -C "$workpath" status --porcelain --untracked-files=no 2>/dev/null | head -1)
    unpushed=$(git -C "$workpath" log HEAD --not --remotes --oneline 2>/dev/null | head -1)
    if [ -n "$dirty" ] || [ -n "$unpushed" ]; then
        echo "ERROR: $workpath has uncommitted changes or commits not on a remote." >&2
        echo "       Refusing to delete it. Commit and push your work, or run" >&2
        echo "       bash tools/get_zalmoxis.sh --force  to discard the checkout." >&2
        exit 1
    fi
fi

# Make room
rm -rf $workpath

# Check SSH access to GitHub
ssh -T git@github.com
if [ $? -eq 1 ]; then
    use_ssh=true
else
    use_ssh=false
fi

# Download
echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    uri="git@github.com:FormingWorlds/Zalmoxis.git"
else
    uri="https://github.com/FormingWorlds/Zalmoxis.git"
fi
echo "    $uri -> $workpath"
git clone "$uri" "$workpath" || { echo "ERROR: git clone failed" >&2; exit 1; }

# Pin the checkout to the fwl-zalmoxis version floor declared in PROTEUS's
# pyproject.toml, so the editable install is reproducible across machines
# and CI instead of tracking whatever the default branch points at.
floor=$(grep -oE 'fwl-zalmoxis>=[0-9][0-9.]*' "$root/pyproject.toml" | head -1 | sed 's/.*>=//')
cd "$workpath" || { echo "ERROR: cannot enter $workpath" >&2; exit 1; }
if [ -n "$floor" ]; then
    echo "Pinning to fwl-zalmoxis floor: $floor"
    git checkout "tags/$floor" || { echo "ERROR: cannot checkout tag $floor" >&2; exit 1; }
else
    echo "WARNING: could not read fwl-zalmoxis floor from pyproject.toml; using HEAD" >&2
fi

# Install zalmoxis package as editable
pip install -U -e . || { echo "ERROR: editable install failed" >&2; exit 1; }

# Back to old folder
cd $root

# Done
echo "Done!"
