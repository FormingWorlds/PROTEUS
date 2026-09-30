#!/bin/bash
# Clone and configure Morrigan as an editable module inside ./Morrigan/.
# Checks out the git tag matching the fwl-morrigan floor in pyproject.toml.
# For standard installations, use `pip install "fwl-proteus[morrigan]"`.

set -euo pipefail

echo "Set up Morrigan..."

portable_realpath() {
    # Keep this helper in sync across the get_* scripts. A path that does not
    # exist yet is rejected by realpath (BSD refuses a missing leaf, GNU a
    # missing parent), so fall through to python3 there too.
    if command -v realpath >/dev/null 2>&1 && realpath "$1" 2>/dev/null; then
        return 0
    fi
    python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$1"
}

# Path to PROTEUS folder
root=$(dirname "$(portable_realpath "$0")")
root=$(portable_realpath "$root/..")

# Refuse to delete a checkout holding local work unless --force is given.
# Guards against overwriting uncommitted changes or unpushed commits.
force=false
for arg in "$@"; do
    [ "$arg" = "--force" ] && force=true
done
workpath="$root/Morrigan/"
if [ -d "$workpath/.git" ] && [ "$force" != true ]; then
    dirty=$(git -C "$workpath" status --porcelain --untracked-files=no 2>/dev/null | head -1)
    unpushed=$(git -C "$workpath" log HEAD --not --remotes --oneline 2>/dev/null | head -1)
    if [ -n "$dirty" ] || [ -n "$unpushed" ]; then
        echo "ERROR: $workpath has uncommitted changes or commits not on a remote." >&2
        echo "       Refusing to delete it. Commit and push your work, or run" >&2
        echo "       bash tools/get_morrigan.sh --force  to discard the checkout." >&2
        exit 1
    fi
fi

# Clone to a temporary directory first so a failed clone does not destroy
# the existing checkout.
tmp_clone="${workpath%/}.tmp.$$"
rm -rf "$tmp_clone"
trap 'rm -rf "$tmp_clone"' EXIT

# Detect SSH access to GitHub. Exit code 1 from ssh -T git@github.com
# indicates authentication succeeded without shell access.
if ssh -T git@github.com; then
    use_ssh=false
else
    if [ $? -eq 1 ]; then
        use_ssh=true
    else
        use_ssh=false
    fi
fi

echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    uri="git@github.com:FormingWorlds/Morrigan.git"
else
    uri="https://github.com/FormingWorlds/Morrigan.git"
fi
echo "    $uri -> $tmp_clone"
git clone "$uri" "$tmp_clone" || { echo "ERROR: git clone failed" >&2; exit 1; }

# Pin checkout to the fwl-morrigan version floor from pyproject.toml.
# Strip comments before extracting to match the active dependency specification.
floor=$(sed 's/#.*//' "$root/pyproject.toml" \
    | grep -oE 'fwl-morrigan>=[0-9][0-9.]*' | head -1 | sed 's/.*>=//' || true)
if [ -n "$floor" ]; then
    echo "Pinning to fwl-morrigan floor: $floor"
    git -C "$tmp_clone" checkout --quiet "tags/$floor" \
        || { echo "ERROR: cannot checkout tag $floor" >&2; exit 1; }
else
    echo "WARNING: could not read fwl-morrigan floor from pyproject.toml; using HEAD" >&2
fi

# Replace existing checkout only after clone and tag checkout succeed
rm -rf "$workpath"
mv "$tmp_clone" "$workpath"
trap - EXIT

# Install morrigan package as editable
pip install -U -e "$workpath" || { echo "ERROR: editable install failed" >&2; exit 1; }

# Done
echo "Done!"
