#!/bin/bash
# Download and setup Morrigan (optional giant-impact accretion module) as
# an editable sibling checkout.
#
# Clones FormingWorlds/Morrigan into ./Morrigan/ inside the PROTEUS root,
# checks out the git tag matching the fwl-morrigan version floor pinned in
# pyproject.toml ([project.optional-dependencies].morrigan), and installs it
# editable. Pinning to the floor tag keeps the editable checkout and the PyPI
# fwl-morrigan release in lock-step instead of tracking the default branch.
#
# For a plain (non-editable) install, `pip install "fwl-proteus[morrigan]"`
# is enough; this script is for developing against a Morrigan checkout.

set -euo pipefail

echo "Set up Morrigan..."

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

echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    uri="git@github.com:FormingWorlds/Morrigan.git"
else
    uri="https://github.com/FormingWorlds/Morrigan.git"
fi
echo "    $uri -> $workpath"
git clone "$uri" "$workpath" || { echo "ERROR: git clone failed" >&2; exit 1; }

# Pin the checkout to the fwl-morrigan version floor declared in PROTEUS's
# pyproject.toml, so the editable install matches the PyPI release across
# machines and CI instead of tracking whatever the default branch points at.
# The floor is written zero-padded (26.07.25) to match the release tag; PEP
# 440 treats that as equal to the normalised PyPI version (26.7.25), so the
# same string serves both the dependency resolver and this checkout.
# Comments are stripped before matching: the pin carries a rationale comment
# above it, and a future comment naming a different version would otherwise be
# picked up first and checked out instead of the real floor. The `|| true`
# keeps a missing pin from aborting under `set -e` before the warning below
# can explain what went wrong.
floor=$(sed 's/#.*//' "$root/pyproject.toml" \
    | grep -oE 'fwl-morrigan>=[0-9][0-9.]*' | head -1 | sed 's/.*>=//' || true)
if [ -n "$floor" ]; then
    echo "Pinning to fwl-morrigan floor: $floor"
    git -C "$workpath" checkout --quiet "tags/$floor" \
        || { echo "ERROR: cannot checkout tag $floor" >&2; exit 1; }
else
    echo "WARNING: could not read fwl-morrigan floor from pyproject.toml; using HEAD" >&2
fi

# Install morrigan package as editable
pip install -U -e "$workpath" || { echo "ERROR: editable install failed" >&2; exit 1; }

# Done
echo "Done!"
