#!/bin/bash
# Download and setup LavAtmos.
#
echo "Set up LavAtmos..."

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
workpath=$root/LavAtmos/

# Already setup?
if [ -n "$LAVA_DIR" ]; then
    echo "WARNING: You already have LavAtmos installed"
    echo "         LAVA_DIR=$LAVA_DIR"
    echo ""
    echo "Installing LavAtmos into '$workpath'..."
    echo ""
    sleep 5
fi

# Refuse to delete a checkout holding local work unless --force is given.
# Keep this guard in sync across the get_* scripts that refresh checkouts.
# Guarded states: modified tracked files, and commits not on any remote.
# Untracked files (build artifacts, egg-info) do not block the refresh.
force=false
for arg in "$@"; do
    [ "$arg" = "--force" ] && force=true
done
if [ -d "$workpath/.git" ] && [ "$force" != true ]; then
    dirty=$(git -C "$workpath" status --porcelain --untracked-files=no 2>/dev/null | head -1)
    unpushed=$(git -C "$workpath" log HEAD --not --remotes --oneline 2>/dev/null | head -1)
    if [ -n "$dirty" ] || [ -n "$unpushed" ]; then
        echo "ERROR: $workpath has uncommitted changes or commits not on a remote." >&2
        echo "       Refusing to delete it. Commit and push your work, or run" >&2
        echo "       bash tools/get_lavatmos.sh --force  to discard the checkout." >&2
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
    uri="git@github.com:FormingWorlds/LavAtmos.git"
else
    uri="https://github.com/FormingWorlds/LavAtmos.git"
fi
echo "    $uri -> $workpath"
git clone "$uri" "$workpath" || { echo "ERROR: git clone failed" >&2; exit 1; }

# Install lavatmos package as editable
# pip install -U -e . || { echo "ERROR: editable install failed" >&2; exit 1; }

# Back to old folder
cd $root

# Inform user
echo " "
echo "You must now run the following command:"
echo "    export LAVA_DIR='$workpath'"
echo "You should also add this command to your shell rc file (e.g. ~/.bashrc)"

