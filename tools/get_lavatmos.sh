#!/bin/bash
# Download and setup LavAtmos.

echo "Set up LavAtmos..."

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
workpath="$root/LavAtmos/"

# Already setup?
if [ -n "$LAVA_DIR" ]; then
    echo "WARNING: You already have LavAtmos installed"
    echo "         LAVA_DIR=$LAVA_DIR"
    echo ""
    echo "Installing LavAtmos into '$workpath'..."
    echo ""
    sleep 5
fi

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
l_url=$(python "$root/tools/_module_pins.py" lavatmos url)
l_ref=$(python "$root/tools/_module_pins.py" lavatmos ref)
if [ -z "$l_url" ] || [ -z "$l_ref" ]; then
    echo "ERROR: could not resolve lavatmos url/ref from pyproject.toml" >&2
    exit 1
fi

echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    # Rewrite https://github.com/ -> git@github.com: for SSH transport.
    uri=${l_url/https:\/\/github.com\//git@github.com:}
else
    uri="$l_url"
fi
echo "    $uri @ $l_ref -> $workpath"
git clone "$uri" "$workpath" || { echo "ERROR: git clone failed" >&2; exit 1; }
git -C "$workpath" checkout --quiet "$l_ref" \
    || { echo "ERROR: cannot checkout $l_ref" >&2; exit 1; }

# Inform user
echo " "
echo "You must now run the following command:"
echo "    export LAVA_DIR='$workpath'"
echo "You should also add this command to your shell rc file (e.g. ~/.bashrc)"
