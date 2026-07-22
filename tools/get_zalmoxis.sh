#!/bin/bash
# Download and setup Zalmoxis as an editable sibling checkout.
#
# Clones FormingWorlds/Zalmoxis into ./Zalmoxis/ inside the PROTEUS root
# and installs it editable into the active Python environment. The
# editable install takes precedence over the PyPI fwl-zalmoxis pin on
# sys.path, so any local edits to Zalmoxis/src/ are picked up by
# `import zalmoxis` without reinstalling.
#
# Checkout target: if PROTEUS is on a feature branch that Zalmoxis also
# has, the matching Zalmoxis branch is installed so the paired code is
# exercised together; otherwise the checkout is pinned to the
# fwl-zalmoxis version floor declared in pyproject.toml. The default
# branch (main) and a detached HEAD always take the floor tag. To develop
# against the latest Zalmoxis, run `git checkout main` inside ./Zalmoxis
# and reinstall.

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

# Paired-branch detection. When PROTEUS is on a feature branch and Zalmoxis has
# a branch of the same name, install that Zalmoxis branch so CI exercises the
# real paired code instead of the released floor tag. In GitHub Actions the
# branch is GITHUB_HEAD_REF (pull_request) or GITHUB_REF_NAME (push /
# workflow_dispatch); locally it is the checked-out branch. The default branch
# and a detached HEAD never pair, so main and tag builds keep the floor tag.
proteus_branch="${GITHUB_HEAD_REF:-${GITHUB_REF_NAME:-}}"
if [ -z "$proteus_branch" ]; then
    proteus_branch=$(git -C "$root" rev-parse --abbrev-ref HEAD 2>/dev/null)
fi

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

cd "$workpath" || { echo "ERROR: cannot enter $workpath" >&2; exit 1; }

# If PROTEUS is on a feature branch that Zalmoxis also has, install that branch;
# otherwise pin to the fwl-zalmoxis version floor declared in PROTEUS's
# pyproject.toml, so main/tag builds are reproducible and unaffected. The full
# clone above already fetched every Zalmoxis branch, so the pairing test reads
# the local remote-tracking refs (no extra network round-trip, nothing to flake).
match_branch=""
case "$proteus_branch" in
    '' | HEAD | main | master)
        : ;;  # default branch or detached HEAD never pairs
    *)
        if git rev-parse --verify --quiet "refs/remotes/origin/$proteus_branch" >/dev/null; then
            match_branch="$proteus_branch"
        fi
        ;;
esac

if [ -n "$match_branch" ]; then
    git checkout "$match_branch" || { echo "ERROR: cannot checkout Zalmoxis branch $match_branch" >&2; exit 1; }
    echo "PROTEUS is on branch '$proteus_branch'; installed paired Zalmoxis branch at $(git rev-parse --short HEAD)"
else
    floor=$(grep -oE 'fwl-zalmoxis>=[0-9][0-9.]*' "$root/pyproject.toml" | head -1 | sed 's/.*>=//')
    if [ -n "$floor" ]; then
        echo "Pinning to fwl-zalmoxis floor: $floor"
        git checkout "tags/$floor" || { echo "ERROR: cannot checkout tag $floor" >&2; exit 1; }
    else
        echo "WARNING: could not read fwl-zalmoxis floor from pyproject.toml; using HEAD" >&2
    fi
fi

# Install zalmoxis package as editable
pip install -U -e . || { echo "ERROR: editable install failed" >&2; exit 1; }

# Back to old folder
cd $root

# Done
echo "Done!"
