#!/bin/bash
# Clone Zalmoxis into ./Zalmoxis/ and install it editable. The checkout is the Zalmoxis branch of the
# same name when PROTEUS is on a feature branch that Zalmoxis also has, and otherwise the release
# tag whose version equals the fwl-zalmoxis floor in pyproject.toml. --force discards local work.

echo "Set up Zalmoxis..."

# Shared helpers: see tools/_get_common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_get_common.sh" || exit 1

# Path to PROTEUS folder
root="$proteus_root"

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

get_parse_args "$@"
workpath="$root/Zalmoxis/"
guard_dirty_checkout "$workpath" get_zalmoxis.sh

# Make room
rm -rf "$workpath"

# Check SSH access to GitHub
use_ssh=$(github_use_ssh)

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
    checkout_floor_tag fwl-zalmoxis "$workpath"
fi

# Install zalmoxis package as editable
pip install -U -e . || { echo "ERROR: editable install failed" >&2; exit 1; }

# Back to old folder
cd "$root"

# Done
echo "Done!"
