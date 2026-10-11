#!/bin/bash
# Clone Aragog into ./aragog/ and install it editable. A new clone, and the one --force puts in
# place of an old one, takes the Aragog branch named like the PROTEUS feature branch, if any, else
# the fwl-aragog floor tag; a re-run never replaces a clone, at most it moves it forward to that tag.

echo "Set up Aragog..."

# Shared helpers: see tools/_get_common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_get_common.sh" || exit 1

# Path to PROTEUS folder
root="$proteus_root"

# Paired-branch detection. When PROTEUS is on a feature branch and aragog has a
# branch of the same name, install that aragog branch so CI exercises the real
# paired code instead of the released floor tag. In GitHub Actions the branch is
# GITHUB_HEAD_REF (pull_request) or GITHUB_REF_NAME (push / workflow_dispatch);
# locally it is the checked-out branch. The default branch and a detached HEAD
# never pair, so main and tag builds keep the floor tag.
proteus_branch="${GITHUB_HEAD_REF:-${GITHUB_REF_NAME:-}}"
if [ -z "$proteus_branch" ]; then
    proteus_branch=$(git -C "$root" rev-parse --abbrev-ref HEAD 2>/dev/null)
fi

get_parse_args "$@"
workpath="$root/aragog/"
reuse_existing_clone fwl-aragog "$workpath" get_aragog.sh
guard_dirty_checkout "$workpath" get_aragog.sh
unchanged=""
if [ -e "$workpath" ]; then
    unchanged="; $workpath is unchanged"
fi

# The new clone replaces the checkout only once it is on its branch or release tag.
tmp_clone="${workpath%/}.tmp.$$"
rm -rf "$tmp_clone"
trap 'rm -rf "$tmp_clone"' EXIT

# Check SSH access to GitHub
use_ssh=$(github_use_ssh)

# Download
echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    uri="git@github.com:FormingWorlds/aragog.git"
else
    uri="https://github.com/FormingWorlds/aragog.git"
fi
echo "    $uri -> $tmp_clone"
git clone "$uri" "$tmp_clone" || { echo "ERROR: git clone failed$unchanged" >&2; exit 1; }

cd "$tmp_clone" || { echo "ERROR: cannot enter $tmp_clone" >&2; exit 1; }

# If PROTEUS is on a feature branch that aragog also has, install that branch;
# otherwise pin to the fwl-aragog version floor declared in PROTEUS's
# pyproject.toml, so main/tag builds are reproducible and unaffected. The full
# clone above already fetched every aragog branch, so the pairing test reads the
# local remote-tracking refs (no extra network round-trip, nothing to flake).
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
    git checkout "$match_branch" || { echo "ERROR: cannot checkout aragog branch $match_branch" >&2; exit 1; }
    echo "PROTEUS is on branch '$proteus_branch'; installed paired aragog branch at $(git rev-parse --short HEAD)"
else
    checkout_floor_tag fwl-aragog "$tmp_clone"
fi

cd "$root" || exit 1
swap_in_checkout "$tmp_clone" "$workpath" || exit 1
trap - EXIT
cd "$workpath" || { echo "ERROR: cannot enter $workpath" >&2; exit 1; }

# Install aragog package as editable
pip install -U -e . || { echo "ERROR: editable install failed" >&2; exit 1; }

# Back to old folder
cd "$root"

# Done
echo "Done!"
