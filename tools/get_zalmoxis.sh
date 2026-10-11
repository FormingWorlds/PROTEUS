#!/bin/bash
# Clone Zalmoxis into ./Zalmoxis/ and install it editable. A new clone, and the one --force puts in
# place of an old one, takes the Zalmoxis branch named like the PROTEUS feature branch, if any, else
# the fwl-zalmoxis floor tag. An existing folder is installed as it is; only --force replaces it.

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
install_existing_clone fwl-zalmoxis "$workpath" get_zalmoxis.sh

# The new clone replaces the checkout only once it is on its branch or release tag.
tmp_clone="${workpath%/}.tmp.$$"
rm -rf "$tmp_clone"
trap 'rm -rf "$tmp_clone"' EXIT

# Check SSH access to GitHub
use_ssh=$(github_use_ssh)

# Download
echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    uri="git@github.com:FormingWorlds/Zalmoxis.git"
else
    uri="https://github.com/FormingWorlds/Zalmoxis.git"
fi
echo "    $uri -> $tmp_clone"
git clone "$uri" "$tmp_clone" ||
    { echo "ERROR: git clone failed$(unchanged_note "$workpath")" >&2; exit 1; }

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
        if git -C "$tmp_clone" rev-parse --verify --quiet "refs/remotes/origin/$proteus_branch" \
            >/dev/null; then
            match_branch="$proteus_branch"
        fi
        ;;
esac

if [ -n "$match_branch" ]; then
    git -C "$tmp_clone" checkout "$match_branch" ||
        { echo "ERROR: cannot checkout Zalmoxis branch $match_branch" >&2; exit 1; }
    echo "PROTEUS is on branch '$proteus_branch'; installed paired Zalmoxis branch at" \
        "$(git -C "$tmp_clone" rev-parse --short HEAD)"
else
    checkout_floor_tag fwl-zalmoxis "$tmp_clone"
fi

swap_in_checkout "$tmp_clone" "$workpath" || exit 1
trap - EXIT
if [ -n "$match_branch" ]; then
    record_clone_stays "$workpath" "paired branch $match_branch"
fi

# Install zalmoxis package as editable
pip install -U -e "${workpath%/}" || { echo "ERROR: editable install failed" >&2; exit 1; }

# Done
echo "Done!"
