#!/bin/bash
# Usage: get_python_module.sh <RepoName> [--force]. Clones FormingWorlds/<RepoName> into ./<RepoName>/
# at the release tag of its fwl-<lowercase name> floor and installs it editable; an existing folder is
# installed as it is. --force replaces it; PROTEUS_MODULE_GIT_BASE sets the URL prefix.

set -euo pipefail

# Shared helpers: see tools/_get_common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_get_common.sh" || exit 1

# The modules without a setup script of their own; each installs as fwl-<lowercase name>.
modules="MORS JANUS CALLIOPE ZEPHYRUS"

get_parse_args "$@"
repo="$get_install_path"
names=0
for arg in "$@"; do [ "$arg" = --force ] || names=$((names + 1)); done
dist=""
spelled=""
wanted=$(printf '%s' "$repo" | tr '[:upper:]' '[:lower:]')
for name in $modules; do
    lower=$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]')
    if [ "$name" = "$repo" ]; then
        dist="fwl-$lower"
    elif [ "$lower" = "$wanted" ]; then
        spelled="$name"
    fi
done
if [ -z "$dist" ] || [ "$names" -ne 1 ]; then
    echo "usage: bash tools/get_python_module.sh <RepoName> [--force]" >&2
    echo "       RepoName is one of: $modules" >&2
    [ -z "$spelled" ] || echo "       The name is case-sensitive: write $spelled, not $repo" >&2
    exit 2
fi

echo "Set up $repo..."
workpath="$proteus_root/$repo"

install_existing_clone "$dist" "$workpath" "get_python_module.sh $repo"

# The new clone replaces the checkout only once it is at the release tag.
tmp_clone="$workpath.tmp.$$"
rm -rf "$tmp_clone"
trap 'rm -rf "$tmp_clone"' EXIT

base="${PROTEUS_MODULE_GIT_BASE:-}"
if [ -z "$base" ]; then
    if [ "$(github_use_ssh)" = true ]; then
        base="git@github.com:"
    else
        base="https://github.com/"
    fi
fi
uri="${base}FormingWorlds/$repo.git"
echo "Cloning $uri -> $tmp_clone"
git clone "$uri" "$tmp_clone" ||
    { echo "ERROR: git clone failed$(unchanged_note "$workpath")" >&2; exit 1; }
checkout_floor_tag "$dist" "$tmp_clone"

swap_in_checkout "$tmp_clone" "$workpath" || exit 1
trap - EXIT

pip install -U -e "$workpath" || { echo "ERROR: editable install failed" >&2; exit 1; }

echo "Done!"
