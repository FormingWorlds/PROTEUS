#!/bin/bash
# Usage: get_python_module.sh <RepoName> [--force]. Clones FormingWorlds/<RepoName> into ./<RepoName>/
# at the release tag of its fwl-<lowercase name> floor in pyproject.toml and installs it editable. A
# set PROTEUS_MODULE_GIT_BASE (such as https://github.com/) is the URL prefix, in place of the SSH probe.

set -euo pipefail

# Shared helpers: see tools/_get_common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_get_common.sh" || exit 1

# The modules without a setup script of their own; each installs as fwl-<lowercase name>.
modules="MORS JANUS CALLIOPE ZEPHYRUS"

get_parse_args "$@"
repo="$get_install_path"
dist=""
for name in $modules; do
    if [ "$name" = "$repo" ]; then
        dist="fwl-$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]')"
    fi
done
if [ -z "$dist" ]; then
    echo "usage: bash tools/get_python_module.sh <RepoName> [--force]" >&2
    echo "       RepoName is one of: $modules" >&2
    exit 2
fi

echo "Set up $repo..."
workpath="$proteus_root/$repo"
guard_dirty_checkout "$workpath" "get_python_module.sh $repo"

# Make room
rm -rf "$workpath"

base="${PROTEUS_MODULE_GIT_BASE:-}"
if [ -z "$base" ]; then
    if [ "$(github_use_ssh)" = true ]; then
        base="git@github.com:"
    else
        base="https://github.com/"
    fi
fi
uri="${base}FormingWorlds/$repo.git"
echo "Cloning $uri -> $workpath"
if ! git clone "$uri" "$workpath"; then
    rm -rf "$workpath"
    echo "ERROR: git clone failed" >&2
    exit 1
fi

# The subshell takes the helper's exit, so a clone left on the default branch is removed.
if ! (checkout_floor_tag "$dist" "$workpath"); then
    rm -rf "$workpath"
    echo "ERROR: removed the new $repo clone, which is not at the $dist release tag" >&2
    exit 1
fi

pip install -U -e "$workpath" || { echo "ERROR: editable install failed" >&2; exit 1; }

echo "Done!"
