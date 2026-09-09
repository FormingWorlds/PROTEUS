# shellcheck shell=bash
#
# Shared shell helpers for the tools/get_*.sh module install scripts.
#
# This file is sourced, never executed. Source it near the top of a get_*
# script, before the first helper call:
#
#     _get_common="$(dirname "${BASH_SOURCE[0]}")/_get_common.sh"
#     [ -f "$_get_common" ] || {
#         echo "ERROR: $_get_common is missing." >&2
#         exit 1
#     }
#     source "$_get_common"
#
# Every caller runs a get_* script by path (docs, install.sh, the CI setup
# action, and proteus install-all through src/proteus/utils/data.py), so
# ${BASH_SOURCE[0]} locates this file without needing a resolved path
# first, which is what makes the bootstrap above safe to run before
# portable_realpath exists.
#
# The helpers target bash 3.2, the version macOS ships, and behave the same
# whether or not the sourcing script enables `set -euo pipefail`. Two of
# them exit the script on failure (guard_dirty_checkout, resolve_module_pin)
# and so must be called as plain commands: inside a command substitution
# the exit would only leave the subshell.
#
# Sourcing this file sets:
#   proteus_tools_dir   absolute path of the tools/ directory
#   proteus_root        absolute path of the PROTEUS checkout root
#
# and defines:
#   portable_realpath     resolve a path, whether or not it exists yet
#   get_parse_args        split --force from an optional install path
#   guard_dirty_checkout  refuse to delete a checkout holding local work
#   github_use_ssh        report whether GitHub accepts an SSH key
#   github_ssh_url        rewrite an https GitHub URL for SSH transport
#   resolve_module_pin    read a module's pinned clone URL and ref

# Resolve a path to an absolute one, with symlinks and .. expanded.
#
# macOS before 13 (Catalina through Monterey) does not ship the coreutils
# realpath, so fall back to python3, which is always present in PROTEUS's
# Python environment. A path that does not exist yet is also rejected by
# realpath (BSD refuses a missing leaf, GNU a missing parent) and takes
# the same fallback: install destinations are resolved before git clone
# creates them, so a missing path must resolve rather than fail.
portable_realpath() {
    if command -v realpath >/dev/null 2>&1 && realpath "$1" 2>/dev/null; then
        return 0
    fi
    python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$1"
}

# The directory holding this file, without calling out to dirname, so that
# sourcing it depends on nothing but bash itself.
_get_common_dir="${BASH_SOURCE[0]%/*}"
if [ "$_get_common_dir" = "${BASH_SOURCE[0]}" ]; then
    _get_common_dir="."
fi
proteus_tools_dir=$(portable_realpath "$_get_common_dir")
proteus_root=$(portable_realpath "$proteus_tools_dir/..")

# Split the script's arguments into the --force switch and an optional
# install-path positional, reported as get_force (true / false) and
# get_install_path (empty when no path was given). Scripts with a fixed
# destination read get_force only; --force may appear before or after the
# path. guard_dirty_checkout honours get_force.
get_parse_args() {
    get_force=false
    get_install_path=""
    for arg in "$@"; do
        if [ "$arg" = "--force" ]; then
            get_force=true
        elif [ -z "$get_install_path" ]; then
            get_install_path="$arg"
        fi
    done
}

# Refuse to delete a checkout that holds local work, unless --force was
# given. Guarded states: modified tracked files, and commits that are on
# no remote. Untracked files (build artifacts, egg-info) do not block the
# refresh, because they are routine in a refreshed checkout.
#
# Usage: guard_dirty_checkout <checkout-path> <script-name> [git pathspec...]
#
# The script name goes into the recovery command in the error message.
# Trailing arguments are passed to git status, which get_socrates.sh uses
# to exclude its regenerable build config from the dirty test. Exits 1
# when the checkout is guarded, so call it as a plain command.
guard_dirty_checkout() {
    local workpath="$1"
    local script="$2"
    shift 2

    if [ "${get_force:-false}" = true ]; then
        return 0
    fi
    if [ ! -d "$workpath/.git" ]; then
        return 0
    fi

    local dirty unpushed
    dirty=$(git -C "$workpath" status --porcelain --untracked-files=no "$@" 2>/dev/null | head -1)
    unpushed=$(git -C "$workpath" log HEAD --not --remotes --oneline 2>/dev/null | head -1)
    if [ -n "$dirty" ] || [ -n "$unpushed" ]; then
        echo "ERROR: $workpath has uncommitted changes or commits not on a remote." >&2
        echo "       Refusing to delete it. Commit and push your work, or run" >&2
        echo "       bash tools/$script --force  to discard the checkout." >&2
        exit 1
    fi
}

# Report whether GitHub accepts an SSH key, as "true" or "false" on stdout.
#
# `ssh -T git@github.com` exits 1 when the key is accepted (GitHub refuses
# the interactive shell it was asked for) and 255 when it is not, so exit
# code 1 is the success signal. The probe honours GIT_SSH_COMMAND, so a
# caller such as CI can make it non-interactive and fast-failing. Its own
# output is sent to stderr, where the user still sees it, so that it
# cannot contaminate the answer read by a command substitution.
github_use_ssh() {
    local rc=0
    ${GIT_SSH_COMMAND:-ssh} -T git@github.com >&2 || rc=$?
    if [ "$rc" -eq 1 ]; then
        echo true
    else
        echo false
    fi
}

# Rewrite an https GitHub URL to its SSH form for cloning over SSH.
# A URL that is not on github.com is returned unchanged.
github_ssh_url() {
    printf '%s\n' "${1/https:\/\/github.com\//git@github.com:}"
}

# Read a module's pinned clone URL and ref from pyproject.toml, through
# tools/_module_pins.py, into module_url and module_ref. A module whose
# pin is missing or empty stops the install here, where the cause is
# named, rather than at a git clone with an empty argument. Exits 1 in
# that case, so call it as a plain command.
resolve_module_pin() {
    local module="$1"
    module_url=$(python "$proteus_tools_dir/_module_pins.py" "$module" url)
    module_ref=$(python "$proteus_tools_dir/_module_pins.py" "$module" ref)
    if [ -z "$module_url" ] || [ -z "$module_ref" ]; then
        echo "ERROR: could not resolve $module url/ref from pyproject.toml" >&2
        exit 1
    fi
}
