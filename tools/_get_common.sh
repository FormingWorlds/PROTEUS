# shellcheck shell=bash
# Shared helpers for the tools/get_*.sh scripts, for bash 3.2 and later. Source this file near
# the top of a script with `|| exit 1`; it sets proteus_tools_dir and proteus_root.

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
    local arg
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

# Usage: guard_dirty_checkout <checkout-path> <script-name> [git pathspec...]. Exits 1, unless
# --force, when the checkout has modified tracked files or commits on no remote; pathspecs go to
# git status. An empty script name marks a clone no script manages, for advice to keep the work.
guard_dirty_checkout() {
    local workpath="$1"
    local script="$2"
    shift 2

    if [ "${get_force:-false}" = true ]; then
        return 0
    fi
    # A worktree or a submodule has a .git file, not a directory.
    if [ ! -e "$workpath/.git" ]; then
        return 0
    fi

    # Branches, and HEAD when it resolves (a clone killed before its first commit has none);
    # tags stay out, since their commits come from the upstream. The probes are not piped,
    # because a pipeline reports git's status only under pipefail.
    local dirty unpushed refs='--branches' rc=0 action='delete it'
    dirty=$(git -C "$workpath" status --porcelain --untracked-files=no "$@" 2>/dev/null) || rc=$?
    if git -C "$workpath" rev-parse --verify --quiet HEAD >/dev/null 2>&1; then
        refs='--branches HEAD'
    fi
    unpushed=$(git -C "$workpath" log $refs --not --remotes --oneline -1 2>/dev/null) || rc=$?
    if [ -z "$script" ]; then
        action='move it to another commit'
    fi
    if [ "$rc" -ne 0 ]; then
        echo "ERROR: git could not report the state of $workpath, so whether" >&2
        echo "       it holds local work is unknown. Refusing to $action." >&2
        if [ -n "$script" ]; then
            echo "       Inspect the checkout, or run" >&2
            echo "       bash tools/$script --force  to discard it." >&2
        else
            echo "       Inspect the checkout with git status and git log." >&2
        fi
        exit 1
    fi
    if [ -n "$dirty" ] || [ -n "$unpushed" ]; then
        echo "ERROR: $workpath has uncommitted changes or commits not on a remote." >&2
        if [ -n "$script" ]; then
            echo "       Refusing to delete it. Commit and push your work, or run" >&2
            echo "       bash tools/$script --force  to discard the checkout." >&2
        else
            echo "       Refusing to move it to another commit. Commit, push or stash" >&2
            echo "       the work first, then run the command again." >&2
        fi
        exit 1
    fi
}

# Report whether GitHub accepts an SSH key, as "true" or "false" on stdout.
#
# `ssh -T git@github.com` exits 1 when the key is accepted (GitHub refuses
# the interactive shell it was asked for) and 255 when it is not, so exit
# code 1 is the success signal. The probe honours GIT_SSH_COMMAND, so a
# caller such as CI can make it non-interactive and fast-failing; the
# expansion is unquoted so that its options reach ssh as separate words,
# which means an option value containing a space does not survive. Its own
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

# Usage: checkout_floor_tag <distribution> <checkout-dir>. Checks out the one tag whose
# version equals the distribution's floor in pyproject.toml (detached HEAD) and prints it.
# Exits 1, with the cause on stderr, when that fails, so call it as a plain command.
checkout_floor_tag() {
    local dist="$1" dir="$2" tags tag commit
    if ! command -v python >/dev/null 2>&1; then
        echo "ERROR: python is not on PATH, so the $dist floor in pyproject.toml cannot be read;" >&2
        echo "       activate the PROTEUS Python environment first" >&2
        exit 1
    fi
    if ! tags=$(git -C "$dir" tag -l); then
        echo "ERROR: git cannot list the tags of $dir; see the git message above" >&2
        exit 1
    fi
    if ! tag=$(printf '%s\n' "$tags" | python "$proteus_tools_dir/_module_pins.py" "$dist" floor-tag); then
        echo "ERROR: cannot choose the $dist release tag in $dir; see the message above" >&2
        exit 1
    fi
    # Without --quiet, git warns when the move leaves commits of a detached HEAD behind.
    if ! git -C "$dir" -c advice.detachedHead=false checkout "tags/$tag"; then
        echo "ERROR: cannot check out tag $tag of $dist in $dir" >&2
        exit 1
    fi
    commit=$(git -C "$dir" rev-parse --short HEAD)
    echo "$dist: checked out release tag $tag at $commit"
}

# Download <dest> from the first of the urls that serves a file with the given
# SHA-256, skipping an empty url. Exits 1, with every url's failure named on
# stderr, when none does.
fetch_verified() {
    local sha256="$1" dest="$2" url got
    shift 2
    if [ -z "$sha256" ]; then
        echo "ERROR: no SHA-256 pin for $(basename "$dest"); check tools/_module_pins.py" >&2
        return 1
    fi
    if ! command -v shasum >/dev/null 2>&1 && ! command -v sha256sum >/dev/null 2>&1; then
        echo "ERROR: neither shasum nor sha256sum is installed; install one of them" >&2
        return 1
    fi
    for url in "$@"; do
        [ -n "$url" ] || continue
        echo "Downloading $url"
        if curl -fLsS --retry 3 --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
            "$url" -o "$dest"; then
            got=$( (shasum -a 256 "$dest" 2>/dev/null || sha256sum "$dest") | awk '{print $1}')
            [ "$got" = "$sha256" ] && return 0
            echo "WARNING: $url served a file with SHA-256 $got, not $sha256" >&2
        else
            echo "WARNING: download from $url failed" >&2
        fi
    done
    rm -f "$dest"
    echo "ERROR: no source served $(basename "$dest") with SHA-256 $sha256" >&2
    return 1
}
