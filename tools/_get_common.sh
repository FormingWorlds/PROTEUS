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

# Put the directory of PYTHON_JULIAPKG_EXE, the Julia PROTEUS uses, first on PATH for this process
# when it names an executable file, so that a bare julia here is that Julia; otherwise change nothing.
use_proteus_julia() {
    local exe="${PYTHON_JULIAPKG_EXE:-}" dir
    if [ -z "$exe" ] || [ ! -f "$exe" ] || [ ! -x "$exe" ]; then
        return 0
    fi
    dir=$(cd "$(dirname "$exe")" && pwd)
    PATH="$dir:$PATH"
    export PATH
    echo "Julia: $exe (PYTHON_JULIAPKG_EXE)"
}

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

# Usage: _local_commits <checkout> <refs...>. Prints, one per line, the commits of the refs that are on
# no remote branch and below no tag that a remote lists; a tag of the user's own stays local work.
_local_commits() {
    local workpath="$1" remote listed excluded=""
    shift
    for remote in $(git -C "$workpath" remote); do
        listed=$(git -C "$workpath" ls-remote --tags "$remote" 2>/dev/null) || continue
        excluded="$excluded $(printf '%s\n' "$listed" | awk '{print "^" $1}')"
    done
    # shellcheck disable=SC2086
    git -C "$workpath" log --ignore-missing "$@" $excluded --not --remotes --oneline
}

# Print the number of lines of $1; an empty text has none.
_count_lines() {
    if [ -z "$1" ]; then
        echo 0
    else
        printf '%s\n' "$1" | awk 'END {print NR}'
    fi
}

# Usage: _local_work_counts <checkout> [git pathspec...]. Sets work_modified and work_local to the
# number of modified tracked files and of local commits (_local_commits) of <checkout>; returns 1
# when git cannot report them. Pathspecs go to git status.
_local_work_counts() {
    local workpath="$1" out refs='--branches'
    shift
    work_modified=0
    work_local=0
    # Not piped, because a pipeline reports git's status only under pipefail.
    out=$(git -C "$workpath" status --porcelain --untracked-files=no "$@" 2>/dev/null) || return 1
    work_modified=$(_count_lines "$out")
    # HEAD counts when it resolves; a clone killed before its first commit has none.
    if git -C "$workpath" rev-parse --verify --quiet HEAD >/dev/null 2>&1; then
        refs='--branches HEAD'
    fi
    out=$(git -C "$workpath" log $refs --not --remotes --oneline -1 2>/dev/null) || return 1
    if [ -n "$out" ]; then
        out=$(_local_commits "$workpath" $refs 2>/dev/null) || return 1
        work_local=$(_count_lines "$out")
    fi
    return 0
}

# Usage: guard_dirty_checkout <checkout-path> <script-name> [git pathspec...]. Exits 1, unless --force,
# on a path that is neither a git clone nor an empty directory, and on a clone with modified tracked
# files or local commits; pathspecs go to git status. An empty script name marks an unmanaged clone.
guard_dirty_checkout() {
    local workpath="$1"
    local script="$2"
    shift 2

    if [ "${get_force:-false}" = true ]; then
        return 0
    fi
    # A worktree or a submodule has a .git file, not a directory.
    if [ ! -e "$workpath/.git" ]; then
        if [ ! -e "$workpath" ] || { [ -d "$workpath" ] && [ -r "$workpath" ] &&
            [ -z "$(ls -A "$workpath" 2>/dev/null)" ]; }; then
            return 0
        fi
        echo "ERROR: $workpath is not a git clone, so it is not replaced." >&2
        if [ -n "$script" ]; then
            echo "       Move it away, or run  bash tools/$script --force  to discard it." >&2
        else
            echo "       Move it away first." >&2
        fi
        exit 1
    fi
    _refuse_local_work "$workpath" "$script" "" "$@"
}

# Usage: _refuse_local_work <checkout> <script-name> <distribution> [git pathspec...]. The local-work
# half of guard_dirty_checkout; with a distribution, the refusal names it, the count of each kind of
# work and the release tag PROTEUS expects (stay_line).
_refuse_local_work() {
    local workpath="$1" script="$2" dist="$3" rc=0 action='delete it' tag found
    shift 3
    _local_work_counts "$workpath" "$@" || rc=$?
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
    if [ "$work_modified" -gt 0 ] || [ "$work_local" -gt 0 ]; then
        if [ -n "$dist" ]; then
            tag=$(_floor_tag_of "$dist" "$workpath" 2>/dev/null) || tag=""
            found="$(_plural "$work_modified" 'modified tracked file') and"
            found="$found $(_plural "$work_local" 'local commit') (on no remote)"
            echo "ERROR: $(stay_line "$dist" "$workpath" "$found" "$(expected_tag_text "$tag")")" >&2
        else
            echo "ERROR: $workpath has uncommitted changes or commits not on a remote." >&2
        fi
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

# Print the one tag of <checkout-dir> whose version equals the floor of <distribution> in
# pyproject.toml; returns 1, with the cause on stderr, otherwise. Usage: <distribution> <checkout-dir>.
_floor_tag_of() {
    local dist="$1" dir="$2" tags
    if ! command -v python >/dev/null 2>&1; then
        echo "ERROR: python is not on PATH, so the $dist floor in pyproject.toml cannot be read;" >&2
        echo "       activate the PROTEUS Python environment first" >&2
        return 1
    fi
    if ! tags=$(git -C "$dir" tag -l); then
        echo "ERROR: git cannot list the tags of $dir; see the git message above" >&2
        return 1
    fi
    if ! printf '%s\n' "$tags" | python "$proteus_tools_dir/_module_pins.py" "$dist" floor-tag; then
        echo "ERROR: cannot choose the $dist release tag in $dir; see the message above" >&2
        return 1
    fi
}

# Usage: checkout_floor_tag <distribution> <checkout-dir>. Checks out the one tag whose
# version equals the distribution's floor in pyproject.toml (detached HEAD) and prints it.
# Exits 1, with the cause on stderr, when that fails, so call it as a plain command.
checkout_floor_tag() {
    local dist="$1" dir="$2" tag commit
    tag=$(_floor_tag_of "$dist" "$dir") || exit 1
    # Without --quiet, git warns when the move leaves commits of a detached HEAD behind.
    if ! git -C "$dir" -c advice.detachedHead=false checkout "tags/$tag"; then
        echo "ERROR: cannot check out tag $tag of $dist in $dir" >&2
        exit 1
    fi
    commit=$(git -C "$dir" rev-parse --short HEAD)
    echo "$dist: checked out release tag $tag at $commit"
}

# Usage: swap_in_checkout <new-dir> <workpath>. Moves an existing <workpath> aside, <new-dir> into its
# place, and only then removes the old copy; a copy that will not go is a warning. Returns 1, with
# the old checkout back at <workpath>, when <new-dir> cannot be moved there.
swap_in_checkout() {
    local new="${1%/}" dest="${2%/}" old="" n=0
    if [ -e "$dest" ] || [ -L "$dest" ]; then
        old="$dest.old.$$"
        while [ -e "$old" ] || [ -L "$old" ]; do
            n=$((n + 1))
            old="$dest.old.$$.$n"
        done
        if ! mv "$dest" "$old"; then
            echo "ERROR: cannot move $dest aside to put the new checkout there; it is unchanged" >&2
            return 1
        fi
    fi
    if ! mv "$new" "$dest"; then
        if [ -z "$old" ]; then
            echo "ERROR: cannot move the new checkout $new to $dest" >&2
        elif mv "$old" "$dest"; then
            echo "ERROR: cannot move the new checkout $new to $dest; $dest is unchanged" >&2
        else
            echo "ERROR: cannot move the new checkout $new to $dest; the old one is at $old" >&2
        fi
        return 1
    fi
    if [ -n "$old" ] && ! rm -rf "$old" 2>/dev/null; then
        echo "WARNING: the old checkout could not be removed completely; delete $old by hand" >&2
    fi
    return 0
}

# Print "<count> <noun>", with an s after the noun unless the count is 1.
_plural() {
    if [ "$1" -eq 1 ]; then
        echo "$1 $2"
    else
        echo "$1 $2s"
    fi
}

# Print the expected half of a stay line for the release tag $1, or say that it is unknown.
expected_tag_text() {
    if [ -n "$1" ]; then
        echo "PROTEUS expects the release tag $1"
    else
        echo "the release tag could not be determined"
    fi
}

# Usage: stay_line <distribution> <clone-dir> <found> <expected>. Prints the one line that says a
# module clone stays as it is, what was found there and what PROTEUS expects.
stay_line() {
    echo "$1: ${2%/} stays as it is: $3; $4"
}

# Usage: report_clone_stays <distribution> <clone-dir> <found> <summary> <expected>. Prints stay_line
# and adds "<clone name> (<summary>)" to the file PROTEUS_CLONE_STAYS names, when set, for install.sh.
report_clone_stays() {
    local dir="${2%/}"
    stay_line "$1" "$dir" "$3" "$5"
    if [ -n "${PROTEUS_CLONE_STAYS:-}" ]; then
        printf '%s (%s)\n' "${dir##*/}" "$4" >>"$PROTEUS_CLONE_STAYS" || true
    fi
}

# Usage: move_clone_to_floor_tag <distribution> <clone-dir> [script-label]. Moves a clone in place to
# the floor release tag when its HEAD is behind it, and leaves a clone on a branch or at or past the
# tag as it is. Exits 1 on local work or a failure; the label names the script that replaces it.
move_clone_to_floor_tag() {
    local dist="$1" dir="${2%/}" label="${3:-}" branch tag tag_commit head describe rc=0
    # A branch is never moved, so its local work needs no guard; its tag comes without a fetch.
    if [ -e "$dir/.git" ] && branch=$(git -C "$dir" symbolic-ref -q --short HEAD 2>/dev/null); then
        tag=$(_floor_tag_of "$dist" "$dir" 2>/dev/null) || tag=""
        report_clone_stays "$dist" "$dir" "on branch $branch" "branch $branch" \
            "$(expected_tag_text "$tag")"
        return 0
    fi
    if [ -e "$dir/.git" ]; then
        # The tags come first, so that a refusal of local work names the tag of the floor.
        if ! git -C "$dir" fetch --tags --quiet; then
            echo "WARNING: cannot fetch the tags of $dir; going on with the tags it has" >&2
        fi
        (_refuse_local_work "$dir" "" "$dist") || rc=1
    else
        (guard_dirty_checkout "$dir" "") || rc=1
    fi
    if [ "$rc" -ne 0 ]; then
        if [ -n "$label" ]; then
            echo "       To replace the clone with a new one instead, run  bash tools/$label --force" >&2
        fi
        exit 1
    fi
    tag=$(_floor_tag_of "$dist" "$dir") || exit 1
    if ! tag_commit=$(git -C "$dir" rev-parse --verify --quiet "refs/tags/$tag^{commit}") ||
        ! head=$(git -C "$dir" rev-parse --verify --quiet HEAD); then
        echo "ERROR: git cannot resolve HEAD and the tag $tag in $dir" >&2
        exit 1
    fi
    if [ "$head" = "$tag_commit" ]; then
        echo "$dist: $dir is at the release tag $tag"
        return 0
    fi
    git -C "$dir" merge-base --is-ancestor "$head" "$tag_commit" || rc=$?
    if [ "$rc" -eq 0 ]; then
        checkout_floor_tag "$dist" "$dir"
        return 0
    elif [ "$rc" -ne 1 ]; then
        echo "ERROR: git cannot compare HEAD of $dir with the tag $tag" >&2
        exit 1
    fi
    describe=$(git -C "$dir" describe --tags --always HEAD 2>/dev/null) || describe="${head:0:7}"
    rc=0
    git -C "$dir" merge-base --is-ancestor "$tag_commit" "$head" || rc=$?
    if [ "$rc" -eq 0 ]; then
        report_clone_stays "$dist" "$dir" "commit ${head:0:7} ($describe), newer than the tag" \
            "commit ${head:0:7}, newer" "$(expected_tag_text "$tag")"
    else
        report_clone_stays "$dist" "$dir" \
            "commit ${head:0:7} ($describe), on a line of history without the tag" \
            "commit ${head:0:7}, other history" "$(expected_tag_text "$tag")"
    fi
    return 0
}

# Usage: reuse_existing_clone <distribution> <clone-dir> <script-label>. Without --force and with a git
# clone at <clone-dir>, moves it (move_clone_to_floor_tag), installs it editable and exits 0; else returns.
reuse_existing_clone() {
    local dist="$1" dir="${2%/}" label="$3"
    if [ "${get_force:-false}" = true ] || [ ! -e "$dir/.git" ]; then
        return 0
    fi
    move_clone_to_floor_tag "$dist" "$dir" "$label"
    pip install -U -e "$dir" || { echo "ERROR: editable install failed" >&2; exit 1; }
    echo "Done!"
    exit 0
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
