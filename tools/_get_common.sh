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
# Each git call must answer for the folder its -C names, not for a repository these variables name.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_COMMON_DIR GIT_NAMESPACE

# Put <PROTEUS root>/.proteus-julia, whose one entry is a link julia to PYTHON_JULIAPKG_EXE, first
# on PATH, so that a bare julia is the Julia PROTEUS uses; unset or empty changes nothing, and a
# value that names no executable file returns 1 with an error.
use_proteus_julia() {
    local exe="${PYTHON_JULIAPKG_EXE:-}" dir="$proteus_root/.proteus-julia" target
    [ -n "$exe" ] || return 0
    if [ ! -f "$exe" ] || [ ! -x "$exe" ]; then
        echo "ERROR: PYTHON_JULIAPKG_EXE=$exe is not an executable file; point it at the Julia PROTEUS uses, or unset it to use the Julia on PATH." >&2
        return 1
    fi
    target="$(cd "$(dirname "$exe")" && pwd)/$(basename "$exe")"
    mkdir -p "$dir" && ln -sfn "$target" "$dir/julia" \
        || { echo "ERROR: could not link $dir/julia to $target." >&2; return 1; }
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

# Usage: guard_dirty_checkout <checkout-path> <script-name> [git pathspec...]. Exits 1, unless
# --force, when the checkout has modified tracked files or commits on no remote; pathspecs go to
# git status. Call it as a plain command: inside a command substitution the exit leaves only that.
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

    # Branches, and HEAD when it resolves: commits can sit on a branch that is
    # not checked out or on a detached HEAD, while a clone killed before its
    # first commit has no HEAD to read. Tags stay out of it, since a clone
    # carries the upstream's tags and the commits they hold are not local work.
    local dirty unpushed refs='--branches' rc=0
    dirty=$(git -C "$workpath" status --porcelain --untracked-files=no "$@" 2>/dev/null) || rc=$?
    if git -C "$workpath" rev-parse --verify --quiet HEAD >/dev/null 2>&1; then
        refs='--branches HEAD'
    fi
    unpushed=$(git -C "$workpath" log $refs --not --remotes --oneline -1 2>/dev/null) || rc=$?
    if [ "$rc" -ne 0 ]; then
        # An incomplete .git, or no git at all. Whether the checkout holds
        # local work is then unknown, so keep it.
        echo "ERROR: git could not report the state of $workpath, so whether" >&2
        echo "       it holds local work is unknown. Refusing to delete it." >&2
        echo "       Inspect the checkout, or run" >&2
        echo "       bash tools/$script --force  to discard it." >&2
        exit 1
    fi
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
    if ! tags=$(git -C "$dir" for-each-ref --format='%(refname:strip=2)' refs/tags); then
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
    # refs/tags/ passes a branch named tags/<tag>; without --quiet git names commits left behind.
    if ! git -C "$dir" -c advice.detachedHead=false checkout --detach "refs/tags/$tag"; then
        echo "ERROR: cannot check out tag $tag of $dist in $dir" >&2
        exit 1
    fi
    commit=$(git -C "$dir" rev-parse --short HEAD)
    echo "$dist: checked out release tag $tag at $commit"
}

# Usage: _proteus_feature_branch. Prints the PROTEUS branch, GITHUB_HEAD_REF or GITHUB_REF_NAME in CI,
# else the branch of the PROTEUS checkout; prints nothing for main, master or a detached HEAD.
_proteus_feature_branch() {
    local branch="${GITHUB_HEAD_REF:-${GITHUB_REF_NAME:-}}"
    if [ -z "$branch" ]; then
        branch=$(git -C "$proteus_root" rev-parse --abbrev-ref HEAD 2>/dev/null) || branch=""
    fi
    case "$branch" in
        '' | HEAD | main | master) ;;
        *) echo "$branch" ;;
    esac
}

# Usage: paired_branch_of <clone-dir>. Prints the PROTEUS feature branch when the origin of the clone
# has a branch of that name, read from its remote-tracking refs without a fetch; else prints nothing.
paired_branch_of() {
    local branch
    branch=$(_proteus_feature_branch)
    if [ -n "$branch" ] &&
        git -C "$1" rev-parse --verify --quiet "refs/remotes/origin/$branch" >/dev/null 2>&1; then
        echo "$branch"
    fi
}

# Usage: _note_paired_without_release <dir> <branch> <tag> <head> <tag-commit> [<final dir>]. For a paired
# branch whose HEAD lacks the commit of the floor tag, prints the line that says so and records the
# summary entry, under <final dir> when the clone is still in its temporary folder.
_note_paired_without_release() {
    local rel=""
    if [ -n "$4" ] && [ -n "$5" ]; then
        rel=$(_relation_to_tag "$1" "$4" "$5")
    fi
    case "$rel" in
        'behind the tag' | 'on a line of history without the tag')
            echo "    This branch does not hold the release $3: the PROTEUS install" \
                "takes that release from PyPI, not this folder."
            record_clone_stays "${6:-$1}" "paired branch $2, without the release $3"
            ;;
    esac
}

# Usage: checkout_paired_or_floor <distribution> <checkout-dir> <repo name>. Checks out the paired
# branch of a new clone (paired_branch_of), else the floor tag; exits 1 on failure, so call it plainly.
checkout_paired_or_floor() {
    local dist="$1" dir="$2" name="$3" branch tag
    branch=$(paired_branch_of "$dir")
    if [ -z "$branch" ]; then
        checkout_floor_tag "$dist" "$dir"
        return 0
    fi
    git -C "$dir" checkout "$branch" ||
        { echo "ERROR: cannot checkout $name branch $branch" >&2; exit 1; }
    echo "PROTEUS is on branch '$branch'; installed paired $name branch at" \
        "$(git -C "$dir" rev-parse --short HEAD)"
    tag=$(_floor_tag_of "$dist" "$dir" 2>/dev/null) || tag=""
    if [ -n "$tag" ]; then
        _note_paired_without_release "$dir" "$branch" "$tag" "$(git -C "$dir" rev-parse HEAD)" \
            "$(git -C "$dir" rev-parse --verify --quiet "refs/tags/$tag^{commit}" || true)" \
            "${dir%.tmp.*}"
    fi
}

# Usage: unchanged_note <path>. Prints "; <path> is unchanged" for an error message when <path> exists.
unchanged_note() { [ ! -e "$1" ] || echo "; $1 is unchanged"; }

# Usage: is_own_clone <dir>. Succeeds when <dir> is the top level of its own git clone. An empty or
# broken .git makes git answer for the repository around <dir>, so prove this before any other git call.
is_own_clone() {
    local dir="${1%/}" top
    top=$(git -C "$dir" rev-parse --show-toplevel 2>/dev/null) || return 1
    [ -n "$top" ] && [ "$top" -ef "$dir" ]
}

# Usage: expected_text <distribution> <tag>. Prints what PROTEUS expects of a module clone: the
# release tag when the clone has it, else the floor version in pyproject.toml.
expected_text() {
    local floor
    if [ -n "$2" ]; then
        echo "PROTEUS expects the release tag $2"
    elif floor=$(python "$proteus_tools_dir/_module_pins.py" "$1" floor 2>/dev/null) &&
        [ -n "$floor" ]; then
        echo "PROTEUS expects release $floor"
    else
        echo "the release tag could not be determined"
    fi
}

# Usage: record_clone_stays <clone-dir> <summary>. Adds "<clone name> (<summary>)" to the file that
# PROTEUS_CLONE_STAYS names, when it is set, for the summary at the end of install.sh.
record_clone_stays() {
    local dir="${1%/}"
    if [ -n "${PROTEUS_CLONE_STAYS:-}" ]; then
        printf '%s (%s)\n' "${dir##*/}" "$2" >>"$PROTEUS_CLONE_STAYS" || true
    fi
}

# Usage: _relation_to_tag <clone-dir> <head> <tag-commit>. Prints where HEAD lies against the tag,
# read with merge-base only; prints nothing when git cannot compare the two.
_relation_to_tag() {
    local rc=0
    git -C "$1" merge-base --is-ancestor "$2" "$3" 2>/dev/null || rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "behind the tag"
        return 0
    fi
    [ "$rc" -eq 1 ] || return 0
    rc=0
    git -C "$1" merge-base --is-ancestor "$3" "$2" 2>/dev/null || rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "newer than the tag"
    elif [ "$rc" -eq 1 ]; then
        echo "on a line of history without the tag"
    fi
}

# Usage: _clone_advice <distribution> <clone-dir> <script-label> <tag> <readable>. Prints the commands
# that move a readable clone to the release tag by hand and reinstall it, and the one that replaces it.
_clone_advice() {
    local dir tag
    dir=$(printf '%q' "${2%/}")
    local how="From the PROTEUS folder, replace"
    if [ "$5" = true ]; then
        echo "    From the PROTEUS folder, move it to the release tag by hand with:"
        echo "        git -C $dir fetch --tags"
        if [ -n "$4" ]; then
            tag=$(printf '%q' "$4")
            echo "        git -C $dir checkout --detach refs/tags/$tag"
        else
            echo "        git -C $dir checkout --detach \"refs/tags/\$(git -C $dir tag -l |" \
                "python tools/_module_pins.py $1 floor-tag)\""
        fi
        echo "        pip install -e $dir"
        how="Or replace"
    fi
    echo "    $how it with a new clone, which deletes everything in it, with:"
    echo "        bash tools/$3 --force"
}

# Usage: _local_changes_note <clone-dir>. Prints ", with local changes" when a tracked file differs from
# HEAD; --no-optional-locks keeps git status from rewriting the index.
_local_changes_note() {
    local changes
    changes=$(git --no-optional-locks -C "$1" status --porcelain --untracked-files=no 2>/dev/null) ||
        changes=""
    [ -z "$changes" ] || echo ", with local changes"
}

# Usage: _commit_text <clone-dir> <head>. Prints "commit <short hash>", followed by what git describe
# names in brackets when that is a tag and not the hash alone.
_commit_text() {
    local desc
    desc=$(git -C "$1" describe --tags --always HEAD 2>/dev/null) || desc=""
    case "$2" in
        "$desc"*) echo "commit ${2:0:7}" ;;
        *) echo "commit ${2:0:7} ($desc)" ;;
    esac
}

# Usage: _report_pinned_clone <distribution> <clone-dir> <script-label> <pin>. Reports an existing
# clone against a commit pin and prints advice when HEAD does not match the pin commit.
_report_pinned_clone() {
    local dist="$1" dir="${2%/}" label="$3" pin="$4" head="" pin_commit="" found="" summary=""
    local qdir qref
    if [ ! -e "$dir/.git" ] && [ ! -L "$dir/.git" ]; then
        found="not a git clone"
        summary="$found"
    elif ! is_own_clone "$dir"; then
        found="not a readable git clone"
        summary="$found"
    else
        head=$(git -C "$dir" rev-parse --verify --quiet HEAD 2>/dev/null) || head=""
        pin_commit=$(git -C "$dir" rev-parse --verify --quiet "$pin^{commit}" \
            2>/dev/null) || pin_commit=""
        if [ -n "$head" ] && [ "$head" = "$pin_commit" ]; then
            echo "$dist: $dir is at the pinned commit ${pin:0:8}$(_local_changes_note "$dir")"
            return 0
        fi
        found=$(_commit_text "$dir" "$head")
        summary="commit ${head:0:7}"
    fi
    echo "$dist: $dir stays as it is: $found; not at the pinned commit ${pin:0:8}"
    record_clone_stays "$dir" "$summary"
    qdir=$(printf '%q' "$dir")
    if [ -n "$head" ]; then
        qref=$(printf '%q' "$pin")
        echo "    From the PROTEUS folder, move it to the pinned commit by hand with:"
        echo "        git -C $qdir fetch origin"
        echo "        git -C $qdir checkout --detach $qref"
        if [ -f "$dir/pyproject.toml" ] || [ -f "$dir/setup.py" ]; then
            echo "        pip install -e $qdir"
        fi
        echo "    Or replace it with a new clone, which deletes everything in it, with:"
    else
        echo "    From the PROTEUS folder, replace it with a new clone, which deletes everything in it, with:"
    fi
    echo "        bash tools/$label --force"
}

# Usage: report_clone_state <dist> <clone-dir> <label> [pin]. Reads <clone-dir> with read-only
# git calls; prints one line for a clone at its tag, newer than it or on the paired branch (with
# get_pairs_branch=true; 2 lines and a summary entry when HEAD lacks the tag), else stay and advice.
report_clone_state() {
    [ -n "${4:-}" ] && { _report_pinned_clone "$1" "$2" "$3" "$4"; return 0; }
    local dist="$1" dir="${2%/}" label="$3" tag="" tag_commit="" head="" rel="" branch=""
    local found summary readable=false
    if [ ! -e "$dir/.git" ] && [ ! -L "$dir/.git" ]; then
        found="not a git clone"
        summary="$found"
    elif ! is_own_clone "$dir"; then
        found="not a readable git clone"
        summary="$found"
    else
        readable=true
        tag=$(_floor_tag_of "$dist" "$dir" 2>/dev/null) || tag=""
        head=$(git -C "$dir" rev-parse --verify --quiet HEAD 2>/dev/null) || head=""
        if [ -n "$tag" ]; then
            tag_commit=$(git -C "$dir" rev-parse --verify --quiet "refs/tags/$tag^{commit}" \
                2>/dev/null) || tag_commit=""
        fi
        if [ -n "$head" ] && [ "$head" = "$tag_commit" ]; then
            echo "$dist: $dir is at the release tag $tag$(_local_changes_note "$dir")"
            return 0
        fi
        branch=$(git -C "$dir" symbolic-ref -q --short HEAD 2>/dev/null) || branch=""
        if [ -n "$branch" ] && [ "${get_pairs_branch:-false}" = true ] &&
            [ "$branch" = "$(paired_branch_of "$dir")" ]; then
            echo "$dist: $dir is on the paired branch $branch, as PROTEUS on that branch expects"
            _note_paired_without_release "$dir" "$branch" "$tag" "$head" "$tag_commit"
            return 0
        fi
        if [ -n "$head" ] && [ -n "$tag_commit" ]; then
            rel=$(_relation_to_tag "$dir" "$head" "$tag_commit")
        fi
        if [ -n "$branch" ]; then
            found="on branch $branch"
            summary="branch $branch"
        elif [ -z "$head" ]; then
            found="a HEAD that git cannot resolve"
            summary="HEAD unknown"
        else
            found=$(_commit_text "$dir" "$head")
            summary="commit ${head:0:7}"
        fi
        case "$rel" in
            '') ;;
            newer*)
                echo "$dist: $dir is newer than the release tag $tag ($summary)"
                return 0
                ;;
            behind*) found="$found, $rel"; summary="$summary, behind" ;;
            *) found="$found, $rel"; summary="$summary, other history" ;;
        esac
    fi
    echo "$dist: $dir stays as it is: $found; $(expected_text "$dist" "$tag")"
    record_clone_stays "$dir" "$summary"
    _clone_advice "$dist" "$dir" "$label" "$tag" "$readable"
}

# Usage: is_empty_dir <path>. Succeeds when <path> is a directory, not a symlink, without any entry.
is_empty_dir() {
    local entries
    { [ -d "$1" ] && [ ! -L "$1" ]; } || return 1
    entries=$(ls -A "$1" 2>/dev/null) || return 1
    [ -z "$entries" ]
}

# Usage: install_existing_clone <dist> <clone-dir> <label> [prepare-fn] [pin]. Returns
# under --force or with nothing or an empty folder at <clone-dir>; else reports what is there, runs the
# prepare function on it, installs it editable as it is and exits 0. No git call writes to it.
install_existing_clone() {
    local dist="$1" dir="${2%/}" label="$3" prepare="${4:-}" pin="${5:-}"
    if [ "${get_force:-false}" = true ] || { [ ! -e "$dir" ] && [ ! -L "$dir" ]; } ||
        is_empty_dir "$dir"; then
        return 0
    fi
    report_clone_state "$dist" "$dir" "$label" "$pin"
    if [ -n "$prepare" ]; then
        "$prepare" "$dir" || exit 1
        echo "$dist: the build step ran in the existing folder $dir"
    fi
    if ! pip install -U -e "$dir"; then
        echo "ERROR: pip could not install $dir (its message is above)" >&2
        if [ ! -f "$dir/pyproject.toml" ] && [ ! -f "$dir/setup.py" ]; then
            echo "       It has no pyproject.toml or setup.py; to replace it with a new clone, run" >&2
            echo "       bash tools/$label --force  from the PROTEUS folder" >&2
        fi
        exit 1
    fi
    echo "Done!"
    exit 0
}

# Usage: swap_in_checkout <new-dir> <workpath>. Moves an existing <workpath> aside (without --force only
# an empty folder), <new-dir> into its place, then removes the old copy; a copy that will not go is a
# warning. Returns 1, with what was at <workpath> back there, when <new-dir> may not or cannot go there.
swap_in_checkout() {
    local new="${1%/}" dest="${2%/}" old="" n=0 back
    if [ "${get_force:-false}" != true ] && { [ -e "$dest" ] || [ -L "$dest" ]; } &&
        ! is_empty_dir "$dest"; then
        echo "ERROR: $dest appeared while the new clone was made; it is unchanged" >&2
        return 1
    fi
    if [ -e "$dest" ] || [ -L "$dest" ]; then
        old="$dest.old.$$"
        while [ -e "$old" ] || [ -L "$old" ]; do
            n=$((n + 1))
            old="$dest.old.$$.$n"
        done
        # A signal during the 2 moves puts the old checkout back; the caller's EXIT trap still runs.
        back="_swap_interrupted $(printf '%q' "$old") $(printf '%q' "$dest")"
        trap "$back 129" HUP
        trap "$back 130" INT
        trap "$back 143" TERM
        if ! mv "$dest" "$old"; then
            trap - HUP INT TERM
            echo "ERROR: cannot move $dest aside to put the new checkout there; it is unchanged" >&2
            return 1
        fi
    fi
    if ! mv "$new" "$dest"; then
        trap - HUP INT TERM
        if [ -z "$old" ]; then
            echo "ERROR: cannot move the new checkout $new to $dest" >&2
        elif mv "$old" "$dest"; then
            echo "ERROR: cannot move the new checkout $new to $dest; $dest is unchanged" >&2
        else
            echo "ERROR: cannot move the new checkout $new to $dest; the old one is at $old" >&2
        fi
        return 1
    fi
    trap - HUP INT TERM
    if [ -n "$old" ] && ! rm -rf "$old" 2>/dev/null; then
        echo "WARNING: the old checkout could not be removed completely; delete $old by hand" >&2
    fi
    return 0
}

# Usage: _swap_interrupted <old> <dest> <status>. The signal handler of swap_in_checkout: moves <old>
# back when <dest> is missing, says what it did and exits with <status>.
_swap_interrupted() {
    trap - HUP INT TERM
    if [ ! -e "$1" ] && [ ! -L "$1" ]; then
        echo "ERROR: interrupted before replacing $2; it is unchanged" >&2
    elif [ ! -e "$2" ] && [ ! -L "$2" ] && mv "$1" "$2"; then
        echo "ERROR: interrupted while replacing $2; the old checkout is back in place" >&2
    else
        echo "ERROR: interrupted while replacing $2; the old checkout is at $1" >&2
    fi
    exit "$3"
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
