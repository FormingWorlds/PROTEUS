#!/bin/bash
# Clone Obliqua at its pin into ./Obliqua/ (or the path given) and set up its Julia project.
# An existing Obliqua folder stays as it is; --force replaces an Obliqua clone at the pin;
# a folder that is no Obliqua project, the PROTEUS folder and the home folder are refused.

set -euo pipefail

# Shared helpers: see tools/_get_common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_get_common.sh" || exit 1
use_proteus_julia

get_force=false
register=false
run_tests=false
dest="$proteus_root/Obliqua"

for arg in "$@"; do
    case "$arg" in
        --force) get_force=true ;;
        --register) register=true ;;
        --test) run_tests=true ;;
        0) ;;
        --*) echo "ERROR: unknown option $arg" >&2; exit 1 ;;
        *) dest="$arg" ;;
    esac
done

case "$dest" in
    "") echo "ERROR: the destination is empty" >&2; exit 1 ;;
    *$'\n'*) echo "ERROR: the destination holds a line break; nothing is changed" >&2; exit 1 ;;
    /*) ;;
    *) dest="$PWD/$dest" ;;
esac

# A trailing "/" or "/." names the same folder.
while :; do
    case "$dest" in
        */.) dest="${dest%/.}" ;;
        ?*/) dest="${dest%/}" ;;
        *) break ;;
    esac
done
[ -n "$dest" ] || dest="/"

# The physical path: the deepest folder of the path that exists, resolved, plus the rest of the path.
base="$dest"
rest=""
while [ ! -d "$base" ]; do
    if [ -e "$base" ] || [ -L "$base" ]; then
        echo "ERROR: $base is a file, a broken link or a folder that cannot be entered; move it away or delete it by hand; it is unchanged" >&2
        exit 1
    fi
    part="${base##*/}"
    base="${base%/*}"
    [ -n "$base" ] || base="/"
    case "$part" in
        "") ;;
        .|..)
            echo "ERROR: $dest goes through a folder that does not exist and then through '$part'; give the path of the folder itself; nothing is changed" >&2
            exit 1 ;;
        *) rest="$part${rest:+/$rest}" ;;
    esac
done
real=$(cd -P "$base" 2>/dev/null && pwd -P) || real=""
if [ -z "$real" ]; then
    echo "ERROR: $base is a file, a broken link or a folder that cannot be entered; move it away or delete it by hand; it is unchanged" >&2
    exit 1
fi
if ! [ "$real" -ef "$base" ]; then
    echo "ERROR: $base resolves to another folder; nothing is changed" >&2
    exit 1
fi
exists=false
if [ -z "$rest" ]; then exists=true; fi
dest="${real%/}${rest:+/$rest}"
[ -n "$dest" ] || dest="/"

if [ "$exists" = true ]; then
    # The PROTEUS folder, the home folder and a folder that holds one of them are never the Obliqua folder.
    for keep in "$proteus_root" "${HOME:-}"; do
        [ -n "$keep" ] || continue
        up=$(cd -P "$keep" 2>/dev/null && pwd -P) || continue
        while :; do
            if [ "$up" -ef "$dest" ]; then
                echo "ERROR: $dest is $keep or a folder that holds it, and cannot be the Obliqua folder; it is unchanged" >&2
                exit 1
            fi
            if [ "$up" = "/" ]; then break; fi
            up=$(dirname "$up")
        done
    done
    if ! is_empty_dir "$dest" && ! grep -q '^name = "Obliqua"' "$dest/Project.toml" 2>/dev/null; then
        echo "ERROR: $dest is not an Obliqua project (no Project.toml that names Obliqua); move it away or delete it by hand; it is unchanged" >&2
        exit 1
    fi
    if [ "$get_force" = true ] && ! is_empty_dir "$dest"; then
        if [ -f "$dest/.git" ]; then
            echo "ERROR: $dest is a folder whose .git is a file (a git worktree or a submodule); move it away or delete it by hand; it is unchanged" >&2
            exit 1
        fi
        if ! is_own_clone "$dest" || ! git -C "$dest" rev-parse --verify --quiet HEAD >/dev/null 2>&1; then
            echo "ERROR: $dest holds files and is not a git clone of its own with a commit; move it away or delete it by hand; it is unchanged" >&2
            exit 1
        fi
    fi
fi

# The forced replace is for the Obliqua clone that the checks saw: a path that was absent or empty gets no forced swap.
if [ "$exists" != true ] || is_empty_dir "$dest"; then get_force=false; fi

if ! command -v julia >/dev/null 2>&1; then
    echo "ERROR: julia is not on PATH. Install Julia first (see https://github.com/FormingWorlds/Obliqua)." >&2
    exit 1
fi

ob_url="${OBLIQUA_GIT_URL:-$(python "$proteus_tools_dir/_module_pins.py" obliqua url || true)}"
ob_ref="${OBLIQUA_GIT_REF:-$(python "$proteus_tools_dir/_module_pins.py" obliqua ref || true)}"
if [ -z "$ob_url" ] || [ -z "$ob_ref" ]; then
    echo "ERROR: could not resolve obliqua url/ref from pyproject.toml" >&2
    exit 1
fi

if [ "$get_force" != true ] && [ "$exists" = true ] &&
    ! is_empty_dir "$dest"; then
    label="get_obliqua.sh"
    [ "$dest" -ef "$proteus_root/Obliqua" ] || label="get_obliqua.sh $(printf '%q' "$dest")"
    get_refuses_git_file=true
    report_clone_state Obliqua "$dest" "$label" "$ob_ref"
    cd "$dest"
    LD_LIBRARY_PATH="" julia --project=. -e 'using Pkg; Pkg.instantiate()' ||
        { echo "ERROR: julia instantiate failed in $dest" >&2; exit 1; }
else
    echo "Cloning Obliqua ($ob_url @ $ob_ref) into $dest..."
    clone_at_ref "$ob_url" "$ob_ref" "$dest"

    cd "$dest"
    echo "Obliqua at $(git -C "$dest" rev-parse --short HEAD)"
    LD_LIBRARY_PATH="" julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()' ||
        { echo "ERROR: julia resolve or instantiate failed in $dest" >&2; exit 1; }
fi

if [ "$register" = true ]; then
    echo "Registering Obliqua into the default Julia environment..."
    LD_LIBRARY_PATH="" julia -e 'using Pkg; Pkg.add(path=".")' ||
        { echo "ERROR: julia Pkg.add failed" >&2; exit 1; }
    LD_LIBRARY_PATH="" julia -e 'using Obliqua; println("Installed to: "*pathof(Obliqua))' ||
        { echo "ERROR: using Obliqua check failed" >&2; exit 1; }
fi

if [ "$run_tests" = true ]; then
    echo "Running Obliqua's own test suite..."
    LD_LIBRARY_PATH="" julia --project=. -e 'using Pkg; Pkg.test()' ||
        { echo "ERROR: julia Pkg.test failed" >&2; exit 1; }
fi
