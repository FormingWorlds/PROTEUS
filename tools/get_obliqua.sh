#!/bin/bash
# Clone Obliqua at its pin into ./Obliqua/ (or the path given) and instantiate its Julia
# project; an existing folder stays as it is, --force replaces it. --register also adds it
# to the default Julia environment, --test runs its own tests. Argument 0 has no effect.

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
    /*) ;;
    *) dest="$PWD/$dest" ;;
esac

if [ "$get_force" = true ] && { [ -e "$dest" ] || [ -L "$dest" ]; }; then
    real=$(cd "$dest" 2>/dev/null && pwd -P) || real=""
    [ -n "$real" ] || { echo "ERROR: --force replaces a folder, and $dest is not one; it is unchanged" >&2; exit 1; }
    for keep in "$proteus_root" "${HOME:-}"; do
        [ -n "$keep" ] || continue
        keep_real=$(cd "$keep" 2>/dev/null && pwd -P) || continue
        case "$keep_real/" in
            "${real%/}/"*) echo "ERROR: --force would delete $keep; $dest is unchanged" >&2; exit 1 ;;
        esac
    done
    if ! is_empty_dir "$dest"; then
        is_own_clone "$dest" ||
            { echo "ERROR: $dest holds files and is not a git clone of its own; move it away or delete it by hand; it is unchanged" >&2; exit 1; }
        grep -q '^name = "Obliqua"' "$dest/Project.toml" 2>/dev/null ||
            { echo "ERROR: $dest is a git clone that is not Obliqua; move it away or delete it by hand; it is unchanged" >&2; exit 1; }
    fi
fi

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

if [ "$get_force" != true ] && { [ -e "$dest" ] || [ -L "$dest" ]; } &&
    ! is_empty_dir "$dest"; then
    label="get_obliqua.sh"
    [ "$dest" = "$proteus_root/Obliqua" ] || label="get_obliqua.sh $(printf '%q' "$dest")"
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
