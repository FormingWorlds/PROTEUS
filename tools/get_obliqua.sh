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

if ! command -v julia >/dev/null 2>&1; then
    echo "ERROR: julia is not on PATH. Install Julia first (see https://github.com/FormingWorlds/Obliqua)." >&2
    exit 1
fi

ob_url="${OBLIQUA_GIT_URL:-$(python "$proteus_tools_dir/_module_pins.py" obliqua url)}"
ob_ref="${OBLIQUA_GIT_REF:-$(python "$proteus_tools_dir/_module_pins.py" obliqua ref)}"
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
    # The new clone replaces the checkout only once it is at the pinned commit.
    tmp_clone="${dest%/}.tmp.$$"
    rm -rf "$tmp_clone"
    trap 'rm -rf "$tmp_clone"' EXIT

    echo "Cloning Obliqua ($ob_url @ $ob_ref) into $dest..."
    git clone "$ob_url" "$tmp_clone" ||
        { echo "ERROR: git clone failed$(unchanged_note "$dest")" >&2; exit 1; }

    git -C "$tmp_clone" checkout --quiet "$ob_ref" ||
        { echo "ERROR: git checkout failed$(unchanged_note "$dest")" >&2; exit 1; }

    swap_in_checkout "$tmp_clone" "$dest" || exit 1
    trap - EXIT

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
