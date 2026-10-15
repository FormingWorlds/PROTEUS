#!/bin/bash
# Download and compile ThermoEngineLite

# Do we have clang?
if ! [ -x "$(command -v clang)" ]; then
  echo 'ERROR: clang is not installed.' >&2
  exit 1
fi

# Do we have pip?
if ! [ -x "$(command -v pip)" ]; then
  echo 'ERROR: pip is not installed.' >&2
  exit 1
fi

# Shared helpers: see tools/_get_common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_get_common.sh" || exit 1

# Path to PROTEUS folder
root="$proteus_root"

get_parse_args "$@"
workpath="$root/ThermoEngineLite/"

# Locate GSL headers and libraries (conda env, Homebrew, or /usr/include/gsl).
# GSL_SYS_INCLUDE lets offline unit tests hide a host /usr/include/gsl on Linux.
gsl_include=""
gsl_lib=""
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "$CONDA_PREFIX/include/gsl" ]; then
    gsl_include="$CONDA_PREFIX/include"
    gsl_lib="$CONDA_PREFIX/lib"
elif command -v brew >/dev/null 2>&1 \
    && brew_gsl=$(brew --prefix gsl 2>/dev/null) \
    && [ -n "$brew_gsl" ] \
    && [ -d "$brew_gsl/include/gsl" ]; then
    gsl_include="$brew_gsl/include"
    gsl_lib="$brew_gsl/lib"
elif [ -d "${GSL_SYS_INCLUDE:-/usr/include}/gsl" ]; then
    gsl_include="${GSL_SYS_INCLUDE:-/usr/include}"
else
    echo "ERROR: GSL headers (gsl/gsl_multiroots.h) not found." >&2
    echo "       Install GSL first, for example:" >&2
    echo "         conda install -c conda-forge gsl" >&2
    echo "         brew install gsl" >&2
    echo "         sudo apt install libgsl-dev" >&2
    exit 1
fi
export CPATH="$gsl_include${CPATH:+:$CPATH}"
if [ -n "$gsl_lib" ]; then
    export LIBRARY_PATH="$gsl_lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi

guard_dirty_checkout "$workpath" get_thermoenginelite.sh

# Make room
rm -rf "$workpath"

# Detect SSH access to GitHub.
use_ssh=$(github_use_ssh)

# Resolve the pinned URL + ref from pyproject.toml.
resolve_module_pin thermoenginelite
l_url="$module_url"
l_ref="$module_ref"

echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    uri=$(github_ssh_url "$l_url")
else
    uri="$l_url"
fi
echo "    $uri @ $l_ref -> $workpath"
git clone "$uri" "$workpath" || { echo "ERROR: git clone failed" >&2; exit 1; }
git -C "$workpath" checkout --quiet "$l_ref" \
    || { echo "ERROR: cannot checkout $l_ref" >&2; exit 1; }

# Compile ThermoEngine
echo "Compiling ThermoEngineLite..."
echo "    This will take ~20 minutes to complete"
cd "$workpath" || exit 1
if ! make devinstall; then
    echo "ERROR: ThermoEngineLite failed to compile (make devinstall failed)." >&2
    exit 1
fi

# Verify import outside the clone so ./thermoengine does not shadow site-packages.
if ! (cd / && python -c "import thermoengine; assert getattr(thermoengine, '__file__', None)" >/dev/null 2>&1); then
    echo "ERROR: ThermoEngineLite failed to install into your Python environment." >&2
    echo "       Check the output above for errors." >&2
    exit 1
fi
